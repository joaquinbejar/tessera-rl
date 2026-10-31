#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Reference training (SPEC 14.2, ADR-0018): PPO on ``tessera/CartPole-v1``
through the Gymnasium vector adapter, with Gymnasium's own ``CartPole-v1``
(``SyncVectorEnv``) as the control.

The PPO is a single-file implementation following the structure and the
CartPole hyperparameters of CleanRL's ``ppo.py`` (MIT License, Copyright
(c) 2019 CleanRL developers, https://github.com/vwxyzjn/cleanrl). It uses
only the pinned PyTorch, NumPy and Gymnasium. Tessera ships no trainer: this
script only validates that the adapter trains like the reference.

For every seed, both backends run the same script with the same seed:

- train for ``--total-steps`` environment steps;
- evaluate the greedy policy on ``--eval-episodes`` episodes of a fresh
  single environment of the same backend (seeds ``EVAL_SEED_BASE + i``).

The gate (exit 1 when it fails):

- every Tessera seed reaches a mean evaluation return of at least
  ``--threshold`` (475, CartPole-v1's ``reward_threshold``);
- the difference of the across-seed means of the training-curve mean
  (the mean, over all updates, of the mean episodic return of the
  episodes that ended in that update), Tessera minus Gymnasium, is within
  ``--margin``. The greedy evaluation saturates at 500 on both backends,
  so it cannot compare them; the curve can.

Seeds and margin come from a 10-seed measurement on each backend (macOS
arm64, the PR records it): the curve mean has a standard deviation
across seeds of 7.0 (Tessera) and 11.6 (Gymnasium), pooled 9.6. With 5
seeds per backend the standard error of the difference is
``9.6 * sqrt(2 / 5) = 6.1``, so a margin of 20 is about 3.3 standard
errors: a false alarm is rare (about 0.1%) while a systematic training
difference of a twentieth of the scale fails.

The JSON report records the seeds, versions, git commit, hyperparameters,
per-seed evaluation returns and the training curves (the mean episodic
return per update).

Usage: reference_training.py [--seeds 1 2 3] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import gymnasium
import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

import tessera_rl
import tessera_rl.gym  # noqa: F401  registers tessera/CartPole-v1

EVAL_SEED_BASE = 10_000


@dataclass(frozen=True)
class Hyper:
    """CleanRL ``ppo.py`` defaults for CartPole-v1."""

    total_steps: int = 500_000
    num_envs: int = 4
    num_steps: int = 128
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


def layer(linear: nn.Linear, std: float = float(np.sqrt(2)), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(linear.weight, std)
    nn.init.constant_(linear.bias, bias)
    return linear


class Agent(nn.Module):
    def __init__(self, obs_dim: int, actions: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer(nn.Linear(64, 64)), nn.Tanh(),
            layer(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer(nn.Linear(64, 64)), nn.Tanh(),
            layer(nn.Linear(64, actions), std=0.01),
        )

    def value(self, x):
        return self.critic(x)

    def act(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)


def make_vector(backend: str, num_envs: int, seed: int):
    if backend == "tessera":
        envs = tessera_rl.gym.make("tessera/CartPole-v1", num_envs=num_envs, seed=seed)
    else:
        envs = gymnasium.make_vec("CartPole-v1", num_envs=num_envs, vectorization_mode="sync")
    return gymnasium.wrappers.vector.RecordEpisodeStatistics(envs)


def make_single(backend: str):
    if backend == "tessera":
        return gymnasium.make("tessera/CartPole-v1")
    return gymnasium.make("CartPole-v1")


def train(backend: str, seed: int, h: Hyper) -> dict:
    """Trains one agent; returns its curve, evaluation and timings."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    envs = make_vector(backend, h.num_envs, seed)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    agent = Agent(obs_dim, int(envs.single_action_space.n))
    optimizer = torch.optim.Adam(agent.parameters(), lr=h.learning_rate, eps=1e-5)

    batch = h.num_envs * h.num_steps
    minibatch = batch // h.num_minibatches
    updates = h.total_steps // batch
    obs = torch.zeros((h.num_steps, h.num_envs, obs_dim))
    actions = torch.zeros((h.num_steps, h.num_envs))
    logprobs = torch.zeros((h.num_steps, h.num_envs))
    rewards = torch.zeros((h.num_steps, h.num_envs))
    dones = torch.zeros((h.num_steps, h.num_envs))
    values = torch.zeros((h.num_steps, h.num_envs))

    started = time.perf_counter()
    env_seconds = 0.0
    t0 = time.perf_counter()
    next_obs, _ = envs.reset(seed=seed)
    env_seconds += time.perf_counter() - t0
    next_obs = torch.tensor(next_obs, dtype=torch.float32)
    next_done = torch.zeros(h.num_envs)
    curve = []
    episodes = 0
    rng = np.random.default_rng(seed)

    for update in range(1, updates + 1):
        if h.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (update - 1.0) / updates) * h.learning_rate
        finished = []
        for step in range(h.num_steps):
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.act(next_obs)
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            t0 = time.perf_counter()
            next_np, reward, term, trunc, info = envs.step(action.numpy())
            env_seconds += time.perf_counter() - t0
            rewards[step] = torch.tensor(np.asarray(reward, dtype=np.float32))
            next_obs = torch.tensor(next_np, dtype=torch.float32)
            next_done = torch.tensor(np.logical_or(term, trunc), dtype=torch.float32)
            if "episode" in info:
                mask = info["_episode"]
                finished.extend(float(r) for r in np.asarray(info["episode"]["r"])[mask])
        episodes += len(finished)
        curve.append(
            {
                "update": update,
                "steps": update * batch,
                "episodes": len(finished),
                "mean_return": float(np.mean(finished)) if finished else None,
            }
        )

        with torch.no_grad():
            next_value = agent.value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            last = 0
            for t in reversed(range(h.num_steps)):
                if t == h.num_steps - 1:
                    nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    nonterminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + h.gamma * next_values * nonterminal - values[t]
                advantages[t] = last = delta + h.gamma * h.gae_lambda * nonterminal * last
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        for _ in range(h.update_epochs):
            order = rng.permutation(batch)
            for start in range(0, batch, minibatch):
                idx = torch.as_tensor(order[start:start + minibatch])
                _, newlogprob, entropy, newvalue = agent.act(b_obs[idx], b_actions.long()[idx])
                ratio = (newlogprob - b_logprobs[idx]).exp()
                adv = b_advantages[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - h.clip_coef, 1 + h.clip_coef)).mean()
                newvalue = newvalue.view(-1)
                v_unclipped = (newvalue - b_returns[idx]) ** 2
                v_clipped = b_values[idx] + torch.clamp(newvalue - b_values[idx], -h.clip_coef, h.clip_coef)
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_returns[idx]) ** 2).mean()
                loss = pg_loss - h.ent_coef * entropy.mean() + v_loss * h.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), h.max_grad_norm)
                optimizer.step()
    envs.close()
    train_seconds = time.perf_counter() - started
    evaluation = evaluate(backend, agent, h)
    return {
        "backend": backend,
        "seed": seed,
        "episodes": episodes,
        "train_seconds": round(train_seconds, 2),
        "env_seconds": round(env_seconds, 2),
        "eval_returns": evaluation,
        "eval_mean": float(np.mean(evaluation)),
        "curve_mean": curve_mean(curve),
        "curve": curve,
    }


def curve_mean(curve: list[dict]) -> float:
    """The mean over updates of the mean episodic return (updates in which
    no episode ended are skipped)."""
    values = [c["mean_return"] for c in curve if c["mean_return"] is not None]
    return float(np.mean(values)) if values else 0.0


def evaluate(backend: str, agent: Agent, h: Hyper, episodes: int | None = None) -> list[float]:
    env = make_single(backend)
    out = []
    for i in range(episodes or ARGS.eval_episodes):
        o, _ = env.reset(seed=EVAL_SEED_BASE + i)
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                logits = agent.actor(torch.tensor(np.asarray(o), dtype=torch.float32))
            o, r, term, trunc, _ = env.step(int(torch.argmax(logits)))
            total += float(r)
            done = term or trunc
        out.append(total)
    env.close()
    return out


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    global ARGS
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--total-steps", type=int, default=Hyper.total_steps)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--threshold", type=float, default=475.0)
    p.add_argument("--margin", type=float, default=20.0)
    p.add_argument("--backends", nargs="+", default=["tessera", "gymnasium"])
    p.add_argument("--out", default="reference-training.json")
    ARGS = p.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    h = Hyper(total_steps=ARGS.total_steps)
    runs = []
    for seed in ARGS.seeds:
        for backend in ARGS.backends:
            r = train(backend, seed, h)
            runs.append(r)
            sys.stderr.write(f"{backend} seed {seed}: eval mean {r['eval_mean']:.1f} in {r['train_seconds']}s\n")

    def values(backend, key):
        return [r[key] for r in runs if r["backend"] == backend]

    tessera = values("tessera", "eval_mean")
    gate = {"threshold": ARGS.threshold, "margin": ARGS.margin}
    gate["every_tessera_seed_reaches_threshold"] = bool(tessera) and min(tessera) >= ARGS.threshold
    gate["tessera_eval_mean"] = float(np.mean(tessera)) if tessera else None
    control = values("gymnasium", "eval_mean")
    gate["gymnasium_eval_mean"] = float(np.mean(control)) if control else None
    t_curve = values("tessera", "curve_mean")
    g_curve = values("gymnasium", "curve_mean")
    if t_curve and g_curve:
        gate["tessera_curve_mean"] = float(np.mean(t_curve))
        gate["gymnasium_curve_mean"] = float(np.mean(g_curve))
        gate["curve_difference"] = gate["tessera_curve_mean"] - gate["gymnasium_curve_mean"]
        gate["difference_within_margin"] = abs(gate["curve_difference"]) <= ARGS.margin
    gate["pass"] = gate["every_tessera_seed_reaches_threshold"] and gate.get("difference_within_margin", True)

    report = {
        "schema": "tessera-reference-training/1",
        "environment": {
            "git": _git("rev-parse", "HEAD"),
            "git_dirty": _git("status", "--porcelain") not in ("", "unknown"),
            "python": platform.python_version(),
            "os": platform.platform(),
            "arch": platform.machine(),
            "tessera_rl": tessera_rl.__version__,
            "gymnasium": gymnasium.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "seeds": ARGS.seeds,
        "eval_seed_base": EVAL_SEED_BASE,
        "eval_episodes": ARGS.eval_episodes,
        "hyperparameters": asdict(h),
        "gate": gate,
        "runs": runs,
    }
    with open(ARGS.out, "w") as f:
        json.dump(report, f, indent=2)
    sys.stderr.write(json.dumps(gate) + "\n")
    return 0 if gate["pass"] else 1


ARGS: argparse.Namespace

if __name__ == "__main__":
    sys.exit(main())
