#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Reference training (SPEC 14.2, ADR-0018): PPO on ``tessera/CartPole-v1``
through the Gymnasium vector adapter, with Gymnasium's own ``CartPole-v1``
(``SyncVectorEnv``) as the control.

Scope: this validates the adapter (SPEC 14.2, ADR-0018); it is not the
SPEC 18 learning test.

The PPO is adapted from CleanRL's ``ppo.py`` (MIT License, Copyright (c)
2019 CleanRL developers, https://github.com/vwxyzjn/cleanrl; the entry and
the license text are in ``THIRD_PARTY.md``): its structure, update and
CartPole hyperparameters. It uses only the pinned PyTorch, NumPy and
Gymnasium. Tessera ships no trainer.

Both backends run in ``SAME_STEP`` auto-reset, so no reset-only step
enters the batch. A truncated episode bootstraps its last reward with the
value of its final observation (``info["final_obs"]``); a terminated one
is terminal. That makes terminated and truncated distinguishable to the
learner.

For every seed, both backends run the same script with the same seed:

- train for ``--total-steps`` environment steps;
- evaluate the greedy policy on ``--eval-episodes`` episodes of a fresh
  single environment of the same backend (seeds ``EVAL_SEED_BASE + i``).

With fixed seeds, one thread and deterministic algorithms, a run is a
fixed number for a given platform and set of versions, not a random draw;
the seeds only stand for typical training runs. The gate (exit 1 when it
fails):

- the median across seeds of the Tessera greedy evaluation mean is at
  least ``--threshold`` (475, CartPole-v1's ``reward_threshold``);
- the difference of the across-seed means of the training-curve mean
  (the mean, over all updates, of the mean episodic return of the
  episodes that ended in that update), Tessera minus Gymnasium, is within
  ``--margin``. The greedy evaluation saturates at 500 on both backends,
  so it cannot compare them; the curve can.

Negative controls: the Tessera adapter deliberately broken (``Mutant``:
terminated and truncated swapped, rewards negated, observations of the
neighbour instance) trains on ``--mutant-seeds`` and is evaluated on the
real environment; each mutant must FAIL the same gate, or the run fails.
The gate also fails closed: Tessera, the control and every mutant must
have run, on the expected seeds (``reference_gate.py``).

The margin comes from the spread of the per-seed curve means measured on
Linux x86-64 (the CI platform) with 10 seeds per backend (the PR records
the per-seed values of both backends): standard deviation 8.3 (Tessera)
and 11.4 (Gymnasium), pooled 10.0. Over the 5 gate seeds the spread of a
difference of means is ``10.0 * sqrt(2 / 5) = 6.3``, and the margin of 20
is about three times that. The mutants land 190 or more below the control
and fail the threshold by far, so the margin separates the two cases
widely. The measured difference was -3.6 on x86-64 and +9.2 on macOS
arm64: no sign of a persistent Tessera-lower bias.

The JSON report records the seeds, versions, git commit, hyperparameters,
per-seed evaluation returns and the training curves.

Usage: reference_training.py [--seeds 1 2 3] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from reference_gate import evaluate_gate, truncation_bootstrap_mask  # noqa: E402

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


SAME_STEP = gymnasium.vector.AutoresetMode.SAME_STEP

# Negative controls: deliberately broken adapters that must fail the gate,
# so the gate is shown to detect what it claims to.
MUTANTS = ("swap_flags", "reward_sign", "obs_roll")


class Mutant(gymnasium.vector.VectorWrapper):
    """A broken adapter: ``swap_flags`` exchanges terminated and truncated,
    ``reward_sign`` negates the rewards, ``obs_roll`` gives each instance
    the observation of its neighbour."""

    def __init__(self, env, kind: str):
        super().__init__(env)
        self.kind = kind

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._obs(obs), info

    def _obs(self, obs):
        return np.roll(obs, 1, axis=0) if self.kind == "obs_roll" else obs

    def step(self, actions):
        obs, reward, term, trunc, info = self.env.step(actions)
        if self.kind == "swap_flags":
            term, trunc = trunc, term
        if self.kind == "reward_sign":
            reward = -np.asarray(reward)
        return self._obs(obs), reward, term, trunc, info


def make_vector(backend: str, num_envs: int, seed: int):
    """Both backends in SAME_STEP auto-reset: no reset-only step enters the
    batch, and the final observation of an ended episode is in
    ``info["final_obs"]``."""
    if backend == "gymnasium":
        envs = gymnasium.make_vec(
            "CartPole-v1",
            num_envs=num_envs,
            vectorization_mode="sync",
            vector_kwargs={"autoreset_mode": SAME_STEP},
        )
    else:
        envs = tessera_rl.gym.make(
            "tessera/CartPole-v1", num_envs=num_envs, seed=seed, autoreset_mode=SAME_STEP
        )
        if backend != "tessera":
            envs = Mutant(envs, backend.removeprefix("mutant:"))
    return gymnasium.wrappers.vector.RecordEpisodeStatistics(envs)


def make_single(backend: str):
    """The evaluation environment: always the backend's real environment
    (a mutant is evaluated on Tessera's)."""
    if backend == "gymnasium":
        return gymnasium.make("CartPole-v1")
    return gymnasium.make("tessera/CartPole-v1")


def train(backend: str, seed: int, h: Hyper, eval_episodes: int = 20) -> dict:
    """Trains one agent; returns its curve, evaluation and timings."""
    if h.total_steps < h.num_envs * h.num_steps:
        raise ValueError(f"total_steps {h.total_steps} is less than one rollout ({h.num_envs * h.num_steps} steps)")
    if eval_episodes <= 0:
        raise ValueError(f"eval_episodes must be positive, got {eval_episodes}")
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
            reward = torch.tensor(np.asarray(reward, dtype=np.float32))
            # A truncated episode is not terminal: bootstrap its last reward
            # with the value of its final observation. A terminated one is,
            # also when it ends on the time-limit step (both flags set).
            boot = truncation_bootstrap_mask(term, trunc)
            if boot.any():
                final = np.stack([info["final_obs"][i] for i in np.flatnonzero(boot)])
                with torch.no_grad():
                    v = agent.value(torch.tensor(final, dtype=torch.float32)).flatten()
                reward[torch.as_tensor(np.flatnonzero(boot))] += h.gamma * v
            rewards[step] = reward
            next_obs = torch.tensor(next_np, dtype=torch.float32)
            # The GAE recursion stops at every episode end.
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
    evaluation = evaluate(backend, agent, eval_episodes)
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


def evaluate(backend: str, agent: Agent, episodes: int) -> list[float]:
    env = make_single(backend)
    out = []
    for i in range(episodes):
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
    p.add_argument("--mutants", nargs="*", default=list(MUTANTS), choices=MUTANTS)
    p.add_argument("--mutant-seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--out", default="reference-training.json")
    ARGS = p.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    h = Hyper(total_steps=ARGS.total_steps)
    mutants = list(ARGS.mutants)
    plan = [(seed, b) for seed in ARGS.seeds for b in ("tessera", "gymnasium")]
    plan += [(seed, "gymnasium") for seed in ARGS.mutant_seeds if seed not in ARGS.seeds]
    plan += [(seed, f"mutant:{m}") for m in mutants for seed in ARGS.mutant_seeds]
    runs = []
    for seed, backend in plan:
        r = train(backend, seed, h, ARGS.eval_episodes)
        runs.append(r)
        sys.stderr.write(f"{backend} seed {seed}: eval mean {r['eval_mean']:.1f} in {r['train_seconds']}s\n")

    gate = evaluate_gate(runs, ARGS.threshold, ARGS.margin, ARGS.seeds, mutants, ARGS.mutant_seeds)

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
        "mutant_seeds": ARGS.mutant_seeds,
        "autoreset_mode": "same_step",
        "eval_seed_base": EVAL_SEED_BASE,
        "eval_episodes": ARGS.eval_episodes,
        "hyperparameters": asdict(h),
        "gate": gate,
        "runs": runs,
    }
    with open(ARGS.out, "w") as f:
        json.dump(report, f, indent=2)
    sys.stderr.write(json.dumps({k: gate[k] for k in ("pass", "every_mutant_fails")}) + "\n")
    sys.stderr.write(json.dumps(gate.get("tessera", gate.get("errors"))) + "\n")
    return 0 if gate["pass"] else 1


ARGS: argparse.Namespace

if __name__ == "__main__":
    sys.exit(main())
