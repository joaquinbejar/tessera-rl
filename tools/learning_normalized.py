# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tier T1n learning protocols (#530): the T1 discrete PPO loop with
observation and reward normalization, as Stable-Baselines3's ``VecNormalize``
(what RL Zoo's ``normalize: true`` means), for protocols such as MountainCar
whose working configuration needs it.

A separate module, so the T1 trainer (``reference_training``) and the T1
training path of ``learning_protocols`` stay unchanged, and with them every
calibrated T1 fingerprint. The fingerprint of a T1n protocol hashes this
module and only the sources it imports (``IMPORTED``).

- Observations: ``RunningNorm`` (clipped z-score, clip 10), updated on every
  rollout step before use, frozen for evaluation.
- Rewards: divided by the running standard deviation of the discounted
  return (``ReturnNorm``), clipped to +-10; no mean subtraction. The return
  accumulator restarts at every episode end. Evaluation scores are raw
  environment rewards.
- Truncation bootstrap: on the normalized reward, with the value of the
  normalized final observation (``bootstrap_rewards``, as tier T2).
- The observation space must be a flat ``Box``: no flattening (T1's
  ``_FlatVector``) is in this path.
- Determinism: the running statistics are float64, updated once per step
  over the instances in index order, so one seed gives the same scores twice.
- The trained parameters and the frozen observation statistics (mean,
  variance, count) are returned as ``agent`` and ``obs_norm``;
  ``learning_test`` saves them beside the score file, so a re-evaluation
  reproduces the frozen normalizer exactly.

Coupling: ``RunningNorm`` and ``bootstrap_rewards`` live in
``learning_continuous``, which tier T2 (Pendulum) hashes whole. A later edit
to either moves both the T1n and the T2 fingerprints.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np

from learning_continuous import RunningNorm, bootstrap_rewards
from reference_training import Agent, layer

#: The functions and classes this module uses from other training modules;
#: the T1n fingerprint hashes their sources, not the whole modules.
IMPORTED = (RunningNorm, bootstrap_rewards, Agent, layer)


@dataclass
class NormalizedHyper:
    """Stable-Baselines3 PPO defaults; a protocol's ``hyper`` overrides them."""

    total_steps: int = 1_000_000
    num_envs: int = 16
    num_steps: int = 16
    learning_rate: float = 3e-4
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_obs: float = 10.0
    clip_reward: float = 10.0


class ReturnNorm:
    """Running variance of the discounted return of each instance; rewards
    are divided by its standard deviation and clipped (``VecNormalize``'s
    reward scaling)."""

    def __init__(self, num_envs: int, gamma: float, clip: float):
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.stats = RunningNorm(1)
        self.gamma = gamma
        self.clip = clip

    def __call__(self, reward, done) -> np.ndarray:
        reward = np.asarray(reward, dtype=np.float64)
        self.returns = self.returns * self.gamma + reward
        self.stats.update(self.returns.reshape(-1, 1))
        scaled = reward / np.sqrt(self.stats.var[0] + 1e-8)
        self.returns[np.asarray(done, dtype=bool)] = 0.0
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)


def train(env_id: str, seed: int, h: NormalizedHyper, eval_episodes: int, eval_seed_base: int) -> dict:
    """Trains one discrete PPO agent on ``env_id`` (Tessera backend) with
    normalization and evaluates it greedily with the frozen observation
    statistics; returns the same fields as ``reference_training.train``."""
    import gymnasium  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    import tessera_rl.gym  # noqa: PLC0415, F401

    if h.total_steps < h.num_envs * h.num_steps:
        raise ValueError(f"total_steps {h.total_steps} is less than one rollout ({h.num_envs * h.num_steps} steps)")
    if eval_episodes <= 0:
        raise ValueError(f"eval_episodes must be positive, got {eval_episodes}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    envs = tessera_rl.gym.make(env_id, num_envs=h.num_envs, seed=seed, autoreset_mode=gymnasium.vector.AutoresetMode.SAME_STEP)
    space = envs.single_observation_space
    if not isinstance(space, gymnasium.spaces.Box) or len(space.shape) != 1:
        envs.close()
        raise ValueError(f"tier T1n needs a flat Box observation space, got {space}")
    envs = gymnasium.wrappers.vector.RecordEpisodeStatistics(envs)
    obs_dim = int(space.shape[0])
    agent = Agent(obs_dim, int(envs.single_action_space.n))
    optimizer = torch.optim.Adam(agent.parameters(), lr=h.learning_rate, eps=1e-5)
    norm = RunningNorm(obs_dim, clip=h.clip_obs)
    scale = ReturnNorm(h.num_envs, h.gamma, h.clip_reward)

    def prepare(raw):
        return norm.apply(np.asarray(raw, dtype=np.float32).reshape(-1, obs_dim))

    def value_of(final):
        with torch.no_grad():
            return agent.value(torch.tensor(prepare(final))).flatten().numpy()

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
    raw, _ = envs.reset(seed=seed)
    norm.update(raw)
    next_obs = torch.tensor(prepare(raw))
    next_done = torch.zeros(h.num_envs)
    episodes = 0
    curve = []
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
            raw, reward, term, trunc, info = envs.step(action.numpy())
            done = np.logical_or(term, trunc)
            scaled = scale(reward, done)
            rewards[step] = torch.tensor(bootstrap_rewards(scaled, term, trunc, info, value_of, h.gamma))
            norm.update(raw)
            next_obs = torch.tensor(prepare(raw))
            next_done = torch.tensor(done, dtype=torch.float32)
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
    norm.frozen = True
    evaluation = evaluate(env_id, agent, norm, eval_episodes, eval_seed_base)
    return {
        "backend": "tessera",
        "seed": seed,
        "episodes": episodes,
        "train_seconds": round(train_seconds, 2),
        "eval_returns": evaluation,
        "eval_mean": float(np.mean(evaluation)),
        "curve_mean": _curve_mean(curve),
        "curve": curve,
        "agent": agent,
        "obs_norm": {"mean": norm.mean.tolist(), "var": norm.var.tolist(), "count": norm.count, "clip": norm.clip},
    }


def _curve_mean(curve: list[dict]) -> float:
    values = [c["mean_return"] for c in curve if c["mean_return"] is not None]
    return float(np.mean(values)) if values else 0.0


def evaluate(env_id: str, agent, norm: RunningNorm, episodes: int, eval_seed_base: int) -> list[float]:
    """Greedy episodes on ``eval_seed_base + i`` with the frozen observation
    statistics; raw environment returns."""
    import gymnasium  # noqa: PLC0415
    import torch  # noqa: PLC0415

    import tessera_rl.gym  # noqa: PLC0415, F401

    if not norm.frozen:
        raise ValueError("evaluation needs frozen observation statistics")
    env = gymnasium.make(env_id)
    out = []
    for i in range(episodes):
        o, _ = env.reset(seed=eval_seed_base + i)
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                logits = agent.actor(torch.tensor(norm.apply(np.asarray(o, dtype=np.float32).reshape(1, -1))))
            o, r, term, trunc, _ = env.step(int(torch.argmax(logits)))
            total += float(r)
            done = term or trunc
        out.append(total)
    env.close()
    return out
