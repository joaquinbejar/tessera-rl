# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tier T2 of the learning protocols (#370, design
docs/reviews/learning-370/DESIGN-T2-r1.md): PPO for continuous actions.

Adapted from CleanRL's ``ppo_continuous_action.py`` (MIT License, Copyright
(c) 2019 CleanRL developers, https://github.com/vwxyzjn/cleanrl; the entry is
in ``THIRD_PARTY.md``): a Gaussian actor (an MLP mean and a state-independent
log standard deviation), clipped PPO with GAE. It mirrors
``reference_training.train`` (SAME_STEP auto-reset, truncated episodes
bootstrapped from ``info["final_obs"]``, deterministic seeding, greedy
evaluation on ``EVAL_SEED_BASE + i``) and leaves that file untouched, so
CartPole's calibration stays valid.

Actions are sampled unclipped and stored unclipped; they are clipped to the
action box only when sent to the environment. Evaluation uses the actor's
mean, clipped.

Observation normalization (``normalize_obs``) is optional and lives inside
the trainer, not in a wrapper: running statistics updated from each
rollout's raw observations, applied to observations and final observations
alike, and frozen for evaluation.
"""

from __future__ import annotations

import pathlib
import random
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from reference_gate import truncation_bootstrap_mask  # noqa: E402


@dataclass
class ContinuousHyper:
    """CleanRL ``ppo_continuous_action.py`` defaults."""

    total_steps: int = 1_000_000
    num_envs: int = 1
    num_steps: int = 2048
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_obs: bool = False


class RunningNorm:
    """Running mean and variance of observations (parallel algorithm),
    applied as a clipped z-score. ``frozen`` stops updates (evaluation)."""

    def __init__(self, dim: int, clip: float = 10.0):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip
        self.frozen = False

    def update(self, batch: np.ndarray) -> None:
        if self.frozen:
            return
        b = np.asarray(batch, dtype=np.float64).reshape(-1, self.mean.size)
        mean, var, n = b.mean(axis=0), b.var(axis=0), b.shape[0]
        delta = mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        self.var = (self.var * self.count + var * n + delta**2 * self.count * n / total) / total
        self.count = total

    def apply(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)


def bootstrap_rewards(reward, term, trunc, info, value_of, gamma: float):
    """Adds ``gamma * V(final_obs)`` to the reward of every instance whose
    episode was truncated and not terminated (``truncation_bootstrap_mask``);
    ``value_of`` maps a batch of final observations to values. Returns a new
    float32 array."""
    out = np.asarray(reward, dtype=np.float32).copy()
    boot = truncation_bootstrap_mask(term, trunc)
    if boot.any():
        idx = np.flatnonzero(boot)
        final = np.stack([np.asarray(info["final_obs"][i], dtype=np.float32) for i in idx])
        out[idx] += gamma * np.asarray(value_of(final), dtype=np.float32).reshape(-1)
    return out


def _agent_class():
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    from torch.distributions.normal import Normal  # noqa: PLC0415

    def layer(linear, std=float(np.sqrt(2)), bias=0.0):
        nn.init.orthogonal_(linear.weight, std)
        nn.init.constant_(linear.bias, bias)
        return linear

    class GaussianAgent(nn.Module):
        def __init__(self, obs_dim: int, act_dim: int):
            super().__init__()
            self.critic = nn.Sequential(
                layer(nn.Linear(obs_dim, 64)), nn.Tanh(),
                layer(nn.Linear(64, 64)), nn.Tanh(),
                layer(nn.Linear(64, 1), std=1.0),
            )  # fmt: skip
            self.actor_mean = nn.Sequential(
                layer(nn.Linear(obs_dim, 64)), nn.Tanh(),
                layer(nn.Linear(64, 64)), nn.Tanh(),
                layer(nn.Linear(64, act_dim), std=0.01),
            )  # fmt: skip
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        def value(self, x):
            return self.critic(x)

        def act(self, x, action=None):
            mean = self.actor_mean(x)
            dist = Normal(mean, self.actor_logstd.expand_as(mean).exp())
            if action is None:
                action = dist.sample()
            return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(x)

    return GaussianAgent


def train(env_id: str, seed: int, h: ContinuousHyper, eval_episodes: int, eval_seed_base: int) -> dict:
    """Trains one Gaussian PPO agent on ``env_id`` (Tessera backend) and
    evaluates its mean action; returns the same fields as
    ``reference_training.train``."""
    import gymnasium  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    import tessera_rl.gym  # noqa: PLC0415

    if h.total_steps < h.num_envs * h.num_steps:
        raise ValueError(f"total_steps {h.total_steps} is less than one rollout ({h.num_envs * h.num_steps} steps)")
    if eval_episodes <= 0:
        raise ValueError(f"eval_episodes must be positive, got {eval_episodes}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    envs = tessera_rl.gym.make(env_id, num_envs=h.num_envs, seed=seed, autoreset_mode=gymnasium.vector.AutoresetMode.SAME_STEP)
    envs = gymnasium.wrappers.vector.RecordEpisodeStatistics(envs)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_space = envs.single_action_space
    act_dim = int(np.prod(act_space.shape))
    low, high = act_space.low, act_space.high
    agent = _agent_class()(obs_dim, act_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=h.learning_rate, eps=1e-5)
    norm = RunningNorm(obs_dim)

    def prepare(raw):
        raw = np.asarray(raw, dtype=np.float32).reshape(-1, obs_dim)
        return norm.apply(raw) if h.normalize_obs else raw

    batch = h.num_envs * h.num_steps
    minibatch = batch // h.num_minibatches
    updates = h.total_steps // batch
    obs = torch.zeros((h.num_steps, h.num_envs, obs_dim))
    actions = torch.zeros((h.num_steps, h.num_envs, act_dim))
    logprobs = torch.zeros((h.num_steps, h.num_envs))
    rewards = torch.zeros((h.num_steps, h.num_envs))
    dones = torch.zeros((h.num_steps, h.num_envs))
    values = torch.zeros((h.num_steps, h.num_envs))

    started = time.perf_counter()
    raw, _ = envs.reset(seed=seed)
    if h.normalize_obs:
        norm.update(raw)
    next_obs = torch.tensor(prepare(raw))
    next_done = torch.zeros(h.num_envs)
    episodes = 0
    curve = []
    rng = np.random.default_rng(seed)

    def value_of(final):
        with torch.no_grad():
            return agent.value(torch.tensor(prepare(final))).flatten().numpy()

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
            # Clipped only at the environment boundary; stored unclipped.
            raw, reward, term, trunc, info = envs.step(np.clip(action.numpy(), low, high))
            if h.normalize_obs:
                norm.update(raw)
            rewards[step] = torch.tensor(bootstrap_rewards(reward, term, trunc, info, value_of, h.gamma))
            next_obs = torch.tensor(prepare(raw))
            next_done = torch.tensor(np.logical_or(term, trunc), dtype=torch.float32)
            if "episode" in info:
                mask = info["_episode"]
                finished.extend(float(r) for r in np.asarray(info["episode"]["r"])[mask])
        episodes += len(finished)
        curve.append({"update": update, "steps": update * batch, "episodes": len(finished), "mean_return": float(np.mean(finished)) if finished else None})

        with torch.no_grad():
            next_value = agent.value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            last = 0
            for t in reversed(range(h.num_steps)):
                if t == h.num_steps - 1:
                    nonterminal, next_values = 1.0 - next_done, next_value
                else:
                    nonterminal, next_values = 1.0 - dones[t + 1], values[t + 1]
                delta = rewards[t] + h.gamma * next_values * nonterminal - values[t]
                advantages[t] = last = delta + h.gamma * h.gae_lambda * nonterminal * last
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_actions = actions.reshape((-1, act_dim))
        b_logprobs, b_advantages = logprobs.reshape(-1), advantages.reshape(-1)
        b_returns, b_values = returns.reshape(-1), values.reshape(-1)
        for _ in range(h.update_epochs):
            order = rng.permutation(batch)
            for start in range(0, batch, minibatch):
                idx = torch.as_tensor(order[start : start + minibatch])
                _, newlogprob, entropy, newvalue = agent.act(b_obs[idx], b_actions[idx])
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
    env = gymnasium.make(env_id)
    evaluation = []
    for i in range(eval_episodes):
        o, _ = env.reset(seed=eval_seed_base + i)
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                mean = agent.actor_mean(torch.tensor(prepare(o)))
            o, r, term, trunc, _ = env.step(np.clip(mean.numpy().reshape(act_space.shape), low, high))
            total += float(r)
            done = term or trunc
        evaluation.append(total)
    env.close()
    return {
        "backend": "tessera",
        "seed": seed,
        "episodes": episodes,
        "train_seconds": round(train_seconds, 2),
        "eval_returns": evaluation,
        "eval_mean": float(np.mean(evaluation)),
        "curve": curve,
    }
