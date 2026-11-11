#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The reference PPO loop for ``tessera/RTS-vsRush-v1`` (stage 5b,
``docs/reviews/ppo-protocols/rts.md``), adapted from CleanRL's ``ppo.py``
as ``tools/reference_training.py`` is (MIT License, Copyright (c) 2019
CleanRL developers; entry in ``THIRD_PARTY.md``), with the conditional
masked policy of ``tools/rts_policy.py``.

No run is authorized by this file. ``train`` refuses to start unless it is
given the exact authorization string of the protocol's frozen sha256, which
exists only after the owner authorizes a run; the unit tests never call it.

Seats: the 16 environments are two vector envs of 8, one with
``seat_swap`` false (side Zero) and one with it true (side One), so each
side gets half the decisions (protocol, training seats).

The entropy bonus applies to the policy's entropy estimate at the chosen
type (``rts_policy.Conditional.log_prob_entropy``): the type head's entropy
plus the target and product heads' at the sampled type. The protocol's
coefficient 0.01 multiplies that quantity.

Tensors are built on the CPU; the protocol run is a CPU run.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import gymnasium
import numpy as np
import torch
from torch import nn

import tessera_rl.gym  # noqa: F401  registers the RTS ids

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rts_policy import Agent  # noqa: E402

ENV_ID = "tessera/RTS-vsRush-v1"
SAME_STEP = gymnasium.vector.AutoresetMode.SAME_STEP


@dataclass(frozen=True)
class Hyper:
    """The protocol's frozen hyperparameters (CleanRL ``ppo.py`` defaults,
    ``num_envs`` 16 and ``num_steps`` 128)."""

    # The owner's budget B (2026-11-11): 840 whole updates of 2,048 decisions.
    total_steps: int = 1_720_320
    num_envs: int = 16
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
    size: int = 16
    max_units: int = 16
    decision_ticks: int = 4
    step_limit: int = 600


class Seats:
    """Two vector envs of ``num_envs / 2``, side Zero then side One, stepped
    together; observations, masks, rewards and flags concatenated in that
    order."""

    def __init__(self, h: Hyper, seed: int):
        if h.num_envs % 2:
            raise ValueError(f"num_envs must be even, got {h.num_envs}")
        half = h.num_envs // 2
        kw = dict(size=h.size, max_units=h.max_units, decision_ticks=h.decision_ticks, step_limit=h.step_limit)
        self.envs = [
            tessera_rl.gym.make(ENV_ID, num_envs=half, seed=seed, autoreset_mode=SAME_STEP, seat_swap=swap, **kw)
            for swap in (False, True)
        ]
        self.half = half

    def reset(self, seed: int):
        outs = [e.reset(seed=seed + k * self.half) for k, e in enumerate(self.envs)]
        return _join_obs([o for o, _ in outs]), _join_masks([i for _, i in outs])

    def step(self, actions: np.ndarray):
        outs = [e.step(actions[k * self.half : (k + 1) * self.half]) for k, e in enumerate(self.envs)]
        obs = _join_obs([o[0] for o in outs])
        reward = np.concatenate([o[1] for o in outs])
        term = np.concatenate([o[2] for o in outs])
        trunc = np.concatenate([o[3] for o in outs])
        infos = [o[4] for o in outs]
        return obs, reward, term, trunc, _join_masks(infos), infos

    def close(self):
        for e in self.envs:
            e.close()


def _join_obs(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _join_masks(infos):
    heads = [i["action_masks"] for i in infos]
    return tuple(np.concatenate([h[k] for h in heads]).astype(bool) for k in range(3))


def tensors(obs):
    return torch.as_tensor(obs["planes"]), torch.as_tensor(obs["globals"].astype(np.float32))


def gae(rewards, values, dones, next_value, next_done, gamma, lam):
    """Generalized advantage estimates over ``[T, N]``; ``dones[t]`` marks
    that the observation at ``t`` starts a new episode (as in CleanRL)."""
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(steps)):
        if t == steps - 1:
            nonterminal, next_values = 1.0 - next_done, next_value
        else:
            nonterminal, next_values = 1.0 - dones[t + 1], values[t + 1]
        delta = rewards[t] + gamma * next_values * nonterminal - values[t]
        advantages[t] = last = delta + gamma * lam * nonterminal * last
    return advantages, advantages + values


def ppo_loss(agent: Agent, h: Hyper, mb: dict) -> tuple[torch.Tensor, dict]:
    """The clipped PPO loss of one minibatch (CleanRL), with the entropy at
    the chosen type (module docs)."""
    dist, value = agent.distribution(mb["planes"], mb["globals"], mb["type_mask"], mb["target_mask"], mb["product_mask"])
    logprob, entropy = dist.log_prob_entropy(mb["actions"])
    ratio = (logprob - mb["logprobs"]).exp()
    adv = mb["advantages"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - h.clip_coef, 1 + h.clip_coef)).mean()
    v_unclipped = (value - mb["returns"]) ** 2
    v_clipped = mb["values"] + torch.clamp(value - mb["values"], -h.clip_coef, h.clip_coef)
    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - mb["returns"]) ** 2).mean()
    loss = pg - h.ent_coef * entropy.mean() + h.vf_coef * v_loss
    return loss, {"pg": float(pg.detach()), "v": float(v_loss.detach()), "entropy": float(entropy.mean().detach())}


def rollout(agent: Agent, envs: Seats, h: Hyper, state):
    """One rollout of ``num_steps`` from ``state`` = (obs, masks, done);
    returns the batch and the next state. A truncated episode's last reward
    is bootstrapped with the value of its final observation."""
    obs, masks, done = state
    T, N, U = h.num_steps, h.num_envs, h.max_units
    buf = {
        "planes": [], "globals": [], "type_mask": [], "target_mask": [], "product_mask": [],
        "actions": [], "logprobs": [], "values": [], "rewards": [], "dones": [],
    }
    for _ in range(T):
        planes, globals_ = tensors(obs)
        tm, gm, pm = (torch.as_tensor(m) for m in masks)
        with torch.no_grad():
            dist, value = agent.distribution(planes, globals_, tm, gm, pm)
            actions = dist.sample()
            logprob, _ = dist.log_prob_entropy(actions)
        for k, v in (("planes", planes), ("globals", globals_), ("type_mask", tm), ("target_mask", gm),
                     ("product_mask", pm), ("actions", actions), ("logprobs", logprob), ("values", value),
                     ("dones", torch.as_tensor(done, dtype=torch.float32))):
            buf[k].append(v)
        obs, reward, term, trunc, masks, infos = envs.step(actions.reshape(N, U * 3).numpy().astype(np.int64))
        reward = torch.as_tensor(reward, dtype=torch.float32)
        boot = np.logical_and(trunc, np.logical_not(term))
        if boot.any():
            final = _final_obs(infos, envs.half)
            idx = np.flatnonzero(boot)
            fp, fg = tensors({k: v[idx] for k, v in final.items()})
            with torch.no_grad():
                reward[torch.as_tensor(idx)] += h.gamma * agent.value(fp, fg)
        buf["rewards"].append(reward)
        done = np.logical_or(term, trunc)
    batch = {k: torch.stack(v) for k, v in buf.items()}
    assert batch["actions"].shape == (T, N, U, 3)
    return batch, (obs, masks, done)


def _final_obs(infos, half):
    """The final observations of both halves, ``[N, ...]`` per leaf. An
    instance that did not end this step has ``None`` there (Gymnasium's
    object array); it gets zeros, which no bootstrap reads."""
    parts = []
    for info in infos:
        fo = info.get("final_obs")
        if fo is None:
            fo = [None] * half
        template = next((o for o in fo if o is not None), None)
        if template is None:
            parts.append(None)
            continue
        parts.append({k: np.stack([o[k] if o is not None else np.zeros_like(template[k]) for o in fo]) for k in ("planes", "globals")})
    filled = next(p for p in parts if p is not None)
    parts = [p if p is not None else {k: np.zeros((half, *v.shape[1:]), v.dtype) for k, v in filled.items()} for p in parts]
    return _join_obs(parts)


def update(agent: Agent, optimizer, h: Hyper, batch: dict, next_value, next_done, rng) -> dict:
    """One PPO update over a rollout (``update_epochs`` × ``num_minibatches``)."""
    advantages, returns = gae(batch["rewards"], batch["values"], batch["dones"], next_value, next_done,
                              h.gamma, h.gae_lambda)
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in batch.items()}
    flat["advantages"], flat["returns"] = advantages.reshape(-1), returns.reshape(-1)
    size = flat["advantages"].shape[0]
    minibatch = size // h.num_minibatches
    stats = {}
    for _ in range(h.update_epochs):
        order = rng.permutation(size)
        for start in range(0, size, minibatch):
            idx = torch.as_tensor(order[start : start + minibatch])
            loss, stats = ppo_loss(agent, h, {k: v[idx] for k, v in flat.items()})
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), h.max_grad_norm)
            optimizer.step()
    return stats


def lr_at(u: int, updates: int, h: Hyper) -> float:
    """The learning rate of update ``u`` of ``updates`` (CleanRL's linear
    anneal from ``learning_rate`` towards 0)."""
    return (1.0 - u / updates) * h.learning_rate if h.anneal_lr else h.learning_rate


def next_state(agent: Agent, state) -> tuple[torch.Tensor, torch.Tensor]:
    """The bootstrap value and done flags after a rollout, from its final
    state ``(obs, masks, done)``: V of the next observation, and whether
    that observation starts a new episode."""
    planes, globals_ = tensors(state[0])
    with torch.no_grad():
        next_value = agent.value(planes, globals_)
    return next_value, torch.as_tensor(state[2], dtype=torch.float32)


def authorize(authorization: str, frozen_sha256: str) -> None:
    """Refuses unless ``authorization`` equals the non-empty frozen sha256."""
    if not frozen_sha256 or authorization != frozen_sha256:
        raise PermissionError("no training run without the frozen protocol's sha256 as authorization")


def train(seed: int, h: Hyper, authorization: str, frozen_sha256: str) -> Agent:
    """A full training run. Refused unless ``authorization`` equals the
    protocol's frozen sha256 (``frozen_sha256``) and both are non-empty: a
    run exists only after the freeze and the owner's authorization.

    One CPU thread (``torch.set_num_threads(1)``), as the protocol's cost
    figure. A smoke run (protocol: at most 3 updates, disclosed) goes through
    here too, with ``Hyper(total_steps=3 * 2048)`` and the same authorization.
    Saving the final policy and the record of seed, Hyper and versions, and
    the evaluation, come in the next PR (the evaluation tool); this function
    returns the agent only."""
    authorize(authorization, frozen_sha256)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    envs = Seats(h, seed)
    agent = Agent(11, h.size, 2, h.max_units, 6, 81, 4)
    optimizer = torch.optim.Adam(agent.parameters(), lr=h.learning_rate, eps=1e-5)
    obs, masks = envs.reset(seed=seed)
    state = (obs, masks, np.zeros(h.num_envs, dtype=bool))
    updates = h.total_steps // (h.num_envs * h.num_steps)
    rng = np.random.default_rng(seed)
    for u in range(updates):
        optimizer.param_groups[0]["lr"] = lr_at(u, updates, h)
        batch, state = rollout(agent, envs, h, state)
        next_value, next_done = next_state(agent, state)
        update(agent, optimizer, h, batch, next_value, next_done, rng)
    envs.close()
    return agent
