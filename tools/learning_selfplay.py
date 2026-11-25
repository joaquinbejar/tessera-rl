# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tier T3c of the learning protocols (#529, design
docs/reviews/529/DESIGN-r1.md, approved in 529-review/REVIEW-design-r1):
one shared masked-PPO policy plays every seat of a cooperative, sequential
game (Lanterns-v1), over the native ``Env``.

The environment runs through ``tessera_rl.make`` (``mode="sync"``,
``outputs="owned"``, ``auto_reset="same_step"``) with no ``policy_map``:
every seat is a learner, and the same parameters choose for each one
(parameter sharing). The PPO is learning_masked's (CleanRL ``ppo.py``,
masked Categorical); its helpers are imported, not copied, and this tier's
fingerprint covers exactly them (``IMPORTED``), as T3b's does.

Rollout rules (rev, REVIEW-design-r1, Q1):

- The episode is one decision sequence across seats, a single-agent MDP
  over the acting steps. On each step the one acting seat (``Step.acting``)
  is chosen; the policy input is **that seat's own observation row only**
  (``seat_obs``), never another seat's row, and its mask is that seat's.
  Every other seat's action row is zero (the engine does not read it).
- One transition is stored per acting step with at least one legal action.
  The reward is shared, so the transition's reward is that step's team
  reward (seat 0's reward, equal to every seat's). Folding rule
  (rev, REVIEW-r1 (a)): a reward from a step where no seat acts is added,
  **undiscounted**, to the **previous** stored transition of the same
  episode, never to the next one, as in T3b; an episode that ends on such
  a step marks that transition terminal. In Lanterns every live step has
  exactly one actor, so the case does not arise there.
- ``auto_reset="same_step"``: the step that ends an episode reports its
  rewards and flags, and its observation is already the next episode's
  first. Truncation is treated as an end with no bootstrap: Lanterns ends
  by its own rules, and make()'s step cap (128) exceeds its longest legal
  game (protocols.toml), so a truncation is not expected; the run counts
  any it sees (``truncations``) instead of hiding them.
- A seat's observation is every leaf of its row, flattened in key order as
  float32.

Score: an episode's undiscounted team return, which in Lanterns is the
final firework score (0..25, ``zero_on_fuse`` false).

Evaluation (``eval_scores``), on seeds ``seed_base + i``, every seat played
by the same kind: ``"trained"`` (greedy shared policy), ``"uniform"`` (a
uniform legal action, drawn from an episode-seeded generator) or
``"heuristic"`` (every seat a heuristic slot: the engine's InfoHeuristic,
``policy::info``).
"""

from __future__ import annotations

import pathlib
import random
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from learning_masked import _agent_class, greedy, legal_entropy, live_rows, masked_logits  # noqa: E402, F401

#: The functions of ``learning_masked`` this tier runs, so its fingerprint
#: covers exactly them (not the rest of that module, which is T3a's).
IMPORTED = (masked_logits, legal_entropy, greedy, _agent_class, live_rows)

#: The player kinds of ``eval_scores``.
KINDS = ("trained", "uniform", "heuristic")


@dataclass
class SelfPlayHyper:
    """learning_masked's CleanRL defaults. ``total_steps`` counts stored
    transitions (acting steps, all seats together)."""

    total_steps: int = 1_000_000
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


def acting_seats(acting) -> np.ndarray:
    """The acting seat of each instance ``[n]``, or -1 where none acts (a
    finished board). More than one acting seat is an error: this tier is
    for sequential games."""
    a = np.asarray(acting, dtype=bool)
    count = a.sum(axis=1)
    if (count > 1).any():
        raise ValueError(f"T3c needs a sequential game: {int(count.max())} seats act on one step")
    return np.where(count == 1, np.argmax(a, axis=1), -1)


def seat_obs(observations: dict, seats) -> np.ndarray:
    """Instance ``i``'s observation row of seat ``seats[i]`` only, every
    leaf flattened in key order: ``[n, d]`` float32. A seat of -1 (none
    acting) reads seat 0's row; its output is never stored."""
    s = np.maximum(np.asarray(seats, dtype=np.int64), 0)
    n = s.shape[0]
    rows = np.arange(n)
    leaves = [np.asarray(observations[k][rows, s], dtype=np.float32).reshape(n, -1) for k in sorted(observations)]
    return np.concatenate(leaves, axis=1)


def seat_mask(masks, seats, actions: int) -> np.ndarray:
    """Instance ``i``'s mask row of seat ``seats[i]`` ``[n, actions]`` as
    bool; all false where no seat acts, all legal without masks."""
    s = np.asarray(seats, dtype=np.int64)
    n = s.shape[0]
    if masks is None:
        out = np.ones((n, actions), dtype=bool)
    else:
        out = np.asarray(masks, dtype=bool)[np.arange(n), np.maximum(s, 0), :actions].copy()
    out[s < 0] = False
    return out


def full_actions(n: int, agents: int, seats, chosen) -> np.ndarray:
    """The ``[n, agents]`` action array: ``chosen[i]`` in the acting seat's
    row, zeros everywhere else."""
    acts = np.zeros((n, agents), dtype=np.int32)
    s = np.asarray(seats, dtype=np.int64)
    live = s >= 0
    acts[np.flatnonzero(live), s[live]] = np.asarray(chosen, dtype=np.int32)[live]
    return acts


def uniform_legal(rng: np.random.Generator, mask: np.ndarray) -> int:
    """A uniformly drawn legal action (0 when none is legal)."""
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal)) if legal.size else 0


class Transitions:
    """Per-instance transition lists with the folding rule of the module
    docstring. Pure Python, so the bookkeeping is tested without torch."""

    def __init__(self, n: int):
        self.lists: list[list[dict]] = [[] for _ in range(n)]
        self._open = [False] * n

    def __len__(self) -> int:
        return sum(len(t) for t in self.lists)

    def record(self, i: int, acted: bool, reward: float, done: bool, item: dict | None) -> None:
        """One step of instance ``i``: a stored transition carrying the
        step's team ``reward`` when a seat ``acted`` (``item`` holds its
        obs, mask, action, logprob and value), otherwise the reward folded
        into the episode's last transition."""
        if acted:
            self.lists[i].append({**(item or {}), "reward": float(reward), "done": bool(done)})
            self._open[i] = True
        elif self._open[i]:
            last = self.lists[i][-1]
            last["reward"] += float(reward)
            last["done"] = last["done"] or bool(done)
        if done:
            self._open[i] = False

    def clear(self) -> None:
        self.lists = [[] for _ in self.lists]
        self._open = [False] * len(self._open)


def advantages(transitions: list[dict], last_value: float, gamma: float, lam: float) -> tuple[list[float], list[float]]:
    """GAE over one instance's transitions, in order; ``last_value`` is the
    value of the instance's current observation, used when the last
    transition is not terminal. Returns (advantages, returns)."""
    adv = [0.0] * len(transitions)
    nxt_value, last = last_value, 0.0
    for t in reversed(range(len(transitions))):
        tr = transitions[t]
        nonterminal = 0.0 if tr["done"] else 1.0
        delta = tr["reward"] + gamma * nxt_value * nonterminal - tr["value"]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
        nxt_value = tr["value"]
    return adv, [a + tr["value"] for a, tr in zip(adv, transitions, strict=True)]


def team_reward(rewards) -> np.ndarray:
    """The shared team reward of each instance: seat 0's (every seat gets
    the same in a cooperative game)."""
    return np.asarray(rewards, dtype=np.float64)[:, 0]


def _make(env_id: str, num_envs: int, seed: int, policy_map, kwargs: dict):
    import tessera_rl  # noqa: PLC0415

    return tessera_rl.make(
        env_id, num_envs=num_envs, seed=seed, workers=0, mode="sync", outputs="owned", auto_reset="same_step", policy_map=policy_map, **kwargs
    )


def _check_stall(steps: int, batch: int, env_id: str) -> None:
    """The stall guard (as T3b's, it fails loudly): raises when the rollout
    has taken more than ``1000 x batch`` steps without filling one batch."""
    if steps > 1000 * batch:
        raise RuntimeError(f"{env_id}: no seat acts any more (no full batch in {steps} steps)")


def train(env_id: str, kwargs: dict, seed: int, h: SelfPlayHyper) -> tuple:
    """Trains one shared masked PPO policy on every seat; returns (agent,
    record)."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = _make(env_id, h.num_envs, seed, None, kwargs)
    agents, n_actions = env.agents, env.action_choices
    if len(env.action_nvec) > 1:
        raise ValueError(f"{env_id}: T3c runs one discrete head, not {env.action_nvec}")
    step = env.reset()
    seats = acting_seats(step.acting)
    obs = seat_obs(step.observations, seats)
    agent = _agent_class()(obs.shape[1], n_actions)
    optimizer = torch.optim.Adam(agent.parameters(), lr=h.learning_rate, eps=1e-5)
    batch = h.num_envs * h.num_steps
    minibatch = max(1, batch // h.num_minibatches)
    updates = max(1, h.total_steps // batch)
    buf = Transitions(h.num_envs)
    totals = np.zeros(h.num_envs)
    curve, episodes, truncations = [], 0, 0
    started = time.perf_counter()
    for update in range(1, updates + 1):
        if h.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (update - 1.0) / updates) * h.learning_rate
        buf.clear()
        finished = []
        guard = 0
        while len(buf) < batch:
            guard += 1
            _check_stall(guard, batch, env_id)
            mask = seat_mask(step.masks, seats, n_actions)
            acting = mask.any(-1)
            with torch.no_grad():
                a, logp, _, v = agent.act(torch.as_tensor(obs), torch.as_tensor(mask))
            a = torch.where(torch.as_tensor(acting), a, torch.zeros_like(a))
            out = env.step(full_actions(h.num_envs, agents, seats, a.numpy()))
            rew = team_reward(out.rewards)
            term = np.asarray(out.episode_terminated, dtype=bool)
            trunc = np.asarray(out.episode_truncated, dtype=bool)
            truncations += int((trunc & ~term).sum())
            done = term | trunc
            for i in range(h.num_envs):
                item = None
                if acting[i]:
                    item = {"obs": obs[i].copy(), "mask": mask[i].copy(), "action": int(a[i]), "logprob": float(logp[i]), "value": float(v[i, 0])}
                buf.record(i, bool(acting[i]), rew[i], bool(done[i]), item)
                totals[i] += rew[i]
                if done[i]:
                    finished.append(float(totals[i]))
                    totals[i] = 0.0
            step = out
            seats = acting_seats(step.acting)
            obs = seat_obs(step.observations, seats)
        episodes += len(finished)
        curve.append({"update": update, "episodes": len(finished), "mean_return": float(np.mean(finished)) if finished else None})
        with torch.no_grad():
            last_values = agent.value(torch.as_tensor(obs)).flatten().numpy()
        rows = []
        for i, lst in enumerate(buf.lists):
            if not lst:
                continue
            adv, ret = advantages(lst, float(last_values[i]), h.gamma, h.gae_lambda)
            rows.extend({**tr, "adv": ad, "ret": rt} for tr, ad, rt in zip(lst, adv, ret, strict=True))
        b_obs = torch.as_tensor(np.stack([r["obs"] for r in rows]))
        b_masks = torch.as_tensor(np.stack([r["mask"] for r in rows]))
        b_actions = torch.as_tensor([r["action"] for r in rows])
        b_logprobs = torch.as_tensor([r["logprob"] for r in rows], dtype=torch.float32)
        b_adv = torch.as_tensor([r["adv"] for r in rows], dtype=torch.float32)
        b_returns = torch.as_tensor([r["ret"] for r in rows], dtype=torch.float32)
        b_values = torch.as_tensor([r["value"] for r in rows], dtype=torch.float32)
        size = len(rows)
        for _ in range(h.update_epochs):
            order = rng.permutation(size)
            for start in range(0, size, minibatch):
                idx = live_rows(torch.as_tensor(order[start : start + minibatch]), b_masks)
                if idx.numel() == 0:
                    continue
                _, newlogprob, entropy, newvalue = agent.act(b_obs[idx], b_masks[idx], b_actions[idx])
                ratio = (newlogprob - b_logprobs[idx]).exp()
                adv = b_adv[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8) if adv.numel() > 1 else adv
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
    env.close()
    record = {"seed": seed, "episodes": episodes, "truncations": truncations, "train_seconds": round(time.perf_counter() - started, 2), "curve": curve}
    return agent, record


def deal_digest(observations: dict) -> str:
    """A digest of every seat's observation at a reset: with two or more
    players each hand is visible to another seat, so distinct deals give
    distinct digests."""
    import hashlib  # noqa: PLC0415

    d = hashlib.sha256()
    for k in sorted(observations):
        d.update(np.ascontiguousarray(observations[k]).tobytes())
    return d.hexdigest()


def eval_scores(env_id: str, kwargs: dict, kind: str, episodes: int, seed_base: int, agent=None, trace: list | None = None) -> list[float]:
    """Per-episode final scores with every seat played by ``kind``
    (``KINDS``), on evaluation seeds ``seed_base + i``. Deterministic given
    the seeds. ``trace``, if given, receives one tuple per episode:
    (episode, deal digest, move-sequence digest, truncated)."""
    import hashlib  # noqa: PLC0415

    if kind not in KINDS:
        raise ValueError(f"kind is one of {KINDS}, not {kind!r}")
    if kind == "trained" and agent is None:
        raise ValueError("a trained evaluation needs the agent")
    probe = _make(env_id, 1, seed_base, None, kwargs)
    agents, n_actions = probe.agents, probe.action_choices
    probe.close()
    policy_map = {s: "heuristic" for s in range(agents)} if kind == "heuristic" else None
    env = _make(env_id, 1, seed_base, policy_map, kwargs)
    out = []
    for i in range(episodes):
        rng = np.random.default_rng(seed_base + i)
        step = env.reset(seed=seed_base + i)
        deal = deal_digest(step.observations)
        moves = hashlib.sha256()
        total, done, truncated = 0.0, False, False
        while not done:
            seats = acting_seats(step.acting)
            chosen = np.zeros(1, dtype=np.int32)
            if kind != "heuristic" and seats[0] >= 0:
                mask = seat_mask(step.masks, seats, n_actions)
                if kind == "trained":
                    import torch  # noqa: PLC0415

                    with torch.no_grad():
                        logits = agent.actor(torch.as_tensor(seat_obs(step.observations, seats)))
                    chosen[0] = int(greedy(logits, torch.as_tensor(mask))[0])
                else:
                    chosen[0] = uniform_legal(rng, mask[0])
            moves.update(np.asarray([seats[0], chosen[0]], dtype=np.int64).tobytes())
            step = env.step(full_actions(1, agents, seats, chosen))
            if env.heuristic_actions is not None:
                moves.update(np.ascontiguousarray(env.heuristic_actions).tobytes())
            total += float(team_reward(step.rewards)[0])
            truncated = bool(step.episode_truncated[0] and not step.episode_terminated[0])
            done = bool(step.episode_terminated[0] or step.episode_truncated[0])
        out.append(total)
        if trace is not None:
            trace.append((i, deal, moves.hexdigest(), truncated))
    env.close()
    return out
