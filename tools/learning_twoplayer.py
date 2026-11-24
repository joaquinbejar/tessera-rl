# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tier T3b of the learning protocols (#370, design
docs/reviews/370/DESIGN-T3b-r1.md, approved in REVIEW-T3b-r1): masked PPO
for one learner against ADR-0079 heuristic slots, over the native ``Env``.

The board games declare only PettingZoo AEC, which refuses
``agents="learner"``, so this tier drives ``tessera_rl.make`` directly
(``mode="sync"``, ``outputs="owned"``, ``auto_reset="same_step"``) with a
``policy_map`` that makes every slot but the learner's a heuristic. The PPO
is learning_masked's (CleanRL ``ppo.py``, masked Categorical; its helpers
are imported, not copied).

Rollout rules:

- ``Env.step`` takes a row for every slot. The learner's row comes from the
  policy, heuristic rows are zeros (the engine ignores them), and an opponent
  played in Python (the uniform-legal evaluations) gets a legal random
  action from its own mask.
- A transition is stored only on a step where the learner acts
  (``Step.acting``) with at least one legal action. The rewards the learner
  receives on the steps in between, the opponent's turns in a sequential
  game, are added **undiscounted** to its last stored transition of the same
  episode (rev, REVIEW-T3b-r1). An episode that ends on such a step marks
  that transition terminal.
- ``auto_reset="same_step"``: the step that ends an episode reports its
  rewards and flags, and its observation is already the next episode's
  first. Truncation is treated as an end, with no bootstrap: these games
  end by their own rules, and a truncated one is scored as it stands.
- The learner's observation is every leaf of its row, flattened in key
  order as float32.

Seats: two-player protocols (``seats = [0, 1]``) put the learner in slot 0
on even seeds and slot 1 on odd seeds, and evaluate half the episodes in
each seat. Tag (``seats = [0]``) keeps the learner in slot 0, a pursuer.

Score (rev, REVIEW-T3b-r1): an episode's score is its return from the
learner's seat, mapped to its sign (+1 win, 0 draw, -1 loss) when the
protocol's ``score`` is ``"sign"`` (the two-player games), or the return
itself for ``"return"`` (Tag), averaged over both seats.
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


@dataclass
class TwoPlayerHyper:
    """learning_masked's CleanRL defaults. ``total_steps`` counts stored
    learner transitions (the learner's own turns)."""

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


def learner_slot(seed: int, seats: list[int]) -> int:
    """The learner's slot for a training seed: ``seats[seed % len(seats)]``
    (even seeds slot 0, odd seeds slot 1 for two seats)."""
    return seats[seed % len(seats)]


def opponent_map(learner: int, agents: int, opponent: str) -> dict[int, str] | None:
    """The ``policy_map``: every slot but ``learner`` is a heuristic when
    ``opponent`` is ``"heuristic"``, and ``learner`` too when it is
    ``"self"`` (the heuristic against itself). ``"python"`` (opponents
    played here) leaves every slot a learner: no map."""
    if opponent == "python":
        return None
    if opponent == "self":
        return {s: "heuristic" for s in range(agents)}
    if opponent != "heuristic":
        raise ValueError(f"opponent is heuristic, self or python, not {opponent!r}")
    return {s: "heuristic" for s in range(agents) if s != learner}


def flat_obs(observations: dict, slot: int) -> np.ndarray:
    """Every leaf of ``slot``'s row, flattened in key order: ``[n, d]``
    float32."""
    leaves = [np.asarray(observations[k][:, slot], dtype=np.float32).reshape(observations[k].shape[0], -1) for k in sorted(observations)]
    return np.concatenate(leaves, axis=1)


def slot_mask(masks, slot: int, n: int, actions: int) -> np.ndarray:
    """``slot``'s mask rows ``[n, actions]`` as bool; all legal without
    masks."""
    if masks is None:
        return np.ones((n, actions), dtype=bool)
    return np.asarray(masks[:, slot, :actions], dtype=bool)


def uniform_legal(rng: np.random.Generator, mask: np.ndarray) -> int:
    """A uniformly drawn legal action (0 when none is legal)."""
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal)) if legal.size else 0


class Transitions:
    """Per-instance transition lists with the reward folding of the module
    docstring. Pure Python, so the bookkeeping is tested without torch."""

    def __init__(self, n: int):
        self.lists: list[list[dict]] = [[] for _ in range(n)]
        self._open = [False] * n

    def __len__(self) -> int:
        return sum(len(t) for t in self.lists)

    def record(self, i: int, acted: bool, reward: float, done: bool, item: dict | None) -> None:
        """One step of instance ``i``: a stored transition when the learner
        ``acted`` (``item`` holds its obs, mask, action, logprob and value),
        otherwise the reward folded into the episode's last transition."""
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


def episode_score(total: float, score: str) -> float:
    """An episode's score: its sign for ``"sign"``, the return for
    ``"return"``."""
    if score == "sign":
        return float(np.sign(total))
    if score == "return":
        return float(total)
    raise ValueError(f"score is sign or return, not {score!r}")


def _make(env_id: str, num_envs: int, seed: int, policy_map, kwargs: dict):
    import tessera_rl  # noqa: PLC0415

    return tessera_rl.make(
        env_id, num_envs=num_envs, seed=seed, workers=0, mode="sync", outputs="owned", auto_reset="same_step", policy_map=policy_map, **kwargs
    )


def _actions(env, learner: int, learner_actions, opponent: str, step, rng) -> np.ndarray:
    """The full action array: the learner's row, zeros for heuristic rows,
    and uniform-legal rows for opponents played in Python."""
    acts = np.zeros((env.num_envs, env.agents), dtype=np.int32)
    acts[:, learner] = learner_actions
    if opponent == "python":
        for s in range(env.agents):
            if s == learner:
                continue
            m = slot_mask(step.masks, s, env.num_envs, env.action_choices)
            for i in range(env.num_envs):
                if step.acting[i, s]:
                    acts[i, s] = uniform_legal(rng, m[i])
    return acts


def _check_stall(steps: int, batch: int, env_id: str, learner: int) -> None:
    """The stall guard (REVIEW-T3b-head-r1: it must fail loudly): raises
    when the rollout has taken more than ``1000 x batch`` steps without
    filling one batch of learner transitions."""
    if steps > 1000 * batch:
        raise RuntimeError(f"{env_id}: the learner in slot {learner} stopped acting (no full batch in {steps} steps)")


def train(env_id: str, kwargs: dict, seed: int, seats: list[int], h: TwoPlayerHyper) -> tuple:
    """Trains one masked PPO learner against heuristic slots; returns
    (agent, learner slot, record)."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    probe = _make(env_id, 1, seed, None, kwargs)
    agents, n_actions = probe.agents, probe.action_choices
    if len(probe.action_nvec) > 1:
        raise ValueError(f"{env_id}: T3b runs one discrete head, not {probe.action_nvec}")
    probe.close()
    learner = learner_slot(seed, seats)
    env = _make(env_id, h.num_envs, seed, opponent_map(learner, agents, "heuristic"), kwargs)
    step = env.reset()
    obs = flat_obs(step.observations, learner)
    agent = _agent_class()(obs.shape[1], n_actions)
    optimizer = torch.optim.Adam(agent.parameters(), lr=h.learning_rate, eps=1e-5)
    batch = h.num_envs * h.num_steps
    minibatch = max(1, batch // h.num_minibatches)
    updates = max(1, h.total_steps // batch)
    buf = Transitions(h.num_envs)
    totals = np.zeros(h.num_envs)
    curve, episodes = [], 0
    started = time.perf_counter()
    for update in range(1, updates + 1):
        if h.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (update - 1.0) / updates) * h.learning_rate
        buf.clear()
        finished = []
        guard = 0
        while len(buf) < batch:
            guard += 1
            _check_stall(guard, batch, env_id, learner)
            mask = slot_mask(step.masks, learner, h.num_envs, n_actions)
            acting = np.asarray(step.acting[:, learner], dtype=bool) & mask.any(-1)
            x = torch.as_tensor(obs)
            with torch.no_grad():
                a, logp, _, v = agent.act(x, torch.as_tensor(mask))
            a = torch.where(torch.as_tensor(acting), a, torch.zeros_like(a))
            out = env.step(_actions(env, learner, a.numpy().astype(np.int32), "heuristic", step, rng))
            rew = np.asarray(out.rewards[:, learner], dtype=np.float64)
            done = np.asarray(out.episode_terminated, dtype=bool) | np.asarray(out.episode_truncated, dtype=bool)
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
            obs = flat_obs(step.observations, learner)
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
    return agent, learner, {"seed": seed, "learner_slot": learner, "episodes": episodes, "train_seconds": round(time.perf_counter() - started, 2), "curve": curve}


def eval_scores(env_id: str, kwargs: dict, seats: list[int], player: str, opponent: str, episodes: int, seed_base: int, score: str, agent=None) -> list[float]:
    """Per-episode scores of ``player`` in the learner's seat against
    ``opponent``, ``episodes`` split evenly over ``seats``, on evaluation
    seeds ``seed_base + i``. ``player`` is ``"trained"`` (greedy ``agent``),
    ``"uniform"`` (uniform legal) or ``"heuristic"`` (the seat is itself a
    heuristic slot); ``opponent`` is ``"heuristic"`` or ``"uniform"``
    (played here). Deterministic given the seeds."""
    import torch  # noqa: PLC0415

    if player == "heuristic" and opponent != "heuristic":
        raise ValueError("a heuristic seat plays against heuristic slots only")
    out = []
    per_seat = [episodes // len(seats) + (1 if k < episodes % len(seats) else 0) for k in range(len(seats))]
    for seat, count in zip(seats, per_seat, strict=True):
        probe = _make(env_id, 1, seed_base, None, kwargs)
        agents, n_actions = probe.agents, probe.action_choices
        probe.close()
        mode = "self" if player == "heuristic" else ("heuristic" if opponent == "heuristic" else "python")
        env = _make(env_id, 1, seed_base, opponent_map(seat, agents, mode), kwargs)
        for i in range(count):
            rng = np.random.default_rng(seed_base + i)
            step = env.reset(seed=seed_base + i)
            total, done = 0.0, False
            while not done:
                mask = slot_mask(step.masks, seat, 1, n_actions)
                acts = 0
                if step.acting[0, seat] and player == "trained":
                    with torch.no_grad():
                        logits = agent.actor(torch.as_tensor(flat_obs(step.observations, seat)))
                    acts = int(greedy(logits, torch.as_tensor(mask))[0])
                elif step.acting[0, seat] and player == "uniform":
                    acts = uniform_legal(rng, mask[0])
                step = env.step(_actions(env, seat, np.asarray([acts], dtype=np.int32), "python" if mode == "python" else "heuristic", step, rng))
                total += float(step.rewards[0, seat])
                done = bool(step.episode_terminated[0] or step.episode_truncated[0])
            out.append(episode_score(total, score))
        env.close()
    return out
