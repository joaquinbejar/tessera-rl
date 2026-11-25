# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tier T3b of the learning protocols (#370, design
docs/reviews/370/DESIGN-T3b-r1.md, approved in REVIEW-T3b-r1): masked PPO
for one learner against ADR-0079 heuristic slots, over the native ``Env``.

The board games declare only PettingZoo AEC, which refuses the learner
filter (``visible="learner"``, #538), so this tier drives ``tessera_rl.make`` directly
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
- Several discrete heads (``MultiDiscrete``, #571 for Survival-v1): a
  factorized policy, one masked Categorical per head over that head's block
  of the mask row (``Env.mask_heads``); an action's log-probability and
  entropy are the sums over heads. A row is live when every head has a
  legal choice. Conditional heads (a head with parents) are refused. One
  head keeps learning_masked's agent and the single-head path unchanged.

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


def action_heads(env) -> list[tuple[int, int]]:
    """Each discrete head's ``(offset, choices)`` in an agent's mask row, in
    head order, checked once (#571): no head has parents, and every block
    lies inside the mask row. Without masks the blocks follow ``action_nvec``."""
    nvec = tuple(env.action_nvec)
    if not nvec:
        raise ValueError(f"{env!r}: T3b needs discrete actions")
    if not env.mask_heads:
        offsets = np.concatenate(([0], np.cumsum(nvec)[:-1])).tolist()
        return [(int(o), int(c)) for o, c in zip(offsets, nvec, strict=True)]
    heads = []
    for offset, parents, choices in env.mask_heads:
        if parents:
            raise ValueError(f"T3b refuses conditional action heads (head at {offset} has parents {parents})")
        if offset < 0 or offset + choices > env._mask_len:
            raise ValueError(f"the head at {offset} with {choices} choices lies outside the mask row of {env._mask_len}")
        heads.append((int(offset), int(choices)))
    if [c for _, c in heads] != list(nvec):
        raise ValueError(f"mask heads {heads} do not match action_nvec {nvec}")
    return heads


def head_masks(mask: np.ndarray, heads: list[tuple[int, int]]) -> list[np.ndarray]:
    """``mask``'s per-head blocks, ``[..., choices]`` each."""
    return [mask[..., o : o + c] for o, c in heads]


def live_heads(mask: np.ndarray, heads: list[tuple[int, int]]) -> np.ndarray:
    """Rows where every head has at least one legal choice."""
    return np.logical_and.reduce([m.any(-1) for m in head_masks(mask, heads)])


def uniform_legal_heads(rng: np.random.Generator, mask: np.ndarray, heads: list[tuple[int, int]]) -> list[int]:
    """One uniform legal choice per head, drawn in head order from ``rng``."""
    return [uniform_legal(rng, m) for m in head_masks(mask, heads)]


def _multihead_agent_class():
    """The factorized masked agent for several heads (#571): learning_masked's
    network shape, with an actor output of ``sum(choices)`` logits."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    from torch.distributions.categorical import Categorical  # noqa: PLC0415

    def layer(linear, std=float(np.sqrt(2)), bias=0.0):
        nn.init.orthogonal_(linear.weight, std)
        nn.init.constant_(linear.bias, bias)
        return linear

    class MultiHeadAgent(nn.Module):
        def __init__(self, obs_dim: int, heads: list[tuple[int, int]]):
            super().__init__()
            self.heads = list(heads)
            width = sum(c for _, c in heads)
            self.critic = nn.Sequential(
                layer(nn.Linear(obs_dim, 64)), nn.Tanh(),
                layer(nn.Linear(64, 64)), nn.Tanh(),
                layer(nn.Linear(64, 1), std=1.0),
            )  # fmt: skip
            self.actor = nn.Sequential(
                layer(nn.Linear(obs_dim, 64)), nn.Tanh(),
                layer(nn.Linear(64, 64)), nn.Tanh(),
                layer(nn.Linear(64, width), std=0.01),
            )  # fmt: skip

        def value(self, x):
            return self.critic(x)

        def _blocks(self, t):
            out, start = [], 0
            for _, c in self.heads:
                out.append(t[..., start : start + c])
                start += c
            return out

        def act(self, x, mask, action=None):
            """``mask`` is the concatenation of the heads' blocks. With
            ``action`` given (the update), every row must be live: a head
            with no legal choice would add a -inf log-probability."""
            logits = self.actor(x)
            parts = list(zip(self._blocks(logits), self._blocks(mask), strict=True))
            if action is not None:
                assert all(bool(m.any(-1).all()) for _, m in parts), "a row with a head that has no legal choice reached the update"
            chosen, logp, ent = [], 0.0, 0.0
            for h, (lg, m) in enumerate(parts):
                dist = Categorical(logits=masked_logits(lg, m))
                a = dist.sample() if action is None else action[..., h]
                chosen.append(a)
                logp = logp + dist.log_prob(a)
                ent = ent + legal_entropy(lg, m)
            return torch.stack(chosen, dim=-1), logp, ent, self.critic(x)

        def greedy(self, x, mask):
            logits = self.actor(x)
            return torch.stack([greedy(lg, m) for lg, m in zip(self._blocks(logits), self._blocks(mask), strict=True)], dim=-1)

    return MultiHeadAgent


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
    if len(env.action_nvec) > 1:
        heads = action_heads(env)
        width = sum(c for _, c in heads)
        acts = np.zeros((env.num_envs, env.agents, len(heads)), dtype=np.int32)
        acts[:, learner] = learner_actions
        if opponent == "python":
            for s in range(env.agents):
                if s == learner:
                    continue
                m = slot_mask(step.masks, s, env.num_envs, width)
                for i in range(env.num_envs):
                    if step.acting[i, s]:
                        acts[i, s] = uniform_legal_heads(rng, m[i], heads)
        return acts
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


def truncation_bootstrap(terminated, truncated, terminal_values, gamma: float) -> np.ndarray:
    """The extra reward that bootstraps a time-limit truncation
    (REVIEW-T3b-head-r1, Tag item (a)): ``gamma * V(s')`` for an instance
    that was truncated and **not** terminated, where ``terminal_values`` are
    the values of the engine's terminal-observation rows (never the
    observation after the auto-reset); 0 for every other instance. It
    applies to every game: the board games never truncate before their own
    end (their step caps exceed their longest games; protocols.toml)."""
    term = np.asarray(terminated, dtype=bool)
    trunc = np.asarray(truncated, dtype=bool)
    return np.where(trunc & ~term, gamma * np.asarray(terminal_values, dtype=np.float64), 0.0)


def terminal_obs(env, learner: int) -> np.ndarray:
    """The learner's flattened terminal-observation rows of the last step,
    through the public ``Env.terminal_observations`` (#545): the engine's
    final observation of an instance that ended, before its same-step
    reset, shaped as ``Step.observations``. Its ``ArgumentError`` (no
    same_step auto-reset, or no whole-pool step yet) propagates and stops
    the run; there is no fallback."""
    return flat_obs(env.terminal_observations, learner)


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
    multi = len(probe.action_nvec) > 1
    heads = action_heads(probe) if multi else None
    if multi:
        n_actions = sum(c for _, c in heads)
    probe.close()
    learner = learner_slot(seed, seats)
    env = _make(env_id, h.num_envs, seed, opponent_map(learner, agents, "heuristic"), kwargs)
    step = env.reset()
    obs = flat_obs(step.observations, learner)
    agent = _multihead_agent_class()(obs.shape[1], heads) if multi else _agent_class()(obs.shape[1], n_actions)
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
            acting = np.asarray(step.acting[:, learner], dtype=bool) & (live_heads(mask, heads) if multi else mask.any(-1))
            x = torch.as_tensor(obs)
            with torch.no_grad():
                a, logp, _, v = agent.act(x, torch.as_tensor(mask))
            a = torch.where(torch.as_tensor(acting).view(-1, *([1] * (a.dim() - 1))), a, torch.zeros_like(a))
            out = env.step(_actions(env, learner, a.numpy().astype(np.int32), "heuristic", step, rng))
            rew = np.asarray(out.rewards[:, learner], dtype=np.float64)
            term = np.asarray(out.episode_terminated, dtype=bool)
            trunc = np.asarray(out.episode_truncated, dtype=bool)
            done = term | trunc
            if (trunc & ~term).any():
                # Bootstrap a time-limit end from the terminal observation; it
                # lands on the episode's closing (terminal) transition.
                with torch.no_grad():
                    tv = agent.value(torch.as_tensor(terminal_obs(env, learner))).flatten().numpy()
                rew = rew + truncation_bootstrap(term, trunc, tv, h.gamma)
            for i in range(h.num_envs):
                item = None
                if acting[i]:
                    action = [int(x) for x in a[i]] if multi else int(a[i])
                    item = {"obs": obs[i].copy(), "mask": mask[i].copy(), "action": action, "logprob": float(logp[i]), "value": float(v[i, 0])}
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
                rows_ = torch.as_tensor(order[start : start + minibatch])
                if multi:
                    idx = rows_[torch.as_tensor(live_heads(b_masks[rows_].numpy(), heads))]
                else:
                    idx = live_rows(rows_, b_masks)
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


#: Redraws of a randomized opening before its episode is skipped (REVIEW of
#: the deterministic-evaluation finding, condition 2).
OPENING_TRIES = 16


def openings(env_id: str, kwargs: dict, episodes: int, seed_base: int, plies: int) -> tuple[list[bytes | None], int]:
    """The randomized evaluation openings (#370 T3b; rev's ruling on the
    deterministic-evaluation finding): for episode ``i``, ``plies`` uniform
    legal plies by both sides, drawn from the episode seed, from a reset with
    seed ``seed_base + i``, returned as the instance record
    (``Env.snapshot(0)``). An opening that ends the game, or leaves the side
    to move no legal move, is redrawn from sub-seed ``(seed_base + i, try)``,
    up to ``OPENING_TRIES``; after that the episode's entry is ``None`` and
    the episode is skipped by every player kind alike. ``plies == 0`` gives
    ``None`` entries with no skips: a normal reset. Returns (records, skipped)."""
    if plies <= 0:
        return [None] * episodes, 0
    env = _make(env_id, 1, seed_base, None, kwargs)
    if len(env.action_nvec) > 1:
        env.close()
        raise ValueError(f"{env_id}: randomized openings support one discrete head")
    n_actions = env.action_choices
    records: list[bytes | None] = []
    skipped = 0
    for i in range(episodes):
        record = None
        for attempt in range(OPENING_TRIES):
            rng = np.random.default_rng([seed_base + i, attempt])
            step = env.reset(seed=seed_base + i)
            ok = True
            for _ in range(plies):
                actor = np.flatnonzero(np.asarray(step.acting[0], dtype=bool))
                if actor.size != 1:
                    ok = False
                    break
                mask = slot_mask(step.masks, int(actor[0]), 1, n_actions)[0]
                if not mask.any():
                    ok = False
                    break
                acts = np.zeros((1, env.agents), dtype=np.int32)
                acts[0, int(actor[0])] = uniform_legal(rng, mask)
                step = env.step(acts)
                if step.episode_terminated[0] or step.episode_truncated[0]:
                    ok = False
                    break
            if ok:
                actor = np.flatnonzero(np.asarray(step.acting[0], dtype=bool))
                if actor.size == 1 and slot_mask(step.masks, int(actor[0]), 1, n_actions)[0].any():
                    record = bytes(env.snapshot(0))
                    break
        if record is None:
            skipped += 1
        records.append(record)
    env.close()
    return records, skipped


def eval_scores(env_id: str, kwargs: dict, seats: list[int], player: str, opponent: str, episodes: int, seed_base: int, score: str, agent=None, opening_plies: int = 0, trace: list | None = None) -> list[float]:
    """Per-episode scores of ``player`` in the learner's seat against
    ``opponent``, ``episodes`` split evenly over ``seats``, on evaluation
    seeds ``seed_base + i``. ``player`` is ``"trained"`` (greedy ``agent``),
    ``"uniform"`` (uniform legal) or ``"heuristic"`` (the seat is itself a
    heuristic slot); ``opponent`` is ``"heuristic"`` or ``"uniform"``
    (played here). Deterministic given the seeds.

    ``opening_plies > 0`` starts episode ``i`` from the randomized opening of
    :func:`openings` for that episode, the same for every player kind and
    both seats (paired comparisons); a skipped opening skips that episode.
    ``trace``, if given, receives one tuple per episode: (seat, episode, the
    move sequence: learner actions and the heuristic slots' actions)."""
    import hashlib  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from tessera_rl import Archive  # noqa: PLC0415

    if player == "heuristic" and opponent != "heuristic":
        raise ValueError("a heuristic seat plays against heuristic slots only")
    out = []
    per_seat = [episodes // len(seats) + (1 if k < episodes % len(seats) else 0) for k in range(len(seats))]
    starts, _ = openings(env_id, kwargs, max(per_seat), seed_base, opening_plies)
    for seat, count in zip(seats, per_seat, strict=True):
        probe = _make(env_id, 1, seed_base, None, kwargs)
        agents, n_actions = probe.agents, probe.action_choices
        multi = len(probe.action_nvec) > 1
        heads = action_heads(probe) if multi else None
        if multi:
            n_actions = sum(c for _, c in heads)
        probe.close()
        mode = "self" if player == "heuristic" else ("heuristic" if opponent == "heuristic" else "python")
        env = _make(env_id, 1, seed_base, opponent_map(seat, agents, mode), kwargs)
        present = [k for k in range(count) if starts[k] is not None]
        if opening_plies > 0 and present:
            env.set_reset_source(Archive(env, [starts[k] for k in present]))
        for i in range(count):
            if opening_plies > 0 and starts[i] is None:
                continue
            rng = np.random.default_rng(seed_base + i)
            if opening_plies > 0:
                env.reset(seed=seed_base + i)
                step = env.reset_from([0], [present.index(i)])
            else:
                step = env.reset(seed=seed_base + i)
            moves = hashlib.sha256()
            total, done = 0.0, False
            while not done:
                mask = slot_mask(step.masks, seat, 1, n_actions)
                acts = [0] * len(heads) if multi else 0
                if step.acting[0, seat] and player == "trained":
                    with torch.no_grad():
                        x = torch.as_tensor(flat_obs(step.observations, seat))
                        if multi:
                            acts = [int(v) for v in agent.greedy(x, torch.as_tensor(mask))[0]]
                        else:
                            acts = int(greedy(agent.actor(x), torch.as_tensor(mask))[0])
                elif step.acting[0, seat] and player == "uniform":
                    acts = uniform_legal_heads(rng, mask[0], heads) if multi else uniform_legal(rng, mask[0])
                moves.update(np.asarray(acts if multi else [acts], dtype=np.int64).tobytes())
                step = env.step(_actions(env, seat, np.asarray([acts], dtype=np.int32), "python" if mode == "python" else "heuristic", step, rng))
                if env.heuristic_actions is not None:
                    moves.update(np.ascontiguousarray(env.heuristic_actions).tobytes())
                total += float(step.rewards[0, seat])
                done = bool(step.episode_terminated[0] or step.episode_truncated[0])
            out.append(episode_score(total, score))
            if trace is not None:
                trace.append((seat, i, moves.hexdigest()))
        if opening_plies > 0 and present:
            env.set_reset_source(None)
        env.close()
    return out
