#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The RTS learning gate's evaluation (stage 5b,
``docs/reviews/ppo-protocols/rts.md``): the final policy saved with its run
record, and its evaluation against Rush.

- ``save``/``load``: the policy at exactly B decisions (``train``'s return),
  as a state dict, beside a JSON record of the seed, the Hyper, the
  protocol's frozen sha256, the thread count and the versions.
- ``evaluate``: deterministic actions (argmax under the conditional masks)
  over the protocol's 400 evaluation seeds, 2,000,000 to 2,000,399, each
  played once: side Zero on even seeds, side One on odd seeds (two pools,
  ``seat_swap`` false and true). A game is a win only when it terminates
  with the learner's reward +1; a loss, a draw and a truncation are not.
- ``verdict``: per training seed, the Wilson 95% lower bound of its win
  rate must be at least 0.60 (owner, 2026-11-11), and every one of the five
  seeds must pass (owner, 2026-11-11).

No evaluation is authorized by this file: ``evaluate`` refuses unless given
the frozen protocol's sha256, as ``train`` does, because the evaluation
seeds are used once. The tests never reach an evaluation loop.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import platform
import sys

import numpy as np
import torch

import tessera_rl
import tessera_rl.gym  # noqa: F401  registers the RTS ids

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rts_policy import Agent  # noqa: E402
from rts_training import ENV_ID, Hyper, authorize, tensors  # noqa: E402

EVAL_SEEDS = range(2_000_000, 2_000_400)
WILSON_Z = 1.959963984540054
GATE = 0.60
TRAINING_SEEDS = 5


def wilson_lower(k: int, n: int, z: float = WILSON_Z) -> float:
    """The lower bound of the Wilson score interval of ``k`` successes in ``n``."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d


def seed_passes(wins: int, n: int = len(EVAL_SEEDS)) -> bool:
    return wilson_lower(wins, n) >= GATE


def verdict(wins_per_seed: dict[int, int], n: int = len(EVAL_SEEDS)) -> dict:
    """The protocol's pass rule: every training seed passes."""
    if sorted(wins_per_seed) != list(range(TRAINING_SEEDS)):
        raise ValueError(f"need the wins of training seeds 0..{TRAINING_SEEDS - 1}, got {sorted(wins_per_seed)}")
    per = {s: {"wins": w, "n": n, "lower": wilson_lower(w, n), "passes": seed_passes(w, n)}
           for s, w in sorted(wins_per_seed.items())}  # fmt: skip
    return {"per_seed": per, "passes": all(v["passes"] for v in per.values())}


def side_of(seed: int) -> bool:
    """``seat_swap`` for an evaluation seed: side Zero on even, side One on odd."""
    return seed % 2 == 1


def outcome(terminated: bool, truncated: bool, reward: float) -> str:
    if terminated and reward > 0:
        return "win"
    if terminated and reward < 0:
        return "loss"
    if terminated:
        return "draw"
    if truncated:
        return "truncated"
    raise ValueError("the game has not ended")


def make_env(h: Hyper, seed: int):
    """One evaluation game's environment: the protocol config, the side by
    the seed's parity."""
    return tessera_rl.gym.make(ENV_ID, num_envs=1, seed=seed, seat_swap=side_of(seed), size=h.size,
                               max_units=h.max_units, decision_ticks=h.decision_ticks, step_limit=h.step_limit)  # fmt: skip


def play(agent: Agent, env, seed: int) -> str:
    """One game with deterministic actions; its outcome for the learner."""
    obs, info = env.reset(seed=seed)
    while True:
        planes, globals_ = tensors(obs)
        masks = tuple(torch.as_tensor(np.asarray(m).astype(bool)) for m in info["action_masks"])
        with torch.no_grad():
            dist, _ = agent.distribution(planes, globals_, *masks)
            actions = dist.sample(greedy=True)
        obs, reward, term, trunc, info = env.step(actions.reshape(1, -1).numpy().astype(np.int64))
        if term[0] or trunc[0]:
            return outcome(bool(term[0]), bool(trunc[0]), float(reward[0]))


def evaluate(agent: Agent, h: Hyper, authorization: str, frozen_sha256: str, seeds=EVAL_SEEDS) -> dict:
    """The final policy over the evaluation seeds, each played once. Refused
    without the frozen protocol's sha256 (the seeds are used once)."""
    authorize(authorization, frozen_sha256)
    torch.set_num_threads(1)
    agent.eval()
    games = {}
    for seed in seeds:
        env = make_env(h, seed)
        try:
            games[seed] = play(agent, env, seed)
        finally:
            env.close()
    wins = sum(1 for o in games.values() if o == "win")
    return {"games": games, "wins": wins, "n": len(games), "lower": wilson_lower(wins, len(games)),
            "passes": seed_passes(wins, len(games))}  # fmt: skip


def record(seed: int, h: Hyper, frozen_sha256: str) -> dict:
    """The run record saved beside the policy."""
    return {
        "training_seed": seed,
        "hyper": dataclasses.asdict(h),
        "protocol_sha256": frozen_sha256,
        "threads": torch.get_num_threads(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                     "tessera_rl": getattr(tessera_rl, "__version__", "unknown")},  # fmt: skip
        "platform": platform.platform(),
    }


def save(agent: Agent, path: pathlib.Path, rec: dict) -> None:
    """The policy's state dict at ``path`` and its record at ``path`` + ``.json``."""
    torch.save(agent.state_dict(), path)
    pathlib.Path(str(path) + ".json").write_text(json.dumps(rec, indent=1, sort_keys=True))


def load(path: pathlib.Path, h: Hyper) -> tuple[Agent, dict]:
    agent = Agent(11, h.size, 2, h.max_units, 6, 81, 4)
    agent.load_state_dict(torch.load(path, weights_only=True))
    return agent, json.loads(pathlib.Path(str(path) + ".json").read_text())
