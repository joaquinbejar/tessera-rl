# SPDX-License-Identifier: MIT OR Apache-2.0
"""Scripted baselines of the logistics environments in Python (#383 stage 4),
from one agent's observation and mask: the same rules as
``crates/tessera-rl-ops`` (``jobshop/baseline.rs``, ``vrp/baseline.rs``),
ties to the lowest index. ``tests/python/test_ops_baselines.py`` checks SPT
and nearest neighbour against the environments' Rust ``Heuristic``.

``main`` prints the evaluation-seed returns behind ``compat/learning/gates.toml``:
each rule and a uniform random policy on the protocol's 20 evaluation
seeds, through ``tessera_rl.gym`` as ``learning_protocols.random_scores``.
"""

from __future__ import annotations

import math

import numpy as np


def round_sqrt(v: int) -> int:
    """``round(sqrt(v))`` exactly, as ``vrp::round_sqrt``."""
    r = math.isqrt(v)
    return r + 1 if v - r * r > r else r


def _jobs(obs) -> np.ndarray:
    return np.asarray(obs["jobs"]).reshape(-1, 5)


def spt(obs, mask) -> int:
    """Shortest processing time: the legal job with the shortest next
    operation."""
    jobs = _jobs(obs)
    return min((j for j in range(len(jobs)) if mask[j]), key=lambda j: (int(jobs[j][1]), j))


def mwkr(obs, mask) -> int:
    """Most work remaining: the legal job with the most work left."""
    jobs = _jobs(obs)
    return min((j for j in range(len(jobs)) if mask[j]), key=lambda j: (-int(jobs[j][2]), j))


def lower_bound(obs) -> int:
    """A makespan lower bound at reset: the largest job's total work or
    machine's total load."""
    machines = np.asarray(obs["machines"]).reshape(-1, 2)
    return max(int(_jobs(obs)[:, 2].max()), int(machines[:, 1].max()))


def nearest(obs, mask) -> int:
    """Nearest feasible neighbour: the closest legal customer, else the
    depot."""
    nodes = np.asarray(obs["nodes"]).reshape(-1, 4)
    here = int(np.asarray(obs["vehicle"]).reshape(-1)[0])
    hx, hy = int(nodes[here][0]), int(nodes[here][1])
    legal = [n for n in range(1, len(nodes)) if mask[n]]
    if not legal:
        return 0
    return min(legal, key=lambda n: (round_sqrt((int(nodes[n][0]) - hx) ** 2 + (int(nodes[n][1]) - hy) ** 2), n))


def evaluate(name: str, policy, episodes: int = 20) -> list[float]:
    """``policy``'s return on a protocol's evaluation seeds."""
    import gymnasium  # noqa: PLC0415

    import learning_protocols as lp  # noqa: PLC0415
    import reference_training as rt  # noqa: PLC0415
    import tessera_rl.gym  # noqa: PLC0415, F401

    p = lp.protocol(name)
    env = gymnasium.make(p["env"], **p.get("env_kwargs", {}))
    out = []
    for i in range(episodes):
        obs, info = env.reset(seed=rt.EVAL_SEED_BASE + i)
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs, info["action_mask"]))
            total += float(r)
            done = term or trunc
        out.append(total)
    env.close()
    return out


def main() -> int:
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import learning_protocols as lp  # noqa: PLC0415

    for name, policies in (("jobshop", (spt, mwkr)), ("vrp", (nearest,))):
        print(f"{name}: random mean {np.mean(lp.random_scores(name, 20)):.2f}")
        for policy in policies:
            print(f"{name}: {policy.__name__} mean {np.mean(evaluate(name, policy)):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
