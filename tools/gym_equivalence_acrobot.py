#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Acrobot equivalence with the pinned Gymnasium (SPEC 18, ADR-0024):
``tessera/Acrobot-v1`` against Gymnasium's ``Acrobot-v1``, under the protocol of
``tools/gym_equivalence.py``.

Primary gate: states sampled from Gymnasium rollouts under a seeded random
policy are injected on both sides (``import_record``: the header then
``[f64; 4]`` little-endian), the same action is applied, and:

- next state: every component within 4 ulps or 1e-12 relative;
- observation (float32): within one float32 ulp of the reference's
  (they come from different sine and cosine implementations);
- reward and terminated: exact, except when the reference height lies
  within 1e-9 of the threshold 1.0, counted, not gated;
- truncated: always false.

The gated set is the 100,000 rollout states plus the predeclared boundary
set. A separate uniform-domain set (fixed seed and size) is reported and is
NOT gated: it has known failures at near-zero velocities, so this port is
not claimed to meet the tolerance over its whole domain. Known sources of
difference from Gymnasium: NumPy's ``sin`` and ``cos`` against the frozen
math, and scalar ``x ** 2`` through the platform ``pow`` against ``x * x``
(ADR-0008 bans ``pow`` in the kernel).

Run ``python tools/gym_equivalence_acrobot.py`` for a JSON report; the
exit status covers the gated set only.
"""

from __future__ import annotations

import json
import platform
import struct
import sys

import gymnasium
import numpy as np
from numpy import cos

import tessera_rl
import tessera_rl.gym  # noqa: F401 (registers tessera/Acrobot-v1)

ID = "tessera/Acrobot-v1"
REF = "Acrobot-v1"
MODES = ("next_step", "same_step", "disabled")
STATE = struct.Struct("<4d")
OBS = 6


def ulps(a: float, b: float) -> int:
    def key(v: float) -> int:
        bits = struct.unpack("<q", struct.pack("<d", v))[0]
        return -(2**63) - bits if bits < 0 else bits

    return abs(key(a) - key(b))


def close(a: float, b: float) -> bool:
    return ulps(a, b) <= 4 or abs(a - b) <= 1e-12 * abs(b)


def _ref_env():
    env = gymnasium.make(REF).unwrapped
    env.reset(seed=0)
    return env


def _ref_step(env, state, action):
    env.state = np.array(state, dtype=np.float64)
    obs, reward, term, trunc, _ = env.step(int(action))
    return np.array(env.state, dtype=np.float64), obs, float(reward), bool(term), bool(trunc)


def sample(n: int, seed: int):
    env = gymnasium.make(REF)
    rng = np.random.default_rng(seed)
    states, actions = [], []
    episode = 0
    env.reset(seed=seed)
    while len(states) < n:
        states.append(np.array(env.unwrapped.state, dtype=np.float64))
        action = int(rng.integers(3))
        actions.append(action)
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            episode += 1
            env.reset(seed=seed + episode)
    env.close()
    return np.array(states), np.array(actions, dtype=np.int32)


def spread(n: int, seed: int):
    """States across the whole valid domain, not only what a random policy
    visits: uniform angles and velocities, so high torques, wraps and
    velocity bounds are exercised."""
    rng = np.random.default_rng(seed)
    pi = np.pi
    states = np.column_stack(
        [
            rng.uniform(-pi, pi, n),
            rng.uniform(-pi, pi, n),
            rng.uniform(-4 * pi, 4 * pi, n),
            rng.uniform(-9 * pi, 9 * pi, n),
        ]
    )
    actions = rng.integers(3, size=n).astype(np.int32)
    # A running episode is never terminal (import validation); keep a margin
    # so the predicate cannot differ by the last ulp of the two cosines.
    keep = -cos(states[:, 0]) - cos(states[:, 1] + states[:, 0]) < 1.0 - 1e-6
    return states[keep], actions[keep]


def _bisect(f, lo, hi):
    """Adjacent doubles around the sign change of ``f`` on ``[lo, hi]``."""
    flo = f(lo) > 0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if (f(mid) > 0) == flo:
            lo = mid
        else:
            hi = mid
    return lo, hi


def boundary_states(states, actions, seed: int = 41, per_case: int = 40):
    """Predeclared deterministic edges (design section 6): pre-states whose
    next height crosses the terminal threshold 1.0 (adjacent doubles, inside
    the 1e-9 band, counted) or 1 +- 2e-9 (gated exactly); whose next angle
    crosses each wrap at +-pi; and velocities at each bound. Pre-states are
    never terminal, as import validation requires."""
    env = _ref_env()
    rng = np.random.default_rng(seed)
    pi = np.pi
    out, acts = [], []

    def add(state, action):
        if height(state) < 1.0 - 1e-6:
            out.append(np.array(state, dtype=np.float64))
            acts.append(int(action))

    def bracket(base, action, axis, measure, target):
        grid = np.linspace(-pi, pi, 257)
        values = []
        for g in grid:
            c = base.copy()
            c[axis] = g
            values.append(measure(_ref_step(env, c, action)[0]) - target)
        for k in range(len(grid) - 1):
            if values[k] * values[k + 1] < 0 and abs(values[k] - values[k + 1]) < 1.0:
                def f(x):
                    c = base.copy()
                    c[axis] = x
                    return measure(_ref_step(env, c, action)[0]) - target

                return grid[k], grid[k + 1], f
        return None

    picks = rng.choice(len(states), size=4 * per_case, replace=False)
    for i in picks[:per_case]:
        base, action = states[i].copy(), int(actions[i])
        for target in (1.0, 1.0 - 2e-9, 1.0 + 2e-9):
            found = bracket(base, action, 0, height, target)
            if found is None:
                continue
            for x in _bisect(found[2], found[0], found[1]):
                c = base.copy()
                c[0] = x
                add(c, action)
    # Wraps: the unwrapped next angle crosses +-pi, so the wrapped one jumps.
    for i in picks[per_case : 2 * per_case]:
        base, action = states[i].copy(), int(actions[i])
        base[2] = float(rng.uniform(3.0, 8.0)) * rng.choice([-1.0, 1.0])
        for axis in (0, 1):
            def signed(next_state, axis=axis):
                return float(np.sin(next_state[axis] / 2.0 - pi / 2.0))  # 0 at the seam

            grid = np.linspace(-pi, pi, 513)
            prev = None
            for g in grid:
                c = base.copy()
                c[axis] = g
                now = _ref_step(env, c, action)[0][axis]
                if prev is not None and abs(now - prev[1]) > pi:
                    def f(x, axis=axis):
                        c = base.copy()
                        c[axis] = x
                        return _ref_step(env, c, action)[0][axis]

                    side = f(prev[0]) > 0
                    lo, hi = prev[0], g
                    for _ in range(200):
                        mid = 0.5 * (lo + hi)
                        if mid in (lo, hi):
                            break
                        if (f(mid) > 0) == side:
                            lo = mid
                        else:
                            hi = mid
                    for x in (lo, hi):
                        c = base.copy()
                        c[axis] = x
                        add(c, action)
                    break
                prev = (g, now)
    # Velocity bounds, both signs, at and next to each bound.
    for i in picks[2 * per_case : 3 * per_case]:
        base, action = states[i].copy(), int(actions[i])
        for axis, bound in ((2, 4 * pi), (3, 9 * pi)):
            for sign in (1.0, -1.0):
                c = base.copy()
                c[axis] = sign * bound
                add(c, action)
                c[axis] = float(np.nextafter(c[axis], 0.0))
                add(c, action)
    env.close()
    return np.array(out), np.array(acts, dtype=np.int32)


def gate_set(n: int, seed: int):
    """The gated set of SPEC 18 and design section 6: rollouts plus edges."""
    a, aa = sample(n, seed)
    b, ba = boundary_states(a, aa)
    return np.concatenate([a, b]), np.concatenate([aa, ba])


def reference(states, actions):
    env = _ref_env()
    out = [_ref_step(env, s, a) for s, a in zip(states, actions, strict=True)]
    env.close()
    return out


def height(s) -> float:
    return float(-cos(s[0]) - cos(s[1] + s[0]))


def one_step(states, actions, mode: str, batch: int = 1000):
    env = tessera_rl.make(ID, num_envs=batch, seed=1, outputs="owned", auto_reset=mode, max_episode_steps=0)
    native = env._native
    native.reset(1)
    templates = [native.snapshot(i)[: -STATE.size] for i in range(batch)]
    results = []
    for start in range(0, len(states), batch):
        chunk = states[start : start + batch]
        acts = np.ones(batch, dtype=np.int32)
        acts[: len(chunk)] = actions[start : start + batch]
        for i in range(batch):
            state = chunk[i] if i < len(chunk) else chunk[0]
            native.import_record(i, templates[i] + STATE.pack(*state))
        out = native.step(acts.astype("<i4").tobytes())
        obs = np.frombuffer(out["observations"][0], dtype="<f4").reshape(batch, OBS)
        rewards = np.frombuffer(out["rewards"], dtype="<f4")
        term = np.frombuffer(out["episode_terminated"], dtype=np.bool_)
        trunc = np.frombuffer(out["episode_truncated"], dtype=np.bool_)
        final = None
        if mode == "same_step":
            final = np.frombuffer(out["terminal_observations"][0], dtype="<f4").reshape(batch, OBS)
        for i in range(len(chunk)):
            if mode == "same_step" and (term[i] or trunc[i]):
                results.append((None, final[i].copy(), float(rewards[i]), bool(term[i]), bool(trunc[i])))
            else:
                nxt = np.array(STATE.unpack(native.snapshot(i)[-STATE.size :]))
                results.append((nxt, obs[i].copy(), float(rewards[i]), bool(term[i]), bool(trunc[i])))
    env.close()
    return results


def compare(ours, ref):
    report = {
        "states": len(ref),
        "state": 0,
        "state_examples": [],
        "obs": 0,
        "reward": 0,
        "terminated": 0,
        "truncated": 0,
        "near_threshold": 0,
        "near_threshold_terminated_agree": 0,
        "max_ulps": 0,
        "max_relative_failing": 0.0,
        "max_absolute": 0.0,
        "max_relative": 0.0,
        "reference_terminations": 0,
        "compared_through_terminal_obs": 0,
    }
    for (n_ours, o_ours, r_ours, t_ours, tr_ours), (n_ref, o_ref, r_ref, t_ref, tr_ref) in zip(ours, ref, strict=True):
        near = abs(height(n_ref) - 1.0) < 1e-9
        report["near_threshold"] += int(near)
        report["reference_terminations"] += int(t_ref)
        report["compared_through_terminal_obs"] += int(n_ours is None)
        if n_ours is not None:
            bad = False
            for a, b in zip(n_ours, n_ref, strict=True):
                report["max_ulps"] = max(report["max_ulps"], ulps(float(a), float(b)))
                report["max_absolute"] = max(report["max_absolute"], abs(float(a) - float(b)))
                if b != 0:
                    report["max_relative"] = max(report["max_relative"], abs(float(a) - float(b)) / abs(float(b)))
                if not close(float(a), float(b)):
                    bad = True
                    rel = abs(a - b) / abs(b) if b != 0 else float("inf")
                    report["max_relative_failing"] = max(report["max_relative_failing"], float(rel))
                    if len(report["state_examples"]) < 5:
                        report["state_examples"].append([float(a), float(b), ulps(float(a), float(b))])
            report["state"] += int(bad)
        if not np.all(np.abs(o_ours - o_ref) <= np.spacing(np.abs(o_ref).astype(np.float32))):
            report["obs"] += 1
        if tr_ours != tr_ref:
            report["truncated"] += 1
        if near:
            report["near_threshold_terminated_agree"] += int(t_ours == t_ref)
            continue
        if r_ours != r_ref:
            report["reward"] += 1
        if t_ours != t_ref:
            report["terminated"] += 1
    return report


def run(n: int = 100_000, seed: int = 2026) -> dict:
    states, actions = gate_set(n, seed)
    ref = reference(states, actions)
    uniform, uniform_actions = spread(n // 5, seed + 1)
    uniform_ref = reference(uniform, uniform_actions)
    return {
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "uniform_seed": seed + 1,
        "states": len(states),
        "boundary_states": len(states) - n,
        "seed": seed,
        "one_step": {mode: compare(one_step(states, actions, mode), ref) for mode in MODES},
        "uniform_domain": {mode: compare(one_step(uniform, uniform_actions, mode), uniform_ref) for mode in MODES},
    }


def gate_passes(report: dict) -> bool:
    keys = ("state", "obs", "reward", "terminated", "truncated")
    return all(all(r[k] == 0 for k in keys) for r in report["one_step"].values())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    result = run(n)
    print(json.dumps(result, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    sys.exit(0 if gate_passes(result) else 1)
