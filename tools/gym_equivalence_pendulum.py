#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Pendulum equivalence with the pinned Gymnasium (SPEC 18, ADR-0024,
ADR-0050): ``tessera/Pendulum-v1`` against Gymnasium's ``Pendulum-v1``.

Gate: rollout states of a seeded uniform float32 policy plus edge states are
injected into both, the same float32 action is applied, and:

- next state: every component within 4 ulps or 1e-12 relative;
- observation (float32): within one float32 ulp of the reference's;
- reward: ADR-0050. ``t`` is the kernel's f64 reward, read through the
  test-hooks accessor; ``g`` is Gymnasium's. Raw: ``|t - g| <= B``. Public:
  ``|f32(t) - f32(g)| <= B + ulp32(g)``. Exact-match counts of both are
  reported;
- terminated and truncated: always false.

``GATE_SEED`` and ``bound`` were committed before the first gate run. A
failure is reported; neither is changed afterwards (ADR-0050, item 4).

Needs a test-hooks build for ``_native._pendulum_rewards``.
"""

from __future__ import annotations

import json
import math
import platform
import struct
import sys

import gymnasium
import numpy as np

import tessera_rl
import tessera_rl.gym  # noqa: F401 (registers tessera/Pendulum-v1)
from tessera_rl import _native

ID = "tessera/Pendulum-v1"
REF = "Pendulum-v1"
MODES = ("next_step", "same_step", "disabled")
STATE = struct.Struct("<2d")
#: Rollout seed base of the gate: unused before it, fixed before its first run.
GATE_SEED = 51_120_001
#: Rollouts in the gate.
GATE_STATES = 100_000
K = float(np.float32(0.001))


def ulps(a: float, b: float) -> int:
    def key(v: float) -> int:
        bits = struct.unpack("<q", struct.pack("<d", v))[0]
        return -(2**63) - bits if bits < 0 else bits

    return abs(key(a) - key(b))


def close(a: float, b: float) -> bool:
    return ulps(a, b) <= 4 or abs(a - b) <= 1e-12 * abs(b)


def ulp32(x: float) -> float:
    return float(np.spacing(np.float32(abs(x))))


def angle_normalize(x: float) -> float:
    """Gymnasium's, with NumPy's floor modulo by the positive 2 pi."""
    b = 2 * np.pi
    r = math.fmod(x + np.pi, b)
    if r != 0.0 and r < 0.0:
        r += b
    elif r == 0.0:
        r = 0.0
    return r - np.pi


def transcription(s, u) -> float:
    """The kernel's reward in Python, products for squares (f32 action term)."""
    c = angle_normalize(float(s[0]))
    d = float(s[1]) * float(s[1])
    a = float(np.float32(0.001) * np.float32(u * u))
    return -((c * c + 0.1 * d) + a)


def bound(s, u, t: float) -> float:
    """ADR-0050's per-state bound ``B``."""
    c = angle_normalize(float(s[0]))
    d = float(s[1]) * float(s[1])
    uu = float(np.float32(u * u))
    a = float(np.float32(0.001) * np.float32(u * u))
    return math.ulp(c * c) + 0.1 * math.ulp(d) + math.ulp(0.1 * d) + K * ulp32(uu) + ulp32(a) + 2 * math.ulp(abs(t))


def _ref_env():
    env = gymnasium.make(REF).unwrapped
    env.reset(seed=0)
    return env


def _ref_step(env, state, action):
    env.state = np.array(state, dtype=np.float64)
    obs, reward, term, trunc, _ = env.step(np.array([action], dtype=np.float32))
    return np.array(env.state, dtype=np.float64), obs, float(reward), bool(term), bool(trunc)


def sample(n: int, seed: int):
    """``n`` (state, float32 action) pairs from seeded rollouts."""
    env = gymnasium.make(REF)
    rng = np.random.default_rng(seed)
    states, actions = [], []
    episode = 0
    env.reset(seed=seed)
    while len(states) < n:
        states.append(np.array(env.unwrapped.state, dtype=np.float64))
        action = np.float32(rng.uniform(-2.0, 2.0))
        actions.append(action)
        _, _, term, trunc, _ = env.step(np.array([action], dtype=np.float32))
        if term or trunc:
            episode += 1
            env.reset(seed=seed + episode)
    env.close()
    return np.array(states), np.array(actions, dtype=np.float32)


def edge_states():
    """Angle wrap edges, speed clips, signed zeros and the action bounds,
    with values 1 and 4 float32 ulps inside them."""
    two = np.float32(2.0)
    inside = [np.nextafter(two, np.float32(0))]
    for _ in range(3):
        inside.append(np.nextafter(inside[-1], np.float32(0)))
    us = [two, -two, np.float32(0.0), np.float32(-0.0), inside[0], -inside[0], inside[3], -inside[3], np.float32(1.0)]
    thetas = [0.0, -0.0, 1e-300, np.pi, -np.pi, np.nextafter(np.pi, 4.0), np.nextafter(np.pi, 0.0), np.nextafter(-np.pi, -4.0), np.nextafter(-np.pi, 0.0), 2 * np.pi, -2 * np.pi, 3 * np.pi, 7.0, -7.0]
    speeds = [0.0, -0.0, 8.0, -8.0, np.nextafter(8.0, 0.0), 7.9, -7.9, 1e-300]
    states, actions = [], []
    for th in thetas:
        for sp in speeds:
            for u in us:
                states.append([th, sp])
                actions.append(u)
    return np.array(states, dtype=np.float64), np.array(actions, dtype=np.float32)


def gate_set(n: int, seed: int):
    states, actions = sample(n, seed)
    edge, edge_actions = edge_states()
    return np.concatenate([states, edge]), np.concatenate([actions, edge_actions])


def reference(states, actions):
    env = _ref_env()
    out = [_ref_step(env, s, a) for s, a in zip(states, actions, strict=True)]
    env.close()
    return out


def kernel_rewards(states, actions):
    rows = b"".join(struct.pack("<ddf", float(s[0]), float(s[1]), float(a)) for s, a in zip(states, actions, strict=True))
    return np.frombuffer(_native._pendulum_rewards(rows), dtype="<f8")


def one_step(states, actions, mode: str, batch: int = 1000):
    """(next state, obs, public reward, terminated, truncated) per state."""
    env = tessera_rl.make(ID, num_envs=batch, seed=1, outputs="owned", auto_reset=mode, max_episode_steps=0)
    native = env._native
    native.reset(1)
    templates = [native.snapshot(i)[: -STATE.size] for i in range(batch)]
    results = []
    for start in range(0, len(states), batch):
        chunk = states[start : start + batch]
        acts = np.zeros(batch, dtype="<f4")
        acts[: len(chunk)] = actions[start : start + batch]
        for i in range(batch):
            state = chunk[i] if i < len(chunk) else chunk[0]
            native.import_record(i, templates[i] + STATE.pack(*state))
        out = native.step(acts.tobytes())
        obs = np.frombuffer(out["observations"][0], dtype="<f4").reshape(batch, 3)
        rewards = np.frombuffer(out["rewards"], dtype="<f4")
        term = np.frombuffer(out["episode_terminated"], dtype=np.bool_)
        trunc = np.frombuffer(out["episode_truncated"], dtype=np.bool_)
        for i in range(len(chunk)):
            nxt = np.array(STATE.unpack(native.snapshot(i)[-STATE.size :]))
            results.append((nxt, obs[i].copy(), float(rewards[i]), bool(term[i]), bool(trunc[i])))
    env.close()
    return results


def compare(ours, ref, states, actions, raw):
    report = {
        "states": len(ref),
        "state": 0,
        "obs": 0,
        "terminated": 0,
        "truncated": 0,
        "max_ulps": 0,
        "speed_clips": 0,
        "kernel_differs_from_public": 0,
        "kernel_differs_from_transcription": 0,
        "raw_inexact": 0,
        "raw_over_bound": 0,
        "raw_max_diff": 0.0,
        "raw_max_ratio": 0.0,
        "public_inexact": 0,
        "public_over_bound": 0,
        "public_max_ratio": 0.0,
    }
    for row, (n_ref, o_ref, g, t_ref, tr_ref), s, a, t in zip(ours, ref, states, actions, raw, strict=True):
        n_ours, o_ours, public, t_ours, tr_ours = row
        t = float(t)
        report["speed_clips"] += int(abs(n_ref[1]) == 8.0)
        for x, y in zip(n_ours, n_ref, strict=True):
            report["max_ulps"] = max(report["max_ulps"], ulps(float(x), float(y)))
            if not close(float(x), float(y)):
                report["state"] += 1
                break
        if not np.all(np.abs(o_ours - o_ref) <= np.spacing(np.abs(o_ref).astype(np.float32))):
            report["obs"] += 1
        report["terminated"] += int(t_ours != t_ref)
        report["truncated"] += int(tr_ours != tr_ref)
        report["kernel_differs_from_public"] += int(float(np.float32(t)) != public)
        report["kernel_differs_from_transcription"] += int(struct.pack("<d", t) != struct.pack("<d", transcription(s, a)))
        b = bound(s, a, t)
        diff = abs(t - g)
        if t != g:
            report["raw_inexact"] += 1
            report["raw_max_diff"] = max(report["raw_max_diff"], diff)
            report["raw_max_ratio"] = max(report["raw_max_ratio"], diff / b)
            report["raw_over_bound"] += int(diff > b)
        fg = float(np.float32(g))
        pdiff = abs(public - fg)
        if public != fg:
            report["public_inexact"] += 1
            limit = b + ulp32(g)
            report["public_max_ratio"] = max(report["public_max_ratio"], pdiff / limit)
            report["public_over_bound"] += int(pdiff > limit)
    return report


def trajectories(states, episodes: int = 1000, length: int = 50, seed: int = 7):
    """Secondary check: divergence over short replayed trajectories."""
    rng = np.random.default_rng(seed)
    starts = states[rng.choice(len(states), size=episodes, replace=False)]
    plans = rng.uniform(-2.0, 2.0, size=(episodes, length)).astype(np.float32)
    ref = _ref_env()
    env = tessera_rl.make(ID, num_envs=episodes, seed=1, outputs="owned", auto_reset="next_step", max_episode_steps=0)
    native = env._native
    native.reset(1)
    for i in range(episodes):
        native.import_record(i, native.snapshot(i)[: -STATE.size] + STATE.pack(*starts[i]))
    max_rel, max_ulps_seen, compared = 0.0, 0, 0
    current = [s.copy() for s in starts]
    for k in range(length):
        native.step(plans[:, k].astype("<f4").tobytes())
        for i in range(episodes):
            current[i] = _ref_step(ref, current[i], plans[i, k])[0]
            ours = np.array(STATE.unpack(native.snapshot(i)[-STATE.size :]))
            compared += 1
            for a, b in zip(ours, current[i], strict=True):
                max_ulps_seen = max(max_ulps_seen, ulps(float(a), float(b)))
                if b != 0:
                    max_rel = max(max_rel, abs(a - b) / abs(b))
    env.close()
    ref.close()
    return {"episodes": episodes, "length": length, "compared_steps": compared, "max_ulps": max_ulps_seen, "max_relative": max_rel}


def truncation(seeds=range(10), limit: int = 200):
    """Both truncate exactly at ``limit``; neither terminates."""
    mismatches, compared = 0, 0
    zero = np.array([0.0], dtype=np.float32)
    for seed in seeds:
        ref = gymnasium.make(REF, max_episode_steps=limit)
        ours = gymnasium.make(ID, max_episode_steps=limit)
        ref.reset(seed=seed)
        ours.reset(seed=seed)
        for k in range(1, limit + 1):
            _, _, rt, rtr, _ = ref.step(zero)
            _, _, ot, otr, _ = ours.step(zero)
            compared += 1
            expect = k == limit
            if (rt, rtr) != (False, expect) or (ot, otr) != (False, expect):
                mismatches += 1
        ref.close()
        ours.close()
    return {"episodes": len(list(seeds)), "limit": limit, "compared_steps": compared, "mismatches": mismatches}


def run(n: int = GATE_STATES, seed: int = GATE_SEED) -> dict:
    states, actions = gate_set(n, seed)
    ref = reference(states, actions)
    raw = kernel_rewards(states, actions)
    return {
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "tessera_rl": tessera_rl.__version__,
        "edge_states": len(states) - n,
        "seed": seed,
        "one_step": {m: compare(one_step(states, actions, m), ref, states, actions, raw) for m in MODES},
        "trajectories": trajectories(states[:n]),
        "truncation": truncation(),
    }


KEYS = ("state", "obs", "terminated", "truncated", "kernel_differs_from_public", "kernel_differs_from_transcription", "raw_over_bound", "public_over_bound")


def gate_passes(report: dict) -> bool:
    one = all(all(r[k] == 0 for k in KEYS) for r in report["one_step"].values())
    return one and report["truncation"]["mismatches"] == 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else GATE_STATES
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else GATE_SEED
    result = run(n, seed)
    print(json.dumps(result, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    sys.exit(0 if gate_passes(result) else 1)
