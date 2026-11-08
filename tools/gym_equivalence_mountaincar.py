#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Mountain car equivalence with the pinned Gymnasium (SPEC 18, ADR-0024):
``tessera/MountainCar-v1`` against Gymnasium's ``MountainCar-v0``.

Primary gate, one-step equivalence, the protocol of
``tools/gym_equivalence.py``: states sampled from Gymnasium rollouts
under a seeded random policy, plus deterministic edge states, are injected
into a Tessera instance (``import_record``: the header then the state,
``[f64; 2]`` little-endian) and into Gymnasium's unwrapped env; the same
action is applied and the results are compared:

- next state: every component within 4 ulps or 1e-12 relative;
- observation (float32): exactly the float32 of our next state; for
  same_step terminal observations, within one float32 ulp of the
  reference's;
- reward and terminated: exact, except when the reference next state lies
  within 1e-9 of the goal (position 0.5 or velocity 0 at the goal),
  counted, not gated;
- truncated: always false (no time limit in one step).

Edge states: next positions landing on the goal, 1 or 4 ulps either side
(inside the exclusion band, counted) and 2e-9 either side (gated); the left wall with leftward velocity (the stop rule); velocities
landing on each speed clip. Every auto-reset mode.

Secondary check, short trajectories (reported, not gated), and truncation
against Gymnasium's ``TimeLimit`` at 200.

Run ``python tools/gym_equivalence_mountaincar.py`` for a JSON report.
"""

from __future__ import annotations

import json
import struct
import sys

import gymnasium
import numpy as np

import tessera_rl
import tessera_rl.gym  # noqa: F401 (registers tessera/MountainCar-v1)

ID = "tessera/MountainCar-v1"
REF = "MountainCar-v0"
MODES = ("next_step", "same_step", "disabled")
STATE = struct.Struct("<2d")
GOAL = 0.5


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
    """``n`` (state, action) pairs from seeded random-policy rollouts."""
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


def _land(env, base, action, axis, target):
    """Nudges ``base[axis]`` until Gymnasium's next ``state[axis]`` is
    ``target`` bit for bit; None if no candidate lands within 400 nudges."""
    cand = base.copy()
    # Coarse: fixed-point on the position (the velocity update barely
    # depends on it), then ulp nudges.
    for _ in range(8):
        got = _ref_step(env, cand, action)[0][axis]
        cand[axis] += target - got
    for _ in range(400):
        got = _ref_step(env, cand, action)[0][axis]
        if got == target:
            return cand
        cand[axis] = np.nextafter(cand[axis], np.inf if got < target else -np.inf)
    return None


def edge_states(per_case: int = 12):
    """Deterministic edge cases: next position on the goal (0, 1 and 4 ulps
    either side), the left wall with leftward velocity, and velocities on
    each speed clip."""
    env = _ref_env()
    rng = np.random.default_rng(99)
    states, actions = [], []
    for _ in range(per_case):
        action = int(rng.integers(3))
        velocity = float(rng.uniform(0.001, 0.069))
        for offset in (-4, -1, 0, 1, 4, -2e-9, 2e-9):
            target = GOAL
            if isinstance(offset, float):
                # Just outside the 1e-9 exclusion band: gated exactly.
                target = GOAL + offset
            for _ in range(abs(offset) if isinstance(offset, int) else 0):
                target = float(np.nextafter(target, np.inf if offset > 0 else -np.inf))
            base = np.array([target - velocity, velocity])
            cand = _land(env, base, action, 0, target)
            if cand is not None:
                states.append(cand)
                actions.append(action)
        # The left wall: position close to -1.2 moving left fast.
        states.append(np.array([-1.2 + float(rng.uniform(0, 0.01)), -float(rng.uniform(0.02, 0.07))]))
        actions.append(action)
        states.append(np.array([-1.2, 0.0]))
        actions.append(0)
        # The speed clips, both signs.
        states.append(np.array([float(rng.uniform(-1.0, 0.3)), 0.07]))
        actions.append(2)
        states.append(np.array([float(rng.uniform(-1.0, 0.3)), -0.07]))
        actions.append(0)
    env.close()
    return np.array(states), np.array(actions, dtype=np.int32)


def gate_set(n: int, seed: int):
    states, actions = sample(n, seed)
    edge, edge_actions = edge_states()
    return np.concatenate([states, edge]), np.concatenate([actions, edge_actions])


def reference(states, actions):
    env = _ref_env()
    out = [_ref_step(env, s, a) for s, a in zip(states, actions, strict=True)]
    env.close()
    return out


def near_threshold(state) -> bool:
    return abs(state[0] - GOAL) < 1e-9 or (state[0] >= GOAL - 1e-9 and abs(state[1]) < 1e-9)


def injection_round_trips(states, batch: int = 64) -> bool:
    env = tessera_rl.make(ID, num_envs=batch, seed=1, outputs="owned", max_episode_steps=0)
    native = env._native
    native.reset(1)
    ok = True
    for i in range(batch):
        native.import_record(i, native.snapshot(i)[: -STATE.size] + STATE.pack(*states[i]))
        ok &= STATE.unpack(native.snapshot(i)[-STATE.size :]) == tuple(states[i])
    obs = np.frombuffer(native.outputs()["observations"][0], dtype="<f4").reshape(batch, 2)
    ok &= np.array_equal(obs, states[:batch].astype(np.float32))
    env.close()
    return bool(ok)


def one_step(states, actions, mode: str, batch: int = 1000):
    """Tessera's one-step results in ``mode``: (next state or None, obs,
    reward, terminated, truncated) per state."""
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
        obs = np.frombuffer(out["observations"][0], dtype="<f4").reshape(batch, 2)
        rewards = np.frombuffer(out["rewards"], dtype="<f4")
        term = np.frombuffer(out["episode_terminated"], dtype=np.bool_)
        trunc = np.frombuffer(out["episode_truncated"], dtype=np.bool_)
        final = None
        if mode == "same_step":
            final = np.frombuffer(out["terminal_observations"][0], dtype="<f4").reshape(batch, 2)
        for i in range(len(chunk)):
            ended = bool(term[i] or trunc[i])
            if mode == "same_step" and ended:
                results.append((None, final[i].copy(), float(rewards[i]), bool(term[i]), bool(trunc[i])))
            else:
                nxt = np.array(STATE.unpack(native.snapshot(i)[-STATE.size :]))
                results.append((nxt, obs[i].copy(), float(rewards[i]), bool(term[i]), bool(trunc[i])))
        if (start // batch) % 20 == 0:
            env.reset(seed=1)
    env.close()
    return results


def compare(ours, ref):
    report = {
        "states": len(ref),
        "state": 0,
        "obs": 0,
        "reward": 0,
        "terminated": 0,
        "truncated": 0,
        "near_threshold": 0,
        "near_threshold_terminated_agree": 0,
        "max_ulps": 0,
        "reference_terminations": 0,
        "wall_stops": 0,
        "speed_clips": 0,
        "compared_through_terminal_obs": 0,
    }
    for (n_ours, o_ours, r_ours, t_ours, tr_ours), (n_ref, o_ref, r_ref, t_ref, tr_ref) in zip(ours, ref, strict=True):
        near = near_threshold(n_ref)
        report["near_threshold"] += int(near)
        report["reference_terminations"] += int(t_ref)
        report["wall_stops"] += int(n_ref[0] == -1.2 and n_ref[1] == 0.0)
        report["speed_clips"] += int(abs(n_ref[1]) == 0.07)
        report["compared_through_terminal_obs"] += int(n_ours is None)
        if n_ours is not None:
            for a, b in zip(n_ours, n_ref, strict=True):
                report["max_ulps"] = max(report["max_ulps"], ulps(float(a), float(b)))
                if not close(float(a), float(b)):
                    report["state"] += 1
                    break
            if not np.array_equal(o_ours, np.asarray(n_ours, dtype=np.float32)):
                report["obs"] += 1
        elif not np.all(np.abs(o_ours - o_ref) <= np.spacing(np.abs(o_ref).astype(np.float32))):
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


def trajectories(states, episodes: int = 1000, length: int = 50, seed: int = 7):
    """Secondary check: divergence over short replayed trajectories."""
    rng = np.random.default_rng(seed)
    starts = states[rng.choice(len(states), size=episodes, replace=False)]
    plans = rng.integers(3, size=(episodes, length)).astype(np.int32)
    ref = _ref_env()
    env = tessera_rl.make(ID, num_envs=episodes, seed=1, outputs="owned", auto_reset="next_step", max_episode_steps=0)
    native = env._native
    native.reset(1)
    for i in range(episodes):
        native.import_record(i, native.snapshot(i)[: -STATE.size] + STATE.pack(*starts[i]))
    ref_traj = []
    for i in range(episodes):
        ref.state = starts[i].copy()
        steps = []
        for a in plans[i]:
            _, _, term, _, _ = ref.step(int(a))
            steps.append((np.array(ref.state, dtype=np.float64), term))
            if term:
                break
        ref_traj.append(steps)
    max_rel, max_ulps_seen, flag_mismatch, compared = 0.0, 0, 0, 0
    alive = np.ones(episodes, dtype=bool)
    for k in range(length):
        out = native.step(plans[:, k].astype("<i4").tobytes())
        term = np.frombuffer(out["episode_terminated"], dtype=np.bool_)
        for i in range(episodes):
            if not alive[i] or k >= len(ref_traj[i]):
                continue
            ref_state, ref_term = ref_traj[i][k]
            ours = np.array(STATE.unpack(native.snapshot(i)[-STATE.size :]))
            compared += 1
            for a, b in zip(ours, ref_state, strict=True):
                max_ulps_seen = max(max_ulps_seen, ulps(float(a), float(b)))
                if b != 0:
                    max_rel = max(max_rel, abs(a - b) / abs(b))
            if bool(term[i]) != bool(ref_term):
                flag_mismatch += 1
            if ref_term or term[i]:
                alive[i] = False
    env.close()
    ref.close()
    return {
        "episodes": episodes,
        "length": length,
        "compared_steps": compared,
        "max_ulps": max_ulps_seen,
        "max_relative": max_rel,
        "flag_mismatches": flag_mismatch,
    }


def truncation(seeds=range(10), limit: int = 200):
    """Truncation against Gymnasium's TimeLimit: the idle action from the
    same seeded reset never reaches the goal, so both truncate exactly at
    ``limit``."""
    mismatches, compared = 0, 0
    for seed in seeds:
        ref = gymnasium.make(REF, max_episode_steps=limit)
        ours = gymnasium.make(ID, max_episode_steps=limit)
        ref.reset(seed=seed)
        ours.reset(seed=seed)
        for k in range(1, limit + 1):
            _, _, rt, rtr, _ = ref.step(1)
            _, _, ot, otr, _ = ours.step(1)
            compared += 1
            expect = k == limit
            if (rt, rtr) != (False, expect) or (ot, otr) != (False, expect):
                mismatches += 1
        ref.close()
        ours.close()
    return {"episodes": len(list(seeds)), "limit": limit, "compared_steps": compared, "mismatches": mismatches}


def run(n: int = 100_000, seed: int = 2026) -> dict:
    states, actions = gate_set(n, seed)
    ref = reference(states, actions)
    return {
        "gymnasium": gymnasium.__version__,
        "tessera_rl": tessera_rl.__version__,
        "edge_states": len(states) - n,
        "seed": seed,
        "one_step": {mode: compare(one_step(states, actions, mode), ref) for mode in MODES},
        "trajectories": trajectories(states[:n]),
        "truncation": truncation(),
    }


def gate_passes(report: dict) -> bool:
    keys = ("state", "obs", "reward", "terminated", "truncated")
    one_step = all(all(r[k] == 0 for k in keys) for r in report["one_step"].values())
    return one_step and report["truncation"]["mismatches"] == 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    result = run(n)
    print(json.dumps(result, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    sys.exit(0 if gate_passes(result) else 1)
