#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""CartPole equivalence with the pinned Gymnasium (SPEC 18, ADR-0024).

Primary gate, one-step equivalence: states are sampled from Gymnasium
``CartPole-v1`` rollouts under a seeded random policy; each state is
injected into a Tessera instance (through ``import_record``: the canonical
record is the header then the state, ``[f64; 4]`` little-endian) and into
Gymnasium's unwrapped env, the same action is applied, and the results are
compared:

- next state: every component within 4 ulps or 1e-12 relative;
- observation (float32): exact;
- reward and terminated: exact, except when the reference next state lies
  within 1e-9 of a termination threshold (counted, not gated);
- truncated: always false here (no time limit in a single step); time
  limits are checked separately against Gymnasium's ``TimeLimit``.

The gated set adds deterministic boundary states whose reference next
``x`` or ``theta`` lands on a threshold or 1 or 4 ulps either side;
terminations are reported by cause.

It runs in each auto-reset mode. In ``same_step`` an ended instance is
reset in the same call, so its next state is compared through the
terminal observation (the step output of ADR-0039) instead.

Secondary check, short trajectories: episodes of 50 steps from injected
initial states with replayed actions, reported with the observed
divergence (not gated: the dynamics amplify one-ulp differences).

Run ``python tools/gym_equivalence.py`` for a JSON report.
"""

from __future__ import annotations

import json
import struct
import sys

import gymnasium
import numpy as np

import tessera_rl
import tessera_rl.gym  # noqa: F401 (registers tessera/CartPole-v1)

MODES = ("next_step", "same_step", "disabled")
STATE = struct.Struct("<4d")


def ulps(a: float, b: float) -> int:
    """Distance in units in the last place between two finite doubles."""

    def key(v: float) -> int:
        bits = struct.unpack("<q", struct.pack("<d", v))[0]
        return -(2**63) - bits if bits < 0 else bits

    return abs(key(a) - key(b))


def close(a: float, b: float) -> bool:
    return ulps(a, b) <= 4 or abs(a - b) <= 1e-12 * abs(b)


def sample(n: int, seed: int):
    """``n`` (state, action) pairs from seeded random-policy rollouts."""
    env = gymnasium.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    states, actions = [], []
    episode = 0
    env.reset(seed=seed)
    while len(states) < n:
        state = np.array(env.unwrapped.state, dtype=np.float64)
        action = int(rng.integers(2))
        states.append(state)
        actions.append(action)
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            episode += 1
            env.reset(seed=seed + episode)
    env.close()
    return np.array(states), np.array(actions, dtype=np.int32)


def boundary_states(states, actions, per_axis: int = 20):
    """Deterministic threshold cases (TESSERA-REVIEWER R-PR23): states whose
    reference next ``x`` or ``theta`` lands exactly on a termination
    threshold, or 1 or 4 ulps either side, on both signs. Built from sampled
    states by solving the (Euler) update one step back and nudging the
    component until Gymnasium's step lands on the target bit for bit."""
    env = gymnasium.make("CartPole-v1").unwrapped
    env.reset(seed=0)
    out_states, out_actions = [], []
    for axis, threshold in ((0, env.x_threshold), (2, env.theta_threshold_radians)):
        rate = axis + 1  # x_dot for x, theta_dot for theta
        for sign in (1.0, -1.0):
            for k in range(per_axis):
                base, action = states[k].copy(), int(actions[k])
                for ulps_off in (-4, -1, 0, 1, 4):
                    target = sign * threshold
                    for _ in range(abs(ulps_off)):
                        target = np.nextafter(target, np.inf if ulps_off * sign > 0 else -np.inf)
                    cand = base.copy()
                    cand[axis] = target - env.tau * cand[rate]
                    for _ in range(200):
                        env.state = cand.copy()
                        env.steps_beyond_terminated = None
                        env.step(action)
                        got = env.state[axis]
                        if got == target:
                            out_states.append(cand.copy())
                            out_actions.append(action)
                            break
                        cand[axis] = np.nextafter(cand[axis], np.inf if got < target else -np.inf)
    env.close()
    return np.array(out_states), np.array(out_actions, dtype=np.int32)


def gate_set(n: int, seed: int):
    """The gated states: ``n`` rollout samples plus the boundary set."""
    states, actions = sample(n, seed)
    edge_states, edge_actions = boundary_states(states, actions)
    return np.concatenate([states, edge_states]), np.concatenate([actions, edge_actions])


def reference(states, actions):
    """Gymnasium's one-step results for each injected state."""
    env = gymnasium.make("CartPole-v1").unwrapped
    env.reset(seed=0)
    out = []
    for state, action in zip(states, actions, strict=True):
        env.state = state.copy()
        env.steps_beyond_terminated = None
        obs, reward, term, trunc, _ = env.step(int(action))
        out.append((np.array(env.state, dtype=np.float64), obs, float(reward), bool(term), bool(trunc)))
    thresholds = (env.x_threshold, env.theta_threshold_radians)
    env.close()
    return out, thresholds


def near_threshold(state, thresholds) -> bool:
    x_thr, theta_thr = thresholds
    return abs(abs(state[0]) - x_thr) < 1e-9 or abs(abs(state[2]) - theta_thr) < 1e-9


def injection_round_trips(states, batch: int = 64) -> bool:
    """An injected state reads back bit for bit (the record layout holds)."""
    env = tessera_rl.make("tessera/CartPole-v1", num_envs=batch, seed=1, outputs="owned", max_episode_steps=0)
    native = env._native
    native.reset(1)
    ok = True
    for i in range(batch):
        native.import_record(i, native.snapshot(i)[: -STATE.size] + STATE.pack(*states[i]))
        ok &= STATE.unpack(native.snapshot(i)[-STATE.size :]) == tuple(states[i])
    obs = np.frombuffer(native.outputs()["observations"][0], dtype="<f4").reshape(batch, 4)
    ok &= np.array_equal(obs, states[:batch].astype(np.float32))
    env.close()
    return bool(ok)


def one_step(states, actions, mode: str, batch: int = 1000):
    """Tessera's one-step results in ``mode``: (next state or None, obs,
    reward, terminated, truncated) per state."""
    env = tessera_rl.make("tessera/CartPole-v1", num_envs=batch, seed=1, outputs="owned", auto_reset=mode, max_episode_steps=0)
    native = env._native
    native.reset(1)
    templates = [native.snapshot(i)[: -STATE.size] for i in range(batch)]
    results = []
    for start in range(0, len(states), batch):
        chunk = states[start : start + batch]
        acts = np.zeros(batch, dtype=np.int32)
        acts[: len(chunk)] = actions[start : start + batch]
        for i in range(batch):
            state = chunk[i] if i < len(chunk) else chunk[0]
            native.import_record(i, templates[i] + STATE.pack(*state))
        out = native.step(acts.astype("<i4").tobytes())
        obs = np.frombuffer(out["observations"][0], dtype="<f4").reshape(batch, 4)
        rewards = np.frombuffer(out["rewards"], dtype="<f4")
        term = np.frombuffer(out["episode_terminated"], dtype=np.bool_)
        trunc = np.frombuffer(out["episode_truncated"], dtype=np.bool_)
        final = None
        if mode == "same_step":
            final = np.frombuffer(out["terminal_observations"][0], dtype="<f4").reshape(batch, 4)
        for i in range(len(chunk)):
            ended = bool(term[i] or trunc[i])
            if mode == "same_step" and ended:
                # Reset in the same call: the next state is the terminal
                # observation (float32), not the record.
                results.append((None, final[i].copy(), float(rewards[i]), bool(term[i]), bool(trunc[i])))
            else:
                nxt = np.array(STATE.unpack(native.snapshot(i)[-STATE.size :]))
                results.append((nxt, obs[i].copy(), float(rewards[i]), bool(term[i]), bool(trunc[i])))
        if (start // batch) % 20 == 0:
            env.reset(seed=1)  # keep header counters fresh
    env.close()
    return results


def compare(ours, ref, thresholds):
    """Mismatch counts of the primary gate."""
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
        "terminations_by_x": 0,
        "terminations_by_theta": 0,
        "compared_through_terminal_obs": 0,
    }
    x_thr, theta_thr = thresholds
    for (n_ours, o_ours, r_ours, t_ours, tr_ours), (n_ref, o_ref, r_ref, t_ref, tr_ref) in zip(ours, ref, strict=True):
        near = near_threshold(n_ref, thresholds)
        report["near_threshold"] += int(near)
        report["reference_terminations"] += int(t_ref)
        report["terminations_by_x"] += int(t_ref and abs(n_ref[0]) > x_thr)
        report["terminations_by_theta"] += int(t_ref and abs(n_ref[2]) > theta_thr)
        report["compared_through_terminal_obs"] += int(n_ours is None)
        if n_ours is not None:
            for a, b in zip(n_ours, n_ref, strict=True):
                report["max_ulps"] = max(report["max_ulps"], ulps(float(a), float(b)))
                if not close(float(a), float(b)):
                    report["state"] += 1
                    break
        # The observation is the float32 of the accepted next state: of ours
        # where we have it (so a rounding-boundary state within the state
        # tolerance cannot fail it spuriously), else within one float32 ulp
        # of the reference's (same_step terminal observations).
        if n_ours is not None:
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
    plans = rng.integers(2, size=(episodes, length)).astype(np.int32)
    ref = gymnasium.make("CartPole-v1").unwrapped
    ref.reset(seed=0)
    env = tessera_rl.make("tessera/CartPole-v1", num_envs=episodes, seed=1, outputs="owned", auto_reset="next_step", max_episode_steps=0)
    native = env._native
    native.reset(1)
    for i in range(episodes):
        native.import_record(i, native.snapshot(i)[: -STATE.size] + STATE.pack(*starts[i]))
    ref_traj = []
    for i in range(episodes):
        ref.state = starts[i].copy()
        ref.steps_beyond_terminated = None
        steps = []
        for a in plans[i]:
            _, _, term, _, _ = ref.step(int(a))
            steps.append((np.array(ref.state), term))
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
    return {"episodes": episodes, "length": length, "compared_steps": compared, "max_ulps": max_ulps_seen, "max_relative": max_rel, "flag_mismatches": flag_mismatch}


def truncation(seeds=range(10), limit: int = 500):
    """Truncation against Gymnasium's TimeLimit: a balancing controller on
    each side from the same seeded reset; both must truncate exactly at
    ``limit`` and never terminate."""

    def balance(o):
        return int(o[2] + 0.25 * o[3] + 0.05 * o[0] + 0.1 * o[1] > 0)

    mismatches, compared = 0, 0
    for seed in seeds:
        ref = gymnasium.make("CartPole-v1", max_episode_steps=limit)
        ours = gymnasium.make("tessera/CartPole-v1", max_episode_steps=limit)
        native = tessera_rl.make(
            "tessera/CartPole-v1", num_envs=1, seed=seed, outputs="owned", auto_reset="next_step", max_episode_steps=limit
        )
        ro, _ = ref.reset(seed=seed)
        oo, _ = ours.reset(seed=seed)
        ns = native.reset()
        no = ns.observations["obs"][0, 0]
        for k in range(1, limit + 2):
            ro, _, rt, rtr, _ = ref.step(balance(ro))
            oo, _, ot, otr, _ = ours.step(balance(oo))
            ns = native.step(np.array([[balance(no)]]))
            no = ns.observations["obs"][0, 0]
            nt, ntr = bool(ns.episode_terminated[0]), bool(ns.episode_truncated[0])
            compared += 1
            expect = k == limit
            if (rt, rtr) != (False, expect) or (ot, otr) != (False, expect) or (nt, ntr) != (False, expect):
                mismatches += 1
            if expect:
                break
        ref.close()
        ours.close()
        native.close()
    return {"episodes": len(list(seeds)), "limit": limit, "compared_steps": compared, "mismatches": mismatches}


def run(n: int = 100_000, seed: int = 2026) -> dict:
    states, actions = gate_set(n, seed)
    ref, thresholds = reference(states, actions)
    report = {
        "gymnasium": gymnasium.__version__,
        "tessera_rl": tessera_rl.__version__,
        "boundary_states": len(states) - n,
        "seed": seed,
        "one_step": {mode: compare(one_step(states, actions, mode), ref, thresholds) for mode in MODES},
        "trajectories": trajectories(states[:n]),
        "truncation": truncation(),
    }
    return report


def gate_passes(report: dict) -> bool:
    keys = ("state", "obs", "reward", "terminated", "truncated")
    one_step = all(all(r[k] == 0 for k in keys) for r in report["one_step"].values())
    return one_step and report["truncation"]["mismatches"] == 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    result = run(n)
    print(json.dumps(result, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    sys.exit(0 if gate_passes(result) else 1)
