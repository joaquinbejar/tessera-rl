# SPDX-License-Identifier: MIT OR Apache-2.0
"""The ``schedule`` pool option (ADR-0076): how chunks are given to workers,
validated at make().

``schedule`` never changes what a step computes, so every output of Sync and
Pipelined mode must match a threadless Sync reference, through several
episode ends (terminations, truncations and auto-resets)."""

import numpy as np
import pytest

import tessera_rl as t

CARTPOLE = "tessera/CartPole-v1"
N = 8
STEPS = 300
ROW_KEYS = ("rewards", "agent_terminated", "agent_truncated", "alive", "acting", "masks")
EPISODE_KEYS = ("episode_terminated", "episode_truncated")
LEAF_KEYS = ("observations", "terminal_observations")
COMMON = dict(num_envs=N, seed=7, outputs="owned", auto_reset="same_step", max_episode_steps=40)


def rows_of(buf, rows):
    if buf is None:
        return [None] * rows
    buf = bytes(buf)
    size = len(buf) // rows
    return [buf[i * size : (i + 1) * size] for i in range(rows)]


def canon(v):
    """A comparable copy of any output value (buffers as bytes)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {k: canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [canon(x) for x in v]
    return bytes(v)


def per_instance(out, rows):
    """Every output field of one step, split into one entry per instance."""
    fields = {k: rows_of(out[k], rows) for k in (*ROW_KEYS, *EPISODE_KEYS)}
    for key in LEAF_KEYS:
        fields[key] = [rows_of(leaf, rows) for leaf in out[key]]
    return fields


def reference():
    """The threadless Sync run: per step, its actions and its fields."""
    sync = t.make(CARTPOLE, **COMMON)
    sync.reset()
    rng = np.random.default_rng(3)
    steps, ended = [], 0
    for _ in range(STEPS):
        acts = rng.integers(0, 2, size=N, dtype=np.int32)
        out = sync._native.step(acts.tobytes())
        ended += sum(out["episode_terminated"]) + sum(out["episode_truncated"])
        steps.append((acts, per_instance(out, N), canon(out)))
    sync.close()
    assert ended >= N, f"only {ended} episodes ended in {STEPS} steps"
    return steps


@pytest.fixture(scope="module")
def ref():
    return reference()


def assert_instance(got, i_got, want, i):
    for key in (*ROW_KEYS, *EPISODE_KEYS):
        assert got[key][i_got] == want[key][i], key
    for key in LEAF_KEYS:
        assert len(got[key]) == len(want[key]), key
        for g, w in zip(got[key], want[key], strict=True):
            assert g[i_got] == w[i], key


@pytest.mark.parametrize("schedule", ["static", "dynamic"])
@pytest.mark.parametrize("workers,chunk", [(2, 1), (3, 2), (7, 3)])
def test_make_schedule_sync_with_workers_every_output_matches(ref, schedule, workers, chunk):
    env = t.make(CARTPOLE, workers=workers, chunk_size=chunk, schedule=schedule, **COMMON)
    env.reset()
    for acts, want, whole in ref:
        out = env._native.step(acts.tobytes())
        assert canon(out) == whole, "a field outside the per-row set differs"
        got = per_instance(out, N)
        for i in range(N):
            assert_instance(got, i, want, i)
    env.close()


@pytest.mark.parametrize("schedule", ["static", "dynamic"])
def test_make_schedule_pipelined_every_output_matches(ref, schedule):
    env = t.make(CARTPOLE, mode="pipelined", workers=3, chunk_size=1, schedule=schedule, **COMMON)
    env.reset()
    for acts, want, _ in ref:
        for half, ids in enumerate(env.halves):
            env._native.send(half, acts[ids.start : ids.stop].tobytes(), env._delivered)
        for half, ids in enumerate(env.halves):
            got = env._native.recv(env._delivered)
            env._delivered = got["seq"]
            assert got["half"] == half
            fields = per_instance(got["outputs"], ids.stop - ids.start)
            for pos, i in enumerate(range(ids.start, ids.stop)):
                assert_instance(fields, pos, want, i)
    env.close()


@pytest.mark.parametrize("bad", ["Dynamic", "steal", ""])
def test_make_schedule_unknown_value_is_an_argument_error(bad):
    with pytest.raises(t.ArgumentError, match="schedule must be static or dynamic"):
        t.make(CARTPOLE, workers=2, schedule=bad, **COMMON)
