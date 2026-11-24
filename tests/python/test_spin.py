# SPDX-License-Identifier: MIT OR Apache-2.0
"""The ``spin`` pool option: the worker spin budget, validated at make().

``spin`` changes only how long a waiting thread spins before it blocks, so
every output of every mode must match a threadless Sync reference, through
several episode ends (terminations, truncations and auto-resets)."""

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


@pytest.mark.parametrize("spin", [0, 2_000, 200_000])
def test_make_spin_sync_with_workers_every_output_matches(ref, spin):
    env = t.make(CARTPOLE, workers=2, chunk_size=2, spin=spin, **COMMON)
    env.reset()
    for acts, want, whole in ref:
        out = env._native.step(acts.tobytes())
        assert canon(out) == whole, "a field outside the per-row set differs"
        got = per_instance(out, N)
        for i in range(N):
            assert_instance(got, i, want, i)
    env.close()


@pytest.mark.parametrize("spin", [0, 200_000])
def test_make_spin_pipelined_every_output_matches(ref, spin):
    env = t.make(CARTPOLE, mode="pipelined", chunk_size=2, spin=spin, **COMMON)
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


@pytest.mark.parametrize("spin", [0, 200_000])
def test_make_spin_async_every_output_matches(ref, spin):
    env = t.make(CARTPOLE, mode="async", batch_size=N, workers=2, spin=spin, **COMMON)
    env.reset()
    for acts, want, _ in ref:
        env._native.send(list(range(N)), acts.tobytes(), env._delivered)
        got = env._native.recv(env._delivered)
        env._delivered = got["seq"]
        assert sorted(got["ids"]) == list(range(N))
        fields = per_instance(got["outputs"], N)
        for pos, i in enumerate(got["ids"]):
            assert_instance(fields, pos, want, i)
    env.close()


def test_make_spin_largest_value_accepted():
    t.make(CARTPOLE, num_envs=2, workers=1, spin=2**32 - 1).close()


@pytest.mark.parametrize("mode_kw", [{"mode": "pipelined"}, {"mode": "async", "batch_size": 2}])
def test_make_spin_split_modes_without_workers_accepted(mode_kw):
    t.make(CARTPOLE, num_envs=4, chunk_size=2, spin=0, **mode_kw).close()


def test_make_spin_sync_workers_zero_raises_argument_error():
    with pytest.raises(t.ArgumentError, match="spin needs workers"):
        t.make(CARTPOLE, num_envs=2, workers=0, spin=10)


def test_make_spin_sync_default_workers_accepted():
    # Without ``workers`` the pool gets the default count (at least one).
    t.make(CARTPOLE, num_envs=2, spin=10).close()


@pytest.mark.parametrize("bad", [2**32, 1.5, "fast", True])
def test_make_spin_wrong_type_or_range_raises_type_error(bad):
    with pytest.raises(TypeError, match="spin"):
        t.make(CARTPOLE, num_envs=2, workers=1, spin=bad)


def test_make_spin_negative_raises_overflow_error():
    with pytest.raises(OverflowError):
        t.make(CARTPOLE, num_envs=2, workers=1, spin=-1)


def test_make_spin_none_without_workers_is_the_default():
    t.make(CARTPOLE, num_envs=2, spin=None).close()
