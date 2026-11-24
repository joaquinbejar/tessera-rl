# SPDX-License-Identifier: MIT OR Apache-2.0
"""Whole-pool snapshots with their context from Python (ADR-0075 item 7,
#465): ``snapshot_all`` in every mode and ``restore_pool(snapshot,
archive)``, an exact continuation that brings back the reset source."""

import numpy as np
import pytest

import tessera_rl as t
from tessera_rl import replay

N = 4
ID = "tessera/CartPole-v1"


def make(**kwargs):
    kwargs.setdefault("num_envs", N)
    kwargs.setdefault("seed", 5)
    return t.make(ID, outputs="owned", **kwargs)


def entries(k=3):
    """`k` CartPole snapshots after 2, 4, ... steps of pushing right, and
    the observation of each."""
    env = make(num_envs=1)
    env.reset()
    snaps, obs = [], []
    for _ in range(k):
        for _ in range(2):
            out = env.step(np.ones(1, dtype=np.int32))
        snaps.append(env.snapshot(0))
        obs.append(np.asarray(out.observations["obs"])[0].copy())
    env.close()
    return snaps, obs


def run(env, steps):
    """Observations and state hashes of `steps` steps that end episodes."""
    seen = []
    for k in range(steps):
        out = env.step(np.full(N, k % 3 != 0, dtype=np.int32))
        seen.append(np.asarray(out.observations["obs"]).copy())
    return seen, [env.state_hash(i) for i in range(N)]


def test_restore_pool_sync_round_trip_restores_the_archive_source():
    snaps, _ = entries(3)
    env = make()
    archive = t.Archive(env, snaps)
    env.set_reset_source(archive, weights=[1, 0, 2])
    env.reset()
    run(env, 7)
    snapshot = env.snapshot_all()
    want, want_hashes = run(env, 60)
    # Move on with fresh resets: restore_pool must bring the source back.
    env.set_reset_source(None)
    run(env, 13)
    out = env.restore_pool(snapshot, archive)
    assert out is not None
    got, got_hashes = run(env, 60)
    assert got_hashes == want_hashes
    for a, b in zip(got, want):
        assert np.array_equal(a, b)
    env.close()


def test_restore_pool_sync_fresh_round_trip():
    env = make()
    env.reset(seed=3)
    run(env, 5)
    snapshot = env.snapshot_all()
    _, want = run(env, 30)
    run(env, 9)
    env.restore_pool(snapshot)
    _, got = run(env, 30)
    assert got == want
    env.close()


def test_restore_pool_refusals_change_nothing():
    snaps, _ = entries(3)
    env = make()
    archive = t.Archive(env, snaps)
    env.set_reset_source(archive)
    env.reset()
    named = env.snapshot_all()
    env.set_reset_source(None)
    fresh = env.snapshot_all()
    env.set_reset_source(archive, weights=[0, 0, 1])
    run(env, 3)
    before = [env.snapshot(i) for i in range(N)]

    with pytest.raises(t.ConfigError):
        env.restore_pool(named)  # the context names an archive
    other = t.Archive(env, entries(2)[0])
    with pytest.raises(t.ConfigError):
        env.restore_pool(named, other)
    with pytest.raises(t.ArgumentError):
        env.restore_pool(fresh, archive)  # fresh resets take no archive
    with pytest.raises(t.ArgumentError):
        env.restore_pool(named, "not an archive")
    stranger = make()
    with pytest.raises(t.ArgumentError, match="only the pool that took it"):
        stranger.restore_pool(named, archive)
    stranger.close()
    with pytest.raises(t.ArgumentError):
        env.restore_all(named)  # Sync: restore_all stays split-phase only

    assert [env.snapshot(i) for i in range(N)] == before
    # The reset source is still the [0, 0, 1] one: every reset lands on entry 2.
    out = env.reset()
    want = make(num_envs=1)
    want.set_reset_source(t.Archive(want, snaps), weights=[0, 0, 1])
    row = np.asarray(want.reset().observations["obs"])[0]
    for i in range(N):
        assert np.array_equal(np.asarray(out.observations["obs"])[i], row)
    want.close()
    env.close()


def test_restore_pool_refused_while_recording(tmp_path):
    env = make()
    env.reset()
    snapshot = env.snapshot_all()
    with pytest.raises(t.ArgumentError, match="not recorded"):
        with replay.record(env, tmp_path / "r.trpl"):
            env.restore_pool(snapshot)
    env.close()


def test_restore_pool_split_phase_modes():
    snaps, obs = entries(2)
    pipe = make(mode="pipelined", chunk_size=2)
    archive = t.Archive(pipe, snaps)
    pipe.set_reset_source(archive, weights=[0, 1])
    before = [np.asarray(h.step.observations["obs"]).copy() for h in pipe.reset()]
    snapshot = pipe.snapshot_all()
    pipe.set_reset_source(None)
    pipe.reset(seed=11)
    halves = pipe.restore_pool(snapshot, archive)
    assert len(halves) == 2
    for h, want in zip(halves, before):
        assert np.array_equal(np.asarray(h.step.observations["obs"]), want)
    # The source came back with the context: a reset lands on entry 1.
    for h in pipe.reset():
        for row in np.asarray(h.step.observations["obs"]):
            assert np.array_equal(row, obs[1])
    pipe.close()

    a = make(mode="async", batch_size=2)
    a.set_reset_source(archive, weights=[1, 0])
    a.reset()
    snapshot = a.snapshot_all()
    a.set_reset_source(None)
    a.reset(seed=11)
    assert a.restore_pool(snapshot, archive) is None
    step = a.reset()
    for row in np.asarray(step.step.observations["obs"]):
        assert np.array_equal(row, obs[0])
    with pytest.raises(t.ConfigError):
        a.restore_pool(snapshot)
    a.close()


def test_restore_pool_host_refuses():
    host_env = t.HostEnv
    with pytest.raises(t.CapabilityError):
        host_env.restore_pool(object.__new__(host_env), None)
