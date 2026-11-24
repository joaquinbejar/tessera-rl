# SPDX-License-Identifier: MIT OR Apache-2.0
"""Archive resets from Python (ADR-0075, #363 head C): ``tessera_rl.Archive``,
``set_reset_source``, ``reset_from``, the modes, the adapters' forwarding,
recording and replay with an archive."""

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
    the observation of each (the state an archive reset starts from)."""
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


def obs_of(out, i):
    return np.asarray(out.observations["obs"])[i]


def test_archive_python_builds_and_reset_from_lands_on_the_entry():
    snaps, obs = entries(3)
    env = make()
    archive = t.Archive(env, snaps)
    assert len(archive) == 3
    assert isinstance(archive.digest, int)
    assert "len=3" in repr(archive)
    # The same snapshots in another env of the same config: the same archive.
    other = make(seed=9)
    assert t.Archive(other, snaps).digest == archive.digest
    other.close()
    env.set_reset_source(archive, weights=[1, 2, 3])
    env.reset()
    out = env.reset_from([2, 0], [1, 2])
    assert np.array_equal(obs_of(out, 2), obs[1])
    assert np.array_equal(obs_of(out, 0), obs[2])
    # Auto-resets draw entries: after an episode ends the observation is one
    # of the entries.
    env.set_reset_source(archive, weights=[0, 0, 1])
    out = env.reset()
    for i in range(N):
        assert np.array_equal(obs_of(out, i), obs[2])
    env.set_reset_source(None)
    env.close()


def test_archive_python_refusals():
    snaps, _ = entries(2)
    env = make()
    archive = t.Archive(env, snaps)
    with pytest.raises(t.ArgumentError):
        env.reset_from([0], [0])  # no archive attached
    with pytest.raises(t.ArgumentError):
        env.set_reset_source(archive, weights=[0, 0])
    with pytest.raises(t.ArgumentError):
        env.set_reset_source(archive, weights=[1])
    with pytest.raises(t.ArgumentError):
        env.set_reset_source(None, weights=[1, 1])
    with pytest.raises(t.ArgumentError):
        env.set_reset_source("not an archive")
    with pytest.raises(t.RecordError):
        t.Archive(env, [b"\x00" * 8])
    with pytest.raises(t.ArgumentError):
        t.Archive(env, [])
    acrobot = t.make("tessera/Acrobot-v1", num_envs=1, outputs="owned")
    with pytest.raises(t.ArgumentError, match="another environment"):
        acrobot.set_reset_source(archive)
    acrobot.close()
    env.set_reset_source(archive)
    env.reset()
    with pytest.raises(t.ArgumentError):
        env.reset_from([0], [5])  # entry out of range
    env.close()


def test_archive_python_split_phase_modes():
    snaps, obs = entries(2)
    pipe = make(mode="pipelined", chunk_size=2)
    archive = t.Archive(pipe, snaps)
    pipe.set_reset_source(archive, weights=[0, 1])
    halves = pipe.reset()
    for h in halves:
        for row in np.asarray(h.step.observations["obs"]):
            assert np.array_equal(row, obs[1])
    with pytest.raises(t.CapabilityError):
        pipe.reset_from([0], [0])
    pipe.close()

    a = make(mode="async", batch_size=2)
    a.set_reset_source(archive, weights=[1, 0])
    step = a.reset()
    for row in np.asarray(step.step.observations["obs"]):
        assert np.array_equal(row, obs[0])
    a.reset_from([1, 3], [1, 1])
    got = a.recv()
    assert sorted(got.ids) == [1, 3]
    for row in np.asarray(got.step.observations["obs"]):
        assert np.array_equal(row, obs[1])
    a.close()


def test_archive_python_adapters_forward():
    snaps, obs = entries(2)
    from tessera_rl import gym as tgym

    v = tgym.TesseraVectorEnv(ID, num_envs=2, seed=1)
    archive = t.Archive(v, snaps)
    v.set_reset_source(archive, weights=[1, 0])
    o, _ = v.reset(seed=1)
    for row in o:
        assert np.array_equal(row, obs[0].reshape(row.shape))
    v.close()


def test_archive_python_host_refuses():
    env = make()
    snaps, _ = entries(1)
    archive = t.Archive(env, snaps)
    env.close()
    # A host environment of the fake protocol refuses archive resets.
    host_env = pytest.importorskip("tessera_rl._host").HostEnv
    assert hasattr(host_env, "set_reset_source")
    with pytest.raises(t.CapabilityError):
        host_env.set_reset_source(object.__new__(host_env), archive)


def test_archive_python_record_and_replay_round_trip(tmp_path):
    # Rev's REVIEW-designC-r1 requirement: the Python recorder records
    # set_reset_source and reset_from, and replay needs the archive.
    snaps, _ = entries(3)
    path = tmp_path / "archive.trpl"
    env = make()
    archive = t.Archive(env, snaps)
    with replay.record(env, path, checkpoint_every=4):
        env.reset(seed=3)
        env.set_reset_source(archive, weights=[1, 0, 2])
        for k in range(40):
            env.step(((np.arange(N) + k) % 2).astype(np.int32))
            if k == 20:
                env.reset_from([1], [2])
    hashes = [env.state_hash(i) for i in range(N)]
    env.close()
    target = make()
    summary = replay.run(path, target, archive=archive)
    assert summary.calls == 1 + 1 + 40 + 1
    assert [target.state_hash(i) for i in range(N)] == hashes
    target.close()
    # Without the archive, or with another: refused before anything runs.
    bare = make()
    with pytest.raises(t.ConfigError):
        replay.run(path, bare)
    other = t.Archive(bare, entries(2)[0])
    with pytest.raises(t.ConfigError):
        replay.run(path, bare, archive=other)
    bare.close()
    # Stepped replay and seek with the archive.
    stepped = make()
    with replay.steps(path, stepped, archive=archive) as it:
        for _ in it:
            pass
    assert [stepped.state_hash(i) for i in range(N)] == hashes
    stepped.close()
