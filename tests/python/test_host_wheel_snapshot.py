# SPDX-License-Identifier: MIT OR Apache-2.0
"""Host snapshots through the test host wheel (ADR-0073, SPEC 14.4.2):
the counter of tests/host-env, exported with `snapshots`
(`make py-test-host`).

- the host object's methods: records, restore, fork, state hash, files;
- the `snapshots` layout key and `HostEnv.snapshots`;
- `HostEnv`'s methods: continuation after restore, `load` of an `encode`
  file, refused records (`RecordError`), damaged files (`PersistError`);
- poison: every snapshot method refuses, `restore_all` and `load` too;
- no `fork_branched`.
"""

import numpy as np
import pytest

import tessera_rl

_package = pytest.importorskip("_tessera_host", reason="needs the test host wheel (make py-test-host)")
host = getattr(_package, "_tessera_host", _package)

COUNTER = "test/FactoryCounter-v1"
N = 3
A = 2


def actions(value=0, n=N):
    return np.full(n * A, value, dtype="<i4").tobytes()


def test_host_wheel_snapshot_layout_reports_snapshots():
    env = host.make(num_envs=N, seed=0)
    assert env.layout()["snapshots"] is True
    wrapped = tessera_rl.make(COUNTER, num_envs=N)
    assert wrapped.snapshots is True
    env.close()
    wrapped.close()


def test_host_wheel_snapshot_object_round_trip_and_state_hash():
    env = host.make(num_envs=N, seed=4)
    env.reset()
    env.step(actions(1))
    records = env.snapshot([0, 2])
    assert [type(r) for r in records] == [bytes, bytes]
    hashes = [env.state_hash(i) for i in (0, 2)]
    assert all(isinstance(h, bytes) and len(h) == 8 for h in hashes)
    env.step(actions(2))
    env.restore([0, 2], records)
    assert [env.state_hash(i) for i in (0, 2)] == hashes
    env.fork(0, [1])
    assert env.state_hash(1) == env.state_hash(0)
    env.close()


def _obs(step):
    return np.array(step.observations["obs"]).copy()


def test_host_wheel_snapshot_restore_continues_identically():
    env = tessera_rl.make(COUNTER, num_envs=N, seed=7)
    env.reset()
    env.step(np.full((N, A), 1))
    record = env.snapshot(1)
    first = [_obs(env.step(np.full((N, A), 2)))[1] for _ in range(2)]
    env.restore([1], [record])
    again = [_obs(env.step(np.full((N, A), 2)))[1] for _ in range(2)]
    assert all((a == b).all() for a, b in zip(first, again))
    env.close()


def test_host_wheel_snapshot_encode_load_into_another_pool():
    a = tessera_rl.make(COUNTER, num_envs=N, seed=1)
    a.reset()
    a.step(np.full((N, A), 1))
    data = a.encode()
    assert data[:8] == b"TESHOST\x1a"
    b = tessera_rl.make(COUNTER, num_envs=N, seed=99)
    b.reset()
    b.load(data)
    assert [b.state_hash(i) for i in range(N)] == [a.state_hash(i) for i in range(N)]
    a.close()
    b.close()


def test_host_wheel_snapshot_refused_record_and_file_change_nothing():
    env = tessera_rl.make(COUNTER, num_envs=N, seed=2)
    env.reset()
    before = [env.state_hash(i) for i in range(N)]
    good = env.snapshot(0)
    with pytest.raises(tessera_rl.RecordError):
        env.restore([0, 1], [good, good[:-1]])
    other = tessera_rl.make(COUNTER, num_envs=N, seed=2, episode=9)
    other.reset()
    with pytest.raises(tessera_rl.RecordError, match="identity digest"):
        other.restore([0], [good])
    data = bytearray(env.encode())
    data[-12] ^= 1
    with pytest.raises(tessera_rl.PersistError):
        env.load(bytes(data))
    assert [env.state_hash(i) for i in range(N)] == before
    env.close()
    other.close()


def test_host_wheel_snapshot_poisoned_refuses_everything():
    env = tessera_rl.make(COUNTER, num_envs=N, seed=0, panic_step=1)
    env.reset()
    record = env.snapshot(0)
    data = env.encode()
    with pytest.raises(tessera_rl.PoisonedError):
        env.step(np.full((N, A), 0))
    assert env.is_poisoned()
    for call in (
        lambda: env.snapshot(0),
        lambda: env.restore([0], [record]),
        lambda: env.restore_all([record] * N),
        lambda: env.fork(0, [1]),
        lambda: env.state_hash(0),
        lambda: env.encode(),
        lambda: env.load(data),
    ):
        with pytest.raises(tessera_rl.PoisonedError):
            call()
    env.close()


def test_host_wheel_snapshot_has_no_fork_branched():
    env = tessera_rl.make(COUNTER, num_envs=N)
    assert not hasattr(env, "fork_branched")
    with pytest.raises(tessera_rl.CapabilityError):
        env.snapshot_all()
    env.close()
