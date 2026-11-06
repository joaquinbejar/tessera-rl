# SPDX-License-Identifier: MIT OR Apache-2.0
"""Persisted files through ``Env.encode`` and ``Env.load`` (ADR-0011), in every mode."""

import threading

import numpy as np
import pytest

import tessera_rl as t

N = 8


def make(mode="sync", env="tessera/Tag-v1", **kwargs):
    if mode == "pipelined":
        kwargs.update(mode="pipelined", workers=2, chunk_size=2)
    elif mode == "async":
        kwargs.update(mode="async", workers=2, batch_size=N)
    return t.make(env, num_envs=N, seed=5, outputs="owned", **kwargs)


def sync_file(env_id="tessera/Tag-v1", steps=3):
    env = make(env=env_id)
    env.reset()
    for _ in range(steps):
        env.step(np.zeros(N * env.agents, dtype=np.int32))
    data = env.encode()
    env.close()
    return data


@pytest.mark.parametrize("env_id", ["tessera/Tag-v1", "tessera/Connect4-v1", "tessera/CartPole-v1"])
def test_persist_round_trip_in_sync_gives_the_same_file(env_id):
    data = sync_file(env_id)
    assert type(data) is bytes
    env = make(env=env_id)
    env.reset()
    env.load(data)
    assert env.encode() == data
    env.close()


@pytest.mark.parametrize("mode", ["pipelined", "async"])
def test_persist_a_sync_file_loads_in_split_modes_and_back(mode):
    data = sync_file()
    env = make(mode)
    env.reset()
    out = env.load(data)
    if mode == "pipelined":
        assert len(out) == 2
    else:
        assert out is None
    assert env.encode() == data
    env.close()


def test_persist_partial_ids_are_sync_only():
    data = sync_file()
    env = make()
    env.reset()
    part = env.encode([3, 1])
    other = make()
    other.reset()
    other.load(part, ids=[0, 2])
    assert other.encode([0, 2]) == env.encode([3, 1])
    with pytest.raises(t.TesseraError):
        other.load(part, ids=[0])
    with pytest.raises(t.TesseraError):
        other.load(part)
    env.close()
    other.close()
    for mode in ("pipelined", "async"):
        split = make(mode)
        split.reset()
        with pytest.raises(t.ArgumentError):
            split.load(data, ids=[0])
        split.close()


@pytest.mark.parametrize("kind", [bytearray, memoryview])
def test_persist_load_takes_exactly_bytes(kind):
    data = sync_file()
    env = make()
    env.reset()
    with pytest.raises(TypeError):
        env.load(kind(data))
    env.load(bytes(kind(data)))
    env.close()


def damaged(data):
    bad = bytearray(data)
    bad[-20] ^= 1
    return bytes(bad)


@pytest.mark.parametrize("mode", ["sync", "pipelined", "async"])
def test_persist_a_damaged_file_raises_and_changes_nothing(mode):
    data = sync_file()
    env = make(mode)
    env.reset()
    before = env.encode()
    with pytest.raises(t.PersistError):
        env.load(damaged(data))
    with pytest.raises(t.PersistError):
        env.load(b"not a file")
    assert issubclass(t.PersistError, t.TesseraError)
    assert env.encode() == before
    env.close()


def test_persist_limits_are_passed_through():
    data = sync_file()
    env = make()
    env.reset()
    with pytest.raises(t.PersistError):
        env.load(data, max_instances=1)
    with pytest.raises(t.PersistError):
        env.load(data, max_decoded_bytes=8)
    env.load(data, max_instances=N, max_decoded_bytes=1 << 30)
    env.close()


def test_persist_another_config_is_refused():
    data = sync_file()
    env = t.make("tessera/Tag-v1", num_envs=N, seed=5, outputs="owned", width=32)
    env.reset()
    with pytest.raises(t.PersistError):
        env.load(data)
    env.close()


def test_persist_load_releases_the_interpreter():
    data = sync_file()
    env = make()
    env.reset()
    done = threading.Event()

    def other():
        done.set()

    thread = threading.Thread(target=other)
    thread.start()
    for _ in range(200):
        env.load(data)
    thread.join()
    assert done.is_set()
    env.close()


def test_persist_no_identity_override_reaches_python():
    names = dir(t) + dir(t.Env) + dir(t._native)
    assert not [n for n in names if "encode_with" in n or "decode_with" in n or "build_id" in n]


@pytest.mark.parametrize("mode", ["pipelined", "async"])
def test_persist_a_refused_file_leaves_views_fresh_and_a_good_one_stales_them(mode):
    data = sync_file()
    env = t.make(
        "tessera/Tag-v1",
        num_envs=N,
        seed=5,
        **(dict(mode="pipelined", workers=2, chunk_size=2) if mode == "pipelined" else dict(mode="async", workers=2, batch_size=N)),
    )
    out = env.reset()
    rewards = out[0].step.rewards if mode == "pipelined" else out.step.rewards
    with pytest.raises(t.PersistError):
        env.load(damaged(data))
    assert not t.is_stale(rewards)
    env.load(data)
    if mode == "pipelined":
        assert t.is_stale(rewards)
    # Async: a gathered view is its own window, stale at the next recv, as after restore_all.
    env.close()
