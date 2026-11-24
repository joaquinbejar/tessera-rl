# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay files from Python (ADR-0072 section 9): ``tessera_rl.replay.record``
and ``tessera_rl.replay.run``."""

import dataclasses
import os

import numpy as np
import pytest

import tessera_rl as t
from tessera_rl import replay

N = 4


def make(env_id, **kwargs):
    return t.make(env_id, num_envs=N, seed=5, outputs="owned", **kwargs)


def actions(env, k):
    return ((np.arange(N * env.agents) + k) % 2).astype(np.int32)


def record_file(path, env_id="tessera/CartPole-v1", steps=20, **kwargs):
    env = make(env_id, **kwargs)
    with replay.record(env, path, checkpoint_every=4):
        env.reset(seed=3)
        for k in range(steps):
            env.step(actions(env, k))
        env.fork(0, [1])
        env.step(actions(env, steps))
    hashes = [env.state_hash(i) for i in range(N)]
    env.close()
    return hashes


@pytest.mark.parametrize("env_id", ["tessera/CartPole-v1", "tessera/Tag-v1"])
def test_replay_round_trip_reproduces_the_state(tmp_path, env_id):
    path = tmp_path / "run.trpl"
    hashes = record_file(path, env_id)
    env = make(env_id)
    summary = replay.run(path, env)
    assert summary.calls == 23
    assert summary.events > summary.calls
    assert not summary.poisoned
    assert not summary.unverified_poison
    assert [env.state_hash(i) for i in range(N)] == hashes
    env.close()


def test_replay_record_writes_atomically_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "run.trpl"
    record_file(path)
    assert path.exists()
    assert not os.path.exists(str(path) + ".tmp")


def test_replay_record_body_raising_writes_nothing(tmp_path):
    path = tmp_path / "run.trpl"
    env = make("tessera/CartPole-v1")
    with pytest.raises(RuntimeError), replay.record(env, path):
        env.reset()
        raise RuntimeError("boom")
    assert not path.exists()
    assert not os.path.exists(str(path) + ".tmp")
    # The environment is usable and can record again.
    with replay.record(env, path):
        env.reset()
    assert path.exists()
    env.close()


def _header_len(b):
    u16 = lambda at: int.from_bytes(b[at : at + 2], "little")  # noqa: E731
    at = 48
    for _ in range(2):  # env_id, library_version
        at += 2 + u16(at)
    at += 2  # asset_count
    at += 2 + u16(at)  # descriptor
    config = int.from_bytes(b[at : at + 4], "little")
    # Action layout (12), flags and host fields (4), the version 1
    # extension, header checksum (8).
    q = at + 4 + config + 12 + 4
    q += 2 + u16(q) + 1  # policy map, has_curriculum
    if b[q] == 1:  # archive block
        q += 1 + 8 + 4 + 4 * int.from_bytes(b[q + 9 : q + 13], "little")
    else:
        q += 1
    return q + 4 + 8  # chunk_size, checksum


def _frames(b):
    at, out = _header_len(b), []
    while at < len(b):
        length = int.from_bytes(b[at + 1 : at + 5], "little")
        out.append((at, b[at]))
        at += 5 + length
    return out


def test_replay_tampered_checkpoint_raises_divergence_with_fields(tmp_path):
    xxhash = pytest.importorskip("xxhash")
    path = tmp_path / "run.trpl"
    record_file(path)
    b = bytearray(path.read_bytes())
    frames = _frames(b)
    checkpoints = [k for k, (_, tag) in enumerate(frames) if tag == 6]
    # Instance 2's hash in the second checkpoint: frame (5), ids marker (1),
    # then 8 bytes per instance.
    b[frames[checkpoints[1]][0] + 5 + 1 + 2 * 8] ^= 1
    end = frames[-1][0]
    start = _header_len(b)
    b[end + 13 : end + 21] = xxhash.xxh3_64_intdigest(bytes(b[start:end])).to_bytes(8, "little")
    path.write_bytes(bytes(b))
    env = make("tessera/CartPole-v1")
    with pytest.raises(replay.ReplayDivergence) as info:
        replay.run(path, env)
    err = info.value
    assert isinstance(err, t.TesseraError)
    assert err.kind == "checkpoint"
    assert err.instance == 2
    assert err.event_index == checkpoints[1]
    assert err.last_matching_checkpoint == checkpoints[0]
    env.close()


def test_replay_divergence_is_a_tessera_error_with_fields():
    err = replay.ReplayDivergence(9, "checkpoint", None, None)
    assert isinstance(err, t.TesseraError)
    assert (err.event_index, err.kind, err.instance, err.last_matching_checkpoint) == (
        9,
        "checkpoint",
        None,
        None,
    )
    assert "event 9" in str(err)


def test_replay_truncated_file_raises_persist_error(tmp_path):
    path = tmp_path / "run.trpl"
    record_file(path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    env = make("tessera/CartPole-v1")
    with pytest.raises(t.PersistError):
        replay.run(path, env)
    env.close()


def test_replay_snapshot_file_raises_persist_error(tmp_path):
    env = make("tessera/CartPole-v1")
    env.reset()
    path = tmp_path / "snapshot.bin"
    path.write_bytes(env.encode())
    with pytest.raises(t.PersistError):
        replay.run(path, env)
    env.close()


def test_replay_into_another_config_raises_persist_error(tmp_path):
    path = tmp_path / "run.trpl"
    record_file(path, "tessera/Tag-v1")
    env = make("tessera/Tag-v1", pursuers=1, evaders=2)
    with pytest.raises(t.PersistError):
        replay.run(path, env)
    env.close()


def test_replay_into_another_num_envs_raises_persist_error(tmp_path):
    path = tmp_path / "run.trpl"
    record_file(path)
    env = t.make("tessera/CartPole-v1", num_envs=N + 1, seed=5)
    with pytest.raises(t.PersistError):
        replay.run(path, env)
    env.close()


def test_replay_load_while_recording_raises_argument_error(tmp_path):
    env = make("tessera/CartPole-v1")
    env.reset()
    data = env.encode()
    with replay.record(env, tmp_path / "run.trpl"):
        with pytest.raises(t.ArgumentError):
            env.load(data)
    env.close()


def test_replay_split_modes_raise_tessera_error(tmp_path):
    env = t.make(
        "tessera/CartPole-v1", num_envs=N, seed=5, mode="pipelined", workers=2, chunk_size=2
    )
    with pytest.raises(t.TesseraError), replay.record(env, tmp_path / "run.trpl"):
        pass
    env.close()


def test_replay_adapter_or_foreign_object_raises_type_error(tmp_path):
    with pytest.raises(TypeError), replay.record(object(), tmp_path / "x"):
        pass


def test_replay_summary_exposes_unverified_poison_with_default_false():
    fields = {f.name: f for f in dataclasses.fields(replay.ReplaySummary)}
    assert list(fields) == ["events", "calls", "poisoned", "unverified_poison", "unclaimed"]
    assert replay.ReplaySummary(events=3, calls=2).unclaimed == ()
    summary = replay.ReplaySummary(events=3, calls=2)
    assert summary.unverified_poison is False
    assert summary.poisoned is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.calls = 5
