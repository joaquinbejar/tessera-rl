# SPDX-License-Identifier: MIT OR Apache-2.0
"""Stepped replay from Python (issue #378): ``tessera_rl.replay.steps`` and
``tessera_rl.render.replay_video``."""

import numpy as np
import pytest

import tessera_rl as t
from tessera_rl import render, replay

N = 2
STEPS = 12


def make():
    return t.make("tessera/CartPole-v1", num_envs=N, seed=5, outputs="owned", workers=0)


def actions(k):
    return ((np.arange(N) + k) % 2).astype(np.int32)


def record_file(path, live_frames=None):
    """Records a reset and STEPS steps (keyframes every 4 calls); with
    ``live_frames``, appends instance 0's frame after every call."""
    env = make()
    with replay.record(env, path, keyframe_every=4, checkpoint_every=3):
        env.reset(seed=3)
        if live_frames is not None:
            live_frames.append(env.render([0])[0])
        for k in range(STEPS):
            env.step(actions(k))
            if live_frames is not None:
                live_frames.append(env.render([0])[0])
    hashes = [env.state_hash(i) for i in range(N)]
    env.close()
    return hashes


def gif_frame_count(data: bytes) -> int:
    """Image descriptors in a GIF, walking its blocks."""
    assert data[:6] in (b"GIF87a", b"GIF89a")
    at = 13
    if data[10] & 0x80:
        at += 3 * (2 << (data[10] & 7))

    def skip_sub_blocks(at):
        while data[at]:
            at += data[at] + 1
        return at + 1

    frames = 0
    while data[at] != 0x3B:
        if data[at] == 0x21:
            at = skip_sub_blocks(at + 2)
        elif data[at] == 0x2C:
            frames += 1
            flags = data[at + 9]
            at += 10
            if flags & 0x80:
                at += 3 * (2 << (flags & 7))
            at = skip_sub_blocks(at + 1)
        else:
            raise AssertionError(f"unexpected GIF block {data[at]:#x}")
    return frames


def test_replay_steps_visits_every_call_and_ends_verified(tmp_path):
    path = tmp_path / "run.trpl"
    hashes = record_file(path)
    env = make()
    with replay.steps(path, env) as it:
        indices = list(it)
    assert len(indices) == STEPS + 1
    assert indices == sorted(set(indices))
    summary = it.summary
    assert summary is not None
    assert summary.verified and summary.stopped_at is None
    assert summary.calls == STEPS + 1
    assert [env.state_hash(i) for i in range(N)] == hashes
    env.close()


def test_replay_steps_every_yields_after_each_group(tmp_path):
    path = tmp_path / "run.trpl"
    record_file(path)
    env = make()
    it = replay.steps(path, env, every=4)
    # 13 calls: stops after calls 4, 8 and 12, then the end.
    assert len(list(it)) == 3
    assert it.summary.verified
    env.close()


def test_replay_steps_closed_early_is_stopped_and_frees_the_pool(tmp_path):
    path = tmp_path / "run.trpl"
    record_file(path)
    env = make()
    it = replay.steps(path, env)
    first = next(it)
    summary = it.close()
    assert summary is it.summary
    assert summary.stopped_at == first + 1
    assert not summary.verified
    # The pool is free again: changing calls work.
    env.reset(seed=1)
    env.step(actions(0))
    # A full replay afterwards still verifies.
    assert replay.run(path, env).verified
    env.close()


def test_replay_steps_mutating_calls_raise_while_active(tmp_path):
    path = tmp_path / "run.trpl"
    record_file(path)
    env = make()
    with replay.steps(path, env) as it:
        next(it)
        with pytest.raises(t.ArgumentError):
            env.step(actions(0))
        with pytest.raises(t.ArgumentError):
            env.reset()
        with pytest.raises(t.ArgumentError):
            env.fork(0, [1])
        with pytest.raises(t.ArgumentError):
            env.import_record(0, env.snapshot(0))
        with pytest.raises(t.ArgumentError):
            replay.run(path, env)
        with pytest.raises(t.ArgumentError):
            replay.steps(path, env)
        with pytest.raises(t.ArgumentError), replay.record(env, tmp_path / "other.trpl"):
            pass
        # The refused calls did not disturb the replay.
        rest = list(it)
    assert len(rest) == STEPS
    assert it.summary.verified
    env.close()


def test_replay_steps_render_and_snapshot_between_yields(tmp_path):
    path = tmp_path / "run.trpl"
    record_file(path)
    env = make()
    with replay.steps(path, env) as it:
        for _ in it:
            assert env.render([0]).shape == (1, 400, 600, 3)
            assert isinstance(env.snapshot(1), bytes)
            env.state_hash(0)
    assert it.summary.verified
    env.close()


def test_replay_steps_seek_to_keyframe_then_end_reaches_recorded_state(tmp_path):
    path = tmp_path / "run.trpl"
    hashes = record_file(path)
    env = make()
    with replay.steps(path, env) as it:
        keyframes = it.keyframes()
        assert keyframes[0] == 0 and len(keyframes) > 1
        # Event 1 is the reset, not a keyframe: refused, the cursor stays.
        with pytest.raises(t.PersistError):
            it.seek(1)
        it.seek(keyframes[-1])
        list(it)
    # Reached the end from a seek: never verified (REVIEW-4c-design-r1 point 3).
    assert it.summary.stopped_at is None and it.summary.started_at == keyframes[-1]
    assert not it.summary.verified
    assert [env.state_hash(i) for i in range(N)] == hashes
    env.close()


def test_replay_steps_every_zero_raises_argument_error(tmp_path):
    path = tmp_path / "run.trpl"
    record_file(path)
    env = make()
    with pytest.raises(t.ArgumentError):
        replay.steps(path, env, every=0)
    # Nothing was started: the pool is free.
    env.reset()
    env.close()


def test_replay_summary_verified_follows_stopped_at():
    assert replay.ReplaySummary(events=1, calls=0).verified
    assert not replay.ReplaySummary(events=1, calls=0, stopped_at=1).verified


def test_render_replay_video_gif_has_one_frame_per_step(tmp_path):
    path = tmp_path / "run.trpl"
    record_file(path)
    env = make()
    out = tmp_path / "run.gif"
    count = render.replay_video(path, env, out, instance=0, fps=25)
    assert count == STEPS + 1
    assert gif_frame_count(out.read_bytes()) == count
    # The pool is free after the video.
    env.reset()
    env.close()


def test_replay_frames_equal_live_frames(tmp_path):
    path = tmp_path / "run.trpl"
    live = []
    record_file(path, live_frames=live)
    env = make()
    replayed = []
    with replay.steps(path, env) as it:
        for _ in it:
            replayed.append(env.render([0])[0])
    assert len(replayed) == len(live) == STEPS + 1
    for a, b in zip(live, replayed, strict=True):
        np.testing.assert_array_equal(a, b)
    env.close()
