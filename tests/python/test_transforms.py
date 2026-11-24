# SPDX-License-Identifier: MIT OR Apache-2.0
"""Observation transforms from Python (ADR-0078, #346 head C): ``make``
takes ``transforms``, the layout and the adapter spaces describe the
transformed observation, and ``epoch_barrier`` and
``freeze_normalization`` drive the running statistics."""

import numpy as np
import pytest

import tessera_rl as t
import tessera_rl.gym as tg
from tessera_rl import pz
from tessera_rl.transforms import FrameStack, Normalize, OneHot


def board(env, s):
    return np.asarray(s.observations["board"])


def test_transforms_frame_stack_shapes_and_order_from_python():
    env = t.make("tessera/Connect4-v1", num_envs=2, seed=1, outputs="owned", transforms=[FrameStack("board", 3)])
    s = env.reset()
    obs = board(env, s)
    assert obs.shape == (2, 2, 3, 2, 6, 7) and obs.dtype == np.uint8
    # Reset fills the stack with the first frame.
    assert (obs[:, :, 0] == obs[:, :, 2]).all()
    a = np.zeros((2, 2), dtype=np.int32)
    a[:, 0] = 3
    s = env.step(a)
    after = board(env, s)
    assert (after[:, :, 1] == obs[:, :, 2]).all(), "frames move toward the front"
    assert not (after[:, :, 2] == obs[:, :, 2]).all(), "the newest frame is the new board"
    env.close()


def test_transforms_one_hot_and_normalize_dtypes():
    env = t.make("tessera/Leduc-v1", num_envs=1, outputs="owned", transforms=[OneHot("view", 4)])
    view = np.asarray(env.reset().observations["view"])
    assert view.dtype == np.bool_ and view.shape[-1] == 4
    env.close()
    env = t.make("tessera/CartPole-v1", num_envs=2, outputs="owned", transforms=[Normalize("", 5.0)])
    # A single-leaf observation is exported as "obs".
    obs = np.asarray(env.reset().observations["obs"])
    assert obs.dtype == np.float32 and obs.shape == (2, 1, 4)
    env.close()


def _cartpole(**kw):
    return t.make("tessera/CartPole-v1", num_envs=4, seed=7, outputs="owned", transforms=[Normalize("", 5.0)], **kw)


def _run(env, steps):
    env.reset()
    for _ in range(steps):
        s = env.step(np.zeros((4, 1), dtype=np.int32))
    return np.asarray(s.observations["obs"]).copy()


def _with_barrier(env):
    """Three steps, a barrier, one more step: the last observation."""
    env.reset()
    for _ in range(3):
        env.step(np.zeros((4, 1), dtype=np.int32))
    env.epoch_barrier()
    s = env.step(np.zeros((4, 1), dtype=np.int32))
    return np.asarray(s.observations["obs"]).copy()


def test_transforms_epoch_barrier_changes_statistics_and_freeze_stops_them():
    # No barrier: the statistics stay the identity (up to the clip).
    plain = _run(_cartpole(), 4)
    learned = _with_barrier(_cartpole())
    assert not np.array_equal(learned, plain), "the barrier applied new statistics"
    assert np.abs(learned).max() <= 5.0
    # The clip reaches the engine: a tight clip bounds the same run.
    tight = t.make("tessera/CartPole-v1", num_envs=4, seed=7, outputs="owned", transforms=[Normalize("", 0.25)])
    clipped = _with_barrier(tight)
    assert np.abs(clipped).max() == np.float32(0.25), "values past the clip are clipped to it"
    frozen = _cartpole()
    frozen.freeze_normalization(True)
    assert np.array_equal(_with_barrier(frozen), plain), "nothing was learned while frozen"


def test_transforms_gymnasium_and_pettingzoo_spaces_are_transformed():
    env = tg.make("tessera/CartPole-v1", num_envs=2, transforms=[FrameStack("", 4)])
    assert env.single_observation_space.shape == (4, 4)
    env.close()
    par = pz.parallel_env("tessera/Tag-v1", transforms=[FrameStack("window", 2)])
    space = par.observation_space(par.possible_agents[0])
    space = space["window"] if hasattr(space, "spaces") else space
    assert space.shape == (2, 3, 5, 5)
    par.close()


def test_transforms_refusals():
    with pytest.raises(t.ConfigError):
        t.make("tessera/CartPole-v1", transforms=[FrameStack("", 1)])
    with pytest.raises(TypeError):
        t.make("tessera/CartPole-v1", transforms=["frame_stack"])
    env = t.make("tessera/Connect4-v1", num_envs=4, mode="pipelined", workers=1, chunk_size=2, transforms=[FrameStack("board", 2)])
    with pytest.raises(t.CapabilityError):
        env.epoch_barrier()
    env.close()
    from tessera_rl import _host

    with pytest.raises(t.CapabilityError, match="transforms"):
        _host._refuse_spin({"transforms": [FrameStack("x", 2)]})


def test_transforms_restore_pool_round_trip_brings_back_frames_and_statistics():
    """Normalize plus FrameStack: ``restore_pool`` restores the frame
    stacks and the running statistics (#465's pool context), so the run
    after it repeats exactly. Chunk accumulators are not continuation state
    (ADR-0078 item 7: empty after a restore), so the snapshot is taken right
    after a barrier, where they are empty too."""
    env = t.make(
        "tessera/CartPole-v1",
        num_envs=4,
        seed=7,
        outputs="owned",
        transforms=[Normalize("", 5.0), FrameStack("", 2)],
    )
    a = np.zeros((4, 1), dtype=np.int32)
    env.reset()
    for i in range(3):
        a[:, 0] = i % 2
        env.step(a)
    env.epoch_barrier()
    snap = env.snapshot_all()

    def tail():
        out = []
        for i in range(4):
            a[:, 0] = (i + 1) % 2
            out.append(np.asarray(env.step(a).observations["obs"]).copy())
            if i == 1:
                env.epoch_barrier()
        return np.stack(out)

    first = tail()
    restored = env.restore_pool(snap)
    obs = np.asarray(restored.observations["obs"])
    # The stack holds two different frames, so the frame stacks came back.
    assert obs.shape == (4, 1, 2, 4)
    assert not np.array_equal(obs[:, :, 0], obs[:, :, 1])
    assert np.array_equal(tail(), first), "an exact continuation"
    env.close()
