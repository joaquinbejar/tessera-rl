# SPDX-License-Identifier: MIT OR Apache-2.0
"""#152 E2: an exact ndarray of C-contiguous little-endian int32 reaches the
native step as a memoryview, with no Python pass over it. Every other input
keeps the converting path; errors stay ``ActionError`` before the pool is
touched."""

import numpy as np
import pytest

import tessera_rl as t


def cartpole(n=2, seed=7):
    env = t.make("tessera/CartPole-v1", num_envs=n, seed=seed)
    env.reset()
    return env


class _Recorder:
    """Wraps an environment's native object; records what ``step`` got."""

    def __init__(self, native):
        self._inner = native
        self.kinds = []

    def step(self, acts):
        self.kinds.append(type(acts).__name__)
        return self._inner.step(acts)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def recorded(env):
    rec = _Recorder(env._native)
    env._native = rec
    return rec


@pytest.mark.parametrize(
    ("actions", "kind"),
    [
        (np.ones((2, 1), dtype="<i4"), "memoryview"),
        (np.ones(2, dtype=np.int32), "memoryview"),
        (np.ones((2, 1), dtype=np.int64), "bytes"),
        (np.ones((2, 1), dtype=">i4"), "bytes"),
        (np.ones((4, 1), dtype=np.int32)[::2], "bytes"),
        (np.ones((1, 2), dtype=np.int32).T, "memoryview"),
        (np.ones((2, 2), dtype=np.int32).T[0], "bytes"),
        (np.ones((2, 1), dtype=np.int32).view(np.matrix), "bytes"),
        ([[1], [1]], "bytes"),
    ],
)
def test_step_int32_contiguous_only_takes_the_memoryview_path(actions, kind):
    env = cartpole()
    rec = recorded(env)
    env.step(actions)
    assert rec.kinds == [kind]


def test_step_int32_and_int64_actions_give_the_same_trajectory():
    a, b = cartpole(8, 3), cartpole(8, 3)
    rng = np.random.default_rng(5)
    for _ in range(200):
        acts = rng.integers(0, 2, size=(8, 1), dtype=np.int32)
        sa, sb = a.step(acts), b.step(acts.astype(np.int64))
        for got, want in ((sa.observations["obs"], sb.observations["obs"]), (sa.rewards, sb.rewards)):
            assert np.array_equal(got, want)
        assert np.array_equal(sa.episode_terminated, sb.episode_terminated)
        assert np.array_equal(sa.episode_truncated, sb.episode_truncated)


def test_step_int32_actions_are_copied_before_the_step():
    a, b = cartpole(2, 3), cartpole(2, 3)
    acts = np.array([[1], [0]], dtype=np.int32)
    sa = a.step(acts)
    acts[:] = 0
    sb = b.step(np.array([[1], [0]], dtype=np.int64))
    assert np.array_equal(sa.observations["obs"], sb.observations["obs"])


@pytest.mark.parametrize("value", [2, -1, 2**31 - 1, -(2**31)])
def test_step_int32_out_of_head_range_raises_the_same_action_error_before_mutation(value):
    env = cartpole()
    before = env.snapshot(0)
    with pytest.raises(t.ActionError) as fast:
        env.step(np.full((2, 1), value, dtype=np.int32))
    assert env.snapshot(0) == before
    with pytest.raises(t.ActionError) as slow:
        env.step(np.full((2, 1), value, dtype=np.int64))
    assert str(fast.value) == str(slow.value)
    assert env.snapshot(0) == before


@pytest.mark.parametrize("shape", [(3, 1), (1,), (0,), ()])
def test_step_int32_wrong_count_raises_the_same_action_error_before_mutation(shape):
    env = cartpole()
    before = env.snapshot(0)
    with pytest.raises(t.ActionError) as fast:
        env.step(np.ones(shape, dtype=np.int32))
    with pytest.raises(t.ActionError) as slow:
        env.step(np.ones(shape, dtype=np.int64))
    assert str(fast.value) == str(slow.value)
    assert env.snapshot(0) == before


@pytest.mark.parametrize(
    ("view", "message"),
    [
        (memoryview(np.ones(2, dtype=np.float32)), "must hold int32"),
        (memoryview(np.ones(2, dtype=np.int64)), "must hold int32"),
        (memoryview(np.ones(2, dtype=np.uint32)), "must hold int32"),
        (memoryview(np.ones(2, dtype=">i4")), "must hold int32"),
        (memoryview(np.ones(8, dtype=np.uint8)), "must hold int32"),
        (memoryview(np.ones(4, dtype=np.int32))[::2], "must be C-contiguous"),
    ],
)
def test_native_step_refuses_a_memoryview_that_is_not_int32_c_contiguous(view, message):
    env = cartpole()
    before = env.snapshot(0)
    with pytest.raises(t.ActionError, match=message):
        env._native.step(view)
    assert env.snapshot(0) == before


def test_native_step_accepts_a_cast_int32_view():
    a, b = cartpole(2, 3), cartpole(2, 3)
    acts = np.array([1, 0], dtype=np.int32)
    a._native.step(memoryview(acts.tobytes()).cast("i"))
    b._native.step(acts.tobytes())
    assert a.snapshot(0) == b.snapshot(0)
    assert a.snapshot(1) == b.snapshot(1)


def test_native_step_a_released_memoryview_raises_value_error():
    env = cartpole()
    before = env.snapshot(0)
    view = memoryview(np.ones(2, dtype=np.int32))
    view.release()
    with pytest.raises(ValueError):
        env._native.step(view)
    assert env.snapshot(0) == before


def test_native_step_continuous_head_refuses_a_memoryview():
    env = t.make("tessera/Pendulum-v1", num_envs=2, seed=1)
    env.reset()
    with pytest.raises(TypeError, match="bytes"):
        env._native.step(memoryview(np.zeros(2, dtype=np.float32)))


def test_native_step_other_objects_still_raise_type_error():
    env = cartpole()
    for other in (bytearray(8), np.ones(2, dtype=np.int32), [1, 1]):
        with pytest.raises(TypeError, match="must be bytes"):
            env._native.step(other)


def test_step_continuous_head_int32_actions_keep_the_converting_path():
    a = t.make("tessera/Pendulum-v1", num_envs=2, seed=1)
    b = t.make("tessera/Pendulum-v1", num_envs=2, seed=1)
    a.reset()
    b.reset()
    a.step(np.array([[1], [-1]], dtype=np.int32))
    b.step(np.array([[1.0], [-1.0]], dtype=np.float32))
    assert a.snapshot(0) == b.snapshot(0)
    assert a.snapshot(1) == b.snapshot(1)


def test_step_int32_releases_the_source_buffer_before_returning():
    # The view is copied and released before the step runs (the interpreter
    # is released only after): once step() returns, no export of the
    # source is left, so it can be resized or freed.
    env = cartpole()
    raw = bytearray(np.ones(2, dtype=np.int32).tobytes())
    acts = np.frombuffer(raw, dtype=np.int32).reshape(2, 1)
    env.step(acts)
    del acts
    raw.extend(b"\0" * 4096)  # BufferError while any export is held
    arr = np.ones((2, 1), dtype=np.int32)
    env.step(arr)
    arr.resize((4096, 1), refcheck=False)
    arr[:] = 7
    del arr
    s = env.step(np.ones((2, 1), dtype=np.int32))
    assert s.rewards.shape == (2, 1)


def test_native_step_releases_a_memoryview_before_returning():
    env = cartpole()
    raw = bytearray(np.ones(2, dtype=np.int32).tobytes())
    view = memoryview(raw).cast("i")
    env._native.step(view)
    view.release()  # BufferError while any export of the view is held
    raw.extend(b"\0" * 4096)
    bad = memoryview(bytearray(12)).cast("i")
    with pytest.raises(t.ActionError):
        env._native.step(bad)
    bad.release()  # also released on the refused path
