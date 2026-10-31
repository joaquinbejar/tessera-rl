# SPDX-License-Identifier: MIT OR Apache-2.0
"""Borrowed outputs (ADR-0003, SPEC 5.3): zero-copy read-only views, the
borrow window (overwrite semantics), staleness, memory lifetime after
close, and the buffer protocol and DLPack exports."""

import gc
import sys

import numpy as np
import pytest

import tessera_rl as t


def cartpole(n=2, seed=7, **kw):
    return t.make("tessera/CartPole-v1", num_envs=n, seed=seed, **kw)


def test_borrowed_is_the_native_default_and_owned_is_opt_in():
    assert cartpole().outputs == "borrowed"
    assert cartpole(outputs="owned").outputs == "owned"
    with pytest.raises(t.ArgumentError):
        cartpole(outputs="lent")


def test_borrowed_arrays_are_read_only_typed_and_zero_copy():
    env = cartpole(3)
    s = env.reset()
    obs = s.observations["obs"]
    assert obs.shape == (3, 1, 4) and obs.dtype == np.float32
    assert s.rewards.dtype == np.float32 and s.alive.dtype == np.bool_
    assert not obs.flags.writeable
    with pytest.raises(ValueError):
        obs[0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        obs.setflags(write=True)
    # Zero copy: two calls in the same window share the same memory.
    again = np.asarray(env._native.outputs()["observations"][0])
    assert again.__array_interface__["data"][0] == obs.__array_interface__["data"][0]


def test_retained_views_are_overwritten_and_stale_after_the_next_step():
    # ADR-0003 rule 2: after the window ends, retained views show the
    # values of later calls.
    env = cartpole(2)
    first = env.reset().observations["obs"]
    kept = first.copy()
    assert not t.is_stale(first)
    second = env.step(np.ones((2, 1), dtype=np.int32)).observations["obs"]
    assert t.is_stale(first)
    assert not t.is_stale(second)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, kept)


def test_owned_arrays_survive_later_steps_and_are_never_stale():
    env = cartpole(2, outputs="owned")
    first = env.reset().observations["obs"]
    kept = first.copy()
    for _ in range(5):
        env.step(np.zeros((2, 1), dtype=np.int32))
    assert np.array_equal(first, kept)
    assert first.flags.writeable
    assert not t.is_stale(first)


def test_staleness_follows_numpy_views_and_memoryviews():
    env = cartpole(4)
    s = env.reset()
    sliced = s.observations["obs"][1:3, :, ::2]
    view = env._native.outputs()["rewards"]
    mem = memoryview(view)
    assert mem.readonly and mem.format == "<f" and mem.shape == (4, 1)
    assert not t.is_stale(sliced) and not t.is_stale(mem)
    env.step(np.zeros((4, 1), dtype=np.int32))
    assert t.is_stale(sliced) and t.is_stale(mem) and view.is_stale()
    with pytest.raises(TypeError):
        t.is_stale(np.zeros(3))


def test_writable_buffer_requests_are_refused():
    env = cartpole()
    env.reset()
    view = env._native.outputs()["rewards"]
    with pytest.raises(ValueError):
        np.frombuffer(view, dtype=np.uint8).setflags(write=True)
    import ctypes

    with pytest.raises((BufferError, TypeError)):
        (ctypes.c_char * 8).from_buffer(view)


def test_retained_views_stay_readable_after_close_and_delete():
    # ADR-0003 rule 1: dropping or closing the environment never frees
    # memory an export references.
    env = cartpole(2)
    env.reset()
    s = env.step(np.ones((2, 1), dtype=np.int32))
    kept = s.observations["obs"].copy()
    env.close()
    assert np.array_equal(s.observations["obs"], kept)
    assert not t.is_stale(s.observations["obs"])
    del env
    gc.collect()
    assert np.array_equal(s.observations["obs"], kept)
    assert s.rewards.tolist() == [[1.0], [1.0]]


def test_dlpack_capsules_versioned_and_legacy_free_their_exports():
    env = cartpole(3)
    env.reset()
    view = env._native.outputs()["rewards"]
    assert view.__dlpack_device__() == (1, 0)
    before = sys.getrefcount(view)
    for _ in range(100):
        cap = view.__dlpack__(max_version=(1, 0))
        assert "dltensor_versioned" in repr(cap)
        del cap
        cap = view.__dlpack__()
        assert '"dltensor"' in repr(cap)
        del cap
    gc.collect()
    assert sys.getrefcount(view) == before
    with pytest.raises(BufferError):
        view.__dlpack__(copy=True)
    with pytest.raises(BufferError):
        view.__dlpack__(dl_device=(2, 0))
    with pytest.raises(TypeError):
        view.__dlpack__(1)


def test_numpy_consumes_dlpack_zero_copy_and_read_only():
    env = cartpole(3)
    env.reset()
    view = env._native.outputs()["observations"][0]
    arr = np.from_dlpack(view)
    assert arr.shape == (3, 1, 4) and arr.dtype == np.float32
    assert not arr.flags.writeable
    assert arr.__array_interface__["data"][0] == np.asarray(view).__array_interface__["data"][0]
    env.close()
    del env, view
    gc.collect()
    assert arr.shape == (3, 1, 4)  # still alive through the export's Arc


def test_tag_borrowed_masks_and_uint8_window():
    env = t.make("tessera/Tag-v1", num_envs=2, seed=3, pursuers=2, evaders=2, radius=1)
    s = env.reset()
    assert s.observations["window"].dtype == np.uint8
    assert s.masks.shape == (2, 8, 5) and not s.masks.flags.writeable
    assert s.masks[:, :4, 0].all()


@pytest.mark.torch
def test_torch_from_dlpack_is_zero_copy_and_reports_read_only_behaviour():
    torch = pytest.importorskip("torch")
    import warnings

    env = cartpole(3)
    env.reset()
    view = env._native.outputs()["observations"][0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tensor = torch.from_dlpack(view)
    assert tuple(tensor.shape) == (3, 1, 4) and tensor.dtype == torch.float32
    assert tensor.data_ptr() == np.asarray(view).__array_interface__["data"][0]
    # PyTorch 2.15 asks for DLPack 1.0 and accepts the read-only flag
    # without warnings, but does not enforce it: the tensor is writable and
    # a write reaches the engine buffer (ADR-0003 rule 7: the borrow rules
    # still apply). The next call overwrites it.
    tensor.add_(100.0)
    assert np.asarray(view)[0, 0, 0] == tensor[0, 0, 0].item()
    env.step(np.ones((3, 1), dtype=np.int32))
    # A raw alias: it shows the overwritten values and is not trackable.
    assert torch.equal(tensor, torch.from_numpy(np.asarray(view).copy()))
    assert not bool((tensor > 50).any())
    with pytest.raises(TypeError):
        t.is_stale(tensor)
