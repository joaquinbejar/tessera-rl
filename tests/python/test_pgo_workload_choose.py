# SPDX-License-Identifier: MIT OR Apache-2.0
"""tools/pgo/workload.py's `choose` (#609): the same actions as before the
scratch array, and no allocation above the mmap threshold per call once warm."""

import importlib.util
import pathlib
import tracemalloc

import numpy as np

TOOLS = pathlib.Path(__file__).parents[2] / "tools" / "pgo"
_spec = importlib.util.spec_from_file_location("workload", TOOLS / "workload.py")
workload = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(workload)


def reference(rng, masks):
    """`choose` before #609: a fresh scores array per call."""
    return np.argmax(rng.random(np.asarray(masks).shape) * np.asarray(masks), axis=-1).astype(np.int32)


def masks_like_go9(rng, n=256):
    m = rng.random((n, 1, 82)) < 0.4
    m[..., 81] = True  # pass is always legal
    return m


def test_choose_masked_matches_the_reference_draw_for_draw():
    mrng = np.random.default_rng(7)
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    for _ in range(50):
        m = masks_like_go9(mrng)
        got, want = workload.choose(a, m, (256, 1), 82), reference(b, m)
        assert got.dtype == np.int32 and np.array_equal(got, want)
        assert np.all(m[np.arange(256), 0, got[:, 0]])


def test_choose_masked_varying_batch_shapes_match_the_reference():
    mrng = np.random.default_rng(3)
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    for n in (256, 17, 256, 1, 17):
        m = masks_like_go9(mrng, n)
        assert np.array_equal(workload.choose(a, m, (n, 1), 82), reference(b, m))


def test_choose_masked_warm_calls_allocate_no_masks_sized_array():
    rng = np.random.default_rng(1)
    m = masks_like_go9(np.random.default_rng(2))
    workload.choose(rng, m, (256, 1), 82)  # warm: the scratch array is made here
    tracemalloc.start()
    try:
        for _ in range(20):
            workload.choose(rng, m, (256, 1), 82)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # A masks-sized float64 array is 256 * 82 * 8 = 167,936 bytes, above the
    # 128 KiB mmap threshold of musl and of glibc's default; before #609 each
    # call made two. What remains is numpy's ufunc cast buffer for the bool
    # mask (8,192 float64, 64 KiB) and the small result, served from the heap.
    assert peak < 128 * 1024, peak


def test_choose_without_masks_is_uniform_and_unchanged():
    a, b = np.random.default_rng(9), np.random.default_rng(9)
    assert np.array_equal(workload.choose(a, None, (8, 2), 5), b.integers(0, 5, (8, 2), dtype=np.int32))
