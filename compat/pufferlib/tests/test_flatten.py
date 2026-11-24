# SPDX-License-Identifier: MIT OR Apache-2.0
"""Observations with several leaves through the PufferLib adapter (SPEC 11.1,
#423): layout-order flattening, ADR-0082 ``Set`` leaves and the dtype rule."""

import numpy as np
import pufferlib.vector
import pytest

import tessera_rl as t
from tessera_rl.puffer import TesseraPufferEnv, flat_dtype, make_puffer, mix_seed

GATHER = "tessera/Gather-v1"
MAX = 8


def reference(num_envs, base):
    """A plain pool at the adapter's instance seeds, owned outputs."""
    return t.make(GATHER, num_envs=num_envs, seed=base, outputs="owned")


def flat_rows(obs: dict, rows: int) -> np.ndarray:
    leaves = [obs["agent"], obs["items"], obs["items_mask"]]
    return np.concatenate([np.asarray(x).reshape((rows, -1)).astype(np.int32) for x in leaves], axis=1)


@pytest.mark.parametrize(
    ("dtypes", "want"),
    [
        (["int32", "int32", "bool"], "int32"),
        (["float32", "bool"], "float32"),
        (["float32", "int32"], "float64"),
        (["uint8", "int8"], "int16"),
        (["uint32", "int32"], "int64"),
        (["float32", "int16"], "float32"),
        (["float64", "uint32"], "float64"),
    ],
)
def test_flat_dtype_promotion_is_numpy_result_type(dtypes, want):
    assert flat_dtype(dtypes) == np.dtype(want)


@pytest.mark.parametrize(
    "dtypes", [["int64", "float32"], ["int64", "float64"], ["uint64", "int8"], ["float32", "int32", "int64"]]
)
def test_flat_dtype_refuses_lossy_promotions(dtypes):
    with pytest.raises(t.ArgumentError):
        flat_dtype(dtypes)


def test_gather_space_is_the_flattened_leaves_in_layout_order():
    env = TesseraPufferEnv(GATHER, num_envs=2, size=6)
    space = env.single_observation_space
    assert space.dtype == np.int32 and space.shape == (2 + MAX * 2 + MAX,)
    assert (space.low == 0).all()
    np.testing.assert_array_equal(space.high, [5] * (2 + 2 * MAX) + [1] * MAX)
    env.close()


@pytest.mark.parametrize("workers", [0, 2])
def test_gather_rows_match_the_engine_leaves(workers):
    env = TesseraPufferEnv(GATHER, num_envs=3, seed=4, workers=workers, log_interval=None)
    obs, _ = env.reset(seed=9)
    ref = reference(3, mix_seed(4, 9))
    s = ref.reset()
    rng = np.random.default_rng(0)
    for _ in range(40):
        np.testing.assert_array_equal(obs, flat_rows(s.observations, 3))
        assert obs.dtype == np.int32
        mask = obs[:, 2 + 2 * MAX :].astype(bool)
        items = obs[:, 2 : 2 + 2 * MAX].reshape((3, MAX, 2))
        assert (items[~mask] == 0).all(), "absent slots are zero"
        a = rng.integers(0, 4, 3, dtype=np.int32)
        obs = env.step(a)[0]
        s = ref.step(a.reshape((3, 1)))
    env.close()
    ref.close()


def test_gather_rows_are_one_buffer_refilled_in_place():
    env = TesseraPufferEnv(GATHER, num_envs=2, log_interval=None)
    obs, _ = env.reset(seed=1)
    assert obs.flags.owndata and obs.flags.c_contiguous
    again = env.step(np.zeros(2, dtype=np.int32))[0]
    assert again is obs
    env.close()


def test_gather_terminal_observations_are_flattened_and_preallocated():
    env = TesseraPufferEnv(GATHER, num_envs=2, log_interval=None)
    env.reset(seed=1)
    first = env.terminal_observations
    assert first.shape == (2, 2 + 3 * MAX) and first.dtype == np.int32
    env.step(np.zeros(2, dtype=np.int32))
    assert env.terminal_observations is first
    env.close()


def test_gather_serial_backend_copies_the_same_rows():
    vec = pufferlib.vector.make(make_puffer(GATHER, num_envs=2, log_interval=None), backend=pufferlib.vector.Serial, num_envs=2)
    native = TesseraPufferEnv(GATHER, num_envs=2, seed=0, log_interval=None)
    obs, _ = vec.reset(seed=3)
    np.testing.assert_array_equal(obs[:2], native.reset(seed=3)[0])
    assert obs.dtype == np.int32 and obs.shape == (4, 2 + 3 * MAX)
    vec.close()
    native.close()


def test_gather_pipelined_halves_match_sync():
    pipe = make_puffer(GATHER, num_envs=4, chunk_size=2, mode="pipelined", log_interval=None)()
    sync = TesseraPufferEnv(GATHER, num_envs=4, log_interval=None)
    whole, _ = sync.reset(seed=2)
    pipe.reset(seed=2)
    first = pipe.recv()[0]
    kept = first.copy()
    pipe.send(np.zeros(2, dtype=np.int32))
    second = pipe.recv()[0]
    assert second is not first, "each half has its own rows"
    np.testing.assert_array_equal(first, kept)
    np.testing.assert_array_equal(np.concatenate([kept, second]), whole)
    pipe.close()
    sync.close()
