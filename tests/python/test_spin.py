# SPDX-License-Identifier: MIT OR Apache-2.0
"""The ``spin`` pool option: the worker spin budget, validated at make()."""

import numpy as np
import pytest

import tessera_rl as t

CARTPOLE = "tessera/CartPole-v1"


def run(env, steps=8):
    env.reset(seed=3)
    out = []
    for i in range(steps):
        obs = env.step(np.full(env.num_envs, i % 2))[0]
        out.append(np.array(obs, copy=True))
    env.close()
    return out


@pytest.mark.parametrize("spin", [0, 2_000, 200_000, 2**32 - 1])
def test_make_spin_sync_with_workers_same_trajectory(spin):
    ref = run(t.make(CARTPOLE, num_envs=4, chunk_size=2))
    got = run(t.make(CARTPOLE, num_envs=4, chunk_size=2, workers=2, spin=spin))
    for a, b in zip(ref, got):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("mode_kw", [{"mode": "pipelined"}, {"mode": "async", "batch_size": 2}])
def test_make_spin_split_modes_without_workers_accepted(mode_kw):
    t.make(CARTPOLE, num_envs=4, chunk_size=2, spin=0, **mode_kw).close()


def test_make_spin_sync_without_workers_raises_argument_error():
    with pytest.raises(t.ArgumentError, match="spin needs workers"):
        t.make(CARTPOLE, num_envs=2, spin=10)


@pytest.mark.parametrize("bad", [2**32, 1.5, "fast", True])
def test_make_spin_wrong_type_or_range_raises_type_error(bad):
    with pytest.raises(TypeError, match="spin"):
        t.make(CARTPOLE, num_envs=2, workers=1, spin=bad)


def test_make_spin_negative_raises_overflow_error():
    with pytest.raises(OverflowError):
        t.make(CARTPOLE, num_envs=2, workers=1, spin=-1)


def test_make_spin_none_without_workers_is_the_default():
    t.make(CARTPOLE, num_envs=2, spin=None).close()
