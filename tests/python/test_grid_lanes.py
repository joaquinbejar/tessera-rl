# SPDX-License-Identifier: MIT OR Apache-2.0
"""``tessera/GridLanes-v1`` from Python (#726): the reference grid world on a
lane-wise state (ADR-0090). Its outputs equal ``tessera/Grid-v1``'s on the
same config and actions; Async and a chunk size that is not a multiple of 8
are refused; the defaults and the Gymnasium adapter build it. The wheel
goldens (``tools/wheel_golden.py``) replay its Rust fixtures."""

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import tessera_rl as t
from tessera_rl import gym as tg

ID = "tessera/GridLanes-v1"
GRID = "tessera/Grid-v1"
ROOMS = b"....#...\n....#...\n........\n....#...\n###.####\n....#...\n........\n....#...\n"
FIELDS = ("rewards", "agent_terminated", "agent_truncated", "alive", "episode_terminated", "episode_truncated", "acting")


def pair(**kw):
    return [t.make(env_id, outputs="owned", **kw) for env_id in (ID, GRID)]


def same(a, b):
    assert a.observations.keys() == b.observations.keys()
    for k in a.observations:
        assert np.array_equal(a.observations[k], b.observations[k]), k
    for f in FIELDS:
        assert np.array_equal(getattr(a, f), getattr(b, f)), f


# 20 instances in chunks of 8: two lane groups and a tail of 4. The walled
# config truncates every 7 steps, so resets land mid-group.
@pytest.mark.parametrize("auto_reset", ["next_step", "same_step"])
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("cfg", [{"size": 6, "max_steps": 40}, {"size": 8, "max_steps": 7, "layout": ROOMS}])
def test_grid_lanes_outputs_equal_grid_on_the_same_actions(auto_reset, workers, cfg):
    lanes, grid = pair(num_envs=20, seed=11, chunk_size=8, workers=workers, auto_reset=auto_reset, **cfg)
    same(lanes.reset(), grid.reset())
    rng = np.random.default_rng(3)
    for _ in range(120):
        actions = rng.integers(0, 4, size=(20, 1), dtype=np.int32)
        same(lanes.step(actions), grid.step(actions))
    lanes.close()
    grid.close()


def test_grid_lanes_default_options_build_and_step():
    # No chunk_size: the default (64) is a multiple of 8.
    env = t.make(ID, num_envs=3, seed=1)
    assert env.agents == 1 and env.action_choices == 4
    env.reset()
    env.step(np.zeros((3, 1), np.int32))
    env.close()


@pytest.mark.parametrize("chunk", [1, 4, 12])
def test_grid_lanes_chunk_size_not_a_multiple_of_8_is_config_error(chunk):
    with pytest.raises(t.ConfigError, match="multiple of 8"):
        t.make(ID, num_envs=16, chunk_size=chunk)


def test_grid_lanes_async_is_config_error():
    with pytest.raises(t.ConfigError, match="Async mode does not support SoA lane environments"):
        t.make(ID, num_envs=16, mode="async", batch_size=8)


def test_grid_lanes_pipelined_equals_sync():
    a = t.make(ID, num_envs=32, seed=4, chunk_size=8, outputs="owned")
    b = t.make(ID, num_envs=32, seed=4, chunk_size=8, outputs="owned", mode="pipelined")
    a.reset()
    b.reset()
    (h0, h1) = b.halves
    rng = np.random.default_rng(5)
    for _ in range(20):
        actions = rng.integers(0, 4, size=(32, 1), dtype=np.int32)
        s = a.step(actions)
        b.send(0, actions[h0.start : h0.stop])
        b.send(1, actions[h1.start : h1.stop])
        r0, r1 = b.recv(), b.recv()
        assert (r0.half, r1.half) == (0, 1)
        pixels = np.concatenate([r0.step.observations["pixels"], r1.step.observations["pixels"]])
        assert np.array_equal(s.observations["pixels"], pixels)
        assert np.array_equal(s.rewards, np.concatenate([r0.step.rewards, r1.step.rewards]))
    a.close()
    b.close()


def test_grid_lanes_is_registered():
    by_id = {r["id"]: r for r in t.envs()}
    assert by_id[ID]["adapters"] == frozenset()


def test_grid_lanes_gym_passes_the_environment_checker():
    env = gymnasium.make(ID, seed=1)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def test_grid_lanes_gym_vector_equals_grid():
    a, b = tg.make(ID, num_envs=10, seed=3), tg.make(GRID, num_envs=10, seed=3)
    oa, _ = a.reset()
    ob, _ = b.reset()
    assert np.array_equal(oa["pixels"], ob["pixels"])
    for _ in range(30):
        actions = np.full(10, 1)
        ra, rb = a.step(actions), b.step(actions)
        assert np.array_equal(ra[0]["pixels"], rb[0]["pixels"])
        for x, y in zip(ra[1:4], rb[1:4]):
            assert np.array_equal(x, y)
    a.close()
    b.close()
