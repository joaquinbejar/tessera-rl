# SPDX-License-Identifier: MIT OR Apache-2.0
"""The reference grid world from Python: ``make`` and the Gymnasium adapter,
which passes Gymnasium's own environment checker."""

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import tessera_rl as t
from tessera_rl import gym as tg

ID = "tessera/Grid-v1"
NORTH = 0


def make(**kw):
    return t.make(ID, num_envs=1, seed=5, outputs="owned", auto_reset="disabled", **kw)


def test_grid_make_has_the_layout_of_the_size():
    env = make(size=5)
    assert env.agents == 1 and env.action_choices == 4
    s = env.reset()
    pixels = s.observations["pixels"]
    assert pixels.shape == (1, 1, 3, 5, 5) and pixels.dtype == np.uint8
    assert pixels[0, 0, 0, 0, 0] == 255, "the agent starts at the top left"
    env.close()


def test_grid_make_walking_off_terminates_with_minus_one():
    env = make()
    env.reset()
    s = env.step(np.full((1, 1), NORTH, dtype=np.int32))
    assert bool(s.episode_terminated[0])
    assert s.rewards[0, 0] == -1.0
    env.close()


@pytest.mark.parametrize("kw", [{"size": 3}, {"size": 65}, {"max_steps": 0}, {"size": 70_000}])
def test_grid_make_rejects_config_out_of_range(kw):
    with pytest.raises(t.ConfigError):
        make(**kw)


def test_grid_gym_passes_the_environment_checker():
    env = gymnasium.make(ID, seed=1)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def test_grid_gym_vector_steps_a_batch():
    env = tg.make(ID, num_envs=2, seed=3)
    env.reset()
    _, rewards, terminated, _, _ = env.step(np.array([NORTH, NORTH]))
    assert rewards.tolist() == [-1.0, -1.0] and terminated.all()
    env.close()


def test_grid_gym_spreads_a_single_bound_over_the_whole_leaf():
    env = gymnasium.make(ID, size=4).unwrapped
    box = env.observation_space["pixels"]
    assert box.shape == (3, 4, 4)
    assert (box.low == 0).all() and (box.high == 255).all()
    env.close()


def test_gym_bounds_are_one_for_all_or_one_per_element():
    leaf = lambda low, high, shape: {"leaves": [("", "u1", shape, low, high)]}  # noqa: E731
    box = tg._single_space(leaf(list(range(6)), [9] * 6, [2, 3]))
    assert box.low.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert tg._single_space(leaf([1, 2, 3, 4], [9] * 4, [4, 1])).low.ravel().tolist() == [1, 2, 3, 4]
    assert (tg._single_space(leaf([1], [9], [2, 2])).low == 1).all()
    with pytest.raises(ValueError, match="reshape"):
        tg._single_space(leaf([1, 2, 3, 4], [9] * 4, [4, 4]))
