# SPDX-License-Identifier: MIT OR Apache-2.0
"""Pendulum from Python: ``make``, the Gymnasium registration and checker."""

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import tessera_rl as t
import tessera_rl.gym as tg
from tessera_rl import _native

ID = "tessera/Pendulum-v1"


def test_pendulum_make_shapes_and_limit():
    env = t.make(ID, num_envs=3, seed=4, outputs="owned")
    assert env.agents == 1 and env.action_choices == 0 and env.action_bounds == (-2.0, 2.0)
    s = env.reset()
    obs = s.observations["obs"]
    assert obs.shape == (3, 1, 3) and obs.dtype == np.float32
    for _ in range(200):
        s = env.step(np.zeros((3, 1), dtype=np.float32))
    assert s.episode_truncated.all() and not s.episode_terminated.any()
    env.close()


def test_pendulum_native_refuses_out_of_range():
    env = t.make(ID, num_envs=1, seed=4, outputs="owned")
    env.reset()
    for bad in (np.nan, np.inf, 2.0000002):
        with pytest.raises(_native.ActionError):
            env.step(np.array([bad], dtype=np.float32))
    env.close()


def test_pendulum_gym_spaces_match_gymnasium_and_pass_the_checker():
    ours = tg.make(ID, num_envs=2)
    ref = gymnasium.make("Pendulum-v1")
    assert ours.single_observation_space == ref.observation_space
    assert ours.single_action_space == ref.action_space
    assert gymnasium.spec(ID).max_episode_steps == 200
    check_env(gymnasium.make(ID).unwrapped, skip_render_check=True)
    ours.close()
