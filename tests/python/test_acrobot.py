# SPDX-License-Identifier: MIT OR Apache-2.0
"""Acrobot from Python: ``make``, the Gymnasium registration and the
env checker."""

import gymnasium
import numpy as np
from gymnasium.utils.env_checker import check_env

import tessera_rl as t
import tessera_rl.gym as tg

ID = "tessera/Acrobot-v1"


def test_acrobot_make_shapes_and_limit():
    env = t.make(ID, num_envs=3, seed=4, outputs="owned")
    assert env.agents == 1 and env.action_choices == 3
    s = env.reset()
    obs = s.observations["obs"]
    assert obs.shape == (3, 1, 6) and obs.dtype == np.float32
    assert np.all(obs[:, 0, 0] > 0.99)  # cos of a small angle
    for _ in range(500):
        s = env.step(np.ones((3, 1), dtype=np.int32))
        assert np.all(s.rewards == -1.0)
    assert s.episode_truncated.all() and not s.episode_terminated.any()
    env.close()


def test_acrobot_gym_spaces_match_gymnasium_and_pass_the_checker():
    ours = tg.make(ID, num_envs=2)
    ref = gymnasium.make("Acrobot-v1")
    assert ours.single_observation_space == ref.observation_space
    assert ours.single_action_space == ref.action_space
    assert gymnasium.spec(ID).max_episode_steps == 500
    check_env(gymnasium.make(ID).unwrapped, skip_render_check=True)
    ours.close()
