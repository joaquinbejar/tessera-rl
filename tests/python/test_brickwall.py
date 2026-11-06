# SPDX-License-Identifier: MIT OR Apache-2.0
"""Brick wall from Python: ``make`` with both observations, and the Gymnasium
adapter, which passes the official environment checker."""

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import tessera_rl as t
import tessera_rl.gym as tg

ID = "tessera/BrickWall-v1"


def _leaf(obs):
    return obs["obs"] if isinstance(obs, dict) else obs


def test_brickwall_make_pixels_by_default_and_state_on_request():
    env = t.make(ID, num_envs=2, seed=5, outputs="owned")
    assert env.agents == 1 and env.action_choices == 3
    pixels = _leaf(env.reset().observations)
    assert pixels.shape == (2, 1, 1, 84, 84) and pixels.dtype == np.uint8
    env.close()
    env = t.make(ID, num_envs=1, seed=5, outputs="owned", obs="state", lives=3)
    vec = _leaf(env.reset().observations)
    assert vec.shape == (1, 1, 116) and vec.dtype == np.float32
    assert vec[0, 0, 6] == 1.0, "the ball is parked at reset"
    assert np.all((vec >= 0) & (vec <= 1))
    env.close()


@pytest.mark.parametrize("kw", [{"frameskip": 0}, {"frameskip": 9}, {"lives": 0}, {"lives": 10}])
def test_brickwall_make_rejects_config_out_of_range(kw):
    with pytest.raises(t.ConfigError):
        t.make(ID, num_envs=1, seed=1, **kw)


def test_brickwall_make_rejects_an_unknown_obs():
    with pytest.raises(t.ArgumentError):
        t.make(ID, num_envs=1, seed=1, obs="rgb")


@pytest.mark.parametrize("obs", ["pixels", "state"])
def test_gym_brickwall_passes_the_env_checker(obs):
    check_env(gymnasium.make(ID, obs=obs).unwrapped, skip_render_check=True)


def test_gym_brickwall_registers_a_4000_step_limit():
    assert gymnasium.spec(ID).max_episode_steps == 4000
    assert tg is not None
