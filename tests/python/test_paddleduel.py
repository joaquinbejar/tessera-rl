# SPDX-License-Identifier: MIT OR Apache-2.0
"""Paddle duel from Python: ``make`` with both observations and the unused
slot, and the PettingZoo adapters, which pass the official API tests."""

import numpy as np
import pytest
from pettingzoo import test as pz_test

import tessera_rl as t
from tessera_rl import pz

ID = "tessera/PaddleDuel-v1"


def _leaf(obs):
    return obs["obs"] if isinstance(obs, dict) else obs


def test_paddleduel_make_pixels_by_default_and_state_on_request():
    env = t.make(ID, num_envs=2, seed=5, outputs="owned", agents=2)
    assert env.agents == 2 and env.action_choices == 3
    pixels = _leaf(env.reset().observations)
    assert pixels.shape == (2, 2, 1, 84, 84) and pixels.dtype == np.uint8
    env.close()
    env = t.make(ID, num_envs=1, seed=5, outputs="owned", obs="state")
    s = env.reset()
    vec = _leaf(s.observations)
    assert vec.shape == (1, 2, 9) and vec.dtype == np.float32
    assert not s.alive[0, 1] and not s.acting[0, 1], "slot 1 is unused with one agent"
    assert np.all(vec[0, 1] == 0)
    env.close()


@pytest.mark.parametrize(
    "kw", [{"agents": 3}, {"frameskip": 9}, {"max_score": 0}, {"max_score": 22}, {"bounce": 11}]
)
def test_paddleduel_make_rejects_config_out_of_range(kw):
    with pytest.raises(t.ConfigError):
        t.make(ID, num_envs=1, seed=1, **kw)


def test_paddleduel_make_rejects_an_unknown_obs():
    with pytest.raises(t.ArgumentError):
        t.make(ID, num_envs=1, seed=1, obs="rgb")


@pytest.mark.parametrize("agents", [1, 2])
@pytest.mark.parametrize("obs", ["pixels", "state"])
def test_pz_paddleduel_passes_the_api_and_seed_tests(agents, obs):
    kw = {"agents": agents, "obs": obs, "max_score": 1, "max_episode_steps": 60}
    pz_test.parallel_api_test(pz.parallel_env(ID, seed=1, **kw), num_cycles=200)
    pz_test.api_test(pz.aec_env(ID, seed=1, **kw), num_cycles=200, verbose_progress=False)
    pz_test.seed_test(lambda: pz.aec_env(ID, **kw), num_cycles=100)
