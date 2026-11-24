# SPDX-License-Identifier: MIT OR Apache-2.0
"""Snake from Python: ``make`` with both observations and the mask, and the
PettingZoo adapters, which pass the official API tests."""

import numpy as np
import pytest
from pettingzoo import test as pz_test

import tessera_rl as t
from tessera_rl import pz

ID = "tessera/Snake-v1"
WEST = 3


def test_snake_make_pixels_by_default_and_state_on_request():
    env = t.make(ID, num_envs=2, seed=5, outputs="owned", agents=2)
    assert env.agents == 2 and env.action_choices == 4
    s = env.reset()
    pixels = s.observations
    pixels = pixels["obs"] if isinstance(pixels, dict) else pixels
    assert pixels.shape == (2, 2, 1, 84, 84) and pixels.dtype == np.uint8
    assert not s.masks[0, 0, WEST], "slot 0 heads east, so west reverses"
    env.close()
    env = t.make(ID, num_envs=1, seed=5, outputs="owned", size=8, obs="state")
    flags = env.reset().observations
    flags = flags["obs"] if isinstance(flags, dict) else flags
    assert flags.shape == (1, 2, 5 * 64) and flags.dtype == np.float32
    assert set(np.unique(flags[0, 0])) <= {0.0, 1.0}
    env.close()


@pytest.mark.parametrize("kw", [{"size": 5}, {"size": 33}, {"agents": 0}, {"agents": 3}])
def test_snake_make_rejects_config_out_of_range(kw):
    with pytest.raises(t.ConfigError):
        t.make(ID, num_envs=1, seed=1, **kw)


def test_snake_make_rejects_an_unknown_obs():
    with pytest.raises(t.ArgumentError):
        t.make(ID, num_envs=1, seed=1, obs="rgb")


@pytest.mark.parametrize("agents", [1, 2])
@pytest.mark.parametrize("obs", ["pixels", "state"])
def test_pz_snake_passes_the_api_and_seed_tests(agents, obs):
    kw = {"agents": agents, "size": 8, "obs": obs, "max_episode_steps": 60}
    pz_test.parallel_api_test(pz.parallel_env(ID, seed=1, **kw), num_cycles=200)
    pz_test.api_test(pz.aec_env(ID, seed=1, **kw), num_cycles=200, verbose_progress=False)
    pz_test.seed_test(lambda: pz.aec_env(ID, **kw), num_cycles=100)


@pytest.mark.parametrize("n", [1, 2])
def test_pz_snake_agents_int_is_the_config_key(n):
    # ADR-0079 deviation 9: an integer `agents` is Snake's own key, passed
    # through to the environment config, in both forms.
    par = pz.parallel_env(ID, seed=1, agents=n, size=8, obs="state")
    aec = pz.aec_env(ID, seed=1, agents=n, size=8, obs="state")
    try:
        assert len(par.possible_agents) == n
        assert len(aec.possible_agents) == n
    finally:
        par.close()
        aec.close()


def test_pz_snake_agents_str_is_the_adapter_option():
    # A string is the adapter option: with no map every slot is a learner,
    # so "learner" and "all" show the same agents (the default count).
    default = pz.parallel_env(ID, seed=1, size=8, obs="state")
    learner = pz.parallel_env(ID, seed=1, agents="learner", size=8, obs="state")
    try:
        assert learner.possible_agents == default.possible_agents
        _, infos = learner.reset(seed=1)
        assert {infos[a]["policy_group"] for a in learner.agents} == {"learner"}
    finally:
        default.close()
        learner.close()
    with pytest.raises(t.ArgumentError):
        pz.parallel_env(ID, seed=1, agents="some", size=8, obs="state")
