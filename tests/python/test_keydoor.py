# SPDX-License-Identifier: MIT OR Apache-2.0
"""Key and door from Python: ``make`` with its two observation leaves, and the
PettingZoo adapters, which pass the official API tests."""

import numpy as np
import pytest
from pettingzoo import test as pz_test

import tessera_rl as t
from tessera_rl import pz

ID = "tessera/GridKeyDoor-v1"
WAIT = 4


def test_keydoor_make_has_both_leaves_masks_and_the_agents():
    env = t.make(ID, num_envs=2, seed=5, outputs="owned", agents=3)
    assert env.agents == 4 and env.action_choices == 5
    s = env.reset()
    assert s.observations["view"].shape == (2, 4, 5, 7, 7)
    assert s.observations["carrying"].shape == (2, 4, 1)
    assert s.masks[0, 0, WAIT], "waiting is always legal"
    env.close()


@pytest.mark.parametrize("kw", [{"size": 5}, {"size": 17}, {"agents": 0}, {"agents": 5}])
def test_keydoor_make_rejects_config_out_of_range(kw):
    with pytest.raises(t.ConfigError):
        t.make(ID, num_envs=1, seed=1, **kw)


def test_keydoor_make_step_limit_truncates():
    env = t.make(ID, num_envs=1, seed=5, outputs="owned", auto_reset="disabled", max_episode_steps=2)
    env.reset()
    a = np.full((1, 4), WAIT, dtype=np.int32)
    env.step(a)
    s = env.step(a)
    assert bool(s.episode_truncated[0]) and not bool(s.episode_terminated[0])
    env.close()


@pytest.mark.parametrize("agents", [1, 3])
def test_pz_keydoor_passes_the_api_and_seed_tests(agents):
    kw = {"agents": agents, "size": 6, "max_episode_steps": 40}
    pz_test.parallel_api_test(pz.parallel_env(ID, seed=1, **kw), num_cycles=200)
    pz_test.api_test(pz.aec_env(ID, seed=1, **kw), num_cycles=200, verbose_progress=False)
    pz_test.seed_test(lambda: pz.aec_env(ID, **kw), num_cycles=100)
