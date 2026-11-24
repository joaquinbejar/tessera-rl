# SPDX-License-Identifier: MIT OR Apache-2.0
"""Gather (``tessera/Gather-v1``, ADR-0082) from Python: the ``Set``
observation's leaves and layout, and the Gymnasium and PettingZoo adapters,
which pass their own checkers."""

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from pettingzoo import test as pz_test

import tessera_rl as t
from tessera_rl import gym as tg
from tessera_rl import pz

ID = "tessera/Gather-v1"
MAX = 8


def make(**kw):
    return t.make(ID, num_envs=2, seed=5, outputs="owned", auto_reset="disabled", **kw)


def test_gather_layout_exports_items_then_mask_and_the_sets_index():
    env = make()
    layout = env._native.layout()
    names = [(name, dtype, tuple(shape)) for name, dtype, shape, *_ in layout["leaves"]]
    assert names == [("agent", "int32", (2,)), ("items", "int32", (MAX, 2)), ("items_mask", "bool", (MAX,))]
    assert layout["sets"] == [("items", MAX, [1, 2])]
    env.close()


def test_gather_absent_slots_are_zero_and_present_rows_on_the_grid():
    env = make(items=5, size=6)
    s = env.reset()
    for _ in range(30):
        items = s.observations["items"][:, 0]
        mask = s.observations["items_mask"][:, 0]
        assert mask.dtype == np.bool_
        assert not mask[:, 5:].any(), "slots past the configured items are absent"
        assert (items[~mask] == 0).all()
        assert ((items[mask] >= 0) & (items[mask] < 6)).all()
        s = env.step(np.zeros((2, 1), dtype=np.int32))
    env.close()


@pytest.mark.parametrize("kw", [{"items": 0}, {"items": 9}, {"size": 3}, {"size": 17}])
def test_gather_make_rejects_config_out_of_range(kw):
    with pytest.raises(t.ConfigError):
        make(**kw)


def test_gather_gym_passes_the_environment_checker():
    env = gymnasium.make(ID, seed=1)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("outputs", ["owned", "borrowed"])
def test_gather_gym_mask_is_multibinary_int8_and_items_a_box(outputs):
    env = tg.make(ID, num_envs=3, seed=2, outputs=outputs)
    space = env.single_observation_space
    assert isinstance(space["items_mask"], gymnasium.spaces.MultiBinary)
    assert space["items_mask"].n == MAX
    assert space["items"].shape == (MAX, 2) and space["items"].dtype == np.int32
    obs, _ = env.reset()
    assert obs["items_mask"].dtype == np.int8 and obs["items_mask"].shape == (3, MAX)
    assert set(np.unique(obs["items_mask"])) <= {0, 1}
    assert env.observation_space.contains(obs)
    obs, *_ = env.step(np.array([0, 1, 2]))
    assert env.observation_space.contains(obs)
    env.close()


def test_gather_pz_passes_the_api_and_seed_tests():
    kw = {"size": 6, "items": 4, "max_episode_steps": 40}
    pz_test.parallel_api_test(pz.parallel_env(ID, seed=1, **kw), num_cycles=200)
    pz_test.api_test(pz.aec_env(ID, seed=1, **kw), num_cycles=200, verbose_progress=False)
    pz_test.seed_test(lambda: pz.aec_env(ID, **kw), num_cycles=100)


def test_gather_pz_mask_is_int8_multibinary():
    env = pz.parallel_env(ID, seed=1)
    obs, _ = env.reset(seed=3)
    (agent,) = env.agents
    assert isinstance(env.observation_space(agent)["items_mask"], gymnasium.spaces.MultiBinary)
    assert obs[agent]["items_mask"].dtype == np.int8
    assert env.observation_space(agent).contains(obs[agent])
    env.close()
