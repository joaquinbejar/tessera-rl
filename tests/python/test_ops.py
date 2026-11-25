# SPDX-License-Identifier: MIT OR Apache-2.0
"""Logistics environments from Python (#383): ``tessera/JobShop-v1`` through
``make`` and Gymnasium, and ``tessera/Vrp-v1`` through ``make`` and the
PettingZoo AEC adapter, with one vehicle and with a fleet."""

import numpy as np
import pytest

import tessera_rl as t
import tessera_rl.gym as tg
from tessera_rl import pz


def first_legal(mask_row):
    return int(np.flatnonzero(mask_row)[0])


def test_jobshop_runs_to_the_end_and_the_return_is_minus_the_makespan():
    env = t.make("tessera/JobShop-v1", num_envs=2, seed=3, outputs="owned", jobs=4, machines=3)
    s = env.reset()
    obs = s.observations
    assert np.asarray(obs["jobs"]).shape == (2, 1, 4, 5)
    assert np.asarray(obs["machines"]).shape == (2, 1, 3, 2)
    total = np.zeros(2, dtype=np.float32)
    for k in range(12):
        masks = np.asarray(s.masks)[:, 0]
        a = np.array([[first_legal(m)] for m in masks], dtype=np.int32)
        s = env.step(a)
        total += np.asarray(s.rewards)[:, 0]
        ended = np.asarray(s.episode_terminated)
        assert ended.all() == (k == 11)
    assert (total < 0).all() and (total == np.round(total)).all()
    env.close()


def test_jobshop_gymnasium_adapter():
    env = tg.make("tessera/JobShop-v1", num_envs=2, seed=1, jobs=3, machines=2)
    assert env.single_action_space.n == 3
    obs, info = env.reset()
    assert info["action_mask"].shape == (2, 3)
    for k in range(6):
        actions = np.array([[first_legal(m)] for m in info["action_mask"]], dtype=np.int64)
        obs, reward, terminated, truncated, info = env.step(actions)
        assert terminated.all() == (k == 5) and not truncated.any()
    env.close()


@pytest.mark.parametrize("vehicles", [1, 3])
def test_vrp_aec_runs_every_vehicle_until_all_are_home(vehicles):
    env = pz.aec_env("tessera/Vrp-v1", customers=10, vehicles=vehicles, grid=30, capacity=20)
    env.reset(seed=4)
    assert len(env.agents) == vehicles
    steps = 0
    for agent in env.agent_iter(max_iter=100):
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            env.step(None)
            continue
        env.step(first_legal(obs["action_mask"]))
        steps += 1
    assert 0 < steps <= 20, "at most 2 * customers steps"
    env.close()


def test_vrp_refusals():
    with pytest.raises(t.ConfigError):
        t.make("tessera/Vrp-v1", capacity=5, max_demand=9)
    with pytest.raises(t.ConfigError):
        t.make("tessera/Vrp-v1", vehicles=9)
    with pytest.raises(t.ConfigError):
        t.make("tessera/JobShop-v1", jobs=65)


def test_vrp_one_vehicle_through_gymnasium_and_fleets_refused():
    env = tg.make("tessera/Vrp-v1", num_envs=2, seed=2, customers=6, grid=30)
    assert env.single_action_space.n == 7
    obs, info = env.reset()
    for _ in range(12):
        actions = np.array([[first_legal(m)] for m in info["action_mask"]], dtype=np.int64)
        obs, reward, terminated, truncated, info = env.step(actions)
    env.close()
    with pytest.raises(t.ArgumentError, match="multi-agent"):
        tg.make("tessera/Vrp-v1", vehicles=3)
