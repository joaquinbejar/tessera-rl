# SPDX-License-Identifier: MIT OR Apache-2.0
"""The PufferLib adapter's ``terminal_observations`` reads
``Env.terminal_observations`` (#545): the same values, dtype and shape as the
engine's raw terminal rows it read before."""

import numpy as np
import pytest

from tessera_rl.puffer import TesseraPufferEnv

CASES = {
    "tessera/CartPole-v1": {"max_episode_steps": 3},
    "tessera/Gather-v1": {"max_episode_steps": 3},
    "tessera/GridKeyDoor-v1": {"agents": 4, "max_episode_steps": 3,
                               "policy_map": ["learner", "heuristic", "learner", "heuristic"]},
}


def raw_rows(env) -> np.ndarray:
    """The previous implementation: the engine's raw rows, flattened and
    gathered like the adapter's."""
    leaves = [np.asarray(leaf) for leaf in env._env._native.outputs()["terminal_observations"]]
    n, a = env._instances, env._agents
    rows = np.concatenate([leaf.reshape((n, a, -1)) for leaf in leaves], axis=2)
    if env._visible is not None:
        rows = rows[:, env._visible]
    return rows.reshape((env.num_agents, *env.single_observation_space.shape)).astype(env.single_observation_space.dtype)


@pytest.mark.parametrize("env_id", sorted(CASES))
def test_puffer_terminal_observations_unchanged(env_id):
    env = TesseraPufferEnv(env_id, num_envs=3, seed=2, log_interval=None, **CASES[env_id])
    try:
        env.reset(seed=4)
        nonzero = 0
        for _ in range(7):
            env.step(np.zeros(env.num_agents, dtype=np.int32))
            got = env.terminal_observations
            want = raw_rows(env)
            assert got.shape == want.shape and got.dtype == want.dtype
            np.testing.assert_array_equal(got, want)
            nonzero += int(got.any())
        assert nonzero > 0, "no terminal row was exercised"
    finally:
        env.close()
