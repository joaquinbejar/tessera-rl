# SPDX-License-Identifier: MIT OR Apache-2.0
"""``tessera/GridLanes-v1`` through the PufferLib adapter (#726): a lane-wise
environment (ADR-0090) with the adapter's defaults, and the same rows as
``tessera/Grid-v1`` on the same actions."""

import numpy as np

import tessera_rl as t
from tessera_rl.puffer import TesseraPufferEnv, mix_seed

LANES = "tessera/GridLanes-v1"
GRID = "tessera/Grid-v1"


def test_grid_lanes_puffer_steps_like_a_plain_grid_pool():
    lanes = TesseraPufferEnv(LANES, num_envs=12, log_interval=None)
    obs, _ = lanes.reset(seed=3)
    assert obs.shape == lanes.observation_space.shape and lanes.num_agents == 12
    rng = np.random.default_rng(1)
    for _ in range(50):
        lanes.send(rng.integers(0, 4, size=12, dtype=np.int32))
        o, r, d, tr, _, _, m = lanes.recv()
        assert o.shape == obs.shape and r.shape == d.shape == tr.shape == m.shape == (12,)
    lanes.close()


def test_grid_lanes_puffer_rows_equal_a_grid_pool():
    # Puffer resets on the same step; Grid-v1 is not on the adapter's
    # allowlist, so the reference is a plain pool at the adapter's seed.
    env = TesseraPufferEnv(LANES, num_envs=12, seed=4, log_interval=None)
    obs, _ = env.reset(seed=9)
    ref = t.make(GRID, num_envs=12, seed=mix_seed(4, 9), outputs="owned", auto_reset="same_step")
    s = ref.reset()
    rng = np.random.default_rng(2)
    for _ in range(80):
        # One leaf keeps its shape: (rows, 3, size, size).
        np.testing.assert_array_equal(obs, s.observations["pixels"].reshape(obs.shape))
        actions = rng.integers(0, 4, size=12, dtype=np.int32)
        obs, rewards, terminals, truncations, *_ = env.step(actions)
        s = ref.step(actions.reshape((12, 1)))
        np.testing.assert_array_equal(rewards, s.rewards[:, 0])
        np.testing.assert_array_equal(terminals, s.agent_terminated[:, 0])
        np.testing.assert_array_equal(truncations, s.agent_truncated[:, 0])
    env.close()
    ref.close()
