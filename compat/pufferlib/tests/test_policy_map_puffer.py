# SPDX-License-Identifier: MIT OR Apache-2.0
"""The PufferLib adapter hides the slots the engine plays (#347). It is here, not
in tests/python, so ``make puffer-test`` runs it: without PufferLib it can only
skip."""

import numpy as np

import tessera_rl as t
from tessera_rl.puffer import TesseraPufferEnv, mix_seed

TAG = "tessera/Tag-v1"
TAG_CFG = {"width": 6, "height": 6, "pursuers": 2, "evaders": 2, "max_episode_steps": 100}
TAG_MAP = ["learner", "learner", "heuristic", "heuristic", "learner", "learner", "learner", "learner"]
VISIBLE = [0, 1, 4, 5, 6, 7]


def test_puffer_hides_heuristic_slots():
    env = TesseraPufferEnv(TAG, num_envs=2, policy_map=TAG_MAP, log_interval=None, **TAG_CFG)
    ref = t.make(TAG, num_envs=2, seed=mix_seed(0, 1), policy_map=TAG_MAP, outputs="owned", **TAG_CFG)
    try:
        # Six visible slots per instance: the learners, not slots 2 and 3.
        assert env.num_agents == 2 * 6
        obs, _ = env.reset(seed=1)
        assert obs.shape == (12, *env.single_observation_space.shape)
        s = ref.reset()
        leaf = next(iter(s.observations.values()))
        np.testing.assert_array_equal(obs, leaf[:, VISIBLE].reshape(obs.shape))
        obs, rewards, terminals, truncations, _ = env.step(np.zeros(12, dtype=np.int32))
        assert rewards.shape == terminals.shape == truncations.shape == (12,)
        assert env.terminal_observations.shape == obs.shape
        s = ref.step(np.zeros((2, 8), dtype=np.int32))
        leaf = next(iter(s.observations.values()))
        np.testing.assert_array_equal(obs, leaf[:, VISIBLE].reshape(obs.shape))
    finally:
        env.close()
        ref.close()
