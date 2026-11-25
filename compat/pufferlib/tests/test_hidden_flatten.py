# SPDX-License-Identifier: MIT OR Apache-2.0
"""Hidden heuristic slots on an environment with several observation leaves
(#552): GridKeyDoor-v1, four simultaneous agents, leaves ``view`` ``u8[5, 7, 7]``
and ``carrying`` ``u8[1]``. The adapter flattens every engine row, then keeps
the visible slots, so a hidden slot is in no flattened row."""

import numpy as np
import pufferlib.vector

import tessera_rl as t
from tessera_rl.puffer import TesseraPufferEnv, make_puffer, mix_seed

KEYDOOR = "tessera/GridKeyDoor-v1"
CFG = {"agents": 4, "size": 8}
#: Episodes end often, so terminal observations differ per slot.
STEPS = {"max_episode_steps": 7}
MAP = ["learner", "heuristic", "learner", "heuristic"]
VISIBLE = [0, 2]
TOTAL = 5 * 7 * 7 + 1
N = 3


def reference(seed, **kw):
    """A plain pool at the adapter's instance seeds and policy map."""
    return t.make(KEYDOOR, num_envs=N, seed=seed, policy_map=MAP, auto_reset="same_step", outputs="owned", **CFG, **kw)


def visible_rows(leaves) -> np.ndarray:
    """The engine leaves ``[N, A, ...]`` flattened in layout order, visible slots only."""
    view, carrying = (np.asarray(leaf) for leaf in leaves)
    rows = np.concatenate([view.reshape((N, 4, -1)), carrying.reshape((N, 4, -1))], axis=2)
    return rows[:, VISIBLE].reshape((N * len(VISIBLE), TOTAL))


def full_actions(actions: np.ndarray) -> np.ndarray:
    full = np.zeros((N, 4), dtype=np.int32)
    full[:, VISIBLE] = actions.reshape((N, len(VISIBLE)))
    return full


def test_hidden_slots_space_and_rows():
    env = TesseraPufferEnv(KEYDOOR, num_envs=N, policy_map=MAP, log_interval=None, **CFG)
    try:
        assert env.num_agents == N * len(VISIBLE)
        space = env.single_observation_space
        assert space.shape == (TOTAL,) and space.dtype == np.uint8
        assert (space.low == 0).all() and (space.high == 1).all()
        obs, _ = env.reset(seed=7)
        assert obs.shape == (N * len(VISIBLE), TOTAL)
    finally:
        env.close()


def test_hidden_slots_rows_match_the_visible_engine_leaves():
    env = TesseraPufferEnv(KEYDOOR, num_envs=N, seed=2, policy_map=MAP, log_interval=None, **CFG, **STEPS)
    ref = reference(mix_seed(2, 7), **STEPS)
    ended = nonzero = 0
    rng = np.random.default_rng(0)
    try:
        obs, _ = env.reset(seed=7)
        s = ref.reset()
        names = [leaf[0] for leaf in ref._native.layout()["leaves"]]
        assert names == ["view", "carrying"]
        for step in range(60):
            np.testing.assert_array_equal(obs, visible_rows([s.observations[n] for n in names]), err_msg=f"step {step}")
            a = rng.integers(0, 5, N * len(VISIBLE), dtype=np.int32)
            obs, rewards, terminals, truncations, _ = env.step(a)
            s = ref.step(full_actions(a))
            np.testing.assert_array_equal(rewards, np.asarray(s.rewards)[:, VISIBLE].reshape(-1))
            np.testing.assert_array_equal(terminals, np.asarray(s.agent_terminated)[:, VISIBLE].reshape(-1))
            terminal = env.terminal_observations
            assert terminal.shape == obs.shape
            np.testing.assert_array_equal(terminal, visible_rows(ref._native.outputs()["terminal_observations"]))
            nonzero += int(terminal.any())
            ended += int(np.asarray(s.episode_truncated).sum() + np.asarray(s.episode_terminated).sum())
        assert ended > 0 and nonzero > 0, "terminal_observations was never exercised"
    finally:
        env.close()
        ref.close()


def test_hidden_slots_serial_backend_copies_the_same_rows():
    creator = make_puffer(KEYDOOR, num_envs=N, policy_map=MAP, log_interval=None, **CFG)
    vec = pufferlib.vector.make(creator, backend=pufferlib.vector.Serial, num_envs=1)
    native = TesseraPufferEnv(KEYDOOR, num_envs=N, seed=0, policy_map=MAP, log_interval=None, **CFG)
    try:
        obs, _ = vec.reset(seed=3)
        np.testing.assert_array_equal(obs, native.reset(seed=3)[0])
        a = np.ones(N * len(VISIBLE), dtype=np.int32)
        np.testing.assert_array_equal(vec.step(a)[0], native.step(a)[0])
    finally:
        vec.close()
        native.close()
