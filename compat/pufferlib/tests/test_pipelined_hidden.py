# SPDX-License-Identifier: MIT OR Apache-2.0
"""Pipelined mode hides a policy map's heuristic slots, as Sync mode does
(#558): a half's rows are its visible slots, copied into arrays the env owns,
and ``send`` takes one action per visible slot."""

import numpy as np
import pytest

import tessera_rl as t
from tessera_rl.puffer import TesseraPufferEnv, make_puffer, mix_seed

TAG = "tessera/Tag-v1"
KEYDOOR = "tessera/GridKeyDoor-v1"
CASES = {
    # The issue's reproduction: slots 2 and 3 of eight hidden.
    TAG: (
        ["learner", "learner", "heuristic", "heuristic"] + ["learner"] * 4,
        {"width": 6, "height": 6, "pursuers": 2, "evaders": 2, "max_episode_steps": 9},
    ),
    # Two leaves, flattened, and slots 1 and 3 of four hidden.
    KEYDOOR: (["learner", "heuristic", "learner", "heuristic"], {"agents": 4, "size": 8, "max_episode_steps": 7}),
}
N, CHUNK = 8, 4


def visible(groups):
    return [s for s, g in enumerate(groups) if g == "learner"]


def flat(observations: dict, names: list, rows: int) -> np.ndarray:
    return np.concatenate([np.asarray(observations[n]).reshape((rows, -1)) for n in names], axis=1)


def expected(step, names, agents, vis):
    """A plain half's outputs, its visible slots only, in the adapter's layout."""
    h = len(np.asarray(step.rewards))
    obs = flat(step.observations, names, h * agents).reshape((h, agents, -1))[:, vis].reshape((h * len(vis), -1))
    rewards = np.asarray(step.rewards)[:, vis].reshape(-1)
    terminals = np.asarray(step.agent_terminated)[:, vis].reshape(-1)
    truncations = np.asarray(step.agent_truncated)[:, vis].reshape(-1)
    masks = None if step.masks is None else np.asarray(step.masks).reshape((h, agents, -1))[:, vis].reshape((h * len(vis), -1))
    return obs, rewards, terminals, truncations, masks


@pytest.mark.parametrize("env_id", [TAG, KEYDOOR])
def test_pipelined_hides_heuristic_slots_half_by_half(env_id):
    groups, cfg = CASES[env_id]
    vis, agents = visible(groups), len(groups)
    env = make_puffer(env_id, num_envs=N, chunk_size=CHUNK, mode="pipelined", workers=2,
                      policy_map=groups, log_interval=1, **cfg)()  # fmt: skip
    base = mix_seed(0, 3)
    ref = t.make(env_id, num_envs=N, seed=base, auto_reset="same_step", mode="pipelined", chunk_size=CHUNK,
                 workers=2, policy_map=groups, **cfg)  # fmt: skip
    names = [leaf[0] for leaf in ref._native.layout()["leaves"]]
    rng = np.random.default_rng(1)
    try:
        assert env.num_agents == N * len(vis)
        assert env.agents_per_batch == N // 2 * len(vis)
        env.reset(seed=3)
        want = list(ref.reset(base))
        arrays = {}
        ended = 0
        for round_ in range(40):
            h = want.pop(0) if want else ref.recv()
            obs, rewards, terminals, truncations, _infos, ids, masks = env.recv()
            w = expected(h.step, names, agents, vis)
            np.testing.assert_array_equal(obs, w[0].reshape(obs.shape).astype(obs.dtype), err_msg=f"round {round_}")
            np.testing.assert_array_equal(rewards, w[1])
            np.testing.assert_array_equal(terminals, w[2])
            np.testing.assert_array_equal(truncations, w[3])
            if w[4] is not None:
                np.testing.assert_array_equal(env.action_masks, w[4])
            assert list(ids) == list(range(h.ids.start * len(vis), h.ids.stop * len(vis)))
            assert obs.shape[0] == env.agents_per_batch
            # Each half's rows are one set of arrays, allocated at construction.
            arrays.setdefault(h.half, obs)
            assert obs is arrays[h.half]
            ended += int(np.asarray(h.step.episode_terminated).sum() + np.asarray(h.step.episode_truncated).sum())
            acts = rng.integers(0, env.single_action_space.n, env.agents_per_batch, dtype=np.int32)
            env.send(acts)
            full = np.zeros((len(h.ids), agents), dtype=np.int32)
            full[:, vis] = acts.reshape((len(h.ids), len(vis)))
            ref.send(h.half, full)
        assert ended > 0, "no episode ended, so terminals and resets were not exercised"
        env.recv()
        ref.recv()
    finally:
        env.close()
        ref.close()


@pytest.mark.parametrize("env_id", [TAG, KEYDOOR])
def test_pipelined_first_halves_equal_the_sync_visible_rows(env_id):
    groups, cfg = CASES[env_id]
    pipe = make_puffer(env_id, num_envs=N, chunk_size=CHUNK, mode="pipelined", workers=1,
                       policy_map=groups, log_interval=None, **cfg)()  # fmt: skip
    sync = TesseraPufferEnv(env_id, num_envs=N, policy_map=groups, log_interval=None, **cfg)
    try:
        whole, _ = sync.reset(seed=5)
        pipe.reset(seed=5)
        first = pipe.recv()[0].copy()
        pipe.send(np.zeros(pipe.agents_per_batch, dtype=np.int32))
        second = pipe.recv()[0]
        np.testing.assert_array_equal(np.concatenate([first, second]), whole)
    finally:
        pipe.close()
        sync.close()
