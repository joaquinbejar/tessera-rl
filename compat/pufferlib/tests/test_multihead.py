# SPDX-License-Identifier: MIT OR Apache-2.0
"""Independent multi-head actions through the PufferLib adapter (#617):
Survival-v1's ``MultiDiscrete([5, 7])``, its flat masks (the heads' rows side
by side, at the cumulative ``nvec`` offsets), hidden heuristic slots with a
head axis, the vector backends and Pipelined, and the refusal of conditional
heads."""

import sys

import numpy as np
import pufferlib.vector
import pytest

import tessera_rl as t
from tessera_rl import puffer
from tessera_rl.puffer import TesseraPufferEnv, make_puffer

SURVIVAL = "tessera/Survival-v1"
CFG = {"agents": 16, "size": 32, "steps": 64, "respawn": 2}
NVEC = [5, 7]
# view 5: a 4 x 11 x 11 window, own (6), the Set's 16 x 3 items and its mask.
OBS_WIDTH = 4 * 11 * 11 + 6 + 16 * 3 + 16


def random_actions(env, rng):
    return np.stack([rng.integers(0, n, env.num_agents) for n in NVEC], axis=1).astype(np.int32)


def test_survival_spaces_are_multidiscrete_and_the_flattened_leaves():
    env = TesseraPufferEnv(SURVIVAL, num_envs=2, log_interval=None, **CFG)
    assert isinstance(env.single_action_space, puffer.gymnasium.spaces.MultiDiscrete)
    assert env.single_action_space.nvec.tolist() == NVEC
    assert env.single_observation_space.shape == (OBS_WIDTH,)
    assert env.num_agents == 2 * 128
    env.close()


def test_survival_steps_and_its_masks_are_the_heads_rows_at_cumulative_offsets():
    env = TesseraPufferEnv(SURVIVAL, num_envs=2, log_interval=None, **CFG)
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    heads = env._env._native.layout()["mask_heads"]
    # Head h's block starts at sum(nvec[:h]) and has nvec[h] columns.
    assert [(offset, size) for offset, _, size in heads] == [(0, 5), (5, 7)]
    assert [offset for offset, _, _ in heads] == list(np.cumsum([0, *NVEC[:-1]]))
    for _ in range(100):
        masks = env.action_masks
        assert masks.shape == (env.num_agents, sum(NVEC))
        obs, rewards, terminals, truncations, _ = env.step(random_actions(env, rng))
        assert obs.shape == (env.num_agents, OBS_WIDTH)
        assert rewards.shape == (env.num_agents,)
    # The move block of a live agent always allows stay, the act block none.
    masks = env.action_masks
    live = masks.any(axis=1)
    assert masks[live, 0].all() and masks[live, 5].all()
    env.close()


def test_survival_masks_equal_the_engines_rows():
    env = TesseraPufferEnv(SURVIVAL, num_envs=2, log_interval=None, **CFG)
    env.reset(seed=3)
    ref = t.make(SURVIVAL, num_envs=2, seed=env.instance_seed_base, outputs="owned", **CFG)
    want = np.asarray(ref.reset().masks).reshape((env.num_agents, -1))
    np.testing.assert_array_equal(env.action_masks, want)
    ref.close()
    env.close()


def test_survival_serial_backend_steps():
    vec = pufferlib.vector.make(
        make_puffer(SURVIVAL, num_envs=2, log_interval=None, **CFG), backend=pufferlib.vector.Serial, num_envs=2
    )
    vec.reset(seed=2)
    rng = np.random.default_rng(2)
    n = vec.num_agents
    for _ in range(20):
        actions = np.stack([rng.integers(0, k, n) for k in NVEC], axis=1).astype(np.int32)
        vec.step(actions)
    vec.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Multiprocessing is tested on Linux at the pins")
def test_survival_multiprocessing_backend_steps():
    vec = pufferlib.vector.make(
        make_puffer(SURVIVAL, num_envs=2, log_interval=None, **CFG),
        backend=pufferlib.vector.Multiprocessing,
        num_envs=2,
        num_workers=2,
    )
    vec.reset(seed=4)
    rng = np.random.default_rng(4)
    n = vec.num_agents
    for _ in range(10):
        vec.step(np.stack([rng.integers(0, k, n) for k in NVEC], axis=1).astype(np.int32))
    vec.close()


def test_survival_hidden_heuristic_slots_keep_the_head_axis():
    groups = ["learner" if a % 2 == 0 else "heuristic" for a in range(128)]
    env = TesseraPufferEnv(SURVIVAL, num_envs=2, log_interval=None, policy_map=groups, **CFG)
    assert env.num_agents == 2 * 64
    env.reset(seed=5)
    rng = np.random.default_rng(5)
    for _ in range(30):
        env.step(random_actions(env, rng))
    assert env.action_masks.shape == (2 * 64, sum(NVEC))
    # The scatter kept the head axis and its order.
    assert_head_order_reaches_the_engine(env)
    env.close()


def assert_head_order_reaches_the_engine(env):
    # Move 6 is out of range (5 choices) and act 6 is not (7): with the heads
    # swapped or the head axis lost, this row would pass the range check.
    bad = np.zeros((env.num_agents, 2), dtype=np.int32)
    bad[0] = (6, 0)
    with pytest.raises(t.ActionError, match="out of range"):
        env.step(bad)


def test_survival_flat_actions_keep_the_head_order():
    env = TesseraPufferEnv(SURVIVAL, num_envs=2, log_interval=None, **CFG)
    env.reset(seed=7)
    assert_head_order_reaches_the_engine(env)
    env.close()


def test_survival_pipelined_halves_have_the_flat_mask_width():
    pipe = pufferlib.vector.make(
        make_puffer(SURVIVAL, num_envs=4, chunk_size=2, workers=1, mode="pipelined", log_interval=None, **CFG)
    )
    pipe.reset(seed=6)
    pipe.recv()
    masks = pipe.action_masks
    assert masks.shape == (pipe.agents_per_batch, sum(NVEC))
    pipe.send(np.zeros((pipe.agents_per_batch, 2), dtype=np.int32))
    pipe.recv()
    pipe.close()


def test_survival_pipelined_hidden_slots_masks_keep_both_heads():
    groups = ["learner" if a % 2 == 0 else "heuristic" for a in range(128)]
    pipe = pufferlib.vector.make(
        make_puffer(SURVIVAL, num_envs=4, chunk_size=2, workers=1, mode="pipelined", log_interval=None,
                    policy_map=groups, **CFG)  # fmt: skip
    )
    pipe.reset(seed=8)
    for _ in range(4):
        pipe.recv()
        masks = pipe.action_masks
        assert masks.shape == (pipe.agents_per_batch, sum(NVEC))
        live = masks.any(axis=1)
        # Stay (column 0) and none (column 5) are legal for every live agent.
        assert live.any() and masks[live, 0].all() and masks[live, 5].all()
        pipe.send(np.zeros((pipe.agents_per_batch, 2), dtype=np.int32))
    pipe.recv()
    pipe.close()

def test_heads_admission_independent_only_naming_the_conditional_head():
    puffer._check_heads("one-head", (5,), [(0, [], 5)])
    puffer._check_heads("survival", (5, 7), [(0, [], 5), (5, [], 7)])
    # LimitOrderBook-v1's mask_heads: head 1 depends on head 0 (6 choices).
    lob = [(0, [], 6), (6, [6], 9), (60, [6, 9], 4), (276, [6], 8)]
    with pytest.raises(t.ArgumentError, match=r"action head 1 \(9 choices\).*parent heads of \[6\] choices"):
        puffer._check_heads("lob", (6, 9, 4, 8), lob)


@pytest.mark.parametrize("env_id", ["tessera/RTS-v1", "tessera/LimitOrderBook-v1"])
def test_unsupported_multihead_ids_are_refused(env_id):
    with pytest.raises(t.ArgumentError):
        make_puffer(env_id)
