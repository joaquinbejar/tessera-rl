# SPDX-License-Identifier: MIT OR Apache-2.0
"""Policy maps and heuristic slots from Python (ADR-0079, stages A and B1):
``make(..., policy_map=...)``, ``layout()["policy_groups"]``,
``Env.heuristic_actions`` in every mode, refusals, and the PettingZoo, Gymnasium and
PufferLib adapters."""

import numpy as np
import pytest

import tessera_rl
from tessera_rl import _native, pz

TAG = "tessera/Tag-v1"
LEDUC = "tessera/Leduc-v1"
# Slots 0 and 1 pursue, 2 and 3 evade.
TAG_CFG = {"width": 6, "height": 6, "pursuers": 2, "evaders": 2, "max_episode_steps": 100}
TAG_MAP = ["learner", "learner", "heuristic", "heuristic", "learner", "learner", "learner", "learner"]

pytestmark = pytest.mark.filterwarnings(
    "ignore:Observation is not a NumPy array:UserWarning",
    "ignore:Observation space for each agent probably should be:UserWarning",
)


def _legal(env, step, t):
    """A legal move per slot from the masks (garbage in heuristic rows)."""
    acts = np.zeros((env.num_envs, env.agents), dtype=np.int32)
    for i in range(env.num_envs):
        for a in range(env.agents):
            legal = np.flatnonzero(step.masks[i, a])
            acts[i, a] = legal[(i + a + t) % len(legal)] if len(legal) else 0
    return acts


def test_policy_map_list_and_dict_give_the_same_groups():
    a = tessera_rl.make(TAG, num_envs=2, workers=0, policy_map=TAG_MAP, **TAG_CFG)
    b = tessera_rl.make(TAG, num_envs=2, workers=0, policy_map={2: "heuristic", 3: "heuristic"}, **TAG_CFG)
    assert a.policy_groups == b.policy_groups == tuple(TAG_MAP)
    assert a._native.layout()["policy_groups"] == TAG_MAP
    plain = tessera_rl.make(TAG, num_envs=2, workers=0, **TAG_CFG)
    assert plain.policy_groups == ("learner",) * 8
    assert plain.heuristic_actions is None
    for env in (a, b, plain):
        env.close()


def test_heuristic_actions_follow_the_engine_and_ignore_submitted_rows():
    env = tessera_rl.make(TAG, num_envs=4, seed=3, workers=2, policy_map=TAG_MAP, **TAG_CFG)
    twin = tessera_rl.make(TAG, num_envs=4, seed=3, workers=0, policy_map=TAG_MAP, **TAG_CFG)
    step, other = env.reset(), twin.reset()
    for t in range(30):
        acts = _legal(env, step, t)
        garbage = acts.copy()
        garbage[:, 2:4] = -12345
        step = env.step(acts)
        other = twin.step(garbage)
        h = env.heuristic_actions
        assert h.shape == (4, 8, 1) and h.dtype == np.int32
        np.testing.assert_array_equal(h, twin.heuristic_actions)
        np.testing.assert_array_equal(step.rewards, other.rewards)
        assert (h[:, [0, 1, 4, 5, 6, 7]] == 0).all()
    env.close()
    twin.close()


def test_policy_map_bad_names_and_heuristics_without_support_are_refused():
    with pytest.raises(_native.ArgumentError):
        tessera_rl.make(TAG, workers=0, policy_map=["learner", "onnx"])
    with pytest.raises(TypeError):
        tessera_rl.make(TAG, workers=0, policy_map="heuristic")
    # CartPole has no maker-side heuristic: CapabilityError.
    with pytest.raises(_native.CapabilityError):
        tessera_rl.make("tessera/CartPole-v1", workers=0, policy_map=["heuristic"])
    # A learner and external map works for any environment.
    env = tessera_rl.make("tessera/CartPole-v1", workers=0, policy_map=["external"])
    assert env.policy_groups == ("external",)
    env.close()


LOB = "tessera/LimitOrderBook-v1"
LOB_MAP = {s: "heuristic" for s in range(1, 8)}


def test_lob_heuristic_slots_fill_1_to_7_deterministically_and_ignore_rows():
    """#470: LimitOrderBook runs heuristic slots (InfoHeuristic) like Tag."""
    env = tessera_rl.make(LOB, num_envs=4, seed=5, workers=2, policy_map=LOB_MAP, agents=8)
    twin = tessera_rl.make(LOB, num_envs=4, seed=5, workers=0, policy_map=LOB_MAP, agents=8)
    # MAX_AGENTS is 16: slots 8 to 15 are learners the config leaves unused.
    assert env.policy_groups[:8] == ("learner",) + ("heuristic",) * 7
    env.reset()
    twin.reset()
    heads = len(env.action_nvec)
    rows = env.agents  # MAX_AGENTS (16) rows; the config runs slots 0 to 7
    zeros = np.zeros((4, rows, heads), dtype=np.int32)
    # Multi-head actions are range-checked per head in Python before the
    # engine sees them, so the ignored rows get in-range values that differ
    # from the zeros (the top value of each head), not -12345 as for Tag.
    garbage = zeros.copy()
    garbage[:, 1:8] = np.asarray(env.action_nvec, dtype=np.int32) - 1
    filled = False
    for _ in range(40):
        step = env.step(zeros)
        other = twin.step(garbage)
        h = env.heuristic_actions
        assert h.shape == (4, rows, heads) and h.dtype == np.int32
        np.testing.assert_array_equal(h, twin.heuristic_actions)
        np.testing.assert_array_equal(step.rewards, other.rewards)
        assert (h[:, 0] == 0).all() and (h[:, 8:] == 0).all()
        filled |= bool((h[:, 1:8] != 0).any())
    assert filled, "the heuristic slots never acted"
    env.close()
    twin.close()


def test_lob_heuristic_slots_build_in_split_modes():
    """Stage B1: LimitOrderBook heuristic slots run in Pipelined and Async too."""
    for mode, extra in (("pipelined", {"chunk_size": 2}), ("async", {"batch_size": 2})):
        env = tessera_rl.make(LOB, num_envs=4, workers=1, mode=mode, policy_map=LOB_MAP, agents=8, **extra)
        assert env.policy_groups[:8] == ("learner",) + ("heuristic",) * 7
        env.close()


def _twin_steps(sync, step, t):
    """Sync's next legal actions and the same rows with garbage in the
    heuristic slots (2 and 3), which the engine never reads."""
    acts = _legal(sync, step, t)
    garbage = acts.copy()
    garbage[:, 2:4] = -12345
    return acts, garbage


def test_policy_map_heuristic_slots_in_pipelined_mode_match_sync():
    sync = tessera_rl.make(TAG, num_envs=8, seed=5, workers=0, policy_map=TAG_MAP, **TAG_CFG)
    env = tessera_rl.make(
        TAG, num_envs=8, seed=5, chunk_size=2, mode="pipelined", workers=2, policy_map=TAG_MAP, **TAG_CFG
    )
    assert env.policy_groups == tuple(TAG_MAP)
    step = sync.reset()
    env.reset()
    for t in range(30):
        acts, garbage = _twin_steps(sync, step, t)
        step = sync.step(acts)
        for half in (0, 1):
            r = env.halves[half]
            env.send(half, garbage[r.start : r.stop])
        for _ in (0, 1):
            out = env.recv()
            r = out.ids
            np.testing.assert_array_equal(out.step.rewards, step.rewards[r.start : r.stop])
        np.testing.assert_array_equal(env.heuristic_actions, sync.heuristic_actions)
    assert (env.heuristic_actions[:, [0, 1, 4, 5, 6, 7]] == 0).all()
    env.close()
    sync.close()


def test_policy_map_heuristic_slots_in_async_mode_match_sync_in_any_order():
    sync = tessera_rl.make(TAG, num_envs=8, seed=5, workers=0, policy_map=TAG_MAP, **TAG_CFG)
    env = tessera_rl.make(
        TAG, num_envs=8, seed=5, mode="async", batch_size=4, workers=2, policy_map=TAG_MAP, **TAG_CFG
    )
    assert env.policy_groups == tuple(TAG_MAP)
    step = sync.reset()
    env.reset()
    rng = np.random.default_rng(11)
    for t in range(30):
        acts, garbage = _twin_steps(sync, step, t)
        step = sync.step(acts)
        order = rng.permutation(8)
        env.send([int(i) for i in order], garbage[order])
        for _ in range(2):
            out = env.recv()
            for k, i in enumerate(out.ids):
                np.testing.assert_array_equal(out.step.rewards[k], step.rewards[int(i)])
        np.testing.assert_array_equal(env.heuristic_actions, sync.heuristic_actions)
    env.close()
    sync.close()


def test_leduc_heuristic_and_privileged_groups():
    env = tessera_rl.make(LEDUC, num_envs=2, workers=0, policy_map=["learner", "heuristic"])
    assert env.policy_groups == ("learner", "heuristic")
    env.close()
    env = tessera_rl.make(LEDUC, num_envs=2, workers=0, policy_map=["learner", "privileged_heuristic"])
    assert env.policy_groups == ("learner", "privileged_heuristic")
    env.close()
    with pytest.raises(_native.CapabilityError):
        tessera_rl.make(LEDUC, workers=0, policy_map=["heuristic", "privileged_heuristic"])


def test_pz_infos_carry_the_policy_group_and_learner_hides_heuristic_slots():
    env = pz.parallel_env(TAG, seed=1, policy_map=TAG_MAP, **TAG_CFG)
    obs, infos = env.reset()
    assert {infos[a]["policy_group"] for a in env.agents} == {"learner", "heuristic"}
    env.close()
    env = pz.parallel_env(TAG, seed=1, agents="learner", policy_map=TAG_MAP, **TAG_CFG)
    obs, infos = env.reset()
    assert env.possible_agents == ["agent_0", "agent_1"]
    assert set(obs) == {"agent_0", "agent_1"}
    for _ in range(5):
        if not env.agents:
            break
        actions = {a: int(np.flatnonzero(obs[a]["action_mask"])[0]) for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        assert set(rewards) <= {"agent_0", "agent_1"}
        for a in infos:
            assert set(infos[a]["heuristic_rewards"]) == {"agent_2", "agent_3"}
    env.close()
    with pytest.raises(_native.ArgumentError):
        pz.parallel_env(TAG, agents="heuristic")
    with pytest.raises(_native.CapabilityError):
        pz.aec_env("tessera/Connect4-v1", agents="learner")


def test_gym_refuses_a_map_without_slot_0_a_learner():
    gym = pytest.importorskip("tessera_rl.gym")
    with pytest.raises(_native.ArgumentError):
        gym.make("tessera/CartPole-v1", policy_map=["external"])


def test_host_ids_refuse_policy_map():
    from tessera_rl import _host

    kwargs = {"policy_map": ["learner"]}
    with pytest.raises(_native.CapabilityError):
        _host._refuse_spin(kwargs)
    with pytest.raises(_native.CapabilityError):
        _host._refuse_spin({"privileged_state": True})
