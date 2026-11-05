# SPDX-License-Identifier: MIT OR Apache-2.0
"""The PettingZoo adapters (ADR-0013): the official API tests for every
declared capability, refusal of undeclared conversions, terminal reward
timing, agent death, a conditional second-player decision, cycle
buffering, and equality with the native pool."""

import numpy as np
import pytest

import pettingzoo.test as pz_test

import tessera_rl
from tessera_rl import _native, pz

TAG = "tessera/Tag-v1"
C4 = "tessera/Connect4-v1"
# A small, crowded Tag so that catches happen within a few steps.
SMALL_TAG = {"width": 4, "height": 4, "pursuers": 2, "evaders": 2, "max_episode_steps": 300}

pytestmark = pytest.mark.filterwarnings(
    "ignore:Observation is not a NumPy array:UserWarning",
    "ignore:Observation space for each agent probably should be:UserWarning",
    "ignore:Environment has not defined a render:UserWarning",
)


def test_pz_tag_passes_the_parallel_api_test():
    pz_test.parallel_api_test(pz.parallel_env(TAG, seed=1, max_episode_steps=50), num_cycles=300)


def test_pz_tag_passes_the_aec_api_test_through_cycle_buffering():
    pz_test.api_test(pz.aec_env(TAG, seed=1, max_episode_steps=50), num_cycles=300, verbose_progress=False)


def test_pz_connect4_passes_the_aec_api_test():
    pz_test.api_test(pz.aec_env(C4, seed=1), num_cycles=300, verbose_progress=False)


def test_pz_small_tag_passes_the_api_tests_with_agents_removed():
    # Crowded enough that evaders are caught and leave mid-episode.
    pz_test.parallel_api_test(pz.parallel_env(TAG, seed=5, **SMALL_TAG), num_cycles=300)
    pz_test.api_test(pz.aec_env(TAG, seed=5, **SMALL_TAG), num_cycles=400, verbose_progress=False)


def test_pz_seed_tests_pass():
    pz_test.parallel_seed_test(lambda: pz.parallel_env(TAG, **SMALL_TAG), num_cycles=200)
    pz_test.seed_test(lambda: pz.aec_env(TAG, **SMALL_TAG), num_cycles=200)
    pz_test.seed_test(lambda: pz.aec_env(C4), num_cycles=200)


def test_pz_construction_does_not_consume_the_first_episode():
    native = tessera_rl.make(TAG, num_envs=1, seed=4, outputs="owned", auto_reset="disabled", **SMALL_TAG)
    first = native.reset()
    obs, _ = pz.parallel_env(TAG, seed=4, **SMALL_TAG).reset()
    for a in obs:
        np.testing.assert_array_equal(obs[a]["window"], first.observations["window"][0, int(a.split("_")[1])])
    native.close()


@pytest.mark.parametrize(
    "make",
    [
        lambda: pz.parallel_env(C4),  # sequential: never Parallel
        lambda: pz.turn_masked_env(C4),  # optional, not declared
        lambda: pz.turn_masked_env(TAG),
        lambda: pz.aec_env("tessera/CartPole-v1"),  # single-agent: Gymnasium only
        lambda: pz.parallel_env("tessera/CartPole-v1"),
        lambda: pz.parallel_env(TAG, num_envs=2),  # fixed by the adapter
        lambda: pz.parallel_env(TAG, auto_reset="same_step"),
    ],
)
def test_pz_undeclared_conversions_are_refused_at_construction(make):
    with pytest.raises(_native.ArgumentError):
        make()


def _random_parallel_episode(env, seed):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    steps = []
    while env.agents:
        actions = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
        live = list(env.agents)
        obs, rewards, terms, truncs, _ = env.step(actions)
        steps.append((live, rewards, terms, truncs, list(env.agents)))
    return steps


def test_pz_tag_death_reports_the_final_reward_once_and_removes_the_agent():
    env = pz.parallel_env(TAG, seed=5, **SMALL_TAG)
    steps = _random_parallel_episode(env, seed=5)
    deaths = [(i, a) for i, (live, _, terms, truncs, after) in enumerate(steps[:-1]) for a in live if terms[a]]
    assert deaths, "an evader is caught before the episode ends"
    for i, agent in deaths:
        _, rewards, _, _, after = steps[i]
        assert rewards[agent] == -1.0, "the caught evader gets its final reward in the step it dies"
        assert agent not in after, "and leaves the agents"
        assert all(agent not in later[0] for later in steps[i + 1 :]), "it is never reported again"


def test_pz_tag_episode_end_flushes_every_live_agent_in_the_same_step():
    env = pz.parallel_env(TAG, seed=5, **SMALL_TAG)
    live, rewards, terms, truncs, after = _random_parallel_episode(env, seed=5)[-1]
    assert after == []
    assert all(terms[a] or truncs[a] for a in live), "every live agent gets its flag in the final step"
    assert set(rewards) == set(live)


def test_pz_parallel_matches_the_native_pool_step_for_step():
    seed, cfg = 9, dict(SMALL_TAG)
    env = pz.parallel_env(TAG, seed=seed, **cfg)
    native = tessera_rl.make(TAG, num_envs=1, seed=seed, outputs="owned", auto_reset="disabled", **cfg)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    ref = native.reset(seed)
    while env.agents:
        actions = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
        acts = np.zeros((1, native.agents), dtype=np.int32)
        for a, v in actions.items():
            acts[0, int(a.split("_")[1])] = v
        obs, rewards, _, _, _ = env.step(actions)
        ref = native.step(acts)
        for a in obs:
            slot = int(a.split("_")[1])
            assert rewards[a] == ref.rewards[0, slot]
            np.testing.assert_array_equal(obs[a]["window"], ref.observations["window"][0, slot])
    native.close()


def test_pz_tag_aec_observations_change_only_at_cycle_boundaries():
    env = pz.aec_env(TAG, seed=3, max_episode_steps=50)
    env.reset(seed=3)
    first = env.agent_selection
    before = {a: env.observe(a)["window"].copy() for a in env.agents}
    env.step(0)  # the first agent of the cycle acts: nothing has stepped yet
    for a in env.agents:
        np.testing.assert_array_equal(env.observe(a)["window"], before[a])
    for _ in range(len(env.agents) - 1):
        env.step(0)
    assert env.agent_selection == first, "a new cycle starts with the first agent"


def _c4_play(env, columns):
    for col in columns:
        env.step(col)


def test_pz_connect4_second_player_decision_depends_on_the_first():
    env = pz.aec_env(C4, seed=0)
    env.reset(seed=0)
    assert env.agent_selection == "agent_0"
    _c4_play(env, [0, 0, 0, 0, 0])  # five pieces in column 0, alternating
    assert env.agent_selection == "agent_1"
    mask = env.observe("agent_1")["action_mask"]
    assert mask[0] == 1, "one cell is left in column 0"
    env.step(0)  # agent_1 fills column 0
    assert env.observe("agent_0")["action_mask"][0] == 0, "agent_0 can no longer play column 0"


def test_pz_connect4_terminal_rewards_reach_both_players_in_the_final_step():
    env = pz.aec_env(C4, seed=0)
    env.reset(seed=0)
    # agent_0 plays column 0 four times, agent_1 column 1: agent_0 wins.
    _c4_play(env, [0, 1, 0, 1, 0, 1, 0])
    assert env.terminations == {"agent_0": True, "agent_1": True}
    assert env.rewards == {"agent_0": 1.0, "agent_1": -1.0}
    while env.agents:
        env.step(None)  # each finished agent leaves
    assert env.agents == []


INVALID = [1.9, -1, 5, 7.5, "2", None, [1], [1, 2], [[1], [1, 2]]]


@pytest.mark.parametrize("bad", INVALID)
def test_pz_parallel_invalid_action_raises_and_changes_nothing(bad):
    env = pz.parallel_env(TAG, seed=2, **SMALL_TAG)
    obs, _ = env.reset(seed=2)
    good = {a: 0 for a in env.agents}
    first = env.agents[0]
    with pytest.raises(_native.ActionError):
        env.step({**good, first: bad})
    # The pool did not step: the same actions now give the same result as a
    # fresh environment's first step.
    ref = pz.parallel_env(TAG, seed=2, **SMALL_TAG)
    ref.reset(seed=2)
    a, b = env.step(good), ref.step(good)
    for agent in a[0]:
        np.testing.assert_array_equal(a[0][agent]["window"], b[0][agent]["window"])
    assert a[1] == b[1]


def test_pz_parallel_refuses_unknown_agent_names():
    env = pz.parallel_env(TAG, seed=2, **SMALL_TAG)
    env.reset(seed=2)
    with pytest.raises(_native.ActionError):
        env.step({**{a: 0 for a in env.agents}, "agent_7": 0})  # a slot that never lived
    with pytest.raises(_native.ActionError):
        env.step({**{a: 0 for a in env.agents}, "nobody": 0})


def _first_legal(mask):
    legal = np.flatnonzero(mask)
    return int(legal[0]) if legal.size else 0


def test_pz_parallel_policy_loop_over_obs_works_across_a_death():
    env = pz.parallel_env(TAG, seed=5, **SMALL_TAG)
    obs, _ = env.reset(seed=5)
    died = False
    while env.agents:
        before = set(env.agents)
        # The natural loop: one action per key of the last observations,
        # including agents that died in the previous step.
        actions = {a: _first_legal(obs[a]["action_mask"]) for a in obs}
        obs, _, terms, truncs, _ = env.step(actions)
        died |= bool(before - set(env.agents)) and bool(env.agents)
    assert died, "an agent died before the end and the loop kept going"


@pytest.mark.parametrize("bad", INVALID)
def test_pz_tag_aec_invalid_action_keeps_the_buffered_cycle(bad):
    env = pz.aec_env(TAG, seed=2, **SMALL_TAG)
    env.reset(seed=2)
    env.step(1)  # the first agent's action is buffered
    second = env.agent_selection
    with pytest.raises(_native.ActionError):
        env.step(bad)
    assert env.agent_selection == second, "the cycle did not advance"
    ref = pz.aec_env(TAG, seed=2, **SMALL_TAG)
    ref.reset(seed=2)
    ref.step(1)
    # Complete the cycle in both with the same actions: the environment
    # steps once, with the same buffered actions, so they stay equal.
    for e in (env, ref):
        for _ in range(len(e.agents) - 1):
            e.step(0)
    for agent in env.agents:
        np.testing.assert_array_equal(env.observe(agent)["window"], ref.observe(agent)["window"])
    assert env.rewards == ref.rewards


@pytest.mark.parametrize("bad", [1.9, -1, 7, 7.5, "2", None, [1], [1, 2], [[1], [1, 2]]])
def test_pz_connect4_invalid_action_raises_and_changes_nothing(bad):
    env = pz.aec_env(C4, seed=0)
    env.reset(seed=0)
    env.step(3)
    before = env.observe("agent_1")["board"].copy()
    with pytest.raises(_native.ActionError):
        env.step(bad)
    assert env.agent_selection == "agent_1"
    np.testing.assert_array_equal(env.observe("agent_1")["board"], before)


def test_pz_connect4_full_column_is_the_engine_noop_not_an_error():
    env = pz.aec_env(C4, seed=0)
    env.reset(seed=0)
    _c4_play(env, [0, 0, 0, 0, 0, 0])  # column 0 is full
    mover = env.agent_selection
    assert env.observe(mover)["action_mask"][0] == 0
    env.step(0)  # legal index, masked illegal: replaced by the no-op (ADR-0028)
    assert env.agent_selection != mover, "the no-op still passed the turn"


LEDUC = "tessera/Leduc-v1"


def test_pz_leduc_passes_the_aec_api_and_seed_tests():
    pz_test.api_test(pz.aec_env(LEDUC, seed=1), num_cycles=300, verbose_progress=False)
    pz_test.seed_test(lambda: pz.aec_env(LEDUC), num_cycles=200)


def test_pz_leduc_is_never_parallel():
    with pytest.raises(_native.ArgumentError):
        pz.parallel_env(LEDUC)


def test_pz_leduc_fold_after_a_raise_pays_the_folders_ante():
    env = pz.aec_env(LEDUC, seed=0)
    env.reset(seed=0)
    assert env.agent_selection == "agent_0"
    assert list(env.observe("agent_0")["action_mask"]) == [0, 1, 1], "no bet to fold against"
    env.step(2)  # raise
    assert list(env.observe("agent_1")["action_mask"]) == [1, 1, 1]
    env.step(0)  # fold
    assert env.terminations == {"agent_0": True, "agent_1": True}
    assert env.rewards == pytest.approx({"agent_0": 1 / 13, "agent_1": -1 / 13})
