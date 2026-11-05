# SPDX-License-Identifier: MIT OR Apache-2.0
"""The PettingZoo adapters (ADR-0013): the official API tests for every
declared capability, refusal of undeclared conversions, terminal reward
timing, agent death, a conditional second-player decision, cycle
buffering, and equality with the native pool."""

import numpy as np
import pytest

pz_test = pytest.importorskip("pettingzoo.test")

import tessera_rl  # noqa: E402
from tessera_rl import _native, pz  # noqa: E402

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
