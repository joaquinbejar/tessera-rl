# SPDX-License-Identifier: MIT OR Apache-2.0
"""``visible=`` is the PettingZoo learner filter, and ``agents=`` the
environment's agent count (#538); ``agents="all"`` and ``agents="learner"``
are the filter's deprecated spelling."""

import warnings

import numpy as np
import pytest

import tessera_rl as t
from tessera_rl import pz

TAG = "tessera/Tag-v1"
TAG_CFG = {"width": 6, "height": 6, "pursuers": 2, "evaders": 2, "max_episode_steps": 100}
TAG_MAP = ["learner", "learner", "heuristic", "heuristic", "learner", "learner", "learner", "learner"]
LOB = "tessera/LimitOrderBook-v1"


def run(env, steps=6):
    """The visible agents and step dicts over a few steps, as plain values."""
    obs, _ = env.reset(seed=4)
    out = [sorted(env.possible_agents), sorted(obs)]
    for _ in range(steps):
        if not env.agents:
            break
        actions = {a: int(np.flatnonzero(obs[a]["action_mask"])[0]) for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        out.append((sorted(rewards.items()), sorted(terms.items()),
                    {a: sorted(infos[a].get("heuristic_rewards", {}).items()) for a in infos}))  # fmt: skip
    env.close()
    return out


@pytest.mark.parametrize("spelling", ["learner", "all"])
def test_deprecated_agents_filter_warns_and_equals_visible(spelling):
    with pytest.warns(t.TesseraDeprecationWarning, match=f"agents='{spelling}'.*visible='{spelling}'") as rec:
        old = pz.parallel_env(TAG, seed=1, agents=spelling, policy_map=TAG_MAP, **TAG_CFG)
    # Attributed to this file's line, not to the package.
    assert [w.filename for w in rec if w.category is t.TesseraDeprecationWarning] == [__file__]
    new = pz.parallel_env(TAG, seed=1, visible=spelling, policy_map=TAG_MAP, **TAG_CFG)
    if spelling == "learner":
        # Tag at this config has four live slots, 2 and 3 hidden.
        assert sorted(new.possible_agents) == ["agent_0", "agent_1"]
    assert run(old) == run(new)


def test_deprecated_filter_through_aec_env_is_attributed_to_the_caller():
    with pytest.warns(t.TesseraDeprecationWarning) as rec:
        env = pz.aec_env(TAG, seed=1, agents="learner", policy_map=TAG_MAP, **TAG_CFG)
    env.close()
    assert [w.filename for w in rec if w.category is t.TesseraDeprecationWarning] == [__file__]


def test_visible_alone_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", t.TesseraDeprecationWarning)
        pz.parallel_env(TAG, seed=1, visible="learner", policy_map=TAG_MAP, **TAG_CFG).close()
        pz.parallel_env(TAG, seed=1, **TAG_CFG).close()


def test_visible_and_a_string_agents_conflict():
    with pytest.raises(t.ArgumentError, match="visible='learner' and agents='learner'"):
        pz.parallel_env(TAG, seed=1, visible="learner", agents="learner", policy_map=TAG_MAP, **TAG_CFG)
    with pytest.raises(t.ArgumentError, match="visible='all' and agents='learner'"):
        pz.parallel_env(TAG, seed=1, visible="all", agents="learner", **TAG_CFG)


@pytest.mark.parametrize("bad", ["nope", "Learner", ""])
def test_visible_refuses_other_values(bad):
    with pytest.raises(t.ArgumentError, match="visible must be"):
        pz.parallel_env(TAG, seed=1, visible=bad, **TAG_CFG)


def test_native_aec_refuses_visible_learner():
    with pytest.raises(t.CapabilityError, match='visible="learner"'):
        pz.aec_env("tessera/Connect4-v1", visible="learner")


def test_agent_count_and_filter_together():
    """The #470 case: LimitOrderBook's own agent count with the filter."""
    env = pz.parallel_env(LOB, seed=2, agents=4, visible="learner", policy_map={1: "heuristic", 2: "heuristic", 3: "heuristic"})
    try:
        assert env.possible_agents == ["agent_0"]
        obs, infos = env.reset(seed=2)
        assert list(obs) == ["agent_0"]
        # Slots 1 to 3 exist and are hidden: four agents, not LobConfig's eight.
        assert set(infos["agent_0"].get("heuristic_rewards", {})) <= {"agent_1", "agent_2", "agent_3"}
        _, _, _, _, infos = env.step({"agent_0": np.zeros(len(env.action_space("agent_0").nvec), dtype=np.int64)})
        assert set(infos["agent_0"]["heuristic_rewards"]) == {"agent_1", "agent_2", "agent_3"}
    finally:
        env.close()
