# SPDX-License-Identifier: MIT OR Apache-2.0
"""ADR-0013 agent death and respawn through the PettingZoo adapters (#71),
over the spawning environment of the test host wheel (`test/Respawn-v1`,
`make py-test-host`): agent 1 dies in step 2 and a new agent spawns into
slot 1 in step 3. The dead agent is reported once, its slot stays out of
`agents` for the step after its death, and the new generation joins under
its own name (identity is (slot, generation))."""

import numpy as np
import pytest

pytest.importorskip("_tessera_host", reason="needs the test host wheel (make py-test-host)")
pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")

ID = "test/Respawn-v1"


def test_respawn_names_are_per_generation():
    assert pz.agent_name(1, 0) == "agent_1"
    assert pz.agent_name(1, 1) == "agent_1_g1"
    assert pz.agent_name(0, 12) == "agent_0_g12"


def test_respawn_parallel_death_once_gap_and_new_generation():
    env = pz.parallel_env(ID, seed=3)
    obs, infos = env.reset(seed=3)
    assert env.agents == ["agent_0", "agent_1"] and env.possible_agents == ["agent_0", "agent_1"]
    acts = {"agent_0": 1, "agent_1": 0}
    obs, rew, term, trunc, info = env.step(acts)
    assert env.agents == ["agent_0", "agent_1"] and not any(term.values())
    # Step 2: agent 1 dies, reported once with its final reward.
    obs, rew, term, trunc, info = env.step(acts)
    assert term == {"agent_0": False, "agent_1": True}
    assert rew == {"agent_0": 2.0, "agent_1": 1.0}
    assert env.agents == ["agent_0"]
    # Step 3: slot 1 was free for the whole step (no action needed for it;
    # the dead agent's action is ignored), and its new agent spawns.
    obs, rew, term, trunc, info = env.step({"agent_0": 0, "agent_1": 1})
    assert "agent_1" not in rew and "agent_1" not in term
    assert env.agents == ["agent_0", "agent_1_g1"]
    assert env.possible_agents == ["agent_0", "agent_1", "agent_1_g1"]
    # The engine's reward and flags for the spawn step (the env pays 0.25).
    assert (rew["agent_1_g1"], term["agent_1_g1"], trunc["agent_1_g1"]) == (0.25, False, False)
    assert obs["agent_1_g1"]["obs"].tolist() == [3, 1]
    assert set(info) >= set(env.agents)
    # Step 4: the new agent acts and is rewarded; the old one never returns.
    obs, rew, term, trunc, info = env.step({"agent_0": 0, "agent_1_g1": 1})
    assert rew == {"agent_0": 1.0, "agent_1_g1": 2.0}
    assert "agent_1" not in env.agents
    env.close()


def test_respawn_parallel_new_generation_must_act():
    env = pz.parallel_env(ID, seed=3)
    env.reset(seed=3)
    for _ in range(3):
        env.step({a: 0 for a in env.agents})
    with pytest.raises(Exception, match="no action for live agents"):
        env.step({"agent_0": 0})
    env.close()


def test_respawn_parallel_episode_end_flushes_every_live_agent():
    env = pz.parallel_env(ID, seed=3)
    env.reset(seed=3)
    for _ in range(5):
        env.step({a: 0 for a in env.agents})
    obs, rew, term, trunc, info = env.step({a: 0 for a in env.agents})
    assert term == {"agent_0": True, "agent_1_g1": True}
    assert env.agents == []
    env.close()


def test_respawn_parallel_reset_keeps_the_generation_name():
    env = pz.parallel_env(ID, seed=3)
    env.reset(seed=3)
    while env.agents:
        env.step({a: 0 for a in env.agents})
    env.reset(seed=4)
    # A slot's generation survives a reset (the engine never lowers it).
    assert env.agents == ["agent_0", "agent_1_g1"] and env.possible_agents == env.agents
    env.close()


def test_respawn_aec_cycle_reports_death_once_and_adds_the_new_agent():
    env = pz.aec_env(ID, seed=3)
    env.reset(seed=3)
    seen, done = [], []
    for agent in env.agent_iter(40):
        _, _, terminated, truncated, _ = env.last()
        seen.append(agent)
        if terminated or truncated:
            done.append(agent)
            env.step(None)
        else:
            env.step(0)
    assert done.count("agent_1") == 1
    assert "agent_1_g1" in seen
    # The new agent appears only after the dead one was removed.
    assert seen.index("agent_1_g1") > len(seen) - 1 - seen[::-1].index("agent_1")
    env.close()


def test_respawn_official_parallel_api_test():
    from pettingzoo.test import parallel_api_test

    env = pz.parallel_env(ID, seed=3)
    parallel_api_test(env, num_cycles=20)
    env.close()


def test_respawn_official_aec_api_test():
    from pettingzoo.test import api_test

    env = pz.aec_env(ID, seed=3)
    api_test(env, num_cycles=20)
    env.close()


@pytest.mark.parametrize("delay", [1, 3])
def test_respawn_slot_stays_empty_until_the_spawn(delay):
    env = pz.parallel_env(ID, seed=3, respawn_delay=delay)
    env.reset(seed=3)
    agents_per_step = []
    for _ in range(2 + delay):
        env.step({a: 0 for a in env.agents})
        agents_per_step.append(list(env.agents))
    assert agents_per_step[1] == ["agent_0"]
    assert all(a == ["agent_0"] for a in agents_per_step[1 : 1 + delay])
    assert agents_per_step[1 + delay] == ["agent_0", "agent_1_g1"]
    assert np.all(np.asarray([len(a) for a in agents_per_step]) >= 1)
    env.close()


def test_respawn_in_the_last_step_is_flushed_and_leaves_no_agent():
    # die 2, delay 4, episode 6: the new agent spawns in the episode's last
    # step and is flushed with the others (SPEC 12).
    env = pz.parallel_env(ID, seed=3, respawn_delay=4)
    env.reset(seed=3)
    for _ in range(5):
        env.step({a: 0 for a in env.agents})
    assert env.agents == ["agent_0"]
    obs, rew, term, trunc, info = env.step({"agent_0": 0})
    assert term == {"agent_0": True, "agent_1_g1": True}
    assert rew == {"agent_0": 1.0, "agent_1_g1": 0.25}
    assert "agent_1_g1" in obs and "agent_1_g1" in env.possible_agents
    assert env.agents == []
    env.close()


def test_respawn_into_a_slot_empty_at_reset_starts_at_generation_1():
    env = pz.parallel_env(ID, seed=3, initial=1, die_step=0, respawn_delay=2)
    env.reset(seed=3)
    assert env.agents == ["agent_0"] and env.possible_agents == ["agent_0"]
    env.step({"agent_0": 0})
    assert env.agents == ["agent_0"]
    obs, rew, term, trunc, info = env.step({"agent_0": 0})
    assert env.agents == ["agent_0", "agent_1_g1"]
    assert env.possible_agents == ["agent_0", "agent_1_g1"]
    assert rew["agent_1_g1"] == 0.25 and obs["agent_1_g1"]["obs"].tolist() == [2, 1]
    env.close()


def test_respawn_native_aec_refuses_a_spawning_environment():
    pool, _ = pz._host_pool(ID, "parallel", 3, {})
    with pytest.raises(pz._native.CapabilityError, match="spawns agents"):
        pz.TesseraAECEnv(ID, seed=3, _pool=pool)


def test_respawn_official_seed_tests():
    from pettingzoo.test import parallel_seed_test, seed_test

    parallel_seed_test(lambda: pz.parallel_env(ID, seed=3), num_cycles=50)
    seed_test(lambda: pz.aec_env(ID, seed=3), num_cycles=50)
