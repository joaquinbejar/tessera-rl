# SPDX-License-Identifier: MIT OR Apache-2.0
"""Teams and roles (#347 stage B4): ``layout()["teams"]`` and
``layout()["roles"]`` per environment and their defaults, the PettingZoo
``metadata["teams"]`` and ``metadata["roles"]`` on the Parallel and AEC
forms, a host id's defaults, and the heuristic slots of PaddleDuel and
Snake."""

import numpy as np
import pytest

import tessera_rl
from tessera_rl import pz

TAG = "tessera/Tag-v1"
# Slots 0 and 1 pursue, 2 to 4 evade, 5 to 7 are unused.
TAG_CFG = {"width": 6, "height": 6, "pursuers": 2, "evaders": 3, "max_episode_steps": 50}
TAG_TEAMS = [0, 0, 1, 1, 1, 0, 0, 0]
TAG_ROLES = ["pursuer", "pursuer", "evader", "evader", "evader", "", "", ""]
PADDLE = "tessera/PaddleDuel-v1"
SNAKE = "tessera/Snake-v1"
TWO = {"agents": 2, "obs": "state"}

pytestmark = pytest.mark.filterwarnings(
    "ignore:Observation is not a NumPy array:UserWarning",
    "ignore:Observation space for each agent probably should be:UserWarning",
)


def _layout(env_id, **kwargs):
    env = tessera_rl.make(env_id, num_envs=2, workers=0, **kwargs)
    layout = env._native.layout()
    env.close()
    return layout["teams"], layout["roles"]


@pytest.mark.parametrize(
    ("env_id", "kwargs", "teams", "roles"),
    [
        (TAG, TAG_CFG, TAG_TEAMS, TAG_ROLES),
        (PADDLE, TWO, [0, 1], ["", ""]),
        (SNAKE, TWO, [0, 1], ["", ""]),
        ("tessera/RTS-v1", {}, [0, 1], ["", ""]),
        ("tessera/RTS-v1", {"seat_swap": True}, [1, 0], ["", ""]),
        ("tessera/CartPole-v1", {}, [0], [""]),
        ("tessera/Leduc-v1", {}, [0, 0], ["", ""]),
        ("tessera/LimitOrderBook-v1", {"agents": 2}, [0] * 16, [""] * 16),
    ],
)
def test_layout_teams_and_roles_per_environment_match(env_id, kwargs, teams, roles):
    assert _layout(env_id, **kwargs) == (teams, roles)


def test_layout_teams_and_roles_reset_and_step_leave_them_unchanged():
    env = tessera_rl.make(TAG, num_envs=2, seed=1, workers=0, **TAG_CFG)
    before = env._native.layout()
    step = env.reset()
    for _ in range(5):
        step = env.step(np.zeros((env.num_envs, env.agents), dtype=np.int32))
    after = env._native.layout()
    assert (after["teams"], after["roles"]) == (before["teams"], before["roles"]) == (TAG_TEAMS, TAG_ROLES)
    assert step.rewards.shape == (2, 8)
    env.close()


def _named(values):
    return {pz.agent_name(slot, 0): v for slot, v in enumerate(values[:5])}


def test_pz_metadata_tag_parallel_and_aec_teams_and_roles_by_agent_name():
    for env in (pz.parallel_env(TAG, seed=1, **TAG_CFG), pz.aec_env(TAG, seed=1, **TAG_CFG)):
        assert env.metadata["teams"] == _named(TAG_TEAMS)
        assert env.metadata["roles"] == _named(TAG_ROLES)
        assert set(env.metadata["teams"]) == set(env.possible_agents)
        env.reset(seed=1)
        assert env.metadata["roles"]["agent_2"] == "evader"
        env.close()


def test_pz_metadata_native_aec_defaults():
    env = pz.aec_env("tessera/Connect4-v1", seed=0)
    assert env.metadata["teams"] == {"agent_0": 0, "agent_1": 0}
    assert env.metadata["roles"] == {"agent_0": "", "agent_1": ""}
    env.close()


def test_pz_metadata_learner_only_keeps_the_visible_agents():
    hidden = {2: "heuristic", 3: "heuristic", 4: "heuristic"}
    env = pz.parallel_env(TAG, seed=1, visible="learner", policy_map=hidden, **TAG_CFG)
    assert env.metadata["teams"] == {"agent_0": 0, "agent_1": 0}
    assert env.metadata["roles"] == {"agent_0": "pursuer", "agent_1": "pursuer"}
    env.close()


def test_host_layout_teams_and_roles_are_the_defaults(monkeypatch):
    from fake_host import FakeHost, install

    install(monkeypatch, {"fake/Counter-v1": [("fake-dist", FakeHost)]})
    env = tessera_rl.make("fake/Counter-v1", num_envs=2, seed=0)
    layout = env._native.layout()
    assert layout["teams"] == [0, 0] and layout["roles"] == ["", ""]
    env.close()
    par = pz.parallel_env("fake/Counter-v1", seed=0)
    assert par.metadata["teams"] == {"agent_0": 0, "agent_1": 0}
    assert par.metadata["roles"] == {"agent_0": "", "agent_1": ""}
    par.close()


def _learner_rows(env, step, t):
    """A legal action for every slot: from the masks when there are any."""
    acts = np.zeros((env.num_envs, env.agents), dtype=np.int32)
    for i in range(env.num_envs):
        for a in range(env.agents):
            if step.masks is None:
                acts[i, a] = (i + a + t) % env.action_choices
            else:
                legal = np.flatnonzero(step.masks[i, a])
                acts[i, a] = legal[(i + a + t) % len(legal)] if len(legal) else 0
    return acts


@pytest.mark.parametrize("env_id", [PADDLE, SNAKE])
def test_heuristic_slot_two_player_acts_as_scripted_and_ignores_rows(env_id):
    kw = {"num_envs": 3, "seed": 7, "workers": 0, **TWO}
    env = tessera_rl.make(env_id, policy_map=["learner", "heuristic"], **kw)
    garbage = tessera_rl.make(env_id, policy_map=["learner", "heuristic"], **kw)
    # A plain pool given the heuristic's actions as slot 1's: the same
    # trajectory, so slot 1 played exactly what heuristic_actions reports.
    plain = tessera_rl.make(env_id, **kw)
    assert env.policy_groups == ("learner", "heuristic")
    step, other, mirror = env.reset(), garbage.reset(), plain.reset()
    acted = False
    for t in range(40):
        acts = _learner_rows(env, step, t)
        junk = acts.copy()
        junk[:, 1] = -12345
        step = env.step(acts)
        other = garbage.step(junk)
        h = env.heuristic_actions
        assert h.shape == (3, 2, 1) and h.dtype == np.int32
        np.testing.assert_array_equal(h, garbage.heuristic_actions)
        np.testing.assert_array_equal(step.rewards, other.rewards)
        assert (h[:, 0] == 0).all()
        copied = acts.copy()
        copied[:, 1] = h[:, 1, 0]
        mirror = plain.step(copied)
        np.testing.assert_array_equal(step.rewards, mirror.rewards)
        for leaf, value in step.observations.items():
            np.testing.assert_array_equal(value, mirror.observations[leaf])
            np.testing.assert_array_equal(value, other.observations[leaf])
        acted |= bool((h[:, 1] != 0).any())
    assert acted, "the heuristic slot never chose a nonzero action"
    for e in (env, garbage, plain):
        e.close()
