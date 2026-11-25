# SPDX-License-Identifier: MIT OR Apache-2.0
"""RTS heuristic slots from Python (#586, ADR-0079): the ``rts`` maker wires
``policy::info``, so a ``heuristic`` group plays the fog-respecting Rush on
its information state, with submitted rows ignored."""

import numpy as np
import pytest

import tessera_rl
from tessera_rl import _native

RTS = "tessera/RTS-v1"
SMALL = {"size": 12, "max_units": 8, "step_limit": 60}
HEADS = 3 * SMALL["max_units"]


def _orders(env):
    """The no-op order for every unit of every slot."""
    return np.zeros((env.num_envs, env.agents, HEADS), dtype=np.int32)


def test_rts_policy_map_heuristic_group_in_layout():
    env = tessera_rl.make(RTS, num_envs=2, workers=0, policy_map=["learner", "heuristic"], **SMALL)
    assert env.policy_groups == ("learner", "heuristic")
    assert env._native.layout()["policy_groups"] == ["learner", "heuristic"]
    env.close()
    env = tessera_rl.make(RTS, num_envs=2, workers=0, policy_map=["learner", "privileged_heuristic"], **SMALL)
    assert env.policy_groups == ("learner", "privileged_heuristic")
    env.close()
    with pytest.raises(_native.CapabilityError):
        tessera_rl.make(RTS, workers=0, policy_map=["heuristic", "privileged_heuristic"], **SMALL)


def test_rts_heuristic_slot_ignores_submitted_rows_and_matches_across_workers():
    env = tessera_rl.make(RTS, num_envs=3, seed=5, workers=2, policy_map=["learner", "heuristic"], **SMALL)
    twin = tessera_rl.make(RTS, num_envs=3, seed=5, workers=0, policy_map=["learner", "heuristic"], **SMALL)
    step, other = env.reset(), twin.reset()
    moved = False
    for t in range(40):
        acts = _orders(env)
        # The Python multi-head path range-checks every slot's actions
        # before the pool, so the ignored rows hold in-range orders that
        # differ from the submitted no-op.
        garbage = acts.copy()
        garbage[:, 1] = np.tile(np.array([1, 40, 2], dtype=np.int32), SMALL["max_units"])
        step = env.step(acts)
        other = twin.step(garbage)
        h = env.heuristic_actions
        assert h.shape == (3, 2, HEADS) and h.dtype == np.int32
        np.testing.assert_array_equal(h, twin.heuristic_actions)
        np.testing.assert_array_equal(step.rewards, other.rewards)
        assert (h[:, 0] == 0).all()
        moved |= bool(h[:, 1].any())
    assert moved, "Rush never issued an order"
    env.close()
    twin.close()


def test_rts_heuristic_slot_plays_like_the_rush_opponent():
    """Slot 1 as a heuristic slot plays the same game as the built-in Rush
    opponent (opponent=1): both run Rush on side One's view at the start of
    each decision, so slot 0's observations and rewards are equal."""
    slot = tessera_rl.make(RTS, num_envs=2, seed=7, workers=0, outputs="owned", policy_map=["learner", "heuristic"], **SMALL)
    bot = tessera_rl.make(RTS, num_envs=2, seed=7, workers=0, outputs="owned", opponent=1, **SMALL)
    a, b = slot.reset(), bot.reset()
    for t in range(40):
        for name in a.observations:
            np.testing.assert_array_equal(np.asarray(a.observations[name])[:, 0], np.asarray(b.observations[name])[:, 0], err_msg=f"{name} t {t}")
        np.testing.assert_array_equal(a.rewards[:, 0], b.rewards[:, 0])
        acts = np.zeros((2, 2, HEADS), dtype=np.int32)
        a, b = slot.step(acts), bot.step(acts)
    slot.close()
    bot.close()
