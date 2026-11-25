# SPDX-License-Identifier: MIT OR Apache-2.0
"""Survival-v1 from Python (#382 stage 4): ``make`` and its layout (a tile
window, own stats and a ``Set`` of visible agents with its mask), the
PettingZoo Parallel API, heuristic slots through ``policy_map`` (the
baseline from each agent's own view), and refusals."""

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test, parallel_seed_test

import tessera_rl
from tessera_rl import pz

ID = "tessera/Survival-v1"
SMALL = {"agents": 16, "size": 32, "steps": 64, "respawn": 2}


def test_survival_make_layout_has_the_window_own_and_the_set():
    env = tessera_rl.make(ID, num_envs=2, seed=1, outputs="owned", **SMALL)
    layout = env._native.layout()
    names = [(name, dtype, tuple(shape)) for name, dtype, shape, *_ in layout["leaves"]]
    assert names == [
        ("window", "uint8", (4, 11, 11)),
        ("own", "int32", (6,)),
        ("others", "int32", (16, 3)),
        ("others_mask", "bool", (16,)),
    ]
    assert [(p, m) for p, m, _ in layout["sets"]] == [("others", 16)]
    assert tuple(env.action_nvec) == (5, 7)
    s = env.reset()
    assert np.asarray(s.observations["window"]).shape == (2, 128, 4, 11, 11)
    assert np.asarray(s.rewards).shape == (2, 128)
    env.close()


def test_survival_pz_passes_the_parallel_api_test():
    parallel_api_test(pz.parallel_env(ID, **SMALL), num_cycles=40)


def test_survival_pz_passes_the_seed_test():
    parallel_seed_test(lambda: pz.parallel_env(ID, **SMALL))


def test_survival_heuristic_slots_play_the_baseline_and_learners_still_act():
    # Every other slot is the in-engine baseline (an InfoHeuristic: it sees
    # only its own window). The learners send stay and none.
    groups = ["learner" if a % 2 == 0 else "heuristic" for a in range(128)]
    env = tessera_rl.make(ID, num_envs=2, seed=3, outputs="owned", policy_map=groups, **SMALL)
    assert env._native.layout()["policy_groups"][:4] == ["learner", "heuristic", "learner", "heuristic"]
    env.reset()
    moved = False
    for _ in range(30):
        s = env.step(np.zeros((2, 128, 2), dtype=np.int32))
        chosen = np.asarray(env.heuristic_actions)
        assert chosen.shape == (2, 128, 2)
        # Only heuristic slots get heuristic actions, within the heads' ranges.
        assert (chosen[:, 0::2] == 0).all()
        assert (chosen[..., 0] < 5).all() and (chosen[..., 1] < 7).all()
        moved |= bool((chosen[:, 1::2, 0] != 0).any())
        assert np.asarray(s.rewards).shape == (2, 128)
    assert moved, "the baseline never moved"
    env.close()


@pytest.mark.parametrize("kw", [{"agents": 15}, {"agents": 129}, {"size": 31}, {"view": 8}, {"water_permille": 400, "forest_permille": 400, "stone_permille": 200}])
def test_survival_make_refuses_config_out_of_range(kw):
    with pytest.raises(tessera_rl.ConfigError):
        tessera_rl.make(ID, **kw)


def test_survival_is_not_a_single_agent_gymnasium_environment():
    import tessera_rl.gym as tg

    with pytest.raises(tessera_rl.ArgumentError):
        tg.make(ID)
