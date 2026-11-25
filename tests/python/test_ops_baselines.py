# SPDX-License-Identifier: MIT OR Apache-2.0
"""The Python baselines of ``tools/ops_baselines.py`` choose the actions of
the Rust ``Heuristic`` of each logistics environment (#383 stage 4): SPT for
JobShop-v1, nearest feasible neighbour for Vrp-v1. Every slot is a heuristic
slot, and each step's ``heuristic_actions`` are compared with the Python
rule on the observation and mask the heuristic saw."""

import pathlib
import sys

import numpy as np
import pytest

import tessera_rl as t

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
import ops_baselines as ob  # noqa: E402


def _agree(env_id, rule, steps, agents, **kwargs):
    env = t.make(env_id, num_envs=4, seed=7, outputs="owned", policy_map=["heuristic"] * agents, **kwargs)
    s = env.reset()
    checked = 0
    for _ in range(steps):
        before = {k: np.asarray(v).copy() for k, v in s.observations.items()}
        masks = np.asarray(s.masks).copy()
        acting = np.asarray(s.acting).copy()
        s = env.step(np.zeros((4, agents), dtype=np.int32))
        chosen = np.asarray(env.heuristic_actions)
        for i in range(4):
            for a in range(agents):
                # A finished episode's row (NextStep: the next step resets
                # it) has no legal action and no heuristic decision.
                if not acting[i, a] or not masks[i, a].any():
                    continue
                obs = {k: v[i, a] for k, v in before.items()}
                assert int(chosen[i, a].reshape(-1)[0]) == rule(obs, masks[i, a]), (i, a)
                checked += 1
    env.close()
    assert checked > 0


def test_ops_baselines_spt_matches_the_jobshop_heuristic():
    _agree("tessera/JobShop-v1", ob.spt, 60, 1, jobs=5, machines=4)


@pytest.mark.parametrize("vehicles", [1, 3])
def test_ops_baselines_nearest_matches_the_vrp_heuristic(vehicles):
    _agree("tessera/Vrp-v1", ob.nearest, 60, 8, customers=12, vehicles=vehicles, grid=40, capacity=20)


def test_ops_baselines_round_sqrt_is_exact():
    for v in range(5000):
        assert ob.round_sqrt(v) == round(v**0.5)


def _jobs_obs(rows):
    """A JobShop observation with the given per-job (next duration, work
    left) and the other features 0."""
    jobs = np.zeros((len(rows), 5), dtype=np.int32)
    for j, (duration, work) in enumerate(rows):
        jobs[j, 1], jobs[j, 2] = duration, work
    return {"jobs": jobs, "machines": np.zeros((2, 2), dtype=np.int32)}


def test_ops_baselines_mwkr_on_the_hand_built_states_of_the_rust_test():
    """The states of tests/jobshop.rs's MWKR test, as observations: the same
    picks (the JobShop gate's bar comes from this Python rule)."""
    # Work left 6, 7 and 4: job 1.
    assert ob.mwkr(_jobs_obs([(5, 6), (3, 7), (2, 4)]), [True, True, True]) == 1
    # Job 1 ran its first operation (4 left): job 0 (6 left).
    assert ob.mwkr(_jobs_obs([(5, 6), (4, 4), (2, 4)]), [True, True, True]) == 0
    # Jobs 0 and 2 tie at 5 left, job 1 done: the lowest index, job 0.
    assert ob.mwkr(_jobs_obs([(4, 5), (0, 0), (1, 5)]), [True, False, True]) == 0
    # SPT on the first state: the shortest next operation, job 2.
    assert ob.spt(_jobs_obs([(5, 6), (3, 7), (2, 4)]), [True, True, True]) == 2
