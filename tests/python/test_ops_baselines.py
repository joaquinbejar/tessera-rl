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
        chosen = np.asarray(env.heuristic_actions())
        for i in range(4):
            for a in range(agents):
                if not acting[i, a]:
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
