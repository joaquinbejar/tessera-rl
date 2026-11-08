# SPDX-License-Identifier: MIT OR Apache-2.0
"""SPEC 18 primary gate for mountain car: one-step equivalence of
``tessera/MountainCar-v1`` with the pinned Gymnasium's ``MountainCar-v0``
over 100,000 states sampled from Gymnasium rollouts plus the edge set, in
every auto-reset mode (``tools/gym_equivalence_mountaincar.py``)."""

import importlib.util
import pathlib

import numpy as np
import pytest

_TOOL = pathlib.Path(__file__).resolve().parents[2] / "tools" / "gym_equivalence_mountaincar.py"
_spec = importlib.util.spec_from_file_location("gym_equivalence_mountaincar", _TOOL)
eq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eq)

N = 100_000
SEED = 2026


@pytest.fixture(scope="module")
def sampled():
    states, actions = eq.gate_set(N, SEED)
    return states, actions, eq.reference(states, actions)


def test_mountaincar_equivalence_sampling_is_deterministic(sampled):
    states, actions, _ = sampled
    again, again_actions = eq.sample(1000, SEED)
    assert np.array_equal(states[:1000], again) and np.array_equal(actions[:1000], again_actions)


def test_mountaincar_equivalence_injection_round_trips(sampled):
    assert eq.injection_round_trips(sampled[0])


@pytest.mark.parametrize("mode", eq.MODES)
def test_mountaincar_equivalence_one_step_matches_gymnasium(sampled, mode):
    states, actions, ref = sampled
    report = eq.compare(eq.one_step(states, actions, mode), ref)
    print(mode, report)
    assert report["states"] > N
    assert report["reference_terminations"] > report["near_threshold"] // 2, "gated goal arrivals"
    assert report["near_threshold"] > 0 and report["near_threshold_terminated_agree"] == report["near_threshold"]
    assert report["wall_stops"] > 0 and report["speed_clips"] > 0
    if mode == "same_step":
        assert report["compared_through_terminal_obs"] == report["reference_terminations"]
    for key in ("state", "obs", "reward", "terminated", "truncated"):
        assert report[key] == 0, (key, report)


def test_mountaincar_equivalence_gate_detects_a_wrong_transition(sampled):
    # Negative control: mirrored actions must fail the comparison.
    states, actions, ref = sampled
    k = 5000
    report = eq.compare(eq.one_step(states[:k], 2 - actions[:k], "next_step"), ref[:k])
    assert report["state"] > 0


def test_mountaincar_equivalence_truncates_with_time_limit():
    assert eq.truncation()["mismatches"] == 0


def test_mountaincar_equivalence_short_trajectories_report(sampled):
    report = eq.trajectories(sampled[0][:N])
    print(report)
    assert report["compared_steps"] > 0 and report["flag_mismatches"] == 0
