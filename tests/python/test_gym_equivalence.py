# SPDX-License-Identifier: MIT OR Apache-2.0
"""SPEC 18 primary gate: CartPole one-step equivalence with the pinned
Gymnasium over 100,000 states sampled from Gymnasium rollouts, in every
auto-reset mode (``tools/gym_equivalence.py``)."""

import importlib.util
import pathlib

import numpy as np
import pytest

_TOOL = pathlib.Path(__file__).resolve().parents[2] / "tools" / "gym_equivalence.py"
_spec = importlib.util.spec_from_file_location("gym_equivalence", _TOOL)
eq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eq)

N = 100_000
SEED = 2026


@pytest.fixture(scope="module")
def sampled():
    states, actions = eq.gate_set(N, SEED)
    ref, thresholds = eq.reference(states, actions)
    return states, actions, ref, thresholds


def test_equivalence_sampling_is_deterministic(sampled):
    states, actions, _, _ = sampled
    again, again_actions = eq.sample(1000, SEED)
    assert np.array_equal(states[:1000], again) and np.array_equal(actions[:1000], again_actions)


def test_equivalence_injection_round_trips(sampled):
    assert eq.injection_round_trips(sampled[0])


@pytest.mark.parametrize("mode", eq.MODES)
def test_equivalence_one_step_matches_gymnasium(sampled, mode):
    states, actions, ref, thresholds = sampled
    report = eq.compare(eq.one_step(states, actions, mode), ref, thresholds)
    print(mode, report)
    assert report["states"] > N  # rollout samples plus the boundary set
    assert report["reference_terminations"] > 0, "the sample must cover episode ends"
    assert report["terminations_by_x"] > 0 and report["terminations_by_theta"] > 0
    assert report["near_threshold"] > 0, "the boundary set must reach the thresholds"
    if mode == "same_step":
        assert report["compared_through_terminal_obs"] == report["reference_terminations"]
    for key in ("state", "obs", "reward", "terminated", "truncated"):
        assert report[key] == 0, (key, report)


def test_equivalence_gate_detects_a_wrong_transition(sampled):
    # Negative control: flipped actions must fail the comparison.
    states, actions, ref, thresholds = sampled
    k = 5000
    report = eq.compare(eq.one_step(states[:k], 1 - actions[:k], "next_step"), ref[:k], thresholds)
    assert report["state"] > 0.9 * k


def test_equivalence_short_trajectories_are_reported(sampled):
    report = eq.trajectories(sampled[0][:N])
    print("trajectories", report)
    assert report["compared_steps"] > 0
    assert np.isfinite(report["max_relative"])


def test_equivalence_boundary_states_land_on_the_thresholds(sampled):
    states, actions, ref, thresholds = sampled
    edge = ref[N:]
    assert len(edge) == 2 * 2 * 20 * 5  # axes x signs x bases x offsets
    on = sum(abs(abs(n[0]) - thresholds[0]) == 0 or abs(abs(n[2]) - thresholds[1]) == 0 for n, *_ in edge)
    assert on == 2 * 2 * 20  # the zero-offset targets land exactly


def test_equivalence_truncation_matches_time_limit():
    report = eq.truncation()
    print("truncation", report)
    assert report["mismatches"] == 0
    assert report["compared_steps"] == report["episodes"] * report["limit"]
