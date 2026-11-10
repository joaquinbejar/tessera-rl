# SPDX-License-Identifier: MIT OR Apache-2.0
"""SPEC 18 primary gate for acrobot: one-step equivalence of
``tessera/Acrobot-v1`` with the pinned Gymnasium over 100,000 rollout states
plus the boundary set, in every auto-reset mode
(``tools/gym_equivalence_acrobot.py``). The tool's uniform-domain set is
reported, not gated."""

import importlib.util
import pathlib

import numpy as np
import pytest

_TOOL = pathlib.Path(__file__).resolve().parents[2] / "tools" / "gym_equivalence_acrobot.py"
_spec = importlib.util.spec_from_file_location("gym_equivalence_acrobot", _TOOL)
eq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eq)

N = 100_000
SEED = 2026


@pytest.fixture(scope="module")
def sampled():
    states, actions = eq.gate_set(N, SEED)
    return states, actions, eq.reference(states, actions)


def test_acrobot_equivalence_sampling_is_deterministic(sampled):
    states, actions, _ = sampled
    again, again_actions = eq.sample(1000, SEED)
    assert np.array_equal(states[:1000], again) and np.array_equal(actions[:1000], again_actions)


@pytest.mark.parametrize("mode", eq.MODES)
def test_acrobot_equivalence_one_step_matches_gymnasium(sampled, mode):
    states, actions, ref = sampled
    report = eq.compare(eq.one_step(states, actions, mode), ref)
    print(mode, report)
    assert report["states"] > N and report["reference_terminations"] > 0
    assert report["near_threshold"] == report["near_threshold_terminated_agree"]
    for key in ("state", "obs", "reward", "terminated", "truncated"):
        assert report[key] == 0, (key, report)


def test_acrobot_equivalence_gate_detects_a_wrong_transition(sampled):
    states, actions, ref = sampled
    k = 5000
    report = eq.compare(eq.one_step(states[:k], 2 - actions[:k], "next_step"), ref[:k])
    assert report["state"] > 0


def test_acrobot_equivalence_obs_mismatch_names_index_component_and_values(sampled):
    states, actions, ref = sampled
    k = 8
    ours = eq.one_step(states[:k], actions[:k], "next_step")
    n, o, r, t, tr = ours[3]
    o = np.array(o, copy=True)
    o[2] = np.nextafter(np.float32(o[2]), np.float32(np.inf)) + np.float32(1e-3)
    ours[3] = (n, o, r, t, tr)
    report = eq.compare(ours, ref[:k])
    assert report["obs"] == 1
    (example,) = report["obs_examples"]
    assert example["index"] == 3 and len(example["state"]) == 4
    (component,) = example["components"]
    assert component["component"] == 2 and component["ulps32"] > 0 and component["relative"] is not None
    assert float.fromhex(component["ours"]) == float(np.float32(o[2]))


def test_acrobot_equivalence_boundary_set_covers_every_class(sampled):
    states, _, ref = sampled
    cover = eq.boundary_coverage(states, ref, N)
    print(cover)
    assert all(cover.values()), cover


def test_acrobot_equivalence_short_trajectories_report(sampled):
    report = eq.trajectories(sampled[0][:N])
    print(report)
    assert report["episodes"] == 1000 and report["length"] == 50
    assert report["compared_steps"] == report["expected_steps"] > 0


def test_acrobot_equivalence_trajectories_stop_at_termination():
    # The first start ends on its first step, the second runs the whole plan.
    starts = np.array([[2.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    plans = np.ones((2, 5), dtype=np.int32)
    report = eq.trajectories(starts, starts=starts, plans=plans)
    assert report["reference_terminated_episodes"] == 1
    assert report["compared_steps"] == report["expected_steps"] == 1 + 5
    assert report["flag_mismatches"] == 0 and report["reward_mismatches"] == 0


def test_acrobot_equivalence_truncates_with_time_limit():
    assert eq.truncation()["mismatches"] == 0


def test_acrobot_equivalence_uniform_set_stays_fixed():
    # Reported, not gated; its size and seed must not drift.
    states, _ = eq.spread(N // 5, SEED + 1)
    assert len(states) == 16_341
