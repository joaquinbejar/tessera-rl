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


# Issue #191: at rollout index 100099 (seed 2026) a reference whose RK4 runs
# on another libm lands theta2 1 ulp away, exactly on -pi, so its sin(theta2)
# is far off in float32 ulps although the gap is about 1e-16. The values
# below are the recorded ones; the rest of the row comes from the reference
# step on this machine.
_START_100099 = [float.fromhex(h) for h in ("0x1.de4a62f41b498p-4", "-0x1.7e6dc4daea8d3p+1", "-0x1.034112ddc520ep+2", "-0x1.c2bd2296565c9p-1")]
_THETA2_OURS = float.fromhex("-0x1.921fb54442d17p+1")
_THETA2_REF = float.fromhex("-0x1.921fb54442d18p+1")
_SIN2_OURS = float.fromhex("-0x1.4698980000000p-51")
_SIN2_REF = float.fromhex("-0x1.1a62640000000p-53")


def _row_100099():
    n, o, r, t, tr = eq._ref_step(eq._ref_env(), _START_100099, 2)
    n_ours, n_ref = n.copy(), n.copy()
    n_ours[1], n_ref[1] = _THETA2_OURS, _THETA2_REF
    o_ours, o_ref = np.array(o, dtype=np.float32), np.array(o, dtype=np.float32)
    o_ours[3], o_ref[3] = _SIN2_OURS, _SIN2_REF
    return (n_ours, o_ours, r, t, tr), (n_ref, o_ref, r, t, tr)


def test_obs_close_recorded_one_ulp_angle_on_minus_pi_passes():
    ours, ref = _row_100099()
    assert eq.ulps32(_SIN2_OURS, _SIN2_REF) > 1_000_000
    assert eq.obs_close(ours[1], ref[1], ours[0], ref[0])
    report = eq.compare([ours], [ref])
    assert report["obs"] == 0 and report["state"] == 0, report


def test_obs_close_trig_gap_beyond_the_angle_difference_fails():
    ours, ref = _row_100099()
    o = ours[1].copy()
    dtheta = abs(_THETA2_OURS - _THETA2_REF)
    o[3] = np.float32(_SIN2_REF - 1.5 * dtheta)
    assert not eq.obs_close(o, ref[1], ours[0], ref[0])
    report = eq.compare([(ours[0], o, *ours[2:])], [ref])
    assert report["obs"] == 1, report


def test_obs_close_non_trig_component_keeps_the_ulp_test():
    ours, ref = _row_100099()
    o = ours[1].copy()
    o[5] = np.nextafter(np.nextafter(o[5], np.float32(np.inf)), np.float32(np.inf))
    assert not eq.obs_close(o, ref[1], ours[0], ref[0])


def test_obs_close_without_next_state_keeps_the_ulp_test():
    ours, ref = _row_100099()
    assert not eq.obs_close(ours[1], ref[1], None, ref[0])


def _trig_row(theta1, theta2):
    """A next state and its float32 observation, both computed here."""
    n = np.array([theta1, theta2, 0.25, -0.5])
    o = np.array([np.cos(n[0]), np.sin(n[0]), np.cos(n[1]), np.sin(n[1]), n[2], n[3]], dtype=np.float32)
    return n, o


@pytest.mark.parametrize("component", [0, 1, 2, 3])
def test_obs_close_trig_two_ulps_off_with_equal_states_fails(component):
    # Far from every zero: rounding allows half a spacing per side, so two
    # float32 ulps with identical next states is a wrong observation.
    n, o_ref = _trig_row(0.7, -2.1)
    o = o_ref.copy()
    for _ in range(2):
        o[component] = np.nextafter(o[component], np.float32(np.inf))
    assert not eq.obs_close(o, o_ref, n.copy(), n)


def test_obs_close_honest_row_needs_the_rounding_term():
    # Both sides compute sin with this libm on angles 1 ulp apart next to
    # -pi; their float32 values are further apart than the angles are, by
    # less than the two half spacings.
    t_ours = float.fromhex("-0x1.921fb54442d1ap+1")
    t_ref = float.fromhex("-0x1.921fb54442d19p+1")
    n_ours, o_ours = _trig_row(0.7, t_ours)
    n_ref, o_ref = _trig_row(0.7, t_ref)
    gap = abs(float(o_ours[3]) - float(o_ref[3]))
    assert gap > abs(t_ours - t_ref) and gap > np.spacing(np.abs(o_ref[3]))
    assert eq.obs_close(o_ours, o_ref, n_ours, n_ref)


def test_angle_gap_wraps_round_the_circle():
    # Not reached by the gated rows (no recorded pair straddles +-pi); kept
    # so that the wrap in angle_gap stays correct if one does.
    assert eq.angle_gap(np.pi, -np.pi) < 1e-15
