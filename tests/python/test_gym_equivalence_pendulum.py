# SPDX-License-Identifier: MIT OR Apache-2.0
"""SPEC 18 and ADR-0050 gate for pendulum, on the committed fresh seed, with
the kernel's raw reward (test-hooks build)."""

import importlib.util
import json
import pathlib
import platform

import numpy as np
import pytest

from tessera_rl import _native

if not hasattr(_native, "_pendulum_rewards"):
    pytest.skip("needs a test-hooks build (make py-test-hooks)", allow_module_level=True)

_TOOL = pathlib.Path(__file__).resolve().parents[2] / "tools" / "gym_equivalence_pendulum.py"
_spec = importlib.util.spec_from_file_location("gym_equivalence_pendulum", _TOOL)
eq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eq)


@pytest.fixture(scope="module")
def sampled():
    states, actions = eq.gate_set(eq.GATE_STATES, eq.GATE_SEED)
    return states, actions, eq.reference(states, actions), eq.kernel_rewards(states, actions)


def test_pendulum_gate_seed_and_size_are_the_committed_ones():
    assert (eq.GATE_SEED, eq.GATE_STATES) == (51_120_001, 100_000)


@pytest.mark.parametrize("mode", eq.MODES)
def test_pendulum_equivalence_matches_gymnasium_within_adr_0050(sampled, mode, capsys):
    states, actions, ref, raw = sampled
    report = eq.compare(eq.one_step(states, actions, mode), ref, states, actions, raw)
    # ADR-0050 asks for the exact-match counts on each platform: in the CI log.
    with capsys.disabled():
        print("\nADR-0050", platform.platform(), platform.libc_ver(), mode, json.dumps(report))
    assert report["states"] > eq.GATE_STATES and report["speed_clips"] > 0
    for key in eq.KEYS:
        assert report[key] == 0, (key, report)


def test_pendulum_equivalence_gate_detects_a_wrong_transition(sampled):
    states, actions, ref, raw = sampled
    k = 5000
    report = eq.compare(eq.one_step(states[:k], -actions[:k], "next_step"), ref[:k], states[:k], actions[:k], raw[:k])
    assert report["state"] > 0


def test_pendulum_equivalence_bound_detects_a_wrong_reward(sampled):
    states, actions, ref, raw = sampled
    k = 5000
    wrong = np.nextafter(raw[:k], np.inf) * (1 + 1e-7)
    report = eq.compare(eq.one_step(states[:k], actions[:k], "next_step"), ref[:k], states[:k], actions[:k], wrong)
    assert report["raw_over_bound"] > 0


def test_pendulum_equivalence_truncates_with_time_limit():
    assert eq.truncation()["mismatches"] == 0


def test_pendulum_equivalence_short_trajectories_report(sampled):
    report = eq.trajectories(sampled[0][: eq.GATE_STATES])
    print(report)
    assert report["compared_steps"] > 0


def _row(theta, thetadot=0.25):
    """A next state and its float32 observation, both computed here."""
    n = np.array([theta, thetadot])
    return n, np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)


def test_trig_bound_value_is_pinned():
    # 0.5 and 0.25 lie in different binades: every term is visible.
    a, b, dtheta = 0.5, 0.25, 1e-9
    s32a, s32b = 2.0**-24, 2.0**-25
    s64a, s64b = 2.0**-53, 2.0**-54
    assert eq.trig_bound(dtheta, a, b) == dtheta + s32a / 2 + s32b / 2 + (s64a + s64b)


@pytest.mark.parametrize("component", [0, 1])
def test_obs_close_trig_two_ulps_off_with_equal_states_fails(component):
    # Far from every zero: rounding allows half a spacing per side, so two
    # float32 ulps with identical next states is a wrong observation.
    n, o_ref = _row(0.7)
    o = o_ref.copy()
    for _ in range(2):
        o[component] = np.nextafter(o[component], np.float32(np.inf))
    assert not eq.obs_close(o, o_ref, n.copy(), n)


def test_obs_close_velocity_is_not_trig():
    # The angles differ by far more than the velocity gap, which a trig
    # component would be allowed; the velocity keeps the ulp test.
    n_ours, o_ours = _row(0.7)
    n_ref, o_ref = _row(0.7 + 1e-3)
    o_ours[:2] = o_ref[:2]
    o_ours[2] = np.float32(o_ref[2] + np.float32(1e-5))
    assert not eq.obs_close(o_ours, o_ref, n_ours, n_ref)
    o_ours[2] = o_ref[2]
    assert eq.obs_close(o_ours, o_ref, n_ours, n_ref)


# Both sides compute sin with this libm on angles 1 ulp apart next to -pi.
# The two sines lie in different float32 binades; their float32 values are
# further apart than the angles by exactly the coarser side's half spacing,
# more than the finer side's. Near a zero the sines are offsets of pi by
# multiples of the angle's ulp, so the finer side rounds alike and no such
# row needs both halves at once; each orientation needs one side's half.
_T_COARSE = float.fromhex("-0x1.921fb54442d1ap+1")
_T_FINE = float.fromhex("-0x1.921fb54442d19p+1")


@pytest.mark.parametrize("ours_coarse", [True, False])
def test_obs_close_honest_row_needs_the_rounding_term(ours_coarse):
    t_ours, t_ref = (_T_COARSE, _T_FINE) if ours_coarse else (_T_FINE, _T_COARSE)
    n_ours, o_ours = _row(t_ours)
    n_ref, o_ref = _row(t_ref)
    gap = abs(float(o_ours[1]) - float(o_ref[1]))
    finer = min(np.spacing(np.abs(o_ours[1])), np.spacing(np.abs(o_ref[1])))
    assert gap - abs(t_ours - t_ref) > float(finer) / 2 and gap > np.spacing(np.abs(o_ref[1]))
    assert eq.obs_close(o_ours, o_ref, n_ours, n_ref)


def test_angle_gap_is_taken_modulo_two_pi():
    # Not reached by the gated rows (no pair is 2 pi apart); kept so the
    # modulo stays correct if one is.
    assert eq.angle_gap(0.5, 0.5 + 2 * np.pi) < 1e-15
