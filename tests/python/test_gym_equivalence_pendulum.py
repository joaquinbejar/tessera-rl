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
