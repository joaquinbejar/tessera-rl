# SPDX-License-Identifier: MIT OR Apache-2.0
"""The reference-training gate fails closed (tools/reference_gate.py)."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

from reference_gate import evaluate_gate, truncation_bootstrap_mask  # noqa: E402

SEEDS = [1, 2, 3]
MUTANTS = ["swap_flags", "reward_sign"]
MUTANT_SEEDS = [1, 2]


def run(backend, seed, eval_mean=500.0, curve_mean=395.0):
    return {"backend": backend, "seed": seed, "eval_mean": eval_mean, "curve_mean": curve_mean}


def good():
    runs = [run(b, s) for s in SEEDS for b in ("tessera", "gymnasium")]
    return runs + [run(f"mutant:{m}", s, eval_mean=10.0, curve_mean=150.0) for m in MUTANTS for s in MUTANT_SEEDS]


def gate(runs):
    return evaluate_gate(runs, 475.0, 20.0, SEEDS, MUTANTS, MUTANT_SEEDS)


def test_truncation_bootstrap_mask_terminal_on_the_time_limit_step_is_not_bootstrapped():
    term = np.array([False, True, True, False])
    trunc = np.array([True, True, False, False])
    assert truncation_bootstrap_mask(term, trunc).tolist() == [True, False, False, False]


def test_reference_gate_matching_runs_with_failing_mutants_pass():
    assert gate(good())["pass"]


def test_reference_gate_missing_control_fails():
    g = gate([r for r in good() if r["backend"] != "gymnasium"])
    assert not g["pass"] and g["errors"]


def test_reference_gate_misspelled_backend_fails():
    assert not gate([dict(r, backend="tesera") if r["backend"] == "tessera" else r for r in good()])["pass"]


def test_reference_gate_missing_tessera_seed_fails():
    assert not gate([r for r in good() if not (r["backend"] == "tessera" and r["seed"] == 3)])["pass"]


def test_reference_gate_missing_mutant_runs_fail():
    assert not gate([r for r in good() if not r["backend"].startswith("mutant:")])["pass"]


def test_reference_gate_incomplete_mutant_matrix_fails():
    g = gate([r for r in good() if not (r["backend"] == "mutant:reward_sign" and r["seed"] == 2)])
    assert not g["pass"] and any("missing" in e for e in g["errors"])


def test_reference_gate_nan_mutant_metrics_fail():
    runs = [dict(r, eval_mean=float("nan"), curve_mean=float("nan")) if r["backend"].startswith("mutant:") else r for r in good()]
    g = gate(runs)
    assert not g["pass"] and not g["every_mutant_fails"]


def test_reference_gate_non_numeric_metric_fails():
    runs = good()
    runs[0] = dict(runs[0], eval_mean=None)
    assert not gate(runs)["pass"]


def test_reference_gate_duplicate_records_fail():
    runs = good() + [run("tessera", 1, eval_mean=10.0, curve_mean=10.0)]
    g = gate(runs)
    assert not g["pass"] and any("records" in e for e in g["errors"])


def test_reference_gate_unexpected_seed_fails():
    assert not gate(good() + [run("tessera", 9)])["pass"]


def test_reference_gate_passing_mutant_fails_the_gate():
    runs = [dict(r, eval_mean=500.0, curve_mean=395.0) if r["backend"] == "mutant:swap_flags" else r for r in good()]
    g = gate(runs)
    assert not g["pass"] and not g["every_mutant_fails"]


def test_reference_gate_curve_difference_beyond_margin_fails():
    assert not gate([dict(r, curve_mean=360.0) if r["backend"] == "tessera" else r for r in good()])["pass"]


def test_reference_gate_median_below_threshold_fails():
    runs = [dict(r, eval_mean=400.0) if r["backend"] == "tessera" and r["seed"] < 3 else r for r in good()]
    assert not gate(runs)["pass"]
