# SPDX-License-Identifier: MIT OR Apache-2.0
"""The reference-training gate fails closed (tools/reference_gate.py)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

from reference_gate import evaluate_gate  # noqa: E402

SEEDS = [1, 2, 3]
MUTANTS = ["swap_flags"]


def run(backend, seed, eval_mean=500.0, curve_mean=395.0):
    return {"backend": backend, "seed": seed, "eval_mean": eval_mean, "curve_mean": curve_mean}


def good():
    runs = [run(b, s) for s in SEEDS for b in ("tessera", "gymnasium")]
    return runs + [run("mutant:swap_flags", s, eval_mean=120.0, curve_mean=150.0) for s in SEEDS]


def test_reference_gate_matching_runs_with_failing_mutants_pass():
    assert evaluate_gate(good(), 475.0, 20.0, SEEDS, MUTANTS)["pass"]


def test_reference_gate_missing_control_fails():
    runs = [r for r in good() if r["backend"] != "gymnasium"]
    gate = evaluate_gate(runs, 475.0, 20.0, SEEDS, MUTANTS)
    assert not gate["pass"] and "error" in gate["tessera"]


def test_reference_gate_misspelled_backend_fails():
    runs = [dict(r, backend="tesera") if r["backend"] == "tessera" else r for r in good()]
    assert not evaluate_gate(runs, 475.0, 20.0, SEEDS, MUTANTS)["pass"]


def test_reference_gate_missing_tessera_seed_fails():
    runs = [r for r in good() if not (r["backend"] == "tessera" and r["seed"] == 3)]
    assert not evaluate_gate(runs, 475.0, 20.0, SEEDS, MUTANTS)["pass"]


def test_reference_gate_missing_mutant_runs_fail():
    runs = [r for r in good() if not r["backend"].startswith("mutant:")]
    assert not evaluate_gate(runs, 475.0, 20.0, SEEDS, MUTANTS)["pass"]


def test_reference_gate_passing_mutant_fails_the_gate():
    runs = [dict(r, eval_mean=500.0, curve_mean=395.0) if r["backend"].startswith("mutant:") else r for r in good()]
    gate = evaluate_gate(runs, 475.0, 20.0, SEEDS, MUTANTS)
    assert not gate["pass"] and not gate["every_mutant_fails"]


def test_reference_gate_curve_difference_beyond_margin_fails():
    runs = [dict(r, curve_mean=360.0) if r["backend"] == "tessera" else r for r in good()]
    assert not evaluate_gate(runs, 475.0, 20.0, SEEDS, MUTANTS)["pass"]


def test_reference_gate_median_below_threshold_fails():
    runs = [dict(r, eval_mean=400.0) if r["backend"] == "tessera" and r["seed"] < 3 else r for r in good()]
    assert not evaluate_gate(runs, 475.0, 20.0, SEEDS, MUTANTS)["pass"]
