# SPDX-License-Identifier: MIT OR Apache-2.0
"""The gate of the reference training (``reference_training.py``), with no
PyTorch dependency so the tests can check it on stub runs.

A run is a dict with ``backend`` (``"tessera"``, ``"gymnasium"`` or
``"mutant:<kind>"``), ``seed``, ``eval_mean`` and ``curve_mean``.

The gate fails closed: before any comparison, the records must be valid
(finite metrics, one record per backend and seed) and complete (Tessera on
exactly the gate seeds, every mutant on exactly the mutant seeds, the
Gymnasium control on all of them). A mutant counts as a measured failure
only on valid, complete records.
"""

from __future__ import annotations

import math

import numpy as np


def truncation_bootstrap_mask(terminated, truncated) -> np.ndarray:
    """The instances whose last reward is bootstrapped with the value of
    the final observation: truncated and not terminated. An episode that
    is both (a terminal state reached on the time limit step) is terminal."""
    return np.logical_and(np.asarray(truncated, dtype=bool), ~np.asarray(terminated, dtype=bool))


def _validate(runs: list[dict], seeds: list[int], mutants: list[str], mutant_seeds: list[int]) -> list[str]:
    errors = []
    seen: dict[tuple[str, int], int] = {}
    for r in runs:
        key = (r.get("backend"), r.get("seed"))
        seen[key] = seen.get(key, 0) + 1
        for metric in ("eval_mean", "curve_mean"):
            v = r.get(metric)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                errors.append(f"{key}: {metric} is not a finite number")
    errors += [f"{key}: {n} records" for key, n in seen.items() if n > 1]
    expected = {("tessera", s) for s in seeds}
    expected |= {("gymnasium", s) for s in set(seeds) | set(mutant_seeds)}
    expected |= {(f"mutant:{m}", s) for m in mutants for s in mutant_seeds}
    errors += [f"{key}: missing" for key in sorted(expected - set(seen))]
    errors += [f"{key}: unexpected" for key in sorted(set(seen) - expected, key=str)]
    if not seeds:
        errors.append("no gate seeds")
    if mutants and not mutant_seeds:
        errors.append("no mutant seeds")
    return errors


def _gate_for(runs: list[dict], backend: str, seeds: list[int], threshold: float, margin: float) -> dict:
    """The gate of ``backend`` (Tessera or a mutant) against the Gymnasium
    control on ``seeds``; the records are already validated."""
    mine = {r["seed"]: r for r in runs if r["backend"] == backend}
    ref = {r["seed"]: r for r in runs if r["backend"] == "gymnasium"}
    seeds = sorted(seeds)
    g: dict = {"backend": backend, "seeds": seeds}
    g["eval_median"] = float(np.median([mine[s]["eval_mean"] for s in seeds]))
    g["median_reaches_threshold"] = g["eval_median"] >= threshold
    g["gymnasium_eval_mean"] = float(np.mean([ref[s]["eval_mean"] for s in seeds]))
    g["curve_mean"] = float(np.mean([mine[s]["curve_mean"] for s in seeds]))
    g["gymnasium_curve_mean"] = float(np.mean([ref[s]["curve_mean"] for s in seeds]))
    g["curve_difference"] = g["curve_mean"] - g["gymnasium_curve_mean"]
    g["difference_within_margin"] = abs(g["curve_difference"]) <= margin
    g["pass"] = g["median_reaches_threshold"] and g["difference_within_margin"]
    return g


def evaluate_gate(
    runs: list[dict],
    threshold: float,
    margin: float,
    seeds: list[int],
    mutants: list[str],
    mutant_seeds: list[int],
) -> dict:
    """Tessera must pass against the control on every gate seed, and every
    negative control (mutant) must fail, on valid and complete records."""
    gate: dict = {"threshold": threshold, "margin": margin}
    errors = _validate(runs, seeds, mutants, mutant_seeds)
    if errors:
        gate["errors"] = errors
        gate["every_mutant_fails"] = False
        gate["pass"] = False
        return gate
    gate["tessera"] = _gate_for(runs, "tessera", seeds, threshold, margin)
    gate["mutants"] = {m: _gate_for(runs, f"mutant:{m}", mutant_seeds, threshold, margin) for m in mutants}
    gate["every_mutant_fails"] = all(not g["pass"] for g in gate["mutants"].values())
    gate["pass"] = gate["tessera"]["pass"] and gate["every_mutant_fails"]
    return gate
