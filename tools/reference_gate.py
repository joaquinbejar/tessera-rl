# SPDX-License-Identifier: MIT OR Apache-2.0
"""The gate of the reference training (``reference_training.py``), with no
PyTorch dependency so the tests can check it on stub runs.

A run is a dict with ``backend`` (``"tessera"``, ``"gymnasium"`` or
``"mutant:<kind>"``), ``seed``, ``eval_mean`` and ``curve_mean``.
"""

from __future__ import annotations

import numpy as np


def _gate_for(runs: list[dict], backend: str, threshold: float, margin: float) -> dict:
    """The gate of ``backend`` (Tessera or a mutant) against the Gymnasium
    control over the seeds both ran. Fails closed: no runs of either side,
    or different seed sets, fail."""
    mine = {r["seed"]: r for r in runs if r["backend"] == backend}
    ref = {r["seed"]: r for r in runs if r["backend"] == "gymnasium"}
    g: dict = {"backend": backend, "seeds": sorted(mine)}
    if not mine or not ref or not set(mine) <= set(ref):
        g["error"] = "missing runs: both the backend and the Gymnasium control are required on the same seeds"
        g["pass"] = False
        return g
    seeds = sorted(mine)
    evals = [mine[s]["eval_mean"] for s in seeds]
    g["eval_median"] = float(np.median(evals))
    g["median_reaches_threshold"] = g["eval_median"] >= threshold
    g["gymnasium_eval_mean"] = float(np.mean([ref[s]["eval_mean"] for s in seeds]))
    g["curve_mean"] = float(np.mean([mine[s]["curve_mean"] for s in seeds]))
    g["gymnasium_curve_mean"] = float(np.mean([ref[s]["curve_mean"] for s in seeds]))
    g["curve_difference"] = g["curve_mean"] - g["gymnasium_curve_mean"]
    g["difference_within_margin"] = abs(g["curve_difference"]) <= margin
    g["pass"] = g["median_reaches_threshold"] and g["difference_within_margin"]
    return g


def evaluate_gate(runs: list[dict], threshold: float, margin: float, seeds: list[int], mutants: list[str]) -> dict:
    """Tessera must pass against the control on every gate seed, and every
    negative control (mutant) must fail."""
    gate: dict = {"threshold": threshold, "margin": margin}
    tessera = _gate_for(runs, "tessera", threshold, margin)
    if tessera.get("seeds") != sorted(seeds):
        tessera["pass"] = False
        tessera.setdefault("error", "Tessera did not run every gate seed")
    gate["tessera"] = tessera
    gate["mutants"] = {m: _gate_for(runs, f"mutant:{m}", threshold, margin) for m in mutants}
    gate["every_mutant_fails"] = all(
        not g["pass"] and "error" not in g for g in gate["mutants"].values()
    )
    gate["pass"] = tessera["pass"] and gate["every_mutant_fails"]
    return gate
