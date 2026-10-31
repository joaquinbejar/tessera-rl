# SPDX-License-Identifier: MIT OR Apache-2.0
"""The SPEC 18 learning-test logic (``learning_test.py``), with no PyTorch
dependency so the tests can check it on stub scores.

- Calibration: the threshold is the 5th percentile (``numpy`` method
  ``"lower"``, so it is an observed score) of at least 40 calibration
  scores.
- A night fails when the median of its seed scores is below the
  threshold.
- An alert (a red job) needs two consecutive failing nights. A missing
  previous night counts as not failing, and is reported.

Everything fails closed: missing, duplicate or non-finite scores, too few
calibration seeds, or a calibration for another budget or environment are
errors, never a pass.
"""

from __future__ import annotations

import math

import numpy as np

MIN_CALIBRATION_SEEDS = 40


def _scores(records: list[dict], seeds: list[int]) -> tuple[list[float], list[str]]:
    errors = []
    by_seed: dict[int, list[float]] = {}
    for r in records:
        by_seed.setdefault(r.get("seed"), []).append(r.get("score"))
    for seed, values in by_seed.items():
        if len(values) > 1:
            errors.append(f"seed {seed}: {len(values)} records")
        if seed not in seeds:
            errors.append(f"seed {seed}: unexpected")
    scores = []
    for seed in seeds:
        values = by_seed.get(seed)
        if not values:
            errors.append(f"seed {seed}: missing")
            continue
        v = values[0]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            errors.append(f"seed {seed}: score is not a finite number")
            continue
        scores.append(float(v))
    return scores, errors


def calibrate(records: list[dict], seeds: list[int]) -> dict:
    """The threshold from the calibration scores of ``seeds``."""
    out: dict = {"seeds": sorted(seeds)}
    scores, errors = _scores(records, seeds)
    if len(seeds) < MIN_CALIBRATION_SEEDS:
        errors.append(f"{len(seeds)} calibration seeds, at least {MIN_CALIBRATION_SEEDS} required")
    if errors:
        out["errors"] = errors
        return out
    out["threshold"] = float(np.percentile(scores, 5, method="lower"))
    out["median"] = float(np.median(scores))
    out["min"] = min(scores)
    out["max"] = max(scores)
    return out


def night(records: list[dict], seeds: list[int], calibration: dict, setup: dict) -> dict:
    """One night: the median of the seed scores against the calibrated
    threshold. ``setup`` (environment id, budget, eval episodes) must match
    the calibration's."""
    out: dict = {"seeds": sorted(seeds)}
    scores, errors = _scores(records, seeds)
    threshold = calibration.get("threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or calibration.get("errors"):
        errors.append("the calibration has no valid threshold")
    for key, value in setup.items():
        if calibration.get("setup", {}).get(key) != value:
            errors.append(f"setup {key}={value!r} differs from the calibration's {calibration.get('setup', {}).get(key)!r}")
    if not seeds:
        errors.append("no seeds")
    if errors:
        out["errors"] = errors
        out["failed"] = True
        return out
    out["threshold"] = float(threshold)
    out["median"] = float(np.median(scores))
    out["scores"] = scores
    out["failed"] = out["median"] < out["threshold"]
    return out


def alert(tonight: dict, previous: dict | None) -> dict:
    """Red only after two consecutive failing nights. A missing or
    unreadable previous night counts as not failing, and is reported."""
    out: dict = {"tonight_failed": bool(tonight.get("failed", True))}
    if previous is None or not isinstance(previous.get("failed"), bool):
        out["previous"] = "missing"
        out["previous_failed"] = False
    else:
        out["previous"] = "found"
        out["previous_failed"] = previous["failed"]
    out["red"] = out["tonight_failed"] and out["previous_failed"]
    return out
