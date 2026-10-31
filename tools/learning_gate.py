# SPDX-License-Identifier: MIT OR Apache-2.0
"""The SPEC 18 learning-test logic (``learning_test.py``), with no PyTorch
dependency so the tests can check it on stub scores.

- Calibration: exactly the seeds 101-140, one score each; the threshold
  is the 5th percentile (``numpy`` method ``"lower"``, so it is an
  observed score). A stored calibration is re-checked: its threshold must
  be the one its records give.
- A night: exactly the seeds 1-5, one score each; it fails when the median
  is below the threshold.
- Alert (a red job): a night with invalid data or setup is red at once.
  Only a valid night below the threshold is debounced: it is red only when
  the previous scheduled night, on the same calibration, failed too. A
  missing or unusable previous night counts as not failing, and is
  reported.
"""

from __future__ import annotations

import math

import numpy as np

CALIBRATION_SEEDS = list(range(101, 141))
NIGHT_SEEDS = [1, 2, 3, 4, 5]


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _scores(records: list[dict], seeds: list[int]) -> tuple[list[float], list[str]]:
    """The scores of exactly ``seeds``, one record each, all finite."""
    errors = []
    if len(set(seeds)) != len(seeds):
        errors.append("the declared seeds repeat")
    by_seed: dict = {}
    for r in records:
        by_seed.setdefault(r.get("seed"), []).append(r.get("score"))
    errors += [f"seed {s}: {len(v)} records" for s, v in by_seed.items() if len(v) > 1]
    errors += [f"seed {s}: unexpected" for s in by_seed if s not in seeds]
    scores = []
    for seed in seeds:
        values = by_seed.get(seed)
        if not values:
            errors.append(f"seed {seed}: missing")
        elif not _finite(values[0]):
            errors.append(f"seed {seed}: score is not a finite number")
        else:
            scores.append(float(values[0]))
    return scores, errors


def calibrate(records: list[dict], seeds: list[int]) -> dict:
    """The threshold from the calibration scores; ``seeds`` must be exactly
    ``CALIBRATION_SEEDS``."""
    out: dict = {"seeds": list(seeds)}
    scores, errors = _scores(records, seeds)
    if list(seeds) != CALIBRATION_SEEDS:
        errors.append(f"the calibration seeds must be exactly {CALIBRATION_SEEDS[0]}-{CALIBRATION_SEEDS[-1]}")
    if errors:
        out["errors"] = errors
        return out
    out["threshold"] = float(np.percentile(scores, 5, method="lower"))
    out["median"] = float(np.median(scores))
    out["min"] = min(scores)
    out["max"] = max(scores)
    return out


def check_calibration(calibration: dict) -> list[str]:
    """Errors of a stored calibration: its records must give its threshold."""
    if calibration.get("errors"):
        return ["the calibration recorded errors"]
    again = calibrate(calibration.get("records") or [], calibration.get("seeds") or [])
    if again.get("errors"):
        return [f"calibration: {e}" for e in again["errors"]]
    if not _finite(calibration.get("threshold")) or calibration["threshold"] != again["threshold"]:
        return [f"the stored threshold {calibration.get('threshold')!r} is not the records' {again['threshold']!r}"]
    return []


def night(records: list[dict], seeds: list[int], calibration: dict, setup: dict) -> dict:
    """One night against the calibration. ``setup`` (environment id, budget,
    eval episodes, fingerprint) must match the calibration's."""
    out: dict = {"seeds": list(seeds)}
    scores, errors = _scores(records, seeds)
    if list(seeds) != NIGHT_SEEDS:
        errors.append(f"the night seeds must be exactly {NIGHT_SEEDS}")
    errors += check_calibration(calibration)
    stored = calibration.get("setup") or {}
    for key in sorted(set(setup) | set(stored)):
        if stored.get(key) != setup.get(key):
            errors.append(f"setup {key}={setup.get(key)!r} differs from the calibration's {stored.get(key)!r}")
    if errors:
        out["errors"] = errors
        out["failed"] = True
        return out
    out["threshold"] = float(calibration["threshold"])
    out["median"] = float(np.median(scores))
    out["scores"] = scores
    out["failed"] = out["median"] < out["threshold"]
    return out


def alert(tonight: dict, identity: dict, previous: dict | None) -> dict:
    """Whether the job is red. ``identity`` is tonight's (``event``,
    ``calibration``: the calibration file's hash); ``previous`` is the last
    scheduled night's whole result (its ``night`` and ``identity``)."""
    out: dict = {"tonight_failed": bool(tonight.get("failed", True))}
    if tonight.get("errors") or not isinstance(tonight.get("failed"), bool):
        out["reason"] = "invalid night: red at once"
        out["red"] = True
        return out
    usable = (
        isinstance(previous, dict)
        and isinstance((previous.get("night") or {}).get("failed"), bool)
        and not (previous.get("night") or {}).get("errors")
        and (previous.get("identity") or {}).get("event") == "schedule"
        and (previous.get("identity") or {}).get("calibration") == identity.get("calibration")
    )
    if not usable:
        out["previous"] = "missing" if previous is None else "unusable (not a valid scheduled night on this calibration)"
        out["previous_failed"] = False
    else:
        out["previous"] = (previous.get("identity") or {}).get("run_id", "found")
        out["previous_failed"] = previous["night"]["failed"]
    out["red"] = out["tonight_failed"] and out["previous_failed"]
    return out


PROVENANCE = ("git", "git_dirty", "python", "arch", "image", "tessera_rl", "gymnasium", "torch", "numpy")


def check_shards(shards: list[dict]) -> list[str]:
    """Errors of a set of calibration score files: one provenance (a clean,
    committed tree; the same versions, platform and setup) for all."""
    if not shards:
        return ["no score files"]
    errors = []
    first = shards[0]
    for i, shard in enumerate(shards):
        env = shard.get("environment") or {}
        if env.get("git") in (None, "", "unknown") or env.get("git_dirty") is not False:
            errors.append(f"shard {i}: not from a clean, committed tree")
        for key in PROVENANCE:
            if env.get(key) != (first.get("environment") or {}).get(key):
                errors.append(f"shard {i}: {key} differs from shard 0")
        if shard.get("setup") != first.get("setup"):
            errors.append(f"shard {i}: setup differs from shard 0")
    return errors
