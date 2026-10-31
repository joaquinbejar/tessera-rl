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

import datetime
import math

import numpy as np

CALIBRATION_SEEDS = list(range(101, 141))
# One full PPO rollout of the reference trainer (num_envs * num_steps): a
# smaller budget trains nothing.
MIN_TOTAL_STEPS = 4 * 128
NIGHT_SEEDS = [1, 2, 3, 4, 5]


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_budget(setup) -> list[str]:
    """Errors of a setup's budget: at least one full rollout, and at least
    one evaluation episode."""
    if not isinstance(setup, dict):
        return ["no setup"]
    errors = []
    steps, episodes = setup.get("total_steps"), setup.get("eval_episodes")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < MIN_TOTAL_STEPS:
        errors.append(f"total_steps {steps!r} is less than one rollout ({MIN_TOTAL_STEPS} steps)")
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes <= 0:
        errors.append(f"eval_episodes {episodes!r} is not a positive integer")
    return errors


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
    budget = check_budget(calibration.get("setup"))
    if budget:
        return [f"calibration: {e}" for e in budget]
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
    errors += check_budget(setup)
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


DEBOUNCE_WINDOW_HOURS = 48


def _date(text):
    try:
        d = datetime.datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return d if d.tzinfo else None


def _run_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _previous_errors(identity: dict, previous) -> list[str]:
    """Why ``previous`` is not the immediately preceding scheduled night of
    this one, on the same calibration (empty when it is)."""
    if not isinstance(previous, dict):
        return ["missing"]
    n = previous.get("night") or {}
    who = previous.get("identity") or {}
    errors = []
    if not isinstance(n.get("failed"), bool) or n.get("errors"):
        errors.append("not a valid night")
    if who.get("event") != "schedule":
        errors.append("not a scheduled run")
    if who.get("calibration") != identity.get("calibration"):
        errors.append("another calibration")
    now, then = _date(identity.get("date")), _date(who.get("date"))
    if now is None or then is None:
        errors.append("no date")
    elif not datetime.timedelta(0) < now - then <= datetime.timedelta(hours=DEBOUNCE_WINDOW_HOURS):
        errors.append(f"not within {DEBOUNCE_WINDOW_HOURS} h before this night")
    mine, theirs = _run_id(identity.get("run_id")), _run_id(who.get("run_id"))
    if mine is None or theirs is None or theirs >= mine:
        errors.append("not an earlier run")
    return errors


def alert(tonight: dict, identity: dict, previous: dict | None) -> dict:
    """Whether the job is red. ``identity`` is tonight's (``event``,
    ``run_id``, ``date``, ``calibration``: the calibration file's hash);
    ``previous`` is the last scheduled night's whole result (its ``night``
    and ``identity``).

    An invalid night is red at once, whatever the event. The debounce (red
    after two failing nights) applies only between two scheduled runs: this
    one and the immediately preceding one (an earlier run id, within
    ``DEBOUNCE_WINDOW_HOURS``), on the same calibration. A manual run
    reports its result but never turns red through the debounce."""
    out: dict = {"tonight_failed": bool(tonight.get("failed", True))}
    if tonight.get("errors") or not isinstance(tonight.get("failed"), bool):
        out["reason"] = "invalid night: red at once"
        out["red"] = True
        return out
    if identity.get("event") != "schedule":
        out["previous"] = "not used: this is not a scheduled run"
        out["previous_failed"] = False
        out["red"] = False
        return out
    errors = _previous_errors(identity, previous)
    if errors:
        out["previous"] = "missing" if errors == ["missing"] else "unusable: " + "; ".join(errors)
        out["previous_failed"] = False
    else:
        out["previous"] = previous["identity"]["run_id"]
        out["previous_failed"] = previous["night"]["failed"]
    out["red"] = out["tonight_failed"] and out["previous_failed"]
    return out


PROVENANCE = ("git", "git_dirty", "os", "python", "arch", "cpu", "image", "tessera_rl", "gymnasium", "torch", "numpy")
SETUP_TYPES = {"env": str, "total_steps": int, "eval_episodes": int, "fingerprint": str}


def _schema_errors(i: int, shard: dict) -> list[str]:
    """The required fields of a score file, present and non-empty."""
    errors = []
    env = shard.get("environment")
    setup = shard.get("setup")
    if not isinstance(env, dict):
        return [f"shard {i}: no environment"]
    if not isinstance(setup, dict):
        return [f"shard {i}: no setup"]
    for key in PROVENANCE:
        value = env.get(key)
        if key == "git_dirty":
            if value is not False:
                errors.append(f"shard {i}: not from a clean tree (git_dirty={value!r})")
        elif not isinstance(value, str) or not value or value == "unknown":
            errors.append(f"shard {i}: environment {key} missing or unknown")
    for key, kind in SETUP_TYPES.items():
        value = setup.get(key)
        if not isinstance(value, kind) or isinstance(value, bool) or (kind is str and not value):
            errors.append(f"shard {i}: setup {key} missing or invalid")
    errors += [f"shard {i}: {e}" for e in check_budget(setup)]
    return errors


def check_shards(shards: list[dict]) -> list[str]:
    """Errors of a set of calibration score files: each one complete (every
    provenance and setup field, a clean committed tree), and one provenance
    and setup for all."""
    if not shards:
        return ["no score files"]
    errors = [e for i, shard in enumerate(shards) for e in _schema_errors(i, shard)]
    if errors:
        return errors
    first = shards[0]
    for i, shard in enumerate(shards):
        for key in PROVENANCE:
            if shard["environment"][key] != first["environment"][key]:
                errors.append(f"shard {i}: {key} differs from shard 0")
        if shard["setup"] != first["setup"]:
            errors.append(f"shard {i}: setup differs from shard 0")
    return errors
