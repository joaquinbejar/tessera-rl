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
#: The largest share of evaluation openings a T3b calibration may skip
#: (REVIEW-T3b-opening-r1); above it, calibrate refuses the game.
MAX_OPENING_SKIP_SHARE = 0.05


def opening_skip_errors(records: list[dict]) -> list[str]:
    """Errors for T3b score records whose skipped openings exceed
    ``MAX_OPENING_SKIP_SHARE`` of the openings drawn (per seed)."""
    errors = []
    for r in records:
        # REVIEW-T3b-skips-r1: with openings on, the counts must be present;
        # a missing count is an error, never a silent pass.
        if r.get("opening_plies", 0) > 0 and ("opening_draws" not in r or "opening_skips" not in r):
            errors.append(f"seed {r.get('seed')}: opening_plies {r['opening_plies']} but no opening_draws or opening_skips")
            continue
        draws, skips = r.get("opening_draws", 0), r.get("opening_skips", 0)
        if draws and skips / draws > MAX_OPENING_SKIP_SHARE:
            errors.append(f"seed {r.get('seed')}: {skips} of {draws} openings skipped, above {MAX_OPENING_SKIP_SHARE:.0%}")
    return errors


#: Tier T3b's calibration seeds (#370, rev's ruling on DESIGN-T3b-r1 section
#: 3): 10 seeds, with the threshold at their minimum (the 5th percentile of
#: 40 needs about 12 h of training at T3b budgets). With exchangeable draws
#: a night (median of the 5 NIGHT_SEEDS) falls below that minimum when the 3
#: lowest of the 15 draws are all night seeds: C(5,3)/C(15,3) = 2.2% per
#: night, the false-red estimate recorded in compat/learning/gates.toml.
T3B_CALIBRATION_SEEDS = list(range(101, 111))


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


def calibrate(records: list[dict], seeds: list[int], expected: list[int] | None = None, threshold: str = "p5") -> dict:
    """The threshold from the calibration scores; ``seeds`` must be exactly
    ``expected`` (``CALIBRATION_SEEDS`` by default). ``threshold`` is
    ``"p5"`` (the 5th percentile, lower method; every tier but T3b) or
    ``"min"`` (T3b: ``T3B_CALIBRATION_SEEDS``, the minimum of 10)."""
    expected = CALIBRATION_SEEDS if expected is None else expected
    out: dict = {"seeds": list(seeds)}
    scores, errors = _scores(records, seeds)
    if list(seeds) != expected:
        errors.append(f"the calibration seeds must be exactly {expected[0]}-{expected[-1]}")
    if threshold not in ("p5", "min"):
        errors.append(f"threshold is p5 or min, not {threshold!r}")
    if errors:
        out["errors"] = errors
        return out
    out["threshold_rule"] = threshold
    out["threshold"] = float(min(scores)) if threshold == "min" else float(np.percentile(scores, 5, method="lower"))
    out["median"] = float(np.median(scores))
    out["min"] = min(scores)
    out["max"] = max(scores)
    return out


#: The default minimum effect: a protocol learns only if the gap over the
#: random policy is at least this fraction of the gap between random and
#: solved (the protocol's ``solved`` score, compat/learning/gates.toml).
MIN_EFFECT = 0.10


def learns(
    scores: list[float],
    random: list[float],
    resamples: int = 10_000,
    seed: int = 0,
    solved: float | None = None,
    min_effect: float = MIN_EFFECT,
) -> dict:
    """The "learns at all" check of a protocol (#370): a seeded bootstrap
    95% interval of median(scores) - mean(random) must lie above 0, and,
    when the protocol names a ``solved`` score, above ``min_effect`` times
    (solved - mean(random)). ``scores`` are the trained seeds' evaluation
    scores, ``random`` the per-episode scores of a uniform random policy on
    the same evaluation seeds.

    Constant scores (zero variance) or a zero-width interval are
    **degenerate**, never "learns": a fixed policy that beats a fixed random
    score by a sliver says nothing about learning (Grid at 150k steps)."""
    s = np.asarray(scores, dtype=np.float64)
    r = np.asarray(random, dtype=np.float64)
    if s.size < 2 or r.size < 2 or not (np.all(np.isfinite(s)) and np.all(np.isfinite(r))):
        return {"learns": False, "error": "learns needs at least two finite scores on each side"}
    rng = np.random.default_rng(seed)
    gap = np.median(rng.choice(s, size=(resamples, s.size)), axis=1) - np.mean(rng.choice(r, size=(resamples, r.size)), axis=1)
    lo, hi = np.percentile(gap, [2.5, 97.5])
    required = 0.0 if solved is None else float(min_effect * (solved - np.mean(r)))
    degenerate = bool(np.ptp(s) == 0 or hi == lo)
    return {
        "learns": bool(lo > 0 and lo > required and not degenerate),
        "degenerate": degenerate,
        "gap": float(np.median(s) - np.mean(r)),
        "ci95": [float(lo), float(hi)],
        "required": required,
        "random_mean": float(np.mean(r)),
    }


def _lower95_mean(values, rng, resamples: int) -> tuple[float, float]:
    """The seeded bootstrap 95% interval of ``mean(values)``."""
    v = np.asarray(values, dtype=np.float64)
    means = np.mean(rng.choice(v, size=(resamples, v.size)), axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def learns_twoplayer(
    vs_heuristic: list[float],
    vs_uniform: list[float],
    uniform: list[float],
    heuristic: list[float],
    resamples: int = 10_000,
    seed: int = 0,
    min_effect: float = MIN_EFFECT,
) -> dict:
    """The tier T3b check (#370, DESIGN-T3b-r1 section 2, REVIEW-T3b-r1).

    ``vs_heuristic`` and ``vs_uniform`` are the trained seeds' scores (one
    mean per seed) against the heuristic and against a uniform-legal
    opponent; ``uniform`` (U) and ``heuristic`` (H) are the per-episode
    scores of a uniform-legal player and of the heuristic itself in the
    learner's seat, against the heuristic, on the same evaluation seeds.

    - **No gate** when the lower bound of the bootstrap 95% interval of
      ``mean(H) - mean(U)`` is at most 0: the heuristic does not beat
      random play, so there is no gap to close.
    - **Threshold 1:** the lower bound of ``mean(vs_heuristic) - mean(U)``
      is at least ``min_effect * (mean(H) - mean(U))``.
    - **Threshold 2:** the lower bound of ``mean(vs_uniform)`` is above 0.
    - Learns only when both hold and the scores are not degenerate
      (constant ``vs_heuristic`` scores, or a zero-width interval)."""
    parts = (vs_heuristic, vs_uniform, uniform, heuristic)
    arrays = [np.asarray(x, dtype=np.float64) for x in parts]
    if any(a.size < 2 or not np.all(np.isfinite(a)) for a in arrays):
        return {"learns": False, "gate": True, "error": "learns_twoplayer needs at least two finite scores in each list"}
    vh, vu, u, hh = arrays
    rng = np.random.default_rng(seed)
    hu = np.mean(rng.choice(hh, size=(resamples, hh.size)), axis=1) - np.mean(rng.choice(u, size=(resamples, u.size)), axis=1)
    hu_lo = float(np.percentile(hu, 2.5))
    if hu_lo <= 0:
        return {"learns": False, "gate": False, "reason": "lower95(H - U) <= 0: no gate", "hu_lower95": hu_lo}
    gap = np.mean(rng.choice(vh, size=(resamples, vh.size)), axis=1) - np.mean(rng.choice(u, size=(resamples, u.size)), axis=1)
    lo, hi = np.percentile(gap, [2.5, 97.5])
    required = float(min_effect * (np.mean(hh) - np.mean(u)))
    u_lo, _ = _lower95_mean(vu, rng, resamples)
    degenerate = bool(np.ptp(vh) == 0 or hi == lo)
    first = bool(lo >= required)
    second = bool(u_lo > 0)
    return {
        "learns": bool(first and second and not degenerate),
        "gate": True,
        "degenerate": degenerate,
        "threshold_1": first,
        "threshold_2": second,
        "ci95_vs_heuristic_minus_u": [float(lo), float(hi)],
        "required": required,
        "lower95_vs_uniform": u_lo,
        "hu_lower95": hu_lo,
    }


def learns_selfplay(
    trained: list[float],
    uniform: list[float],
    heuristic: list[float],
    resamples: int = 10_000,
    seed: int = 0,
    min_effect: float = MIN_EFFECT,
) -> dict:
    """The tier T3c check (#529, DESIGN-r1 section 2, REVIEW-design-r1).

    ``trained`` are the trained seeds' scores (one mean final score per
    seed); ``uniform`` (U) and ``heuristic`` (H) are the per-episode scores
    of uniform-legal play and of heuristic slots on every seat, on the same
    evaluation seeds. One threshold: the game is cooperative, so there is no
    opponent to beat.

    - **No gate** when the lower bound of the bootstrap 95% interval of
      ``mean(H) - mean(U)`` is at most 0.
    - **Threshold:** the lower bound of ``mean(trained) - mean(U)`` is at
      least ``min_effect * (mean(H) - mean(U))``.
    - Learns only when it holds and the scores are not degenerate (constant
      ``trained`` scores, or a zero-width interval)."""
    arrays = [np.asarray(x, dtype=np.float64) for x in (trained, uniform, heuristic)]
    if any(a.size < 2 or not np.all(np.isfinite(a)) for a in arrays):
        return {"learns": False, "gate": True, "error": "learns_selfplay needs at least two finite scores in each list"}
    t, u, hh = arrays
    rng = np.random.default_rng(seed)
    hu = np.mean(rng.choice(hh, size=(resamples, hh.size)), axis=1) - np.mean(rng.choice(u, size=(resamples, u.size)), axis=1)
    hu_lo = float(np.percentile(hu, 2.5))
    if hu_lo <= 0:
        return {"learns": False, "gate": False, "reason": "lower95(H - U) <= 0: no gate", "hu_lower95": hu_lo}
    gap = np.mean(rng.choice(t, size=(resamples, t.size)), axis=1) - np.mean(rng.choice(u, size=(resamples, u.size)), axis=1)
    lo, hi = np.percentile(gap, [2.5, 97.5])
    required = float(min_effect * (np.mean(hh) - np.mean(u)))
    degenerate = bool(np.ptp(t) == 0 or hi == lo)
    threshold = bool(lo >= required)
    return {
        "learns": bool(threshold and not degenerate),
        "gate": True,
        "degenerate": degenerate,
        "threshold": threshold,
        "ci95_trained_minus_u": [float(lo), float(hi)],
        "required": required,
        "hu_lower95": hu_lo,
    }


def no_verdict_streak(previous: dict | None) -> int:
    """Consecutive nights without a verdict, counting tonight, from the
    previous night's result. A night on an uncalibrated host gives no
    verdict; a run of them means the runner changed and the gate is silent,
    so the count is reported and annotated, never ignored."""
    if not isinstance(previous, dict):
        return 1
    night = previous.get("night") or {}
    if not str(night.get("verdict", "")).startswith("uncalibrated host"):
        return 1
    before = night.get("no_verdict_nights")
    return (before if isinstance(before, int) and before >= 1 else 1) + 1


def host_mismatch(calibration: dict, cpu: str) -> str | None:
    """Why tonight gives no verdict on this host, or None. Learning
    thresholds are host-specific (compat/learning/protocols.toml): a night
    on another CPU than the calibration's is reported, never judged."""
    calibrated = (calibration.get("environment") or {}).get("cpu")
    if not calibrated:
        return "uncalibrated host: the calibration records no CPU"
    if calibrated != cpu:
        return f"uncalibrated host: no verdict (calibrated on {calibrated!r}, this is {cpu!r})"
    return None


def check_calibration(calibration: dict) -> list[str]:
    """Errors of a stored calibration: its records must give its threshold."""
    if calibration.get("errors"):
        return ["the calibration recorded errors"]
    budget = check_budget(calibration.get("setup"))
    if budget:
        return [f"calibration: {e}" for e in budget]
    if calibration.get("threshold_rule") == "min":
        # T3b (#370): 10 seeds, the threshold at their minimum; recomputing
        # with T3B_CALIBRATION_SEEDS still enforces the exact seed list.
        again = calibrate(calibration.get("records") or [], calibration.get("seeds") or [], expected=T3B_CALIBRATION_SEEDS, threshold="min")
    else:
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
