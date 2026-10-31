# SPDX-License-Identifier: MIT OR Apache-2.0
"""The SPEC 18 learning-test logic fails closed (tools/learning_gate.py)."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

from learning_gate import (  # noqa: E402
    CALIBRATION_SEEDS,
    NIGHT_SEEDS,
    alert,
    calibrate,
    check_calibration,
    check_shards,
    night,
)

SETUP = {"env": "tessera/CartPole-v1", "total_steps": 150000, "eval_episodes": 20, "fingerprint": "ab" * 32}
ROOT = pathlib.Path(__file__).resolve().parents[2]


def records(seeds, score=lambda s: 450.0 + (s % 50)):
    return [{"seed": s, "score": score(s)} for s in seeds]


def calibration():
    c = calibrate(records(CALIBRATION_SEEDS), CALIBRATION_SEEDS)
    c["records"] = records(CALIBRATION_SEEDS)
    c["setup"] = dict(SETUP)
    return c


def good_night(score=lambda s: 500.0):
    return night(records(NIGHT_SEEDS, score=score), NIGHT_SEEDS, calibration(), SETUP)


IDENTITY = {"event": "schedule", "calibration": "c1"}


def previous(failed, event="schedule", cal="c1", errors=None):
    n = {"failed": failed}
    if errors:
        n["errors"] = errors
    return {"night": n, "identity": {"event": event, "calibration": cal, "run_id": "41"}}


# -- calibration --------------------------------------------------------------


def test_learning_calibrate_threshold_is_the_lower_fifth_percentile():
    c = calibrate(records(CALIBRATION_SEEDS, score=float), CALIBRATION_SEEDS)
    # 40 scores 101..140: the 5th percentile, method "lower", is the 2nd.
    assert c["threshold"] == 102.0 and "errors" not in c


def test_learning_calibrate_one_seed_repeated_40_times_fails():
    assert calibrate(records([101] * 40), [101] * 40)["errors"]


def test_learning_calibrate_other_seeds_fail():
    seeds = list(range(201, 241))
    assert calibrate(records(seeds), seeds)["errors"]
    assert calibrate(records(CALIBRATION_SEEDS[:39]), CALIBRATION_SEEDS[:39])["errors"]


def test_learning_calibrate_missing_duplicate_or_nan_scores_fail():
    assert calibrate(records(CALIBRATION_SEEDS[1:]), CALIBRATION_SEEDS)["errors"]
    assert calibrate(records(CALIBRATION_SEEDS) + records([101]), CALIBRATION_SEEDS)["errors"]
    bad = records(CALIBRATION_SEEDS)
    bad[3]["score"] = float("nan")
    assert calibrate(bad, CALIBRATION_SEEDS)["errors"]


def test_learning_check_calibration_tampered_threshold_fails():
    c = calibration()
    c["threshold"] = c["threshold"] - 50
    assert check_calibration(c)


def test_learning_check_calibration_tampered_records_fail():
    c = calibration()
    c["records"][0]["score"] = 0.0
    c["records"][1]["score"] = 0.0
    assert check_calibration(c)


def test_learning_check_shards_one_provenance_passes():
    env = {"git": "a" * 40, "git_dirty": False, "torch": "2.15.0+cpu", "arch": "x86_64"}
    assert check_shards([{"environment": env, "setup": SETUP}] * 4) == []


def test_learning_check_shards_dirty_unknown_or_mixed_fail():
    env = {"git": "a" * 40, "git_dirty": False, "torch": "2.15.0+cpu"}
    assert check_shards([{"environment": dict(env, git_dirty=True), "setup": SETUP}])
    assert check_shards([{"environment": dict(env, git="unknown"), "setup": SETUP}])
    assert check_shards([{"environment": env, "setup": SETUP}, {"environment": dict(env, git="b" * 40), "setup": SETUP}])
    assert check_shards([{"environment": env, "setup": SETUP}, {"environment": dict(env, torch="2.14.0"), "setup": SETUP}])
    assert check_shards([{"environment": env, "setup": SETUP}, {"environment": env, "setup": dict(SETUP, fingerprint="cd" * 32)}])
    assert check_shards([])


# -- night --------------------------------------------------------------------


def test_learning_night_median_above_threshold_passes():
    assert good_night()["failed"] is False


def test_learning_night_median_below_threshold_fails():
    scores = {1: 500.0, 2: 500.0, 3: 100.0, 4: 100.0, 5: 100.0}
    n = good_night(score=scores.get)
    assert n["failed"] is True and "errors" not in n


def test_learning_night_two_low_seeds_still_pass():
    scores = {1: 500.0, 2: 500.0, 3: 500.0, 4: 100.0, 5: 100.0}
    assert good_night(score=scores.get)["failed"] is False


def test_learning_night_single_or_other_seeds_fail():
    assert night(records([1], score=lambda s: 500.0), [1], calibration(), SETUP)["errors"]
    assert night(records([6, 7, 8, 9, 10], score=lambda s: 500.0), [6, 7, 8, 9, 10], calibration(), SETUP)["errors"]


def test_learning_night_nan_score_is_an_error():
    scores = {1: 500.0, 2: float("nan"), 3: 500.0, 4: 500.0, 5: 500.0}
    assert good_night(score=scores.get)["errors"]


def test_learning_night_other_setup_or_fingerprint_fails():
    for other in (dict(SETUP, total_steps=500000), dict(SETUP, fingerprint="cd" * 32)):
        n = night(records(NIGHT_SEEDS, score=lambda s: 500.0), NIGHT_SEEDS, calibration(), other)
        assert n["failed"] is True and n["errors"]


def test_learning_night_calibration_without_fingerprint_fails():
    c = calibration()
    del c["setup"]["fingerprint"]
    assert night(records(NIGHT_SEEDS, score=lambda s: 500.0), NIGHT_SEEDS, c, SETUP)["errors"]


def test_learning_night_tampered_calibration_fails():
    c = calibration()
    c["threshold"] = 0.0
    assert night(records(NIGHT_SEEDS, score=lambda s: 500.0), NIGHT_SEEDS, c, SETUP)["errors"]


# -- alert --------------------------------------------------------------------


def test_learning_alert_invalid_night_is_red_at_once():
    a = alert({"failed": True, "errors": ["seed 2: score is not a finite number"]}, IDENTITY, None)
    assert a["red"] is True


def test_learning_alert_one_failing_night_is_not_red():
    assert alert({"failed": True}, IDENTITY, previous(False))["red"] is False


def test_learning_alert_two_consecutive_failing_scheduled_nights_are_red():
    assert alert({"failed": True}, IDENTITY, previous(True))["red"] is True


def test_learning_alert_missing_previous_counts_as_not_failing_and_is_reported():
    a = alert({"failed": True}, IDENTITY, None)
    assert a["red"] is False and a["previous"] == "missing"


def test_learning_alert_previous_dispatch_run_does_not_count():
    a = alert({"failed": True}, IDENTITY, previous(True, event="workflow_dispatch"))
    assert a["red"] is False and a["previous"].startswith("unusable")


def test_learning_alert_previous_on_another_calibration_does_not_count():
    assert alert({"failed": True}, IDENTITY, previous(True, cal="c0"))["red"] is False


def test_learning_alert_previous_invalid_night_does_not_count():
    assert alert({"failed": True}, IDENTITY, previous(True, errors=["bad"]))["red"] is False
    assert alert({"failed": True}, IDENTITY, {"night": {"failed": "yes"}})["red"] is False


def test_learning_alert_passing_night_after_a_failure_is_not_red():
    assert alert({"failed": False}, IDENTITY, previous(True))["red"] is False


# -- the committed calibration -----------------------------------------------


def test_learning_committed_calibration_is_valid():
    c = json.loads((ROOT / "compat" / "learning" / "cartpole.json").read_text())
    assert check_calibration(c) == []
    assert {k: c["setup"][k] for k in ("env", "total_steps", "eval_episodes")} == {
        k: SETUP[k] for k in ("env", "total_steps", "eval_episodes")
    }
    assert isinstance(c["setup"]["fingerprint"], str) and len(c["setup"]["fingerprint"]) == 64
    env = c["environment"]
    assert env["git_dirty"] is False and len(env["git"]) == 40
    assert env["os"].startswith("Linux") and env["arch"] == "x86_64"
