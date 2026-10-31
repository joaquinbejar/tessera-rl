# SPDX-License-Identifier: MIT OR Apache-2.0
"""The SPEC 18 learning-test logic fails closed (tools/learning_gate.py)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

from learning_gate import alert, calibrate, night  # noqa: E402

CAL_SEEDS = list(range(101, 141))
NIGHT_SEEDS = [1, 2, 3, 4, 5]
SETUP = {"env": "tessera/CartPole-v1", "total_steps": 150000, "eval_episodes": 20, "fingerprint": "ab" * 32}


def records(seeds, score=lambda s: 450.0 + (s % 50)):
    return [{"seed": s, "score": score(s)} for s in seeds]


def calibration():
    c = calibrate(records(CAL_SEEDS), CAL_SEEDS)
    c["setup"] = dict(SETUP)
    return c


def test_learning_calibrate_threshold_is_the_lower_fifth_percentile():
    c = calibrate(records(CAL_SEEDS, score=float), CAL_SEEDS)
    # 40 scores 101..140: the 5th percentile, method "lower", is the 2nd.
    assert c["threshold"] == 102.0 and "errors" not in c


def test_learning_calibrate_fewer_than_40_seeds_fails():
    assert calibrate(records(CAL_SEEDS[:39]), CAL_SEEDS[:39])["errors"]


def test_learning_calibrate_missing_duplicate_or_nan_scores_fail():
    assert calibrate(records(CAL_SEEDS[1:]), CAL_SEEDS)["errors"]
    assert calibrate(records(CAL_SEEDS) + records([101]), CAL_SEEDS)["errors"]
    bad = records(CAL_SEEDS)
    bad[3]["score"] = float("nan")
    assert calibrate(bad, CAL_SEEDS)["errors"]


def test_learning_night_median_above_threshold_passes():
    n = night(records(NIGHT_SEEDS, score=lambda s: 500.0), NIGHT_SEEDS, calibration(), SETUP)
    assert n["failed"] is False


def test_learning_night_median_below_threshold_fails():
    scores = {1: 500.0, 2: 500.0, 3: 100.0, 4: 100.0, 5: 100.0}
    n = night(records(NIGHT_SEEDS, score=scores.get), NIGHT_SEEDS, calibration(), SETUP)
    assert n["failed"] is True


def test_learning_night_two_low_seeds_still_pass():
    scores = {1: 500.0, 2: 500.0, 3: 500.0, 4: 100.0, 5: 100.0}
    assert night(records(NIGHT_SEEDS, score=scores.get), NIGHT_SEEDS, calibration(), SETUP)["failed"] is False


def test_learning_night_other_setup_fails():
    n = night(records(NIGHT_SEEDS, score=lambda s: 500.0), NIGHT_SEEDS, calibration(), dict(SETUP, total_steps=500000))
    assert n["failed"] is True and n["errors"]


def test_learning_night_other_fingerprint_fails():
    n = night(records(NIGHT_SEEDS, score=lambda s: 500.0), NIGHT_SEEDS, calibration(), dict(SETUP, fingerprint="cd" * 32))
    assert n["failed"] is True and any("fingerprint" in e for e in n["errors"])


def test_learning_night_calibration_without_fingerprint_fails():
    c = calibration()
    del c["setup"]["fingerprint"]
    assert night(records(NIGHT_SEEDS, score=lambda s: 500.0), NIGHT_SEEDS, c, SETUP)["failed"] is True


def test_learning_night_invalid_calibration_fails():
    c = calibration()
    del c["threshold"]
    assert night(records(NIGHT_SEEDS, score=lambda s: 500.0), NIGHT_SEEDS, c, SETUP)["failed"] is True


def test_learning_night_missing_seed_fails():
    assert night(records(NIGHT_SEEDS[:4], score=lambda s: 500.0), NIGHT_SEEDS, calibration(), SETUP)["failed"] is True


def test_learning_alert_one_failing_night_is_not_red():
    assert alert({"failed": True}, {"failed": False})["red"] is False


def test_learning_alert_two_consecutive_failing_nights_are_red():
    assert alert({"failed": True}, {"failed": True})["red"] is True


def test_learning_alert_missing_previous_counts_as_not_failing_and_is_reported():
    a = alert({"failed": True}, None)
    assert a["red"] is False and a["previous"] == "missing"
    assert alert({"failed": True}, {"failed": "yes"})["previous"] == "missing"


def test_learning_alert_passing_night_after_a_failure_is_not_red():
    assert alert({"failed": False}, {"failed": True})["red"] is False


def test_learning_committed_calibration_is_valid():
    import json

    root = pathlib.Path(__file__).resolve().parents[2]
    c = json.loads((root / "compat" / "learning" / "cartpole.json").read_text())
    again = calibrate(c["records"], CAL_SEEDS)
    assert "errors" not in c and "errors" not in again
    assert again["threshold"] == c["threshold"]
    assert {k: c["setup"][k] for k in ("env", "total_steps", "eval_episodes")} == {k: SETUP[k] for k in ("env", "total_steps", "eval_episodes")}
    fp = c["setup"]["fingerprint"]
    assert isinstance(fp, str) and len(fp) == 64
