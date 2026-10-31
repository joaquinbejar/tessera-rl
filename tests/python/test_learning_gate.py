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
    check_budget,
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


IDENTITY = {"event": "schedule", "calibration": "c1", "run_id": "42", "date": "2026-11-02T06:10:00+00:00"}


def previous(failed, event="schedule", cal="c1", errors=None, run_id="41", date="2026-11-01T06:10:00+00:00"):
    n = {"failed": failed}
    if errors:
        n["errors"] = errors
    return {"night": n, "identity": {"event": event, "calibration": cal, "run_id": run_id, "date": date}}


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


ENV = {
    "git": "a" * 40,
    "git_dirty": False,
    "os": "Linux-7.0.0-x86_64-with-glibc2.41",
    "python": "3.11.18",
    "arch": "x86_64",
    "cpu": "12th Gen Intel(R) Core(TM) i7-12700H",
    "image": "rust:1.99.1@sha256:09027f87",
    "tessera_rl": "0.1.0",
    "gymnasium": "1.4.0",
    "torch": "2.15.0+cpu",
    "numpy": "2.4.6",
}


def test_learning_check_shards_one_provenance_passes():
    assert check_shards([{"environment": ENV, "setup": SETUP}] * 4) == []


def test_learning_check_shards_missing_fields_fail():
    # The schema is checked before the coherence: identical incomplete
    # shards must not pass.
    minimal = {"environment": {"git": "a" * 40, "git_dirty": False}, "setup": {"env": "tessera/CartPole-v1"}}
    assert check_shards([minimal] * 4)
    for key in ENV:
        if key != "git_dirty":
            env = {k: v for k, v in ENV.items() if k != key}
            assert check_shards([{"environment": env, "setup": SETUP}] * 4), key
            assert check_shards([{"environment": dict(ENV, **{key: ""}), "setup": SETUP}] * 4), key
    for key in SETUP:
        setup = {k: v for k, v in SETUP.items() if k != key}
        assert check_shards([{"environment": ENV, "setup": setup}] * 4), key
    assert check_shards([{"environment": ENV, "setup": dict(SETUP, total_steps="150000")}] * 4)
    assert check_shards([{"setup": SETUP}] * 4)


def test_learning_check_shards_dirty_unknown_or_mixed_fail():
    env = ENV
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


def test_learning_alert_manual_run_never_turns_red_through_the_debounce():
    manual = dict(IDENTITY, event="workflow_dispatch")
    a = alert({"failed": True}, manual, previous(True))
    assert a["red"] is False and a["previous"].startswith("not used")


def test_learning_alert_manual_invalid_night_is_still_red():
    assert alert({"failed": True, "errors": ["bad"]}, dict(IDENTITY, event="workflow_dispatch"), None)["red"] is True


def test_learning_alert_stale_previous_does_not_count():
    a = alert({"failed": True}, IDENTITY, previous(True, date="2020-01-01T06:10:00+00:00"))
    assert a["red"] is False and "within" in a["previous"]


def test_learning_alert_previous_without_date_or_later_run_does_not_count():
    assert alert({"failed": True}, IDENTITY, previous(True, date="yesterday"))["red"] is False
    assert alert({"failed": True}, IDENTITY, previous(True, date="2026-11-01T06:10:00"))["red"] is False
    assert alert({"failed": True}, IDENTITY, previous(True, run_id="43"))["red"] is False
    assert alert({"failed": True}, IDENTITY, previous(True, run_id="42"))["red"] is False
    assert alert({"failed": True}, IDENTITY, previous(True, date="2026-11-02T07:00:00+00:00"))["red"] is False


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


# -- budget -------------------------------------------------------------------

BAD_BUDGETS = [
    {"total_steps": -1},
    {"total_steps": 0},
    {"total_steps": 511},
    {"eval_episodes": -1},
    {"eval_episodes": 0},
    {"total_steps": True},
]


def test_learning_budget_negative_zero_or_sub_rollout_fails():
    assert check_budget(SETUP) == []
    assert check_budget(dict(SETUP, total_steps=512)) == []
    for bad in BAD_BUDGETS:
        assert check_budget(dict(SETUP, **bad)), bad


def test_learning_budget_is_checked_in_shards_night_and_calibration():
    for bad in BAD_BUDGETS:
        setup = dict(SETUP, **bad)
        assert check_shards([{"environment": ENV, "setup": setup}] * 4), bad
        assert night(records(NIGHT_SEEDS, score=lambda s: 500.0), NIGHT_SEEDS, dict(calibration(), setup=setup), setup)["errors"], bad
        assert check_calibration(dict(calibration(), setup=setup)), bad


def test_learning_trainer_rejects_a_sub_rollout_budget():
    pytest = __import__("pytest")
    pytest.importorskip("torch")
    import reference_training as rt

    for steps, episodes in ((-1, 20), (0, 20), (511, 20), (512, 0), (512, -1)):
        with pytest.raises(ValueError):
            rt.train("tessera", 1, rt.Hyper(total_steps=steps), episodes)
