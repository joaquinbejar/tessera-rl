# SPDX-License-Identifier: MIT OR Apache-2.0
"""The nightly wall-clock gate (SPEC 19.4, #374): the paired change and its
flag, the accepted-base state machine and the qualification rules, driven by
fake runners. No benchmark runs here."""

import importlib.util
import pathlib

import pytest

_TOOL = pathlib.Path(__file__).resolve().parents[2] / "tools" / "bench" / "nightly.py"
_spec = importlib.util.spec_from_file_location("nightly", _TOOL)
nightly = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nightly)

CASE = "rust/cartpole/inline/n1024"
OTHER = "rust/tag/inline/n256"


def fake_run(speed: dict, noise: float = 0.0):
    """A runner whose throughput per side is ``speed[side]``, with a
    deterministic +-noise wobble per call."""
    calls = {"n": 0}

    def run(side, case):
        calls["n"] += 1
        wobble = noise * (1 if calls["n"] % 2 else -1)
        return speed[side] * (1 + wobble)

    return run


def night(head, base, slow_by=0.0, noise=0.002, reasons=(), cases=(CASE,)):
    """One measured night: the head is ``slow_by`` slower than the base."""
    run = fake_run({"base": 1000.0, "head": 1000.0 / (1 + slow_by)}, noise)
    raw = nightly.measure(run, cases, pairs=10)
    changes = {c: nightly.paired_change(r["base"], r["head"], resamples=2000) for c, r in raw.items()}
    return {"head": head, "base": base, "reasons": list(reasons), "cases": changes}


def run_nights(plan, binding=True, acks=None):
    """``plan``: (head, slow_by, reasons) per night; returns the verdicts and
    the final state."""
    state = nightly.initial_state("c0")
    verdicts = []
    for head, slow_by, reasons in plan:
        n = night(head, state["accepted_base"], slow_by, reasons=reasons)
        state, v = nightly.decide(state, n, acks or [], binding)
        verdicts.append(v)
    return verdicts, state


# -- the change ---------------------------------------------------------------------


def test_nightly_paired_change_is_positive_when_the_head_is_slower():
    c = nightly.paired_change([100.0] * 10, [90.0] * 10, resamples=500)
    assert c["change"] == pytest.approx(100 / 90 - 1)
    assert c["ci95"][0] == pytest.approx(c["change"])
    assert nightly.flagged(c)


def test_nightly_flag_needs_the_whole_interval_above_three_percent():
    assert not nightly.flagged({"ci95": [0.029, 0.2]})
    assert not nightly.flagged({"ci95": [0.03, 0.2]})
    assert nightly.flagged({"ci95": [0.0301, 0.2]})


def test_nightly_paired_change_refuses_bad_input():
    with pytest.raises(ValueError):
        nightly.paired_change([1.0], [1.0])
    with pytest.raises(ValueError):
        nightly.paired_change([1.0, 1.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        nightly.paired_change([1.0, 1.0], [1.0, 1.0, 1.0])


def test_nightly_measure_interleaves_base_and_head():
    seen = []
    nightly.measure(lambda side, case: seen.append(side) or 1.0, (CASE,), pairs=4)
    assert seen == ["base", "head", "head", "base", "base", "head", "head", "base"]


# -- the state machine --------------------------------------------------------------


def test_nightly_persistent_regression_goes_red_on_the_second_night_and_holds_the_base():
    v, state = run_nights([("c1", 0.10, ()), ("c2", 0.10, ())])
    assert [x["status"] for x in v] == ["flagged", "flagged"]
    assert [x["red"] for x in v] == [False, True]
    # Both nights compared against the held base c0, not against c1.
    assert [x["base"] for x in v] == ["c0", "c0"]
    assert state["accepted_base"] == "c0"


def test_nightly_regression_fixed_on_the_second_night_clears_and_advances():
    v, state = run_nights([("c1", 0.10, ()), ("c2", 0.0, ())])
    assert [x["status"] for x in v] == ["flagged", "passed"]
    assert not any(x["red"] for x in v)
    assert state["accepted_base"] == "c2"
    assert state["flag_streak"] == {}


def test_nightly_noise_only_never_flags_and_the_base_advances():
    v, state = run_nights([(f"c{i}", 0.005, ()) for i in range(1, 6)])
    assert all(x["status"] == "passed" for x in v)
    assert state["accepted_base"] == "c5"


def test_nightly_acknowledgement_moves_the_base_to_the_acknowledged_head():
    acks = [{"commit": "c2", "cases": [CASE], "reason": "accepted cost of a feature"}]
    v, state = run_nights([("c1", 0.10, ()), ("c2", 0.10, ()), ("c3", 0.0, ())], acks=acks)
    assert [x["status"] for x in v] == ["flagged", "acknowledged", "passed"]
    assert not any(x["red"] for x in v)
    assert v[2]["base"] == "c2"


def test_nightly_non_qualifying_night_between_two_flags_does_not_count():
    plan = [("c1", 0.10, ()), ("c2", 0.10, ("load over 4.0",)), ("c3", 0.10, ())]
    v, state = run_nights(plan)
    assert [x["status"] for x in v] == ["flagged", "non-qualifying", "flagged"]
    # Consecutive means consecutive qualifying nights: red on the third.
    assert [x["red"] for x in v] == [False, False, True]
    assert state["accepted_base"] == "c0"


def test_nightly_non_qualifying_night_never_advances_the_base():
    v, state = run_nights([("c1", 0.0, ("other jobs on the host",))])
    assert v[0]["status"] == "non-qualifying"
    assert state["accepted_base"] == "c0"


def test_nightly_no_new_commit_runs_as_a_and_never_flags_or_moves():
    state = nightly.initial_state("c0")
    n = night("c0", "c0", 0.0)
    state, v = nightly.decide(state, n, [], True)
    assert v["status"] == "a/a" and not v["red"]
    assert state["accepted_base"] == "c0"


def test_nightly_red_is_only_reported_on_a_candidate_machine():
    v, _ = run_nights([("c1", 0.10, ()), ("c2", 0.10, ())], binding=False)
    assert v[1]["red"] is False and v[1]["would_be_red"] is True


def test_nightly_streaks_are_per_case():
    state = nightly.initial_state("c0")
    reds = []
    for head, slow in [
        ("c1", {CASE: 0.10, OTHER: 0.0}),
        ("c2", {CASE: 0.0, OTHER: 0.10}),
        ("c3", {CASE: 0.10, OTHER: 0.0}),
    ]:
        cases = {}
        for case, s in slow.items():
            raw = nightly.measure(fake_run({"base": 1000.0, "head": 1000.0 / (1 + s)}, 0.002), (case,), 10)
            cases[case] = nightly.paired_change(raw[case]["base"], raw[case]["head"], resamples=2000)
        state, v = nightly.decide(state, {"head": head, "base": state["accepted_base"], "reasons": [], "cases": cases}, [], True)
        assert v["status"] == "flagged"
        reds.append(v["red"])
    # A different case each night: no case is flagged on two consecutive
    # nights, so a streak ends on a night its case is not flagged.
    assert reds == [False, False, False]


def test_nightly_weekly_base_moves_every_seven_qualifying_nights():
    v, state = run_nights([(f"c{i}", 0.0, ()) for i in range(1, 8)])
    assert state["qualifying_nights"] == 7
    assert state["weekly_base"] == "c7"
    v, state6 = run_nights([(f"c{i}", 0.0, ()) for i in range(1, 7)])
    assert state6["weekly_base"] == "c0"


# -- qualification ------------------------------------------------------------------


def test_nightly_qualify_reasons():
    assert nightly.qualify([0.5, 0.7], [], None) == []
    assert "load over" in nightly.qualify([0.5, 9.0], [], None)[0]
    assert "no load average" in nightly.qualify([None], [], None)[0]
    assert "other jobs" in nightly.qualify([0.5], ["Runner.Worker 123"], None)[0]
    wide = {CASE: {"half_width": 0.02}}
    assert "A/A half-width" in nightly.qualify([0.5], [], wide)[0]
    assert nightly.qualify([0.5], [], {CASE: {"half_width": 0.01}}) == []


def test_nightly_ack_file_parses_and_matches(tmp_path):
    p = tmp_path / "ack.toml"
    p.write_text('[[ack]]\ncommit = "abc"\ncases = []\nreason = "r"\n')
    acks = nightly.load_acks(p)
    assert nightly.acknowledged(acks, "abc") == ["*"]
    assert nightly.acknowledged(acks, "def") is None
    assert nightly.load_acks(tmp_path / "missing.toml") == []


def test_nightly_repo_ack_file_is_valid():
    acks = nightly.load_acks()
    assert isinstance(acks, list)
    for ack in acks:
        assert set(ack) >= {"commit", "reason"}
