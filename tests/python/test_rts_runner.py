# SPDX-License-Identifier: MIT OR Apache-2.0
"""The RTS learning gate's runner (tools/rts_runner.py). Nothing real runs:
train and evaluate are replaced by tripwires or recorders in every test."""

import dataclasses
import json
import pathlib
import sys

import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

import rts_evaluation as ev  # noqa: E402
import rts_runner as rr  # noqa: E402
import rts_training  # noqa: E402
from rts_policy import Agent  # noqa: E402
from rts_training import Hyper  # noqa: E402


class Reached(Exception):
    pass


def tripwire(*_a, **_k):
    raise Reached


@pytest.fixture
def tripped(monkeypatch):
    monkeypatch.setattr(rts_training, "train", tripwire)
    monkeypatch.setattr(ev, "evaluate", tripwire)


def protocol_result(wins):
    return {"games": {}, "wins": wins, "n": 400, "lower": ev.wilson_lower(wins, 400), "protocol": True,
            "seeds": {"first": 2_000_000, "last": 2_000_399, "count": 400}, "passes": ev.seed_passes(wins, 400)}  # fmt: skip


@pytest.fixture
def recorded(monkeypatch):
    """train returns an untrained agent marked with B decisions; evaluate
    returns a protocol result with 300 wins. Records the calls."""
    calls = []

    def fake_train(seed, h, a, f):
        calls.append(("train", seed, h == Hyper()))
        torch.manual_seed(seed)
        agent = Agent(11, h.size, 2, h.max_units, 6, 81, 4)
        agent.decisions_trained = h.total_steps
        return agent

    def fake_evaluate(agent, h, a, f, seeds=ev.EVAL_SEEDS):
        calls.append(("evaluate", type(agent).__name__))
        return protocol_result(300)

    monkeypatch.setattr(rts_training, "train", fake_train)
    monkeypatch.setattr(ev, "evaluate", fake_evaluate)
    return calls


@pytest.mark.parametrize("authorization,frozen", [("", ""), ("x", "y"), ("x", ""), ("", "y")])
def test_rts_runner_refuses_without_the_frozen_hash_before_any_work(tmp_path, tripped, authorization, frozen):
    with pytest.raises(PermissionError):
        rr.run_seed(0, tmp_path, authorization, frozen)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("seed", [-1, 5, 7])
def test_rts_runner_refuses_a_seed_outside_the_five(tmp_path, tripped, seed):
    with pytest.raises(rr.Refused):
        rr.run_seed(seed, tmp_path, "h", "h")


@pytest.mark.parametrize("existing", ["result", "policy"])
def test_rts_runner_refuses_a_second_run_of_a_seed_before_any_work(tmp_path, tripped, existing):
    path = rr.result_path(tmp_path, 2) if existing == "result" else rr.policy_path(tmp_path, 2)
    path.write_text("{}")
    with pytest.raises(rr.Refused, match="used once"):
        rr.run_seed(2, tmp_path, "h", "h")


def test_rts_runner_trains_saves_checks_evaluates_and_writes_once(tmp_path, recorded):
    r = rr.run_seed(3, tmp_path, "h", "h")
    assert recorded == [("train", 3, True), ("evaluate", "Agent")]
    saved = json.loads(rr.result_path(tmp_path, 3).read_text())
    assert saved["wins"] == 300 and saved["protocol"] is True and r["wins"] == 300
    rec = saved["record"]
    assert rec["protocol_sha256"] == "h" and rec["training_seed"] == 3
    assert rec["decisions_trained"] == Hyper().total_steps and rec["hyper"] == dataclasses.asdict(Hyper())
    assert rr.policy_path(tmp_path, 3).exists()
    with pytest.raises(rr.Refused, match="used once"):
        rr.run_seed(3, tmp_path, "h", "h")
    assert len(recorded) == 2


def _rec(**over):
    rec = {"protocol_sha256": "h", "hyper": dataclasses.asdict(Hyper()), "decisions_trained": Hyper().total_steps}
    rec.update(over)
    return rec


def test_rts_runner_record_must_be_the_protocols_run():
    rr.check_record(_rec(), "h")
    for bad in (_rec(protocol_sha256="other"), _rec(hyper=dataclasses.asdict(Hyper(total_steps=2048))),
                _rec(decisions_trained=Hyper().total_steps - 2048), _rec(decisions_trained=None)):
        with pytest.raises(rr.Refused):
            rr.check_record(bad, "h")


def test_rts_runner_refuses_a_policy_trained_short(tmp_path, recorded, monkeypatch):
    def short_train(seed, h, a, f):
        agent = Agent(11, h.size, 2, h.max_units, 6, 81, 4)
        agent.decisions_trained = h.total_steps - 2048
        return agent

    monkeypatch.setattr(rts_training, "train", short_train)
    with pytest.raises(rr.Refused, match="decisions"):
        rr.run_seed(1, tmp_path, "h", "h")
    assert ("evaluate", "Agent") not in recorded and not rr.result_path(tmp_path, 1).exists()


def test_rts_runner_refuses_a_non_protocol_evaluation(tmp_path, recorded, monkeypatch):
    monkeypatch.setattr(ev, "evaluate", lambda *a, **k: {**protocol_result(300), "protocol": False})
    with pytest.raises(rr.Refused, match="protocol"):
        rr.run_seed(0, tmp_path, "h", "h")
    assert not rr.result_path(tmp_path, 0).exists()


def test_rts_runner_verdict_over_the_five_result_files(tmp_path):
    with pytest.raises(rr.Refused, match="no result"):
        rr.verdict(tmp_path)
    for s in range(5):
        rr.result_path(tmp_path, s).write_text(json.dumps(protocol_result(260)))
    assert rr.verdict(tmp_path)["passes"]
    rr.result_path(tmp_path, 4).write_text(json.dumps(protocol_result(259)))
    assert not rr.verdict(tmp_path)["passes"]
    rr.result_path(tmp_path, 4).write_text(json.dumps({**protocol_result(400), "protocol": False}))
    with pytest.raises(rr.Refused, match="not over the protocol"):
        rr.verdict(tmp_path)
