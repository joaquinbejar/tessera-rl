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


PROTOCOL_TEXT = b"the frozen protocol, as the owner authorized it\n"
H = __import__("hashlib").sha256(PROTOCOL_TEXT).hexdigest()


@pytest.fixture(autouse=True)
def _frozen(monkeypatch, tmp_path):
    """Tests stand in a protocol file and set the repository's constant to
    its hash; the real constant stays None until the freeze PR."""
    monkeypatch.setattr(rts_training, "FROZEN_PROTOCOL_SHA256", H)


@pytest.fixture
def proto(tmp_path):
    path = tmp_path / "rts.md"
    path.write_bytes(PROTOCOL_TEXT)
    return path


def tripwire(*_a, **_k):
    raise Reached


@pytest.fixture
def tripped(monkeypatch):
    monkeypatch.setattr(rts_training, "train", tripwire)
    monkeypatch.setattr(ev, "evaluate", tripwire)


def protocol_result(wins, seed=0):
    games = {str(s): ("win" if i < wins else "loss") for i, s in enumerate(ev.EVAL_SEEDS)}
    rec = {"protocol_sha256": H, "hyper": dataclasses.asdict(Hyper()), "decisions_trained": Hyper().total_steps,
           "training_seed": seed}  # fmt: skip
    return {"games": games, "wins": wins, "n": 400, "lower": ev.wilson_lower(wins, 400), "protocol": True,
            "seeds": {"first": 2_000_000, "last": 2_000_399, "count": 400}, "passes": ev.seed_passes(wins, 400),
            "record": rec}  # fmt: skip


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
        # S2: the evaluated policy is the one load returned, not the trained object.
        calls.append(("evaluate", getattr(agent, "loaded_from_disk", False)))
        r = protocol_result(300)
        del r["record"]
        return r

    real_load = ev.load

    def marking_load(path, h):
        agent, rec = real_load(path, h)
        agent.loaded_from_disk = True
        return agent, rec

    monkeypatch.setattr(rts_training, "train", fake_train)
    monkeypatch.setattr(ev, "evaluate", fake_evaluate)
    monkeypatch.setattr(ev, "load", marking_load)
    return calls


@pytest.mark.parametrize("authorization,frozen", [("", ""), ("x", "y"), ("x", ""), ("", "y")])
def test_rts_runner_refuses_without_the_frozen_hash_before_any_work(tmp_path, tripped, proto, authorization, frozen):
    with pytest.raises(PermissionError):
        rr.run_seed(0, tmp_path / "out", proto, authorization, frozen)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("seed", [-1, 5, 7])
def test_rts_runner_refuses_a_seed_outside_the_five(tmp_path, tripped, proto, seed):
    with pytest.raises(rr.Refused):
        rr.run_seed(seed, tmp_path, proto, H, H)


@pytest.mark.parametrize("existing", ["result", "policy", "lock"])
def test_rts_runner_refuses_a_second_run_of_a_seed_before_any_work(tmp_path, tripped, proto, existing):
    path = {"result": rr.result_path, "policy": rr.policy_path, "lock": rr.lock_path}[existing](tmp_path, 2)
    path.write_text("{}")
    with pytest.raises(rr.Refused, match="used once"):
        rr.run_seed(2, tmp_path, proto, H, H)


def test_rts_runner_trains_saves_checks_evaluates_and_writes_once(tmp_path, recorded, proto):
    r = rr.run_seed(3, tmp_path, proto, H, H)
    assert recorded == [("train", 3, True), ("evaluate", True)]
    assert rr.lock_path(tmp_path, 3).exists()
    saved = json.loads(rr.result_path(tmp_path, 3).read_text())
    assert saved["wins"] == 300 and saved["protocol"] is True and r["wins"] == 300
    rec = saved["record"]
    assert rec["protocol_sha256"] == H and rec["training_seed"] == 3
    assert rec["decisions_trained"] == Hyper().total_steps and rec["hyper"] == dataclasses.asdict(Hyper())
    assert rr.policy_path(tmp_path, 3).exists()
    with pytest.raises(rr.Refused, match="used once"):
        rr.run_seed(3, tmp_path, proto, H, H)
    assert len(recorded) == 2


def _rec(**over):
    rec = {"protocol_sha256": H, "hyper": dataclasses.asdict(Hyper()), "decisions_trained": Hyper().total_steps}
    rec.update(over)
    return rec


def test_rts_runner_record_must_be_the_protocols_run():
    rr.check_record(_rec(), H)
    for bad in (_rec(protocol_sha256="other"), _rec(hyper=dataclasses.asdict(Hyper(total_steps=2048))),
                _rec(decisions_trained=Hyper().total_steps - 2048), _rec(decisions_trained=None)):
        with pytest.raises(rr.Refused):
            rr.check_record(bad, H)


def test_rts_runner_refuses_a_policy_trained_short(tmp_path, recorded, proto, monkeypatch):
    def short_train(seed, h, a, f):
        agent = Agent(11, h.size, 2, h.max_units, 6, 81, 4)
        agent.decisions_trained = h.total_steps - 2048
        return agent

    monkeypatch.setattr(rts_training, "train", short_train)
    with pytest.raises(rr.Refused, match="decisions"):
        rr.run_seed(1, tmp_path, proto, H, H)
    assert not any(c[0] == "evaluate" for c in recorded) and not rr.result_path(tmp_path, 1).exists()


def test_rts_runner_refuses_a_non_protocol_evaluation(tmp_path, recorded, proto, monkeypatch):
    monkeypatch.setattr(ev, "evaluate", lambda *a, **k: {**protocol_result(300), "protocol": False})
    with pytest.raises(rr.Refused, match="protocol"):
        rr.run_seed(0, tmp_path, proto, H, H)
    assert not rr.result_path(tmp_path, 0).exists()


def test_rts_runner_refuses_a_protocol_file_of_other_bytes(tmp_path, tripped):
    """R1 (a): the run is tied to the protocol text: another file's bytes are
    refused before any work, even with the constant and both arguments
    equal; the right file passes the guard (and meets the tripwire)."""
    other = tmp_path / "other.md"
    other.write_bytes(PROTOCOL_TEXT + b" ")
    with pytest.raises(rr.Refused, match="hashes to"):
        rr.run_seed(0, tmp_path / "out", other, H, H)
    good = tmp_path / "rts.md"
    good.write_bytes(PROTOCOL_TEXT)
    with pytest.raises(Reached):
        rr.run_seed(0, tmp_path / "out", good, H, H)


def test_rts_runner_refuses_while_the_repository_has_no_frozen_hash(tmp_path, tripped, proto, monkeypatch):
    monkeypatch.setattr(rts_training, "FROZEN_PROTOCOL_SHA256", None)
    with pytest.raises(PermissionError, match="not frozen"):
        rr.run_seed(0, tmp_path / "out", proto, H, H)


def _five(tmp_path, wins=260):
    for s in range(5):
        rr.result_path(tmp_path, s).write_text(json.dumps(protocol_result(wins, seed=s)))


def test_rts_runner_verdict_over_the_five_result_files(tmp_path):
    with pytest.raises(rr.Refused, match="no result"):
        rr.verdict(tmp_path, H)
    _five(tmp_path)
    assert rr.verdict(tmp_path, H)["passes"]
    rr.result_path(tmp_path, 4).write_text(json.dumps(protocol_result(259, seed=4)))
    assert not rr.verdict(tmp_path, H)["passes"]


@pytest.mark.parametrize("case", ["not_protocol", "n_50", "other_sha", "other_seed", "games_short",
                                  "wins_edited", "passes_edited", "no_record"])  # fmt: skip
def test_rts_runner_verdict_refuses_a_file_that_is_not_its_seeds_protocol_result(tmp_path, case):
    """R2 and S1: each file is checked against the frozen protocol."""
    _five(tmp_path)
    r = protocol_result(260, seed=2)
    if case == "not_protocol":
        r["protocol"] = False
    elif case == "n_50":
        r["n"] = 50
    elif case == "other_sha":
        r["record"]["protocol_sha256"] = "0" * 64
    elif case == "other_seed":
        r["record"]["training_seed"] = 3
    elif case == "games_short":
        r["games"].pop("2000399")
    elif case == "wins_edited":
        r["wins"] = 300
    elif case == "passes_edited":
        r = protocol_result(259, seed=2)
        r["passes"] = True
    elif case == "no_record":
        del r["record"]
    rr.result_path(tmp_path, 2).write_text(json.dumps(r))
    with pytest.raises(rr.Refused):
        rr.verdict(tmp_path, H)


def test_rts_runner_cli_exits_and_reports(tmp_path, tripped, proto, capsys):
    """S3: main() refuses with exit 2 and "refused:", prints the verdict and
    exits 0, and a missing argument exits non-zero (argparse)."""
    assert rr.main(["seed", "0", "--out", str(tmp_path / "o"), "--protocol", str(proto),
                    "--authorization", "x", "--frozen", "y"]) == 2  # fmt: skip
    assert "refused:" in capsys.readouterr().err
    assert rr.main(["verdict", "--out", str(tmp_path / "none"), "--frozen", H]) == 2
    _five(tmp_path)
    assert rr.main(["verdict", "--out", str(tmp_path), "--frozen", H]) == 0
    assert json.loads(capsys.readouterr().out)["passes"] is True
    with pytest.raises(SystemExit) as e:
        rr.main(["seed", "0", "--out", str(tmp_path)])
    assert e.value.code != 0
