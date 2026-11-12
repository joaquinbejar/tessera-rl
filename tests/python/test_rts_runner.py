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
    returns a protocol result with 300 wins. Records the calls; with
    ``calls.out`` set to the run directory it also records that the seed's
    lock exists (holding the hash) when train starts, and the evaluation
    marker when evaluate starts."""

    class Calls(list):
        out = None
        old = None

    calls = Calls()

    def fake_train(seed, h, a, f):
        if calls.out is not None:
            lock = rr.lock_path(calls.out, seed)
            held = json.loads(lock.read_text()) if lock.exists() else {}
            calls.append(("lock_before_train", held.get("sha256") == H and held.get("pid") == __import__("os").getpid()))
        if calls.old is not None:
            taken = rr.taken_path(calls.old, seed)
            calls.append(("taken_before_train", taken.exists() and taken.read_text() == str(calls.out.resolve())))
        calls.append(("train", seed, h == Hyper()))
        torch.manual_seed(seed)
        agent = Agent(11, h.size, 2, h.max_units, 6, 81, 4)
        agent.decisions_trained = h.total_steps
        return agent

    def fake_evaluate(agent, h, a, f, seeds=ev.EVAL_SEEDS):
        if calls.out is not None:
            calls.append(("marker_before_evaluate", any(calls.out.glob("seed-*.evaluating"))))
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
    recorded.out = tmp_path
    r = rr.run_seed(3, tmp_path, proto, H, H)
    assert recorded == [("lock_before_train", True), ("train", 3, True), ("marker_before_evaluate", True),
                        ("evaluate", True)]  # fmt: skip
    assert rr.lock_path(tmp_path, 3).exists()
    saved = json.loads(rr.result_path(tmp_path, 3).read_text())
    assert saved["wins"] == 300 and saved["protocol"] is True and r["wins"] == 300
    rec = saved["record"]
    assert rec["protocol_sha256"] == H and rec["training_seed"] == 3
    assert rec["decisions_trained"] == Hyper().total_steps and rec["hyper"] == dataclasses.asdict(Hyper())
    assert rr.policy_path(tmp_path, 3).exists()
    with pytest.raises(rr.Refused, match="used once"):
        rr.run_seed(3, tmp_path, proto, H, H)
    assert len(recorded) == 4


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



# -- crash rule: the evaluation marker and one rerun before it (TR, pre-read 5) --


def _dead_pid():
    p = __import__("subprocess").Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _crashed(dirpath, seed, marker=False, result=False, rerun=False, pid=None, host=None):
    import socket

    dirpath.mkdir(parents=True, exist_ok=True)
    holder = {"sha256": H, "pid": _dead_pid() if pid is None else pid, "host": host or socket.gethostname()}
    rr.lock_path(dirpath, seed).write_text(json.dumps(holder))
    if marker:
        rr.marker_path(dirpath, seed).write_text(H)
    if result:
        rr.result_path(dirpath, seed).write_text("{}")
    if rerun:
        rr.rerun_path(dirpath, seed).write_text("x")


def test_rts_runner_reruns_a_seed_that_crashed_before_evaluation_once(tmp_path, recorded, proto):
    old, new = tmp_path / "a", tmp_path / "b"
    _crashed(old, 1)
    recorded.out, recorded.old = new, old
    r = rr.run_seed(1, new, proto, H, H, rerun_of=old)
    assert ("taken_before_train", True) in recorded
    assert r["rerun_of"] == str(old.resolve())
    assert rr.rerun_path(new, 1).read_text() == str(old.resolve())
    assert rr.lock_path(old, 1).exists()  # the crashed directory is kept
    assert rr.taken_path(old, 1).read_text() == str(new.resolve())
    assert ("train", 1, True) in recorded
    # F1: once. A second rerun of the same crashed directory is refused before train.
    trains = sum(1 for c in recorded if c[0] == "train")
    with pytest.raises(rr.Refused, match="already rerun"):
        rr.run_seed(1, tmp_path / "c", proto, H, H, rerun_of=old)
    assert sum(1 for c in recorded if c[0] == "train") == trains


@pytest.mark.parametrize("case", ["live_pid", "other_host", "plain_lock"])
def test_rts_runner_refuses_to_rerun_a_seed_that_may_still_run(tmp_path, tripped, proto, case):
    """F2: a lock without a marker is also a live run: refused while its pid
    lives on this host, on another host, or when the lock names no process."""
    old = tmp_path / "a"
    if case == "plain_lock":
        old.mkdir()
        rr.lock_path(old, 1).write_text(H)
    else:
        _crashed(old, 1, pid=__import__("os").getpid() if case == "live_pid" else None,
                 host="elsewhere" if case == "other_host" else None)  # fmt: skip
    why = {"live_pid": "still running", "other_host": "not this host", "plain_lock": "does not name its process"}[case]
    with pytest.raises(rr.Refused, match=why):
        rr.run_seed(1, tmp_path / "b", proto, H, H, rerun_of=old)
    assert not rr.taken_path(old, 1).exists()


@pytest.mark.parametrize("case", ["marker", "result", "second_rerun", "same_dir", "no_lock"])
def test_rts_runner_refuses_a_rerun_after_evaluation_began_or_twice(tmp_path, tripped, proto, case):
    old, new = tmp_path / "a", tmp_path / "b"
    if case == "no_lock":
        old.mkdir()
    else:
        _crashed(old, 1, marker=case == "marker", result=case == "result", rerun=case == "second_rerun")
    target = old if case == "same_dir" else new
    with pytest.raises(rr.Refused):
        rr.run_seed(1, target, proto, H, H, rerun_of=old)


def test_rts_runner_verdict_finds_a_rerun_seed_in_its_own_directory(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for s in range(4):
        rr.result_path(a, s).write_text(json.dumps(protocol_result(260, seed=s)))
    rr.result_path(b, 4).write_text(json.dumps(protocol_result(260, seed=4)))
    v = rr.verdict([a, b], H)
    assert v["passes"] and v["sources"][4]["dir"] == str(b.resolve())
    rr.result_path(b, 3).write_text(json.dumps(protocol_result(260, seed=3)))
    with pytest.raises(rr.Refused, match="several"):
        rr.verdict([a, b], H)



def test_rts_runner_verdict_checks_a_reruns_link_both_ways(tmp_path):
    """F1: a result naming rerun_of counts only if the crashed directory's
    rerun-taken names this directory; the verdict lists it."""
    a, old, other = tmp_path / "a", tmp_path / "old", tmp_path / "other"
    for d in (a, old, other):
        d.mkdir()
    for s in range(4):
        rr.result_path(a, s).write_text(json.dumps(protocol_result(260, seed=s)))
    r = protocol_result(260, seed=4)
    r["rerun_of"] = str(old.resolve())
    rr.result_path(other, 4).write_text(json.dumps(r))
    rr.rerun_path(other, 4).write_text(str(old.resolve()))
    rr.taken_path(old, 4).write_text(str((tmp_path / "elsewhere").resolve()))
    with pytest.raises(rr.Refused, match="whose rerun is not this directory"):
        rr.verdict([a, other], H)
    rr.taken_path(old, 4).write_text(str(other.resolve()))
    v = rr.verdict([a, other], H)
    assert v["passes"] and v["sources"][4] == {"dir": str(other.resolve()), "rerun_of": str(old.resolve())}



def test_rts_runner_a_not_fresh_directory_does_not_spend_the_rerun(tmp_path, tripped, proto):
    """(b): the new directory's lock is taken before the crashed directory's
    rerun-taken; a directory that was not fresh fails first and the crashed
    directory stays available for its one rerun."""
    old, new = tmp_path / "a", tmp_path / "b"
    _crashed(old, 1)
    new.mkdir()
    rr.lock_path(new, 1).write_text("{}")
    with pytest.raises(rr.Refused, match="already taken"):
        rr.run_seed(1, new, proto, H, H, rerun_of=old)
    assert not rr.taken_path(old, 1).exists()


@pytest.mark.parametrize("case", ["result_names_but_no_file", "file_but_result_silent", "disagree"])
def test_rts_runner_verdict_needs_both_rerun_links_to_agree(tmp_path, case):
    """(c): the result's rerun_of and the directory's seed-<s>.rerun-of must
    agree, so dropping or editing one of them is refused."""
    a, old, other = tmp_path / "a", tmp_path / "old", tmp_path / "other"
    for d in (a, old, other):
        d.mkdir()
    for s in range(4):
        rr.result_path(a, s).write_text(json.dumps(protocol_result(260, seed=s)))
    r = protocol_result(260, seed=4)
    rr.taken_path(old, 4).write_text(str(other.resolve()))
    if case != "file_but_result_silent":
        r["rerun_of"] = str(old.resolve())
    if case != "result_names_but_no_file":
        rr.rerun_path(other, 4).write_text(str((tmp_path / "x").resolve()) if case == "disagree" else str(old.resolve()))
    rr.result_path(other, 4).write_text(json.dumps(r))
    with pytest.raises(rr.Refused, match="disagrees"):
        rr.verdict([a, other], H)
