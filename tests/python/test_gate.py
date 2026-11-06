# SPDX-License-Identifier: MIT OR Apache-2.0
"""The P2-G1 gate tool (examples/alphazero_connect4/gate.py)."""

import argparse
import ast
import builtins
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

EXAMPLE = pathlib.Path(__file__).parents[2] / "examples" / "alphazero_connect4"
REPO = EXAMPLE.parents[1]
sys.path.insert(0, str(EXAMPLE))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import artifacts  # noqa: E402
import attempts  # noqa: E402
import gate  # noqa: E402
import loop  # noqa: E402
import train_step  # noqa: E402
from test_loop import bot_binary, make_root, run_loop  # noqa: E402

HELDOUT = REPO / "compat" / "connect4" / "openings-heldout.txt"
IMAGE = "rust:1.99.1@sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137"


REAL_EVAL = os.path.realpath(gate.EVAL)


@pytest.fixture(scope="module")
def bot():
    return bot_binary()


@pytest.fixture(autouse=True)
def synthetic_eval(tmp_path, monkeypatch):
    """Every test plays a synthetic 500-opening stand-in for the blind set,
    and any open of the real file fails the test."""
    lines = [x for x in HELDOUT.read_text().splitlines() if x and not x.startswith("#")]
    fake = tmp_path / "synthetic-eval.txt"
    fake.write_text("".join(f"{j} x {lines[j % len(lines)].split()[2]}\n" for j in range(500)))
    monkeypatch.setattr(gate, "EVAL", str(fake))
    monkeypatch.setattr(gate, "EVAL_SHA256", artifacts.sha256_file(str(fake)))
    real_open = builtins.open

    def guard(path, *a, **k):
        if isinstance(path, (str, os.PathLike)) and os.path.realpath(path) == REAL_EVAL:
            raise AssertionError("the real evaluation file was opened")
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", guard)


# ------------------------------------------------------------------ rule


@pytest.mark.parametrize("wins, ok", [(1000, True), (815, True), (814, False), (0, False)])
def test_gate_pass_rule_is_815_wins_with_no_interval(wins, ok):
    r = gate.passes(wins, 1000)
    assert r["pass"] is ok and r["wins_at_least_815"] is ok and r["rule_id"] == "adr-0048-wins-815"
    assert not any("wilson" in k or "lower" in k for k in r), "the interval decides nothing"
    short = gate.passes(999, 999)
    assert short["pass"] is None and short["wins_at_least_815"] is None and short["invalid_because"]


def test_gate_development_has_no_pass_at_any_level():
    r = gate.passes(1000, 1000, "development")
    assert r["pass"] is None and r["wins_at_least_815"] is True
    assert gate.passes(1000, 1000, None)["pass"] is None


@pytest.mark.parametrize("wins, games, ok", [(336, 400, True), (335, 400, False), (336, 399, False), (401, 400, False),
                                             (-1, 400, False), (336.0, 400, False)])  # fmt: skip
def test_gate_eligibility_is_336_of_400(wins, games, ok):
    if ok:
        gate.check_eligible(wins, games)
    else:
        with pytest.raises(gate.Refused):
            gate.check_eligible(wins, games)


# --------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def done_root(tmp_path_factory, bot):
    """A completed tiny loop root."""
    root = make_root(tmp_path_factory.mktemp("gate"), bot)
    r = run_loop(root)
    assert r.returncode == 0, r.stderr
    return root


def pretend_formal(monkeypatch, bot, medium=True, registered=True):
    """The loop and the runtime as a formal, registered run, which a test
    cannot be."""
    real_looks, real_load = loop.Loop.looks, gate.ckpt.load

    def eligible_looks(self, gens, entries):
        return [{**e, "eligible": True} for e in real_looks(self, gens, entries)]

    def as_medium(path):
        model, meta = real_load(path)
        return model, {**meta, "size": "medium"}

    monkeypatch.setattr(loop, "is_formal", lambda cfg: True)
    # A tiny look plays two games, and no solver binary is pinned yet.
    monkeypatch.setattr(gate, "ELIGIBLE_WINS", 0)
    monkeypatch.setattr(gate, "LOOK_GAMES", 2)
    monkeypatch.setattr(gate, "SOLVER_SLOTS_SHA256", "0" * 64)
    monkeypatch.setattr(loop.Loop, "looks", eligible_looks)
    if medium:
        # Only the gate's view: the loop's identities read the real size.
        import types

        monkeypatch.setattr(gate, "ckpt", types.SimpleNamespace(load=as_medium))
    monkeypatch.setattr(gate.h2h, "runtime", lambda: {"registered": registered, "image_digest": IMAGE})
    monkeypatch.setitem(gate.look.REGISTERED_BOTS, IMAGE, artifacts.sha256_file(bot))


@pytest.fixture
def as_formal(monkeypatch, bot):
    pretend_formal(monkeypatch, bot)


@pytest.fixture
def eval_trap(monkeypatch):
    """Records every open of the real evaluation file."""
    opened = []
    real_open = builtins.open
    target = os.path.realpath(gate.EVAL)

    def spy(path, *a, **k):
        if isinstance(path, (str, os.PathLike)) and os.path.realpath(path) == target:
            opened.append(str(path))
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    return opened


def selected_root(tmp_path, done_root):
    root = tmp_path / "root"
    shutil.copytree(done_root, root)
    m = loop.Loop(str(root)).select()
    path = root / "selection-manifest.json"
    return root, path, artifacts.sha256_file(str(path)), m


def formal_argv(root, sha, bot, *extra, bot_sha=None):
    journal = pathlib.Path(root).parent / "attempts.jsonl"
    return [str(a) for a in ("--root", root, "--selection-sha256", sha, "--attempt", "formal",
                             "--attempts-journal", journal, "--create-journal", "--bot", bot,
                             "--bot-sha256", bot_sha or artifacts.sha256_file(bot), "--source-sha", "t",
                             "--budget-core-hours", 1, "--wall-cap-s", 2400, *extra)]  # fmt: skip


def preflight_and_claim(argv):
    args = gate.parse(argv)
    lp = loop.Loop(args.root)
    lock = lp.owner()
    try:
        record = gate.preflight(args, lp)
        gate.claim(args, lp, record)
    finally:
        lock.close()
    return args, record


# --------------------------------------------------------------- formal


def test_gate_formal_preflight_and_claim_on_the_committed_selection(tmp_path, done_root, as_formal, bot, eval_trap):
    root, _, sha, m = selected_root(tmp_path, done_root)
    assert m["formal"] and m["selected"]
    _, record = preflight_and_claim(formal_argv(root, sha, bot))
    assert record["selected"] == m["selected"] and record["checkpoint_sha256"] == m["selected"]["checkpoint"]
    assert json.loads((root / "gate" / "claim.json").read_text()) == record
    audit = [e for e in loop.Loop(str(root)).selection() if e["type"] == "gate-claim"]
    assert len(audit) == 1 and audit[0]["selection_sha256"] == sha
    assert eval_trap == []


DAMAGES = ["sha", "not-formal", "no-selection", "tampered-looks", "tampered-pick", "broken-ledger", "evicted",
           "bot-sha", "bot-unregistered", "runtime", "small-net", "search", "budget", "unpinned", "ineligible", "look-count"]  # fmt: skip


@pytest.mark.parametrize("damage", DAMAGES)
def test_gate_refuses_before_the_evaluation_set_is_touched(tmp_path, done_root, bot, eval_trap, damage, monkeypatch):
    pretend_formal(monkeypatch, bot, medium=damage != "small-net", registered=damage != "runtime")
    root, path, sha, m = selected_root(tmp_path, done_root)
    extra, bot_sha = [], None

    def rewrite(**change):
        path.write_bytes(artifacts.canonical_json({**json.loads(path.read_text()), **change}))
        return artifacts.sha256_file(str(path))

    if damage == "sha":
        sha = "0" * 64
    elif damage == "not-formal":
        sha = rewrite(formal=False)
    elif damage == "no-selection":
        sha = rewrite(selected=None)
    elif damage == "tampered-looks":
        sha = rewrite(eligible_looks=[{**e, "wins": e["wins"] + 1} for e in m["eligible_looks"]])
    elif damage == "tampered-pick":
        other = next(e for e in m["eligible_looks"] if e["checkpoint"] != m["selected"]["checkpoint"])
        sha = rewrite(selected={k: other[k] for k in ("generation", "checkpoint", "wins")})
    elif damage == "broken-ledger":
        (root / "ledger.json").unlink()
    elif damage == "evicted":
        d = root / f"gen-{m['selected']['generation'] - 1:03d}" / "train-0"
        desc = json.loads((d / "attempt.json").read_text())
        (d / "attempt.json").write_bytes(artifacts.canonical_json({**desc, "evicted": {"name": "x"}}))
    elif damage == "bot-sha":
        bot_sha = "0" * 64
    elif damage == "bot-unregistered":
        monkeypatch.setitem(gate.look.REGISTERED_BOTS, IMAGE, "0" * 64)
    elif damage == "search":
        extra = ["--sims", "50"]
    elif damage == "budget":
        extra = ["--wall-cap-s", "4000"]  # 1 core-hour cannot fund it
    elif damage == "unpinned":
        monkeypatch.setattr(gate, "SOLVER_SLOTS_SHA256", None)
    elif damage == "ineligible":
        monkeypatch.setattr(gate, "ELIGIBLE_WINS", 3)
    elif damage == "look-count":
        monkeypatch.setattr(gate, "LOOK_GAMES", 400)
    with pytest.raises(SystemExit):  # Refused, or a refusing loop Stop
        preflight_and_claim(formal_argv(root, sha, bot, *extra, bot_sha=bot_sha))
    assert not (root / "gate").exists(), "a refused preflight claims nothing"
    assert not (root.parent / "attempts.jsonl").exists(), "and spends no attempt"
    assert eval_trap == []


def test_gate_second_claim_is_refused_whatever_paths_it_names(tmp_path, done_root, as_formal, bot, eval_trap):
    root, _, sha, _ = selected_root(tmp_path, done_root)
    preflight_and_claim(formal_argv(root, sha, bot))
    with pytest.raises(gate.Refused, match="already claimed"):
        preflight_and_claim(formal_argv(root, sha, bot))
    with pytest.raises(SystemExit):  # the formal paths are fixed
        gate.parse(formal_argv(root, sha, bot, "--out", tmp_path / "elsewhere", "--ledger", tmp_path / "l.json"))
    with pytest.raises(gate.Refused, match="already claimed"):
        gate.main(formal_argv(root, sha, bot))
    assert eval_trap == []


def test_gate_refused_reservation_never_unblinds(tmp_path, done_root, as_formal, bot, eval_trap, monkeypatch):
    """A reservation the ledger refuses stops the formal run after the
    claim and before the child exists: the eval file is never opened,
    and the claim stays consumed."""
    import supervisor
    from ledger import BudgetExceeded

    root, _, sha, _ = selected_root(tmp_path, done_root)

    def refuse(self, run, cores, deadline_s):
        raise BudgetExceeded(f"{run}: refused for the test")

    monkeypatch.setattr(supervisor.Ledger, "reserve", refuse)
    with pytest.raises(BudgetExceeded):
        gate.main(formal_argv(root, sha, bot))
    assert (root / "gate" / "claim.json").exists() and eval_trap == []
    with pytest.raises(gate.Refused, match="already claimed"):
        gate.main(formal_argv(root, sha, bot))


def scripted_play(wins):
    def play(evaluator, bot_, openings, *rest):
        assert len(openings) == 500
        logs = [[j, first, 1.0 if 2 * j + (not first) < wins else -1.0, []] for j in range(500) for first in (True, False)]
        pairs = [sum(1 for x in logs[2 * j : 2 * j + 2] if x[2] > 0) for j in range(500)]
        return {"results": [x[2] for x in logs], "logs": logs, "pairs": pairs,
                "by_colour": {"candidate_first": [0, 0, 0], "bot_first": [0, 0, 0]}}  # fmt: skip

    return play


def test_gate_formal_work_revalidates_then_records_true_or_false(tmp_path, done_root, as_formal, bot, monkeypatch):
    """The supervised work with the games replaced by a scripted result:
    the claim is revalidated before the evaluation set is read, and a
    complete gate records pass true or false, never null."""
    root, _, sha, _ = selected_root(tmp_path, done_root)
    args, _ = preflight_and_claim(formal_argv(root, sha, bot))
    order = []
    real_check, real_openings = gate.check_candidate, gate.formal_openings
    monkeypatch.setattr(gate, "check_candidate", lambda *a: (order.append("check"), real_check(*a))[1])
    monkeypatch.setattr(gate, "formal_openings", lambda: (order.append("openings"), real_openings())[1])
    monkeypatch.setattr(gate.game, "BotClient", lambda path: type("B", (), {"close": lambda self: None})())
    for wins, expected in ((815, True), (814, False)):
        monkeypatch.setattr(gate.botmatch, "play", scripted_play(wins))
        args.out = str(tmp_path / f"out-{wins}")
        os.makedirs(args.out)
        assert gate.work(args, None) == 0
        assert not (pathlib.Path(args.out) / "gate.json").exists(), "the child writes a provisional report only"
        report = json.loads((pathlib.Path(args.out) / gate.PROVISIONAL).read_text())
        assert report["gate_run"] is True and report["games"] == 1000 and "pass" not in report
        assert report["provisional_pass"] is expected and report["rule"]["wins"] == wins
        assert report["openings_sha256_at_end"] == gate.EVAL_SHA256
        assert report["selected"]["checkpoint"] == report["checkpoint"]
        assert report["rule_id"] == gate.RULE_ID and report["rule"]["pass"] is expected and report["meets_815"] is expected
        assert "wilson_lower_z1.96" not in report["rule"] and "decide" in report["descriptive"]["note"]
        assert report["attempt"]["label"] == "formal" and report["look_evidence"]["games"] == 2
        assert report["wdl"] == [wins, 0, 1000 - wins] and sum(report["pair_wins"].values()) == 500
        assert set(report["by_colour"]) == {"candidate_first", "bot_first"} and "distinct_games" in report
    assert order[:2] == ["check", "openings"]


def test_gate_formal_work_refuses_a_claim_for_another_selection(tmp_path, done_root, as_formal, bot, eval_trap):
    root, _, sha, _ = selected_root(tmp_path, done_root)
    args, _ = preflight_and_claim(formal_argv(root, sha, bot))
    args.selection_sha256 = "0" * 64
    with pytest.raises(gate.Refused, match="another selection"):
        gate.work(args, None)
    assert eval_trap == []


def test_gate_refuses_a_real_non_formal_loop(tmp_path, done_root, bot, eval_trap):
    root, path, sha, _ = selected_root(tmp_path, done_root)
    with pytest.raises(gate.Refused, match="not formal"):
        preflight_and_claim(formal_argv(root, sha, bot))
    assert eval_trap == [] and not (root / "gate").exists()


# ------------------------------------------------------------ diagnostic


def diag_argv(tmp_path, bot, openings, *extra):
    c = tmp_path / "C.pt"
    if not c.exists():
        assert train_step.main(["init", "--size", "tiny", "--seed", "1", "--out", str(c), "--source-sha", "t"]) == 0
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return [str(a) for a in ("--any-openings", "--openings", openings, "--checkpoint", c, "--out", out,
                             "--ledger", tmp_path / "l.json", "--bot", bot, "--bot-sha256", artifacts.sha256_file(bot),
                             "--source-sha", "t", "--budget-core-hours", 1, "--wall-cap-s", 120, *extra)]  # fmt: skip


def test_gate_diagnostic_run_never_opens_the_evaluation_file(tmp_path, bot, eval_trap):
    args = gate.parse(diag_argv(tmp_path, bot, HELDOUT, "--limit", 2, "--sims", 2, "--batch-size", 2, "--capacity", 64))
    assert gate.work(args, None) == 0
    report = json.loads((tmp_path / "out" / gate.PROVISIONAL).read_text())
    assert report["gate_run"] is False and report["provisional_pass"] is None and report["selected"] is None
    assert report["games"] == 4 and report["openings_covered"] == 2 and sum(report["pair_wins"].values()) == 2
    assert [x[0] for x in json.loads((tmp_path / "out" / "gate-log.json").read_text())] == [0, 0, 1, 1]
    assert eval_trap == []


@pytest.mark.parametrize("alias", ["path", "symlink", "hardlink", "copy"])
def test_gate_diagnostic_refuses_the_evaluation_set_and_its_aliases(tmp_path, bot, eval_trap, alias):
    target = tmp_path / "alias.txt"
    if alias == "path":
        target = pathlib.Path(gate.EVAL)
    elif alias == "symlink":
        target.symlink_to(gate.EVAL)
    elif alias == "hardlink":
        # On the stand-in's own file system.
        import tempfile

        target = pathlib.Path(tempfile.mkdtemp(dir=tmp_path)) / "alias.txt"
        os.link(gate.EVAL, target)
    else:
        shutil.copyfile(gate.EVAL, target)
        eval_trap.clear()
    args = gate.parse(diag_argv(tmp_path, bot, target))
    try:
        with pytest.raises(gate.Refused, match="never plays the evaluation"):
            gate.work(args, None)
    finally:
        if alias == "hardlink":
            shutil.rmtree(target.parent)
    # Aliases are refused by device and inode, before any open.
    assert eval_trap == []


def test_gate_modes_do_not_mix(tmp_path, bot):
    with pytest.raises(SystemExit):
        gate.parse(diag_argv(tmp_path, bot, HELDOUT, "--root", tmp_path))
    with pytest.raises(SystemExit):  # no default fixture
        gate.parse([a for a in diag_argv(tmp_path, bot, HELDOUT) if a not in ("--openings", str(HELDOUT))])
    with pytest.raises(SystemExit):  # formal takes the selected checkpoint only
        gate.parse(formal_argv(tmp_path, "0" * 64, bot, "--checkpoint", tmp_path / "C.pt"))


# ------------------------------------------------------------- blindness


def test_gate_evaluation_set_is_named_only_by_the_gate():
    """Within the example and its tests, only gate.py names the evaluation
    file. The P2-M3 pilot (PR 1) timed a randomly initialized net there
    before any training, and never selects or tunes. test_connect4_pins
    only hashes the file against the ADR-0042 pin. tools/connect4-ref
    generates and calibrates the set (ADR-0042)."""
    tests = (p for p in (REPO / "tests" / "python").glob("*.py") if p.name != "test_gate.py")
    files = sorted(EXAMPLE.glob("*.py")) + sorted(tests)
    users = sorted(p.name for p in files if "openings-eval" in p.read_text())
    assert users == ["gate.py", "pilot.py", "test_connect4_pins.py"]
    assert "netmod.build(args.eval_size, seed=args.seed)" in (EXAMPLE / "pilot.py").read_text()
    # In gate.py, EVAL is read only by the formal readers; the diagnostic
    # branch only stats and resolves it.
    tree = ast.parse((EXAMPLE / "gate.py").read_text())
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    named = {f.name for f in fns if any(isinstance(x, ast.Name) and x.id == "EVAL" for x in ast.walk(f))}
    assert named == {"formal_openings", "eval_sha_now", "diagnostic_openings", "compare_openings"}
    diag = next(f for f in fns if f.name == "diagnostic_openings")
    calls = {ast.unparse(c.func) for c in ast.walk(diag) if isinstance(c, ast.Call)
             if any(isinstance(a, ast.Name) and a.id == "EVAL" for a in c.args)}  # fmt: skip
    assert calls <= {"os.stat", "os.path.realpath"}


# ------------------------------------------------------------- finalize


def test_gate_verdict_only_after_an_ok_settlement(tmp_path, done_root, as_formal, bot, monkeypatch):
    """A provisional PASS becomes the gate result only with exit 0 and an ok
    ledger entry; a failure after the report was written leaves pass null."""
    from ledger import Ledger

    root, _, sha, _ = selected_root(tmp_path, done_root)
    args, _ = preflight_and_claim(formal_argv(root, sha, bot))
    monkeypatch.setattr(gate.botmatch, "play", scripted_play(1000))
    monkeypatch.setattr(gate.game, "BotClient", lambda path: type("B", (), {"close": lambda self: None})())
    for case, (rc, status) in {"ok": (0, "ok"), "killed": (137, "failed: killed at the deadline"),
                               "failed-exit": (1, "ok")}.items():  # fmt: skip
        args.out = str(tmp_path / f"out-{case}")
        args.ledger = str(tmp_path / f"ledger-{case}.json")
        os.makedirs(args.out)
        led = Ledger(args.ledger, 1)
        i, _ = led.reserve("gate 20261103T000000Z", 1, 60)
        assert gate.work(args, None) == 0  # the provisional report exists
        led.settle(i, 1, status)
        final = gate.finalize(args, rc, "gate")
        on_disk = json.loads((pathlib.Path(args.out) / "gate.json").read_text())
        assert on_disk == final and "provisional_pass" not in final
        if case == "ok":
            assert final["pass"] is True and final["attempt_invalid_because"] == []
        else:
            assert final["pass"] is None and final["attempt_invalid_because"]
    # No provisional report at all (killed before the games ended).
    args.out = str(tmp_path / "out-empty")
    os.makedirs(args.out)
    assert gate.finalize(args, 137, "gate")["pass"] is None


def test_gate_claim_syncs_the_parent_entry_before_anything_else(tmp_path, done_root, as_formal, bot, monkeypatch):
    root, _, sha, _ = selected_root(tmp_path, done_root)
    order = []
    real_mkdir, real_sync, real_write = os.mkdir, gate.fsync_dir, gate.artifacts.write_atomic
    monkeypatch.setattr(gate.os, "mkdir", lambda p, *a: (order.append(("mkdir", p)), real_mkdir(p, *a))[1])
    monkeypatch.setattr(gate, "fsync_dir", lambda p: (order.append(("fsync", p)), real_sync(p))[1])
    monkeypatch.setattr(gate.artifacts, "write_atomic", lambda p, b: (order.append(("write", p)), real_write(p, b))[1])
    preflight_and_claim(formal_argv(root, sha, bot))
    g = str(root / "gate")
    assert order[:4] == [("mkdir", g), ("fsync", str(root)), ("write", os.path.join(g, "claim.json")), ("fsync", g)]
    assert order[-1] == ("fsync", str(root))


def test_gate_distinct_games_are_move_sequences():
    openings = ["4455", "44", "445"]
    logs = [
        [0, True, 1.0, [("c", 0, [1, 2]), ("b", 1, [])]],
        [0, False, -1.0, [("b", 0, []), ("c", 1, [9, 9])]],  # same moves, roles and visits swapped
        [1, True, 1.0, [("c", 4, []), ("b", 4, []), ("c", 0, []), ("b", 1, [])]],  # "44"+"55..." = "4455"+...
        [2, True, 1.0, [("c", 4, [])]],
    ]
    assert gate.distinct_games(openings, logs) == 2


@pytest.mark.parametrize("bad", [("--budget-core-hours", "nan"), ("--budget-core-hours", "inf"),
                                 ("--budget-core-hours", "0"), ("--budget-core-hours", "-1"),
                                 ("--wall-cap-s", "0"), ("--wall-cap-s", "-5"), ("--wall-cap-s", "nan"),
                                 ("--wall-cap-s", "1"), ("--cores", "0"), ("--cores", "2")])  # fmt: skip
def test_gate_invalid_resources_never_consume_the_claim(tmp_path, done_root, as_formal, bot, eval_trap, bad):
    root, _, sha, _ = selected_root(tmp_path, done_root)
    with pytest.raises(gate.Refused):
        preflight_and_claim(formal_argv(root, sha, bot, *bad))
    assert not (root / "gate").exists() and eval_trap == []


def test_gate_finalize_recovers_after_the_parent_died(tmp_path, done_root, as_formal, bot, monkeypatch):
    """The child settled ok but the parent died before gate.json: the
    read-only finalize rebuilds it under the same rules, idempotently."""
    from ledger import Ledger

    root, _, sha, _ = selected_root(tmp_path, done_root)
    assert pytest.raises(gate.Refused, gate.recover, str(root))  # nothing claimed
    args, _ = preflight_and_claim(formal_argv(root, sha, bot))
    monkeypatch.setattr(gate.botmatch, "play", scripted_play(1000))
    monkeypatch.setattr(gate.game, "BotClient", lambda path: type("B", (), {"close": lambda self: None})())
    os.makedirs(args.out)
    led = Ledger(args.ledger, args.budget_core_hours)
    i, _ = led.reserve("gate 20261103T000000Z", 1, 60)
    with pytest.raises(gate.Refused, match="no settled"):
        gate.recover(str(root))  # still running
    assert gate.work(args, None) == 0
    led.settle(i, 1, "ok")
    first = gate.main(["finalize", "--root", str(root)])
    assert first == 0
    final = json.loads((pathlib.Path(args.out) / "gate.json").read_text())
    assert final["pass"] is True and final["settlement"]["status"] == "ok"
    assert gate.recover(str(root)) == final  # idempotent, unchanged
    # A failed settlement recovers as pass null.
    os.remove(pathlib.Path(args.out) / "gate.json")
    doc = json.loads(pathlib.Path(args.ledger).read_text())
    doc["entries"][-1]["status"] = "failed: killed"
    pathlib.Path(args.ledger).write_text(json.dumps(doc))
    assert gate.recover(str(root))["pass"] is None


def test_gate_finalize_and_recover_never_rewrite_the_ledger(tmp_path, done_root, as_formal, bot, monkeypatch):
    import fcntl

    from ledger import Ledger

    root, _, sha, _ = selected_root(tmp_path, done_root)
    args, _ = preflight_and_claim(formal_argv(root, sha, bot, "--budget-core-hours", "0.75", "--wall-cap-s", "2400"))
    monkeypatch.setattr(gate.botmatch, "play", scripted_play(1000))
    monkeypatch.setattr(gate.game, "BotClient", lambda path: type("B", (), {"close": lambda self: None})())
    os.makedirs(args.out)
    led = Ledger(args.ledger, args.budget_core_hours)
    i, _ = led.reserve("gate 20261103T000000Z", 1, 60)
    ledger = pathlib.Path(args.ledger)
    running = ledger.read_bytes()
    # Refusal while running: the entry is never settled by the recovery.
    with pytest.raises(gate.Refused, match="no settled"):
        gate.recover(str(root))
    assert ledger.read_bytes() == running
    assert gate.work(args, None) == 0
    led.settle(i, 1, "ok")
    before = ledger.read_bytes()
    # The normal path.
    gate.finalize(args, 0, "gate")
    assert ledger.read_bytes() == before
    os.remove(pathlib.Path(args.out) / "gate.json")
    # Owner busy: refused, nothing written.
    with open(root / "loop.lock", "a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(SystemExit, match="another loop"):
            gate.recover(str(root))
    assert not (pathlib.Path(args.out) / "gate.json").exists()
    # Success, then an idempotent repeat.
    first = gate.recover(str(root))
    assert first["pass"] is True and gate.recover(str(root)) == first
    assert ledger.read_bytes() == before
    doc = json.loads(before)
    assert doc["budget_core_hours"] == 0.75 and doc["entries"][-1]["status"] == "ok"


# --------------------------------------------------------------- attempts


def journal_args(path, label, checkpoint, create=False, root="/r"):
    return (str(path), "B", "F", create, label, root, checkpoint)


def settled(block, disposition="evaluated-failed"):
    attempts.settle(block, disposition, {"claim_sha256": "c", "report_sha256": "r", "ledger_sha256": "l"})


def test_attempts_admission_follows_the_lifecycle(tmp_path):
    j = tmp_path / "j.jsonl"
    with pytest.raises(attempts.JournalError, match="no journal"):
        attempts.claim(*journal_args(j, "formal", "a"))
    first = attempts.claim(*journal_args(j, "formal", "a", create=True))
    assert first["journal_created"] is True and first["journal"] == os.path.realpath(j)
    with pytest.raises(attempts.JournalError, match="one formal"):
        attempts.claim(*journal_args(j, "formal", "b"))
    with pytest.raises(attempts.JournalError, match="active or interrupted"):
        attempts.claim(*journal_args(j, "development", "b"))
    settled(first)
    settled(first, "evaluated-passed")  # the first terminal event stands
    assert sum('"settled"' in x for x in j.read_text().splitlines()) == 1
    with pytest.raises(attempts.JournalError, match="one formal"):
        attempts.claim(*journal_args(j, "formal", "b"))
    with pytest.raises(attempts.JournalError, match="already evaluated"):
        attempts.claim(*journal_args(j, "development", "a"))
    dev = attempts.claim(*journal_args(j, "development", "b"))
    assert dev["journal_created"] is False
    with pytest.raises(attempts.JournalError, match="active or interrupted"):
        attempts.claim(*journal_args(j, "development", "c"))


def test_attempts_development_refused_while_the_set_is_blind(tmp_path):
    with pytest.raises(attempts.JournalError, match="still blind"):
        attempts.claim(*journal_args(tmp_path / "j", "development", "a", create=True))


def test_attempts_two_roots_racing_one_formal_claim(tmp_path):
    j = tmp_path / "j.jsonl"

    def go(i):
        try:
            return attempts.claim(*journal_args(j, "formal", f"c{i}", create=True, root=f"/r{i}"))
        except attempts.JournalError:
            return None

    with ThreadPoolExecutor(8) as pool:
        won = [x for x in pool.map(go, range(8)) if x]
    assert len(won) == 1
    assert sum('"claimed"' in x for x in j.read_text().splitlines()) == 1


@pytest.mark.parametrize("damage", ["truncated", "malformed", "chain", "unknown", "header", "empty", "double"])
def test_attempts_journal_fails_closed(tmp_path, damage):
    j = tmp_path / "j.jsonl"
    block = attempts.claim(*journal_args(j, "formal", "a", create=True))
    raw = j.read_bytes()
    lines = raw.splitlines(keepends=True)
    if damage == "truncated":
        j.write_bytes(raw[:-3])
    elif damage == "malformed":
        j.write_bytes(lines[0] + b"not json\n")
    elif damage == "chain":
        j.write_bytes(lines[0] + lines[1].replace(b'"prev":"', b'"prev":"0'))
    elif damage == "header":
        j.write_bytes(lines[0].replace(b'"benchmark":"B"', b'"benchmark":"X"') + lines[1])
    elif damage == "empty":
        j.write_bytes(b"")
    elif damage == "unknown":
        rec = {"event": "other", "attempt": block["id"], "prev": artifacts.sha256(lines[1].rstrip(b"\n"))}
        j.write_bytes(raw + artifacts.canonical_json(rec).rstrip(b"\n") + b"\n")
    else:
        rec = {"event": "claimed", "attempt": block["id"], "label": "formal", "prev": artifacts.sha256(lines[1].rstrip(b"\n"))}
        j.write_bytes(raw + artifacts.canonical_json(rec).rstrip(b"\n") + b"\n")
    with pytest.raises(attempts.JournalError):
        attempts.claim(*journal_args(j, "development", "b"))


def test_attempts_a_lost_journal_is_a_visible_departure_and_aliases_are_one(tmp_path):
    j = tmp_path / "j.jsonl"
    attempts.claim(*journal_args(j, "formal", "a", create=True))
    link = tmp_path / "alias.jsonl"
    link.symlink_to(j)
    with pytest.raises(attempts.JournalError, match="one formal"):
        attempts.claim(*journal_args(link, "formal", "b"))
    j.unlink()
    link.unlink()
    with pytest.raises(attempts.JournalError, match="no journal"):
        attempts.claim(*journal_args(j, "formal", "b"))
    assert attempts.claim(*journal_args(j, "formal", "b", create=True))["journal_created"] is True


def development_argv(root, sha, bot, checkpoint):
    argv = formal_argv(root, sha, bot)
    argv[argv.index("formal")] = "development"
    return [*argv, "--development-checkpoint", checkpoint]


def fake_bot(monkeypatch, wins):
    monkeypatch.setattr(gate.botmatch, "play", scripted_play(wins))
    monkeypatch.setattr(gate.game, "BotClient", lambda path: type("B", (), {"close": lambda self: None})())


def finish_attempt(args, status="ok", rc=0):
    """The child's work and its settlement, as the supervisor would do them."""
    from ledger import Ledger

    os.makedirs(args.out)
    assert gate.work(args, None) == 0
    led = Ledger(args.ledger, args.budget_core_hours)
    i, _ = led.reserve("gate 20261103T000000Z", 1, 60)
    led.settle(i, 1, status)
    return gate.finalize(args, rc, "gate")


def test_gate_development_after_a_settled_formal_has_null_pass(tmp_path, done_root, as_formal, bot, monkeypatch):
    root, _, sha, m = selected_root(tmp_path, done_root)
    checkpoint = m["selected"]["checkpoint"]
    with pytest.raises(SystemExit):
        preflight_and_claim(development_argv(root, sha, bot, "f" * 64))
    fake_bot(monkeypatch, 900)
    args, _ = preflight_and_claim(formal_argv(root, sha, bot))
    with pytest.raises(attempts.JournalError, match="active or interrupted"):
        preflight_and_claim(development_argv(root, sha, bot, checkpoint))
    assert finish_attempt(args)["pass"] is True
    lines = [json.loads(x) for x in pathlib.Path(args.attempts_journal).read_text().splitlines()]
    assert lines[-1]["event"] == "settled" and lines[-1]["disposition"] == "evaluated-passed"
    assert lines[-1]["claim_sha256"] and lines[-1]["report_sha256"] and lines[-1]["ledger_sha256"]
    # The same checkpoint has an entry: refused, and nothing is claimed.
    with pytest.raises(attempts.JournalError, match="already evaluated"):
        preflight_and_claim(development_argv(root, sha, bot, checkpoint))
    assert not (root / f"gate-dev-{checkpoint[:16]}").exists()


def test_gate_development_report_reads_as_no_pass(tmp_path, done_root, as_formal, bot, monkeypatch):
    root, _, sha, m = selected_root(tmp_path, done_root)
    checkpoint = m["selected"]["checkpoint"]
    journal = pathlib.Path(root).parent / "attempts.jsonl"
    first = attempts.claim(str(journal), gate.EVAL_SHA256, gate.RULE_FAMILY, True, "formal", "/other", "c" * 64)
    settled(first)
    args, _ = preflight_and_claim(development_argv(root, sha, bot, checkpoint))
    assert args.gate_dir == str(root / f"gate-dev-{checkpoint[:16]}")
    fake_bot(monkeypatch, 900)
    final = finish_attempt(args)
    assert final["pass"] is None and final["rule"]["pass"] is None and final["meets_815"] is True
    assert final["claim"] == gate.DEVELOPMENT_CLAIM and final["attempt"]["label"] == "development"
    last = json.loads(journal.read_text().splitlines()[-1])
    assert last["disposition"] == "evaluated-passed"  # met the score; it closes nothing
    assert gate.recover(str(root), checkpoint) == final


def interrupted_root(tmp_path, done_root, bot):
    root, _, sha, _ = selected_root(tmp_path, done_root)
    args, _ = preflight_and_claim(formal_argv(root, sha, bot))
    return root, args, pathlib.Path(args.attempts_journal)


def test_gate_recovery_waits_for_a_live_child_after_the_parent_died(tmp_path, done_root, as_formal, bot):
    """The ledger still says running, and the recorded group is alive: the
    attempt stays interrupted, and development stays refused."""
    from ledger import Ledger

    root, args, journal = interrupted_root(tmp_path, done_root, bot)
    Ledger(args.ledger, 1).reserve("gate 20261103T000000Z", 1, 60)
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        artifacts.write_atomic(os.path.join(args.gate_dir, gate.SPAWN),
                               artifacts.canonical_json({"host": socket.gethostname(), "pgid": proc.pid}))  # fmt: skip
        with pytest.raises(gate.Refused, match="not confirmed gone"):
            gate.recover(str(root))
        assert not any('"settled"' in x for x in journal.read_text().splitlines())
    finally:
        proc.kill()
        proc.wait()
    final = gate.recover(str(root))
    assert final["pass"] is None and final["attempt_invalid_because"]
    last = json.loads(journal.read_text().splitlines()[-1])
    assert last["event"] == "settled" and last["disposition"] == "consumed-no-verdict"


@pytest.mark.parametrize("spawn", [{"pgid": None, "host": None}, {"pgid": 1, "host": "another-host"}])
def test_gate_recovery_keeps_an_unaccounted_group_interrupted(tmp_path, done_root, as_formal, bot, spawn):
    """A spawn with no recorded group, or from another host, is unconfirmed."""
    from ledger import Ledger

    root, args, journal = interrupted_root(tmp_path, done_root, bot)
    Ledger(args.ledger, 1).reserve("gate 20261103T000000Z", 1, 60)
    spawn = {**spawn, "host": spawn["host"] or socket.gethostname()}
    artifacts.write_atomic(os.path.join(args.gate_dir, gate.SPAWN), artifacts.canonical_json(spawn))
    with pytest.raises(gate.Refused, match="not confirmed gone"):
        gate.recover(str(root))
    assert not any('"settled"' in x for x in journal.read_text().splitlines())


def test_gate_settled_as_not_confirmed_does_not_settle_the_journal(tmp_path, done_root, as_formal, bot, monkeypatch):
    root, _, sha, _ = selected_root(tmp_path, done_root)
    args, _ = preflight_and_claim(formal_argv(root, sha, bot))
    fake_bot(monkeypatch, 900)
    final = finish_attempt(args, f"failed: {gate.supervisor.NOT_CONFIRMED}", 1)
    assert final["pass"] is None
    assert not any('"settled"' in x for x in pathlib.Path(args.attempts_journal).read_text().splitlines())


def test_gate_crash_before_any_run_spends_the_attempt(tmp_path, done_root, as_formal, bot):
    """Claimed, nothing reserved: settled with no verdict, and no new formal."""
    root, args, journal = interrupted_root(tmp_path, done_root, bot)
    assert gate.recover(str(root))["pass"] is None
    assert json.loads(journal.read_text().splitlines()[-1])["disposition"] == "consumed-no-verdict"
    with pytest.raises(attempts.JournalError, match="one formal"):
        attempts.claim(str(journal), gate.EVAL_SHA256, gate.RULE_FAMILY, False, "formal", "/other", "b" * 64)


def test_gate_old_report_and_recovery_keep_the_adr_0042_meaning(tmp_path, done_root, as_formal, bot):
    """No ``rule_id`` and no journal block: 850 wins stay false, not 815's true."""
    from ledger import Ledger

    root, args, _ = interrupted_root(tmp_path, done_root, bot)
    claim_path = pathlib.Path(args.gate_dir) / "claim.json"
    old_claim = {k: v for k, v in json.loads(claim_path.read_text()).items() if k not in ("attempt", "rule_id", "look_evidence")}
    claim_path.write_bytes(artifacts.canonical_json(old_claim))
    os.makedirs(args.out)
    old = {"kind": "p2-g1-gate", "gate_run": True, "games": 1000, "provisional_pass": False,
           "rule": {"wins": 850, "games": 1000, "wilson_lower_z1.96": 0.82, "wins_at_least_900": False,
                    "lower_bound_at_least_0.90": False, "pass": False, "invalid_because": None}}  # fmt: skip
    artifacts.write_atomic(os.path.join(args.out, gate.PROVISIONAL), artifacts.canonical_json(old))
    led = Ledger(args.ledger, 1)
    i, _ = led.reserve("gate 20261103T000000Z", 1, 60)
    led.settle(i, 1, "ok")
    final = gate.recover(str(root))
    assert final["pass"] is False and final["rule"] == old["rule"] and "rule_id" not in final
    with pytest.raises(gate.Refused, match="not a complete valid"):
        gate.settled_attempt(args.gate_dir)


# ---------------------------------------------------------------- compare


@pytest.mark.parametrize("b, c, p", [(0, 0, 1.0), (5, 0, 0.0625), (3, 1, 0.625), (1, 3, 0.625), (4, 4, 1.0)])
def test_gate_mcnemar_by_hand(b, c, p):
    assert gate.mcnemar_exact(b, c) == pytest.approx(p, abs=1e-15)


def test_gate_join_uses_the_candidate_seat():
    """A swapped colour would report two discordant slots here."""
    logs = [[0, True, 1.0, []], [0, False, -1.0, []], [1, True, 0.0, []], [1, False, 1.0, []]]
    solver = {(0, True): "win", (0, False): "loss", (1, True): "draw", (1, False): "win"}
    r = gate.join(logs, solver)
    assert r["candidate_only_win_slots"] == [] and r["solver_only_win_slots"] == []
    swapped = {(j, not first): v for (j, first), v in solver.items()}
    assert gate.join(logs, swapped)["candidate_only_win_slots"] == [[0, "candidate_first"], [1, "candidate_second"]]


def solver_slots_binary():
    root = pathlib.Path(__file__).parents[2]
    if shutil.which("cargo") is None:
        pytest.skip("no cargo")
    subprocess.run(["cargo", "build", "-q", "--release", "--locked", "-p", "tessera-rl-connect4-ref", "--bin", "solver_slots"],
                   cwd=root, check=True)  # fmt: skip
    out = subprocess.run(["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=root, check=True,
                         capture_output=True, text=True).stdout  # fmt: skip
    return str(pathlib.Path(json.loads(out)["target_directory"]) / "release" / "solver_slots")


def test_solver_slots_names_the_candidate_seat_on_held_out_rows(tmp_path):
    """The first openings' rows from the stored per-slot run; openings 0 and 1
    differ by seat, so a swapped colour fails."""
    lines = [x for x in HELDOUT.read_text().splitlines() if x and not x.startswith("#")][:3]
    fixture = tmp_path / "openings.txt"
    fixture.write_text("\n".join(lines) + "\n")
    out = tmp_path / "slots.tsv"
    subprocess.run([solver_slots_binary(), str(fixture), str(out), "1"], check=True)
    assert out.read_text() == ("0\tcandidate_first\twin\n0\tcandidate_second\tdraw\n1\tcandidate_first\twin\n"
                               "1\tcandidate_second\tloss\n2\tcandidate_first\twin\n2\tcandidate_second\twin\n")  # fmt: skip


@pytest.mark.skipif("TESSERA_SOLVER_SLOTS_FULL" not in os.environ, reason="minutes of CPU: run by hand")
def test_solver_slots_all_held_out_rows_match_the_digest(tmp_path):
    """All 400 rows, per slot, against the digest of the canonical rows of the
    stored run (held-out only, no line data)."""
    out = tmp_path / "slots.tsv"
    subprocess.run([solver_slots_binary(), str(HELDOUT), str(out), "5"], check=True)
    assert artifacts.sha256_file(str(out)) == "519ab797c610fc09f71f568399f9ecc0acf0c4aed2239cd3c58f09b22a49ac19"


def solver_rows(wins):
    rows = []
    for j in range(500):
        for k, seat in enumerate(("candidate_first", "candidate_second")):
            rows.append(f"{j}\t{seat}\t{'win' if 2 * j + k < wins else 'loss'}\n")
    return "".join(rows)


class FakeClock:
    def left(self):
        return 30.0


@pytest.fixture
def compared(tmp_path, done_root, as_formal, bot, monkeypatch):
    """A settled formal attempt of 817 wins and a scripted solver binary."""
    root, _, sha, _ = selected_root(tmp_path, done_root)
    args, _ = preflight_and_claim(formal_argv(root, sha, bot))
    fake_bot(monkeypatch, 817)
    assert finish_attempt(args)["pass"] is True
    rows = tmp_path / "rows.tsv"
    rows.write_text(solver_rows(815))
    binary = tmp_path / "solver_slots"
    binary.write_text(f'#!/bin/sh\ncat "{rows}" > "$2"\n')
    binary.chmod(0o755)
    monkeypatch.setattr(gate, "SOLVER_SLOTS_SHA256", artifacts.sha256_file(str(binary)))
    monkeypatch.setattr(gate, "SOLVER_SOURCES", {})
    ns = argparse.Namespace(solver_slots_bin=str(binary), solver_slots_bin_sha256=gate.SOLVER_SLOTS_SHA256, threads=1)
    return ns, args.gate_dir, rows, binary, root


def comparison(directory):
    return json.loads((pathlib.Path(directory) / "run" / "comparison.json").read_text())


def test_gate_compare_publishes_and_leaves_gate_json(compared):
    ns, directory, *_ = compared
    gate_json = pathlib.Path(directory) / "run" / "gate.json"
    before = gate_json.read_bytes()
    assert gate.compare_work(ns, directory, FakeClock()) == 0
    r = comparison(directory)
    assert r["outcome"] == "complete" and r["solver_wins"] == 815 and r["matches_published_815"] is True
    assert r["candidate_only_win_slots"] == [[407, "candidate_second"], [408, "candidate_first"]]
    assert r["solver_only_win_slots"] == [] and r["mcnemar_exact_two_sided_nominal"] == 0.5
    assert r["inputs"]["log_sha256"] and r["inputs"]["gate_json_sha256"] == artifacts.sha256(before)
    assert gate_json.read_bytes() == before


def test_gate_compare_with_another_solver_total_is_invalid_and_keeps_the_bar(compared):
    ns, directory, rows, *_ = compared
    before = (pathlib.Path(directory) / "run" / "gate.json").read_bytes()
    rows.write_text(solver_rows(814))
    assert gate.compare_work(ns, directory, FakeClock()) == 0
    r = comparison(directory)
    assert r["outcome"] == "invalid" and r["matches_published_815"] is False and "mcnemar_exact_two_sided_nominal" not in r
    assert (pathlib.Path(directory) / "run" / "gate.json").read_bytes() == before


@pytest.mark.parametrize("damage", ["short", "duplicate", "outcome", "seat"])
def test_gate_compare_with_bad_rows_is_invalid(compared, damage):
    ns, directory, rows, *_ = compared
    good = solver_rows(815).splitlines(keepends=True)
    bad = {"short": good[:-1], "duplicate": good[:-1] + good[:1],
           "outcome": good[:-1] + ["499\tcandidate_second\tmaybe\n"],
           "seat": good[:-1] + ["499\tfirst\twin\n"]}  # fmt: skip
    rows.write_text("".join(bad[damage]))
    assert gate.compare_work(ns, directory, FakeClock()) == 0
    assert comparison(directory)["outcome"] == "invalid"


def test_gate_compare_failed_solver_publishes_nothing(compared):
    ns, directory, _, binary, _ = compared
    binary.write_text("#!/bin/sh\nexit 3\n")
    assert gate.compare_work(ns, directory, FakeClock()) == 1
    assert not (pathlib.Path(directory) / "run" / "comparison.json").exists()


@pytest.mark.parametrize("damage", ["log-bytes", "log-slots", "counts", "invalid-attempt", "complete-flag-only", "no-gate-json", "digest"])
def test_gate_compare_refuses_an_unusable_attempt(compared, damage, monkeypatch):
    ns, directory, *_ = compared
    run = pathlib.Path(directory) / "run"
    report = json.loads((run / "gate.json").read_text())
    log = json.loads((run / "gate-log.json").read_text())

    def write_log(rows):
        data = artifacts.canonical_json(rows)
        (run / "gate-log.json").write_bytes(data)
        (run / "gate.json").write_bytes(artifacts.canonical_json({**report, "log_sha256": artifacts.sha256(data)}))

    if damage == "log-bytes":
        (run / "gate-log.json").write_bytes(b"[]")
    elif damage == "log-slots":
        write_log(log[:-1] + log[:1])
    elif damage == "counts":
        log[-1][2] = 1.0
        write_log(log)
    elif damage == "invalid-attempt":
        (run / "gate.json").write_bytes(artifacts.canonical_json({**report, "attempt_invalid_because": ["x"]}))
    elif damage == "complete-flag-only":
        (run / "gate.json").write_bytes(artifacts.canonical_json({**report, "exit_code": 137, "complete": True}))
    elif damage == "no-gate-json":
        (run / "gate.json").unlink()
    else:
        monkeypatch.setattr(gate, "EVAL_SHA256", "0" * 64)
    with pytest.raises(gate.Refused):
        gate.compare_work(ns, directory, FakeClock())
    assert not (run / "comparison.json").exists()


def compare_argv(root, ns):
    return ["compare", "--root", str(root), "--solver-slots-bin", ns.solver_slots_bin, "--solver-slots-bin-sha256",
            ns.solver_slots_bin_sha256, "--threads", "1", "--wall-cap-s", "60", "--budget-core-hours", "1"]  # fmt: skip


def test_gate_compare_refuses_until_pinned_and_a_wrong_binary(compared, monkeypatch):
    ns, _, _, binary, root = compared
    pinned = gate.SOLVER_SLOTS_SHA256
    monkeypatch.setattr(gate, "SOLVER_SLOTS_SHA256", None)
    with pytest.raises(gate.Refused, match="until an approved build"):
        gate.main(compare_argv(root, ns))
    monkeypatch.setattr(gate, "SOLVER_SLOTS_SHA256", pinned)
    binary.write_text("#!/bin/sh\n")
    with pytest.raises(gate.Refused, match="not the pinned one"):
        gate.main(compare_argv(root, ns))


def test_gate_compare_rerun_is_idempotent_and_never_replaces(compared):
    ns, directory, _, _, root = compared
    assert gate.compare_work(ns, directory, FakeClock()) == 0
    before = (pathlib.Path(directory) / "run" / "comparison.json").read_bytes()
    assert gate.main(compare_argv(root, ns)) == 0
    assert (pathlib.Path(directory) / "run" / "comparison.json").read_bytes() == before
    doc = json.loads(before)
    doc["inputs"]["checkpoint"] = "other"
    (pathlib.Path(directory) / "run" / "comparison.json").write_bytes(artifacts.canonical_json(doc))
    with pytest.raises(gate.Refused, match="never replaced"):
        gate.main(compare_argv(root, ns))
