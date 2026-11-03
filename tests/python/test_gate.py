# SPDX-License-Identifier: MIT OR Apache-2.0
"""The P2-G1 gate tool (examples/alphazero_connect4/gate.py)."""

import ast
import builtins
import json
import os
import pathlib
import shutil
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

EXAMPLE = pathlib.Path(__file__).parents[2] / "examples" / "alphazero_connect4"
REPO = EXAMPLE.parents[1]
sys.path.insert(0, str(EXAMPLE))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import artifacts  # noqa: E402
import gate  # noqa: E402
import loop  # noqa: E402
import train_step  # noqa: E402
from test_loop import bot_binary, make_root, run_loop  # noqa: E402

HELDOUT = REPO / "compat" / "connect4" / "openings-heldout.txt"
IMAGE = "rust:1.99.1@sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137"


@pytest.fixture(scope="module")
def bot():
    return bot_binary()


# ------------------------------------------------------------------ rule


@pytest.mark.parametrize(
    "wins, count_ok, lb, ok",
    [(1000, True, None, True), (919, True, 0.9004424261087147, True), (918, True, 0.8993524099554585, False),
     (900, True, None, False), (899, False, None, False), (0, False, None, False)],
)  # fmt: skip
def test_gate_pass_rule_needs_900_wins_and_the_two_sided_lower_bound(wins, count_ok, lb, ok):
    r = gate.passes(wins, 1000)
    assert r["pass"] is ok and r["wins_at_least_900"] is count_ok
    assert r["lower_bound_at_least_0.90"] is (wins >= 919)
    if lb is not None:
        assert r["wilson_lower_z1.96"] == pytest.approx(lb, abs=1e-15)
    short = gate.passes(999, 999)
    assert short["pass"] is None and short["invalid_because"], "a short gate has no verdict"


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
    return [str(a) for a in ("--root", root, "--selection-sha256", sha, "--bot", bot,
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
           "bot-sha", "bot-unregistered", "runtime", "small-net", "search", "budget"]  # fmt: skip


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
    with pytest.raises(SystemExit):  # Refused, or a refusing loop Stop
        preflight_and_claim(formal_argv(root, sha, bot, *extra, bot_sha=bot_sha))
    assert not (root / "gate").exists(), "a refused preflight claims nothing"
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
    for wins, expected in ((919, True), (918, False)):
        monkeypatch.setattr(gate.botmatch, "play", scripted_play(wins))
        args.out = str(tmp_path / f"out-{wins}")
        os.makedirs(args.out)
        assert gate.work(args, None) == 0
        report = json.loads((pathlib.Path(args.out) / "gate.json").read_text())
        assert report["gate_run"] is True and report["games"] == 1000
        assert report["pass"] is expected and report["rule"]["wins"] == wins
        assert report["openings_sha256_at_end"] == gate.EVAL_SHA256
        assert report["selected"]["checkpoint"] == report["checkpoint"]
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
    report = json.loads((tmp_path / "out" / "gate.json").read_text())
    assert report["gate_run"] is False and report["pass"] is None and report["selected"] is None
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
        # On the evaluation file's own file system (target/ is ignored).
        import tempfile

        (REPO / "target").mkdir(exist_ok=True)
        target = pathlib.Path(tempfile.mkdtemp(dir=REPO / "target")) / "alias.txt"
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
    assert named == {"formal_openings", "eval_sha_now", "diagnostic_openings"}
    diag = next(f for f in fns if f.name == "diagnostic_openings")
    calls = {ast.unparse(c.func) for c in ast.walk(diag) if isinstance(c, ast.Call)
             if any(isinstance(a, ast.Name) and a.id == "EVAL" for a in c.args)}  # fmt: skip
    assert calls <= {"os.stat", "os.path.realpath"}
    assert artifacts.sha256_file(gate.EVAL) == gate.EVAL_SHA256
