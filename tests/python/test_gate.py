# SPDX-License-Identifier: MIT OR Apache-2.0
"""The P2-G1 gate tool (examples/alphazero_connect4/gate.py)."""

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


@pytest.fixture(scope="module")
def bot():
    return bot_binary()


@pytest.mark.parametrize(
    "wins, ok",
    [(1000, True), (919, True), (918, False), (900, False), (899, False), (0, False)],
)
def test_gate_pass_rule_needs_900_wins_and_the_two_sided_lower_bound(wins, ok):
    r = gate.passes(wins, 1000)
    assert r["pass"] is ok
    assert r["spec_rule"] is (wins >= 900)
    # 919 is the first count whose z = 1.96 lower bound reaches 0.90.
    assert r["internal_criterion"] is (wins >= 919)
    assert not gate.passes(999, 999)["pass"], "a short gate never passes"


@pytest.fixture(scope="module")
def formal_root(tmp_path_factory, bot):
    """A completed tiny loop root, treated as formal with eligible looks
    (the registered runtime cannot exist in a test)."""
    root = make_root(tmp_path_factory.mktemp("gate"), bot)
    r = run_loop(root)
    assert r.returncode == 0, r.stderr
    return root


@pytest.fixture
def as_formal(monkeypatch):
    real = loop.Loop.looks

    def eligible_looks(self, gens, entries):
        return [{**e, "eligible": True} for e in real(self, gens, entries)]

    monkeypatch.setattr(loop, "is_formal", lambda cfg: True)
    monkeypatch.setattr(loop.Loop, "looks", eligible_looks)


def selected_root(tmp_path, formal_root):
    root = tmp_path / "root"
    shutil.copytree(formal_root, root)
    m = loop.Loop(str(root)).select()
    assert m["formal"] and m["selected"]
    path = root / "selection-manifest.json"
    return root, path, artifacts.sha256_file(str(path)), m


def test_gate_accepts_the_committed_formal_selection(tmp_path, formal_root, as_formal):
    root, path, sha, m = selected_root(tmp_path, formal_root)
    assert gate.check_selection(str(root), str(path), sha, m["selected"]["checkpoint"])["selected"] == m["selected"]


@pytest.mark.parametrize("damage", ["sha", "not-formal", "no-selection", "tampered-looks", "tampered-pick",
                                    "checkpoint", "broken-ledger", "owner"])  # fmt: skip
def test_gate_refuses_a_non_formal_or_unverified_selection(tmp_path, formal_root, as_formal, damage):
    root, path, sha, m = selected_root(tmp_path, formal_root)
    ckpt_sha = m["selected"]["checkpoint"]

    def rewrite(**change):
        doc = {**json.loads(path.read_text()), **change}
        path.write_bytes(artifacts.canonical_json(doc))
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
        other = next(e for e in m["eligible_looks"] if e["checkpoint"] != ckpt_sha)
        sha = rewrite(selected={k: other[k] for k in ("generation", "checkpoint", "wins")})
        ckpt_sha = other["checkpoint"]
    elif damage == "checkpoint":
        ckpt_sha = "f" * 64
    elif damage == "broken-ledger":
        (root / "ledger.json").unlink()
    if damage == "owner":
        import fcntl

        with open(root / "loop.lock", "a+") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            with pytest.raises(SystemExit, match="another loop"):
                gate.check_selection(str(root), str(path), sha, ckpt_sha)
        return
    with pytest.raises(gate.Refused):
        gate.check_selection(str(root), str(path), sha, ckpt_sha)


def test_gate_refuses_a_real_non_formal_loop(tmp_path, formal_root):
    root = tmp_path / "root"
    shutil.copytree(formal_root, root)
    m = loop.Loop(str(root)).select()
    path = root / "selection-manifest.json"
    with pytest.raises(gate.Refused, match="not formal"):
        gate.check_selection(str(root), str(path), artifacts.sha256_file(str(path)), m["final_checkpoint"]["checkpoint"])


def gate_args(tmp_path, bot, *extra, checkpoint=None):
    c = checkpoint or tmp_path / "C.pt"
    if not os.path.exists(c):
        assert train_step.main(["init", "--size", "tiny", "--seed", "1", "--out", str(c), "--source-sha", "t"]) == 0
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return [str(a) for a in ("--checkpoint", c, "--bot", bot, "--bot-sha256", artifacts.sha256_file(bot),
                             "--out", out, "--source-sha", "t", "--ledger", tmp_path / "gate-ledger.json",
                             "--budget-core-hours", 1, "--wall-cap-s", 120, *extra)]  # fmt: skip


def test_gate_formal_mode_refuses_before_reading_the_evaluation_openings(tmp_path, formal_root, as_formal, bot, monkeypatch):
    root, _, sha, m = selected_root(tmp_path, formal_root)
    opened = []
    real_open = builtins.open

    def spy(path, *a, **k):
        if os.path.abspath(str(path)) == os.path.abspath(gate.EVAL):
            opened.append(path)
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    # A tiny net is never a gate candidate, and the runtime is not the
    # registered one: refused, with the evaluation file never opened.
    sel = root / f"gen-{m['selected']['generation'] - 1:03d}" / "train-0" / "C.pt"
    args = gate.parse(gate_args(tmp_path, bot, "--root", root, "--selection-sha256", sha, checkpoint=sel))
    with pytest.raises(gate.Refused):
        gate.work(args, None)
    assert opened == []


def test_gate_diagnostic_mode_cannot_open_the_evaluation_file(tmp_path, bot, monkeypatch):
    real_open = builtins.open

    def guard(path, *a, **k):
        assert os.path.abspath(str(path)) != os.path.abspath(gate.EVAL), "the evaluation file was opened"
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", guard)
    args = gate.parse(gate_args(tmp_path, bot, "--diagnostic-openings", HELDOUT, "--limit", 2, "--sims", 2,
                                "--batch-size", 2, "--capacity", 64))  # fmt: skip
    assert gate.work(args, None) == 0
    report = json.loads((tmp_path / "out" / "gate.json").read_text())
    assert report["formal"] is False and report["pass"] is None and report["not_a_gate_because"]
    assert report["games"] == 4 and report["selected"] is None and report["openings_covered"] == 2
    assert sum(report["pair_wins"].values()) == 2 and 1 <= report["distinct_games"] <= 4


def test_gate_diagnostic_mode_refuses_the_evaluation_set_by_any_path(tmp_path, bot):
    copy = tmp_path / "copy.txt"
    shutil.copyfile(gate.EVAL, copy)
    for path in (gate.EVAL, copy):
        args = gate.parse(gate_args(tmp_path, bot, "--diagnostic-openings", path))
        with pytest.raises(gate.Refused, match="never plays the evaluation"):
            gate.work(args, None)
    with pytest.raises(SystemExit):
        gate.parse(gate_args(tmp_path, bot, "--diagnostic-openings", HELDOUT, "--root", tmp_path))
    with pytest.raises(SystemExit):
        gate.parse(gate_args(tmp_path, bot))  # formal needs the selection


def test_gate_is_one_shot_per_ledger(tmp_path, bot):
    from ledger import Ledger

    args = gate.parse(gate_args(tmp_path, bot, "--root", tmp_path, "--selection-sha256", "0" * 64))
    gate.one_shot(args)  # no ledger yet
    led = Ledger(args.ledger, 1)
    i, _ = led.reserve("gate 20261103T000000Z", 1, 60)
    led.settle(i, 1, "failed: exit 1")
    with pytest.raises(gate.Refused, match="already ran"):
        gate.one_shot(args)
    with pytest.raises(gate.Refused, match="already ran"):
        gate.main(gate_args(tmp_path, bot, "--root", tmp_path, "--selection-sha256", "0" * 64))


def test_gate_evaluation_openings_are_read_by_the_gate_only():
    """Only the gate plays a candidate on them. The P2-M3 pilot (PR 1)
    timed a randomly initialized net there before any training; it never
    selects or tunes, and no other tool may join it."""
    users = sorted(p.name for p in EXAMPLE.glob("*.py") if "openings-eval" in p.read_text())
    assert users == ["gate.py", "pilot.py"]
    assert "netmod.build(args.eval_size, seed=args.seed)" in (EXAMPLE / "pilot.py").read_text()
    assert artifacts.sha256_file(gate.EVAL) == gate.EVAL_SHA256
