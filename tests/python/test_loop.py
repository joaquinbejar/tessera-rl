# SPDX-License-Identifier: MIT OR Apache-2.0
"""The P2-M3 resumable loop (examples/alphazero_connect4, PR 3).

Covered:
- the registered LR schedule against its formula;
- window composition, order and seeds;
- admission against the ledger (unit, with K_max lifted), and the real
  configuration stopping with reason K_max, not budget;
- the selection rule and log deduplication on scripted logs, and no
  selection without an eligible look;
- a tiny end-to-end loop, and resumes after crashes at each boundary
  reproducing the logical identities;
- a changed input refused, and a second owner refused.
"""

import fcntl
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

EXAMPLE = pathlib.Path(__file__).parents[2] / "examples" / "alphazero_connect4"
sys.path.insert(0, str(EXAMPLE))

import artifacts  # noqa: E402
import ckpt  # noqa: E402
import loop  # noqa: E402
import schedule  # noqa: E402
import seeds  # noqa: E402
import train_step  # noqa: E402
from ledger import Ledger  # noqa: E402


def test_loop_lr_schedule_matches_the_registered_formula():
    def formula(t):
        return 0.002 + 0.5 * (0.02 - 0.002) * (1 + math.cos(math.pi * t / 30000))

    assert schedule.HORIZON == 30000 and schedule.K_MAX == 30 and schedule.STEPS == 1000
    for t in (0, 15000, 29999, schedule.global_step(7, 123)):
        assert math.isclose(schedule.lr(t), formula(t), rel_tol=1e-15)
    assert math.isclose(schedule.lr(0), 0.02, rel_tol=1e-15)
    assert math.isclose(schedule.lr(15000), 0.011, rel_tol=1e-12)
    assert 0.002 < schedule.lr(29999) < 0.002 + 1e-10
    with pytest.raises(ValueError):
        schedule.lr(30000)


def test_loop_generation_and_sample_seeds_are_separate_domains():
    assert seeds.generation(20261103, 4) == seeds.derive(20261103, "generation", 4)
    values = {seeds.generation(20261103, 4), seeds.sample(20261103, 4), seeds.env(20261103, 4), seeds.shuffle(20261103, 4)}
    assert len(values) == 4


def bot_binary():
    root = pathlib.Path(__file__).parents[2] / "target"
    for p in (root / "release" / "tessera-rl-connect4-ref", root / "debug" / "tessera-rl-connect4-ref"):
        if p.exists():
            return str(p)
    if shutil.which("cargo") is None:
        pytest.skip("no connect4-ref binary and no cargo")
    subprocess.run(["cargo", "build", "-q", "-p", "tessera-rl-connect4-ref"], cwd=root.parent, check=True)
    return bot_binary()


@pytest.fixture(scope="module")
def bot():
    return bot_binary()


def make_root(tmp_path, bot, **over):
    c0 = tmp_path / "c0src" / "C0.pt"
    c0.parent.mkdir(parents=True)
    assert train_step.main(["init", "--size", "tiny", "--seed", "7", "--out", str(c0), "--source-sha", "test"]) == 0
    opts = {
        "--root": tmp_path / "root", "--c0": c0, "--bot": bot, "--source-sha": "test", "--k-max": 2,
        "--games": 2, "--sims": 2, "--batch-size": 2, "--capacity": 64, "--steps": 2, "--train-batch": 8,
        "--selfplay-cores": 1, "--selfplay-wall-s": 60, "--train-wall-s": 60, "--look-wall-s": 60,
        "--look-openings": 1, "--budget-core-hours": 1,
    }  # fmt: skip
    opts.update(over)
    argv = ["init"] + [x for k, v in opts.items() for x in (k, str(v))]
    assert loop.main(argv) == 0
    return tmp_path / "root"


def identities(root):
    """The logical identities of a loop root: checkpoints, data, looks."""
    lp = loop.Loop(str(root))
    entries = lp.ledger.entries()
    out = []
    for k in range(lp.cfg["k_max"]):
        sp, _ = lp.phase_state(k, "selfplay", entries)
        tr, _ = lp.phase_state(k, "train", entries)
        lk, _ = lp.phase_state(k, "look", entries)
        out.append(
            (
                artifacts.load_manifest(sp)[0]["games_digest"] if sp else None,
                ckpt.manifest(os.path.join(tr, "C.pt"))["sha256"] if tr else None,
                json.load(open(os.path.join(lk, "look.json")))["log_sha256"] if lk else None,
            )
        )
    looks = [(e["generation"], e["checkpoint"]) for e in lp.selection() if e["type"] == "look"]
    return out, looks


def run_loop(root, crash=None):
    env = {k: v for k, v in os.environ.items() if k != "PILOT_INHERITED_THREAD_VARS"}
    env["LOOP_TEST_INPROCESS"] = "1"
    if crash:
        env["LOOP_TEST_CRASH"] = crash
    return subprocess.run(
        [sys.executable, str(EXAMPLE / "loop.py"), "run", "--root", str(root)],
        capture_output=True, text=True, timeout=300, env=env,
    )  # fmt: skip


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory, bot):
    root = make_root(tmp_path_factory.mktemp("clean"), bot)
    r = run_loop(root)
    assert r.returncode == 0, r.stderr
    return root, json.loads(r.stdout.strip().splitlines()[-1]), identities(root)


def test_loop_tiny_end_to_end_stops_at_k_max(clean_run):
    root, result, want = clean_run
    assert result["stop"] == "K_max"
    assert all(all(x is not None for x in g) for g in want[0])
    assert [g for g, _ in want[1]] == [1, 2]
    # The window of generation 1 is generations 0 and 1, oldest first.
    lp = loop.Loop(str(root))
    c2 = ckpt.manifest(str(root / "gen-001" / "train-0" / "C.pt"))
    assert len(c2["data_manifests"]) == 2 and c2["t_first"] == schedule.global_step(1, 0)
    assert c2["sample_seed"] == seeds.sample(lp.cfg["base"], 1)
    assert c2["parent_sha256"] == ckpt.manifest(str(root / "gen-000" / "train-0" / "C.pt"))["sha256"]
    stops = [e for e in lp.selection() if e["type"] == "stop"]
    assert [e["reason"] for e in stops] == ["K_max"]
    # Test looks are never eligible: no selection, so no gate.
    assert lp.select()["selected"] is None


@pytest.mark.parametrize(
    "crash",
    ["during-selfplay", "after-selfplay", "during-train", "after-train", "during-look", "after-look",
     "after-selection-append", "before-state"],
)  # fmt: skip
def test_loop_resume_after_a_crash_reproduces_the_logical_identities(tmp_path, bot, clean_run, crash):
    _, _, want = clean_run
    root = make_root(tmp_path, bot)
    r = run_loop(root, crash=crash)
    assert r.returncode == 137, r.stderr
    r = run_loop(root)
    assert r.returncode == 0, r.stderr
    assert identities(root) == want
    looks = [e for e in loop.Loop(str(root)).selection() if e["type"] == "look"]
    assert len({e["checkpoint"] for e in looks}) == len(looks) == 2
    if crash.startswith("during-"):
        # The interrupted attempt was charged and superseded by a new one.
        entries = loop.Loop(str(root)).ledger.entries()
        assert any(e["status"].startswith("interrupted") for e in entries)


def test_loop_incomplete_attempt_is_redone_in_a_new_directory(tmp_path, bot):
    root = make_root(tmp_path, bot, **{"--k-max": 1})
    # A killed self-play attempt: output without a manifest.
    d = root / "gen-000" / "selfplay-0"
    d.mkdir(parents=True)
    (d / "shard-w000-r0000.bin").write_bytes(b"partial")
    r = run_loop(root)
    assert r.returncode == 0, r.stderr
    assert (root / "gen-000" / "selfplay-1" / "manifest.json").exists()
    assert (d / "shard-w000-r0000.bin").read_bytes() == b"partial", "never overwritten"


def test_loop_refuses_a_changed_input_and_a_second_owner(tmp_path, bot):
    root = make_root(tmp_path, bot, **{"--k-max": 1})
    assert run_loop(root).returncode == 0
    # A changed shard makes its self-play attempt incomplete: training of a
    # later generation could not use it, and the cache rebuild sees it.
    lp = loop.Loop(str(root))
    shard = next((root / "gen-000" / "selfplay-0").glob("shard-*.bin"))
    shard.write_bytes(shard.read_bytes() + b"x")
    done, _ = lp.phase_state(0, "selfplay", lp.ledger.entries())
    assert done is None
    with open(root / "loop.lock", "a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        r = run_loop(root)
    assert r.returncode != 0 and "another loop" in r.stderr


def scripted(tmp_path, entries, k_max=30):
    root = tmp_path / "s"
    root.mkdir(parents=True)
    cfg = {"k_max": k_max, "budget_core_hours": 200.0, "early_stop_wins": 380}
    (root / "config.json").write_text(json.dumps(cfg))
    with open(root / "selection.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return loop.Loop(str(root))


def look_entry(g, wins, eligible=True, ok=True, sha=None):
    return {"type": "look", "generation": g, "checkpoint": sha or f"sha{g}", "wins": wins, "eligible": eligible, "ledger_ok": ok}


def test_loop_selection_rule_ties_to_the_later_generation_and_needs_eligible_looks(tmp_path):
    entries = [look_entry(1, 300), look_entry(2, 350), look_entry(3, 350)]
    entries += [look_entry(4, 399, eligible=False), look_entry(5, 399, ok=False)]
    lp = scripted(tmp_path, entries)
    m = lp.select()
    assert m["selected"] == {"generation": 3, "checkpoint": "sha3", "wins": 350}
    assert [e["generation"] for e in m["eligible_looks"]] == [1, 2, 3]
    lp2 = scripted(tmp_path / "b", [look_entry(1, 300, eligible=False)])
    assert lp2.select()["selected"] is None


def test_loop_early_stop_needs_two_consecutive_eligible_generations(tmp_path):
    lp = scripted(tmp_path, [look_entry(1, 381), look_entry(2, 379), look_entry(3, 385)])
    assert lp.early_stop(1) is None  # generations 1, 2: 381, 379
    with pytest.raises(loop.Stop) as stop:
        lp.append_selection(look_entry(4, 380))
        lp.early_stop(3)  # generations 3, 4: 385, 380
    assert stop.value.reason == "early" and stop.value.inputs["wins"] == [385, 380]


def test_loop_admission_against_the_ledger_with_k_max_lifted(tmp_path):
    """With about 4.4 settled core-hours per generation, generation 30 is
    admitted and the one that would cross 200 is refused (the real config
    stops at K_max = 30 before that)."""
    root = tmp_path / "adm"
    root.mkdir()
    cfg = {
        "k_max": 60, "budget_core_hours": 200.0, "selfplay_cores": 32, "selfplay_wall_s": 900,
        "train_wall_s": 1200, "look_wall_s": 900, "early_stop_wins": 380,
    }  # fmt: skip
    (root / "config.json").write_text(json.dumps(cfg))
    res = loop.reservations(cfg)
    assert sum(res.values()) == 30900  # exactly, never a rounded 8.58 h
    led = Ledger(str(root / "ledger.json"), 200.0)

    def admitted(generations_done):
        remaining = (200.0 - 4.4 * generations_done) * 3600
        return sum(res.values()) <= remaining

    assert admitted(29), "generation 30 is admitted"
    first_refused = next(g for g in range(60) if not admitted(g))
    assert first_refused == 44
    # The loop's own check reads the ledger: seed it and ask.
    for g in range(44):
        i, _ = led.reserve(f"g{g:03d}-selfplay-0", 1, 4.4 * 3600)
        led.settle(i, 4.4 * 3600, "ok")
    lp = loop.Loop(str(root))
    assert lp.remaining_core_s(lp.ledger.entries()) < sum(res.values())
    with pytest.raises(loop.Stop) as stop:
        lp.step()
    assert stop.value.reason == "budget"


def test_loop_real_configuration_stops_at_k_max_not_budget():
    res = loop.reservations({"selfplay_cores": 32, "selfplay_wall_s": 900, "train_wall_s": 1200, "look_wall_s": 900})
    projected = 4.4  # core-hours per generation, from the PR 2 run
    remaining = 200.0 - projected * (schedule.K_MAX - 1)
    assert remaining * 3600 >= sum(res.values()), "the last generation is fundable"
    assert schedule.K_MAX == 30


def test_loop_assets_are_content_addressed_and_substitutes_refused(tmp_path, clean_run):
    import assets

    root, _, _ = clean_run
    m1 = assets.pack(str(root), str(tmp_path / "a"), "p2-m3-train-test")
    m2 = assets.pack(str(root), str(tmp_path / "b"), "p2-m3-train-test")
    assert [a["sha256"] for a in m1["assets"]] == [a["sha256"] for a in m2["assets"]]
    assert len(m1["assets"]) == 6 and m1["tag"] == "p2-m3-train-test"
    first = tmp_path / "a" / m1["assets"][0]["name"]
    assets.verify(str(tmp_path / "a" / "assets-manifest.json"), str(first))
    first.write_bytes(first.read_bytes() + b"x")
    with pytest.raises(artifacts.Incomplete, match="broken"):
        assets.verify(str(tmp_path / "a" / "assets-manifest.json"), str(first))
