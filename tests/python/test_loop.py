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

import numpy as np
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
    """The logical identities of a loop root: data, checkpoints, looks."""
    lp = loop.Loop(str(root))
    entries = lp.ledger.entries()
    gens = lp.chain(entries)
    out = [
        tuple(g[p]["identity"].get(k) if g[p] else None for p, k in
              (("selfplay", "games_digest"), ("train", "checkpoint_sha256"), ("look", "log_sha256")))
        for g in gens
    ]  # fmt: skip
    looks = [(e["generation"], e["checkpoint"]) for e in lp.looks(gens, entries)]
    return out, looks


def run_loop(root, crash=None, inprocess=True):
    env = {k: v for k, v in os.environ.items() if k != "PILOT_INHERITED_THREAD_VARS"}
    if inprocess:
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
    # A test run is never formal: no eligible look, no selection, no gate.
    assert lp.cfg["formal"] is False
    m = lp.select()
    assert m["selected"] is None and m["formal"] is False
    assert m["final_checkpoint"]["generation"] == 2


@pytest.mark.parametrize(
    "crash",
    ["during-selfplay", "output-before-settle-selfplay", "after-selfplay", "during-train",
     "output-before-settle-train", "after-train", "during-look", "output-before-settle-look", "after-look",
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
    assert [e["generation"] for e in looks] == [1, 2], "one audit entry per generation"
    if crash.startswith(("during-", "output-before-settle")):
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


def copy_root(src, dst):
    shutil.copytree(src, dst)
    return dst


@pytest.mark.parametrize("change", ["shard", "c0", "checkpoint", "look-log"])
def test_loop_refuses_a_changed_committed_input_after_completion(tmp_path, clean_run, change):
    root = copy_root(clean_run[0], tmp_path / "root")
    target = {
        "shard": next((root / "gen-000" / "selfplay-0").glob("shard-*.bin")),
        "c0": root / "c0" / "C.pt",
        "checkpoint": root / "gen-000" / "train-0" / "C.pt",
        "look-log": root / "gen-001" / "look-0" / "look-log.json",
    }[change]
    target.write_bytes(target.read_bytes() + b"x")
    r = run_loop(root)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["stop"] == "incomplete input"
    # select refuses too.
    with pytest.raises(loop.Stop, match="incomplete input"):
        loop.Loop(str(root)).select()


def test_loop_refuses_a_second_owner(tmp_path, clean_run):
    root = copy_root(clean_run[0], tmp_path / "root")
    with open(root / "loop.lock", "a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        r = run_loop(root)
        assert r.returncode != 0 and "another loop" in r.stderr
        with pytest.raises(SystemExit, match="another loop"):
            loop.Loop(str(root)).select()


@pytest.mark.parametrize("damage", ["failed", "deleted"])
def test_loop_refuses_descriptors_without_their_ok_settlement(tmp_path, clean_run, damage):
    import assets

    root = copy_root(clean_run[0], tmp_path / "root")
    ledger = root / "ledger.json"
    if damage == "failed":
        doc = json.loads(ledger.read_text())
        for e in doc["entries"]:
            if e["run"].split("-")[1] in ("selfplay", "train"):
                e["status"] = "failed: exit 1"
        ledger.write_text(json.dumps(doc))
    else:
        ledger.unlink()
    r = run_loop(root)
    assert r.returncode == 0, r.stderr
    last = json.loads(r.stdout.strip().splitlines()[-1])
    assert last["stop"] == "incomplete input" and "settlement" in last["inputs"]["reason"]
    with pytest.raises(loop.Stop, match="incomplete input"):
        loop.Loop(str(root)).select()
    with pytest.raises(SystemExit, match="broken history"):
        assets.pack(str(root), str(tmp_path / "out"), "p2-m3-train-test")


def test_loop_look_reference_needs_its_settled_origin(tmp_path, clean_run):
    root = copy_root(clean_run[0], tmp_path / "root")
    lp = loop.Loop(str(root))
    gens = lp.chain(lp.ledger.entries())
    gens[1]["c_k1"] = gens[0]["look"]["identity"]["checkpoint_sha256"]
    shutil.rmtree(root / "gen-001" / "look-0")
    assert lp.reuse_look(gens, 1)
    doc = json.loads((root / "ledger.json").read_text())
    for e in doc["entries"]:
        if e["run"].startswith("g000-look-0 "):
            e["status"] = "failed: exit 1"
    (root / "ledger.json").write_text(json.dumps(doc))
    with pytest.raises(loop.Stop, match="incomplete input"):
        lp.resolve(1, "look", lp.ledger.entries())


def test_loop_asset_operations_hold_the_owner_lock(tmp_path, clean_run):
    import assets

    root = copy_root(clean_run[0], tmp_path / "root")
    out = tmp_path / "out"
    m = assets.pack(str(root), str(out), "p2-m3-train-test")
    sp = next(a for a in m["assets"] if a["attempt"] == "g000-selfplay-0")
    shards = sorted((root / "gen-000" / "selfplay-0").glob("shard-*"))
    with open(root / "loop.lock", "a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        for op in (lambda: assets.pack(str(root), str(out), "p2-m3-train-test"),
                   lambda: assets.evict(str(root), str(out / sp["name"])),
                   lambda: assets.restore(str(root), str(out / sp["name"]))):  # fmt: skip
            with pytest.raises(SystemExit, match="another loop"):
                op()
    assert sorted((root / "gen-000" / "selfplay-0").glob("shard-*")) == shards
    # Idle: evict then restore succeed, and an interrupted eviction (mark
    # written, files partly left) resumes by running it again.
    assets.evict(str(root), str(out / sp["name"]))
    assert not list((root / "gen-000" / "selfplay-0").glob("shard-*"))
    assets.restore(str(root), str(out / sp["name"]))
    assert sorted((root / "gen-000" / "selfplay-0").glob("shard-*")) == shards
    assets.evict(str(root), str(out / sp["name"]))
    assets.evict(str(root), str(out / sp["name"]))
    assets.restore(str(root), str(out / sp["name"]))
    assert "evicted" not in json.loads((root / "gen-000" / "selfplay-0" / "attempt.json").read_text())


def test_loop_selection_comes_from_evidence_not_the_log(tmp_path, clean_run):
    root = copy_root(clean_run[0], tmp_path / "root")
    # A forged audit entry for a checkpoint with no look: ignored.
    lp = loop.Loop(str(root))
    lp.append_selection({"type": "look", "generation": 9, "checkpoint": "forged", "wins": 400, "eligible": True})
    assert lp.select()["selected"] is None
    looks = lp.looks(lp.chain(lp.ledger.entries()), lp.ledger.entries())
    assert {e["checkpoint"] for e in looks} != {"forged"} and len(looks) == 2


def test_loop_evicted_window_input_must_be_restored_then_resumes(tmp_path, bot, clean_run):
    import assets

    root = make_root(tmp_path, bot)
    r = run_loop(root, crash="after-look")  # generation 0 done, generation 1 not started
    assert r.returncode == 137
    out = tmp_path / "assets"
    m = assets.pack(str(root), str(out), "p2-m3-train-test")
    sp = next(a for a in m["assets"] if a["attempt"] == "g000-selfplay-0")
    assert sp["command"][0] == "selfplay.py"
    assets.evict(str(root), str(out / sp["name"]))
    assert not list((root / "gen-000" / "selfplay-0").glob("shard-*"))
    r = run_loop(root)
    assert json.loads(r.stdout.strip().splitlines()[-1])["stop"] == "incomplete input"
    assets.restore(str(root), str(out / sp["name"]))
    r = run_loop(root)
    assert json.loads(r.stdout.strip().splitlines()[-1])["stop"] == "K_max"
    assert identities(root) == clean_run[2]
    # The manifest only grows; a substitute is refused.
    m2 = assets.pack(str(root), str(out), "p2-m3-train-test")
    assert {a["name"] for a in m["assets"]} <= {a["name"] for a in m2["assets"]}
    bad = out / sp["name"]
    bad.write_bytes(bad.read_bytes() + b"x")
    with pytest.raises(artifacts.Incomplete, match="broken"):
        assets.verify(assets.load_manifest(str(root)), str(bad))


def test_loop_reuses_a_look_at_a_repeated_checkpoint(tmp_path, clean_run):
    root = copy_root(clean_run[0], tmp_path / "root")
    lp = loop.Loop(str(root))
    gens = lp.chain(lp.ledger.entries())
    # Generation 1 trained a checkpoint identical to generation 0's (forced).
    gens[1]["c_k1"] = gens[0]["look"]["identity"]["checkpoint_sha256"]
    gens[1]["look"] = None
    shutil.rmtree(root / "gen-001" / "look-0")
    assert lp.reuse_look(gens, 1)
    desc = json.loads((root / "gen-001" / "look-ref" / "attempt.json").read_text())
    assert desc["ref"] == "g000-look-0"


def test_loop_supervised_runner_path(tmp_path, bot):
    """One generation through the real supervised tools (no in-process)."""
    root = make_root(tmp_path, bot, **{"--k-max": 1})
    r = run_loop(root, inprocess=False)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["stop"] == "K_max"
    entries = loop.Loop(str(root)).ledger.entries()
    assert [e["run"].split()[0] for e in entries] == ["g000-selfplay-0", "g000-train-0", "g000-look-0"]
    assert all(e["status"] == "ok" for e in entries)


def look_entry(g, wins, eligible=True):
    return {"generation": g, "checkpoint": f"sha{g}", "wins": wins, "eligible": eligible}


def test_loop_selection_rule_ties_to_the_later_generation():
    eligible = [look_entry(1, 300), look_entry(2, 350), look_entry(3, 350)]
    assert loop.choose(eligible) == {"generation": 3, "checkpoint": "sha3", "wins": 350}
    assert loop.choose([]) is None


def test_loop_early_stop_needs_two_consecutive_eligible_generations(tmp_path, clean_run):
    root = copy_root(clean_run[0], tmp_path / "root")
    lp = loop.Loop(str(root))
    lp.cfg["early_stop_wins"] = 0
    gens, entries = lp.chain(lp.ledger.entries()), lp.ledger.entries()
    # Not formal: no look is eligible, so no early stop.
    lp.early_stop(gens, entries)
    lp.looks = lambda gens, entries: [look_entry(1, 381), look_entry(2, 379), look_entry(3, 385), look_entry(4, 380)]
    lp.cfg["early_stop_wins"] = 380
    with pytest.raises(loop.Stop) as stop:
        lp.early_stop(gens, entries)
    assert stop.value.reason == "early" and stop.value.inputs["generations"] == [3, 4]


def test_loop_formal_flag_needs_the_registered_configuration():
    cfg = dict(loop.FORMAL)
    assert loop.is_formal(cfg)
    for key, bad in (("k_max", 2), ("steps", 2), ("c0_sha256", "x"), ("any_openings", True), ("games", 4)):
        assert not loop.is_formal({**cfg, key: bad}), key


def test_loop_admission_against_the_ledger_with_k_max_lifted(tmp_path):
    """With about 4.4 settled core-hours per generation, generation 30 is
    admitted and the one that would cross 200 is refused (the real config
    stops at K_max = 30 before that)."""
    root = tmp_path / "adm"
    root.mkdir()
    (root / "c0").mkdir()
    c0 = str(root / "c0" / "C.pt")
    assert train_step.main(["init", "--size", "tiny", "--seed", "7", "--out", c0, "--source-sha", "t"]) == 0
    c0sha = ckpt.manifest(str(root / "c0" / "C.pt"))["sha256"]
    cfg = {
        "c0_sha256": c0sha, "k_max": 60, "budget_core_hours": 200.0, "selfplay_cores": 32, "selfplay_wall_s": 900,
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


def test_loop_assets_are_content_addressed(tmp_path, clean_run):
    import assets

    a = copy_root(clean_run[0], tmp_path / "a")
    b = copy_root(clean_run[0], tmp_path / "b")
    m1 = assets.pack(str(a), str(tmp_path / "out-a"), "p2-m3-train-test")
    m2 = assets.pack(str(b), str(tmp_path / "out-b"), "p2-m3-train-test")
    assert [x["sha256"] for x in m1["assets"]] == [x["sha256"] for x in m2["assets"]]
    assert len(m1["assets"]) == 6 and m1["tag"] == "p2-m3-train-test"
    with pytest.raises(SystemExit, match="release"):
        assets.pack(str(a), str(tmp_path / "out-c"), "another-tag")


def test_loop_look_is_eligible_only_with_the_registered_bot_and_a_medium_net(monkeypatch, bot):
    import look

    args = look.parse(["--checkpoint", "x", "--bot", bot, "--bot-sha256", "y", "--out", "o", "--source-sha", "t",
                       "--ledger", "l", "--budget-core-hours", "1", "--wall-cap-s", "1"])  # fmt: skip
    observed = {"registered": True, "image_digest": "img"}
    ok = ("botsha", "medium", look.HELDOUT_SHA256, args, 200)
    assert not look.is_eligible(observed, *ok), "no registered bot yet"
    monkeypatch.setitem(look.REGISTERED_BOTS, "img", "botsha")
    assert look.is_eligible(observed, *ok)
    assert not look.is_eligible(observed, "other", "medium", look.HELDOUT_SHA256, args, 200)
    assert not look.is_eligible(observed, "botsha", "tiny", look.HELDOUT_SHA256, args, 200)
    assert not look.is_eligible({**observed, "registered": False}, *ok)
    assert not look.is_eligible(observed, "botsha", "medium", look.HELDOUT_SHA256, args, 1)


def test_loop_look_refuses_an_unexpected_bot_binary(tmp_path, bot):
    import look

    c = tmp_path / "C.pt"
    assert train_step.main(["init", "--size", "tiny", "--seed", "1", "--out", str(c), "--source-sha", "t"]) == 0
    out = tmp_path / "o"
    out.mkdir()
    args = look.parse(["--checkpoint", str(c), "--bot", bot, "--bot-sha256", "0" * 64, "--out", str(out),
                       "--source-sha", "t", "--ledger", str(tmp_path / "l"), "--budget-core-hours", "1",
                       "--wall-cap-s", "1"])  # fmt: skip
    with pytest.raises(SystemExit, match="not the expected"):
        look.work(args, None)


def test_loop_lazy_augmentation_matches_the_materialized_pool():
    import augment

    rng = np.random.default_rng(1)
    planes = (rng.random((5, 2, 6, 7)) < 0.3).astype(np.uint8)
    policy = rng.dirichlet(np.ones(7), size=5)
    z = rng.choice([-1.0, 0.0, 1.0], size=5)
    p, pol, v = augment.augment(planes.astype(np.float32), policy, z)
    model = __import__("net").build("tiny", seed=1)
    calls = {}

    def capture(x):
        calls.setdefault("x", []).append(x.clone())
        return orig(x)

    orig = model.forward
    model.forward = capture
    losses, counts = train_step.train_window(model, planes, policy, z, generation=0, base=7, steps=3, batch=4)
    idx = np.random.Generator(np.random.PCG64(seeds.sample(7, 0))).integers(0, 10, size=(3, 4))
    for s, x in enumerate(calls["x"]):
        assert torch.equal(x, torch.from_numpy(p[idx[s]]))


def test_loop_formal_training_needs_exactly_the_registered_steps(tmp_path):
    args = train_step.parse(["train-window", "--checkpoint", "c", "--data", "d", "--out", str(tmp_path / "o.pt"),
                             "--generation", "0", "--base", "1", "--steps", "5", "--formal", "--source-sha", "t",
                             "--ledger", "l", "--budget-core-hours", "1", "--wall-cap-s", "1"])  # fmt: skip
    with pytest.raises(SystemExit, match="exactly"):
        train_step.work_window(args, None)
