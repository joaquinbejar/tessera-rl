# SPDX-License-Identifier: MIT OR Apache-2.0
"""The P2-M3 self-play baseline (examples/alphazero_connect4, PR 2).

Covered:
- seeds are domain-separated, W-independent and collision-checked;
- targets are mover-relative, and visit policies legal, on scripted games;
- self-play data is identical for the same seeds, and per game the same
  for W = 1 and W = 2 (full sample arrays);
- manifests are atomic, and an incomplete run is refused;
- a killed run is charged and stays incomplete;
- a tiny end-to-end run: C0, self-play, one step to C1, a head-to-head.
"""

import json
import os
import pathlib
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
import h2h  # noqa: E402
import seeds  # noqa: E402
import selfplay  # noqa: E402
import train_step  # noqa: E402
from h2h import wilson_lower  # noqa: E402


def run(script, *args, env=None):
    return subprocess.run(
        [sys.executable, str(EXAMPLE / script), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=120,
        env={k: v for k, v in {**os.environ, **(env or {})}.items() if k != "PILOT_INHERITED_THREAD_VARS"},
    )


def test_selfplay_seeds_are_domain_separated_and_collision_checked():
    a = seeds.env(1, 5)
    assert a == seeds.env(1, 5)
    assert len({seeds.env(1, 5), seeds.search(1, 5), seeds.shuffle(1, 5), seeds.env(2, 5), seeds.env(1, 6)}) == 5
    assert seeds.canonical_bytes(1, "env", 5).endswith((5).to_bytes(8, "little"))
    with pytest.raises(ValueError):
        seeds.derive(1, "worker", 0)
    # Every reserved range is enforced.
    for name, base in seeds.RESERVED:
        orig = seeds.hashlib.sha256

        class Fixed:
            value = base + 7

            def __init__(self, data):
                pass

            def digest(self):
                return self.value.to_bytes(8, "little") + bytes(24)

        seeds.hashlib.sha256 = Fixed
        try:
            with pytest.raises(seeds.SeedCollision, match=name):
                seeds.env(1, 0)
        finally:
            seeds.hashlib.sha256 = orig


def scripted(moves):
    it = iter(moves)

    def choose(board):
        col = next(it)
        visits = [0] * 7
        visits[col] = 1
        return col, visits

    return choose


def test_selfplay_targets_are_mover_relative_on_a_scripted_win_and_draw():
    win = selfplay.play(scripted([0, 1, 0, 1, 0, 1, 0]), env_seed=3)
    # Seven plies; the first player (plies 0, 2, 4, 6) wins.
    assert win["z"].tolist() == [1, -1, 1, -1, 1, -1, 1]
    draw_moves = [int(c) - 1 for c in "547125662261271266215743771576315353334444"]
    draw = selfplay.play(scripted(draw_moves), env_seed=3)
    assert len(draw["z"]) == 42 and (draw["z"] == 0).all()
    # Planes are the mover's: own stones in plane 0.
    assert win["planes"][2, 0].sum() == 1 and win["planes"][2, 1].sum() == 1


def make_c0(tmp_path, size="tiny"):
    path = tmp_path / "C0.pt"
    assert train_step.main(["init", "--size", size, "--seed", "7", "--out", str(path), "--source-sha", "test"]) == 0
    return path


class NoClock:
    def check(self, projected_s, what):
        pass


def inprocess(module, argv):
    """The tool's work, without the supervisor (its own tests cover it)."""
    return module.work(module.parse([str(a) for a in argv]), NoClock())


def selfplay_args(tmp_path, c0, out, workers, games=4, extra=()):
    return [
        "--checkpoint", c0, "--games", games, "--workers", workers, "--run-seed", 11,
        "--out", out, "--source-sha", "test", "--sims", 2, "--batch-size", 2, "--capacity", 64, "--round-games", 2,
        "--ledger", tmp_path / f"ledger-{pathlib.Path(out).name}.json", "--cores", max(2, workers),
        "--budget-core-hours", 1, "--wall-cap-s", 120, *extra,
    ]  # fmt: skip


def selfplay_run(tmp_path, c0, out, workers, games=4, extra=()):
    return run("selfplay.py", *selfplay_args(tmp_path, c0, out, workers, games, extra))


def games_by_id(directory):
    manifest, _ = artifacts.load_manifest(directory)
    out = {}
    for entry in manifest["files"]:
        arrays, _ = artifacts.decode_shard((pathlib.Path(directory) / entry["name"]).read_bytes())
        for g in np.unique(arrays["g"]):
            sel = arrays["g"] == g
            out[int(g)] = {k: v[sel] for k, v in arrays.items()}
    return manifest, out


def test_selfplay_is_repeatable_and_per_game_independent_of_the_worker_count(tmp_path):
    if (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()) < 2:
        pytest.skip("needs 2 cores")
    c0 = make_c0(tmp_path)
    runs = {}
    for name, w in (("w1a", 1), ("w2", 2), ("w1b", 1)):
        assert inprocess(selfplay, selfplay_args(tmp_path, c0, tmp_path / name, w)) == 0
        runs[name] = games_by_id(tmp_path / name)
    m1, g1 = runs["w1a"]
    assert artifacts.load_manifest(tmp_path / "w1a")[1] == artifacts.load_manifest(tmp_path / "w1b")[1]
    m2, g2 = runs["w2"]
    assert m1["games_digest"] == m2["games_digest"]
    assert sorted(g1) == sorted(g2) == list(range(4))
    for g in g1:
        for k in ("planes", "visits", "z", "ply"):
            assert np.array_equal(g1[g][k], g2[g][k]), (g, k)
    # Visit policies are legal: no visits on a full column.
    for sample in g1.values():
        full = sample["planes"][:, :, 0, :].sum(axis=1) > 0
        assert (sample["visits"][full] == 0).all()
    assert m1["checkpoint_size"] == "tiny" and m1["seed_derivation"] == seeds.DERIVATION


def test_selfplay_incomplete_or_changed_runs_are_refused(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    data = artifacts.encode_shard({"z": np.zeros(3, dtype=np.int8)}, {})
    artifacts.write_atomic(str(d / "s.bin"), data)
    with pytest.raises(artifacts.Incomplete, match="no manifest"):
        artifacts.load_manifest(str(d))
    artifacts.write_manifest(str(d), {"files": [{"name": "s.bin", "sha256": artifacts.sha256(data)}]})
    artifacts.load_manifest(str(d))
    assert not list(d.glob("*.tmp")), "no temporary file left"
    (d / "s.bin").write_bytes(data + b"x")
    with pytest.raises(artifacts.Incomplete, match="does not match"):
        artifacts.load_manifest(str(d))
    with pytest.raises(artifacts.Incomplete):
        train_step.load_samples([str(d)])


def test_selfplay_refuses_a_used_output_directory(tmp_path):
    c0 = make_c0(tmp_path)
    out = tmp_path / "used"
    out.mkdir()
    (out / "manifest.json").write_text("{}")
    r = selfplay_run(tmp_path, c0, out, 1, games=1)
    assert r.returncode != 0 and "not empty" in r.stderr
    assert not (tmp_path / "ledger-used.json").exists(), "refused before anything is reserved"


def test_selfplay_rejects_hash_consistent_bad_samples_and_non_finite_state(tmp_path):
    good = {
        "planes": np.zeros((2, 2, 6, 7), dtype=np.uint8),
        "visits": np.ones((2, 7), dtype=np.uint32),
        "z": np.array([1, -1], dtype=np.int8),
        "g": np.zeros(2, dtype=np.uint32),
        "ply": np.arange(2, dtype=np.uint16),
    }
    bad = [
        {**good, "z": np.array([1, 7], dtype=np.int8)},
        {**good, "visits": np.zeros((2, 7), dtype=np.uint32)},
        {**good, "planes": np.full((2, 2, 6, 7), 3, dtype=np.uint8)},
        {k: v for k, v in good.items() if k != "ply"},
        {**good, "z": np.array([np.nan, 1.0])},  # NaN outcome, float dtype
    ]
    for i, arrays in enumerate(bad):
        d = tmp_path / f"bad{i}"
        d.mkdir()
        data = artifacts.encode_shard(arrays, {})
        artifacts.write_atomic(str(d / "s.bin"), data)
        artifacts.write_manifest(str(d), {"files": [{"name": "s.bin", "sha256": artifacts.sha256(data)}]})
        with pytest.raises(artifacts.Incomplete):
            train_step.load_samples([str(d)])
    model = __import__("net").build("tiny", seed=1)
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(ckpt.NonFinite):
        ckpt.save(model, str(tmp_path / "nan.pt"), "tiny", {})
    assert not (tmp_path / "nan.pt").exists()


def test_selfplay_killed_run_is_charged_and_incomplete(tmp_path):
    c0 = make_c0(tmp_path)
    out = tmp_path / "killed"
    r = selfplay_run(tmp_path, c0, out, 1, games=100000, extra=("--wall-cap-s", 6))
    assert r.returncode == 124, r.stderr
    with pytest.raises(artifacts.Incomplete):
        artifacts.load_manifest(str(out))
    (entry,) = json.loads((tmp_path / "ledger-killed.json").read_text())["entries"]
    assert entry["status"].startswith("killed") and 0 < entry["wall_s"] <= 6


def test_selfplay_wilson_lower_bound():
    assert wilson_lower(0.5, 400) < 0.5
    assert wilson_lower(0.55, 400) > 0.5
    assert abs(wilson_lower(0.541, 400) - 0.5) < 0.005


def test_selfplay_tiny_end_to_end(tmp_path):
    c0 = make_c0(tmp_path)
    data = tmp_path / "data"
    assert inprocess(selfplay, selfplay_args(tmp_path, c0, data, 1, games=2)) == 0
    c1 = tmp_path / "C1.pt"
    targs = train_step.parse([str(a) for a in (
        "train", "--checkpoint", c0, "--data", data, "--out", c1, "--run-seed", 11,
        "--batch", 16, "--source-sha", "test", "--ledger", tmp_path / "ledger-train.json",
        "--budget-core-hours", 1, "--wall-cap-s", 60,
    )])  # fmt: skip
    assert train_step.work_train(targs, NoClock()) == 0
    _, m1 = ckpt.load(str(c1))
    _, m0 = ckpt.load(str(c0))
    assert m1["parent_sha256"] == m0["sha256"] and m1["steps"] >= 1
    assert m1["dropped_per_epoch"] == m1["augmented_samples"] - m1["steps_per_epoch"] * 16
    assert m1["data_manifests"] == [artifacts.load_manifest(str(data))[1]]
    out = tmp_path / "h2h"
    out.mkdir()
    assert inprocess(h2h, (
        "--a", c1, "--b", c0, "--out", out, "--source-sha", "test", "--any-openings", "--require-size", "any", "--limit", 1,
        "--sims", 2, "--batch-size", 2, "--capacity", 64,
        "--ledger", tmp_path / "ledger-h2h.json", "--budget-core-hours", 1, "--wall-cap-s", 60,
    )) == 0  # fmt: skip
    report = json.loads((out / "h2h.json").read_text())
    assert report["games"] == 2 and sum(report["wdl"]) == 2
    assert report["a"] == m1["sha256"] and report["b"] == m0["sha256"]
    assert report["any_openings"] is True and report["pass"] is None
    assert report["criterion_applicable"] is False
    assert report["provenance"]["source_sha"] == "test" and m1["provenance"]["config_sha256"]
    # The pinned dev set with a limit is a diagnostic run, never a pass,
    # whatever the games give (Codex's 4-game probe).
    out3 = tmp_path / "h2h3"
    out3.mkdir()
    assert inprocess(h2h, (
        "--a", c1, "--b", c0, "--out", out3, "--source-sha", "test", "--require-size", "any", "--limit", 2,
        "--sims", 2, "--batch-size", 2, "--capacity", 64,
        "--ledger", tmp_path / "ledger-h2h3.json", "--budget-core-hours", 1, "--wall-cap-s", 60,
    )) == 0  # fmt: skip
    r3 = json.loads((out3 / "h2h.json").read_text())
    assert r3["games"] == 4 and r3["criterion_applicable"] is False and r3["pass"] is None
    # A checkpoint is never overwritten, the parent least of all.
    with pytest.raises(SystemExit, match="parent"):
        inprocess_train = train_step.parse([str(a) for a in (
            "train", "--checkpoint", c0, "--data", data, "--out", c0, "--run-seed", 11, "--batch", 16,
            "--source-sha", "test", "--ledger", tmp_path / "l.json", "--budget-core-hours", 1, "--wall-cap-s", 60,
        )])  # fmt: skip
        train_step.work_train(inprocess_train, NoClock())
    with pytest.raises(FileExistsError):
        ckpt.save(ckpt.load(str(c1))[0], str(c1), "tiny", {})
    # Refused on a non-medium net without the test flag (the dev set is
    # the default openings file).
    (tmp_path / "h2h2").mkdir()
    with pytest.raises(SystemExit, match="medium"):
        inprocess(h2h, (
            "--a", c1, "--b", c0, "--out", tmp_path / "h2h2", "--source-sha", "test", "--limit", 1,
            "--ledger", tmp_path / "ledger-h2h2.json", "--budget-core-hours", 1, "--wall-cap-s", 60,
        ))  # fmt: skip
    with pytest.raises(SystemExit, match="dev set"):
        inprocess(h2h, (
            "--a", c1, "--b", c0, "--out", tmp_path / "h2h2", "--source-sha", "test", "--limit", 1, "--require-size", "any",
            "--openings", EXAMPLE.parents[1] / "compat" / "connect4" / "openings-heldout.txt",
            "--ledger", tmp_path / "ledger-h2h2.json", "--budget-core-hours", 1, "--wall-cap-s", 60,
        ))  # fmt: skip


def test_selfplay_diverged_training_writes_no_checkpoint(tmp_path, monkeypatch):
    c0 = make_c0(tmp_path)
    data = tmp_path / "data"
    assert inprocess(selfplay, selfplay_args(tmp_path, c0, data, 1, games=2)) == 0
    monkeypatch.setitem(train_step.HYPER, "lr", 1e30)
    out = tmp_path / "C1.pt"
    targs = train_step.parse([str(a) for a in (
        "train", "--checkpoint", c0, "--data", data, "--out", out, "--run-seed", 11, "--epochs", 50,
        "--batch", 16, "--source-sha", "test", "--ledger", tmp_path / "l.json", "--budget-core-hours", 1,
        "--wall-cap-s", 60,
    )])  # fmt: skip
    assert train_step.work_train(targs, NoClock()) == train_step.EXIT_DIVERGED
    assert not out.exists() and not (tmp_path / "C1.json").exists()
