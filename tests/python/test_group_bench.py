# SPDX-License-Identifier: MIT OR Apache-2.0
"""The ADR-0045 §5 measurement tool (examples/alphazero_connect4)."""

import json
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

EXAMPLE = pathlib.Path(__file__).parents[2] / "examples" / "alphazero_connect4"
sys.path.insert(0, str(EXAMPLE))

import group_bench  # noqa: E402
import train_step  # noqa: E402


class NoClock:
    def check(self, projected_s, what):
        pass


def test_group_bench_verdict_reads_only_complete_named_cells():
    def c(kind, w, t, median, kept=7):
        return {"kind": kind, "workers": w, "trees": t, "rounds_kept": kept,
                "games_per_hour_per_allocated_core": {"median": median}}  # fmt: skip

    cells = [c("single", 32, 1, 100.0), c("group", 32, 8, 125.0), c("single", 1, 1, 90.0), c("group", 1, 8, 200.0)]
    v = group_bench.verdict(cells, 32, 7)
    assert v["ratio"] == 1.25 and v["adopted"] is True and v["complete"]
    assert v["diagnostic_ratio_w1"] == pytest.approx(200 / 90)
    cells[1] = c("group", 32, 8, 119.0)
    assert group_bench.verdict(cells, 32, 7)["adopted"] is False
    cells[2] = c("single", 1, 1, 90.0, kept=6)  # a short cell is missing
    assert group_bench.verdict(cells, 32, 7)["adopted"] == "incomplete"


def test_group_bench_tiny_run_writes_every_cell(tmp_path):
    c0 = tmp_path / "S.pt"
    assert train_step.main(["init", "--size", "tiny", "--seed", "3", "--out", str(c0), "--source-sha", "test"]) == 0
    out = tmp_path / "out"
    out.mkdir()
    args = group_bench.parse([str(a) for a in (
        "--checkpoint", c0, "--out", out, "--source-sha", "test", "--sims", 2, "--batch-size", 2,
        "--capacity", 64, "--trees", 2, 8, "--warmup", 0, "--repeats", 1, "--games-per-slot", 1,
        "--matched-roots", 2, "--any-size", "--ledger", tmp_path / "l.json", "--cores", 1,
        "--budget-core-hours", 1, "--wall-cap-s", 120,
    )])  # fmt: skip
    assert group_bench.work(args, NoClock()) == 0
    report = json.loads((out / "group-bench.json").read_text())
    kinds = {(c["kind"], c["workers"], c["trees"]) for c in report["cells"]}
    assert kinds == {("single", 1, 1), ("group", 1, 2), ("group", 1, 8)}
    assert all(c["rounds_kept"] == 1 and c["moves_per_game"]["n"] == 1 for c in report["cells"])
    assert report["verdict"]["complete"] is True
    assert set(report["matched_work"]) >= {"single", "group", "speedup"}
    assert set(report["memory_bytes_per_worker"]) == {"2", "8"}


def test_group_bench_same_logical_games_with_a_canned_evaluator():
    """With an evaluator that depends on the leaf only, a group plays the
    same games as independent trees (ADR-0045 equivalence)."""
    import numpy as np

    def canned(obs, legal, hashes):
        w = np.stack([1 + ((hashes >> np.uint64(8 * a)) & np.uint64(0xFF)).astype(np.float64) for a in range(7)], axis=1)
        return w / w.sum(axis=1, keepdims=True), ((hashes >> np.uint64(48)) & np.uint64(0xFFFF)) / 32767.5 - 1.0

    class Ev:
        def __init__(self):
            self.leaves = 0
            self.net_ns = 0
            self.batches = []

        def __call__(self, obs, legal, hashes):
            self.leaves += len(hashes)
            self.batches.append(len(hashes))
            return canned(obs, legal, hashes)

    ev = Ev()
    group_bench._W.update(evaluator=ev, cfg={"run_seed": 5, "sims": 8, "batch_size": 4, "capacity": 256})
    single = [(g["g"], g["moves"]) for g in group_bench.single_games([0, 1, 2])]
    grouped = [(g["g"], g["moves"]) for g in group_bench.group_games([0, 1, 2])]
    assert single == grouped
