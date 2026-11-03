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


def c(kind, w, t, median, kept=7):
    return {"kind": kind, "workers": w, "trees": t, "rounds_kept": kept, "stopped": None,
            "games_per_hour_per_allocated_core": {"median": median}}  # fmt: skip


def formal_inputs(cores=32):
    cells = [c(k, w, t, 100.0) for k, w, t in group_bench.declared_cells(cores, [4, 8, 16])]
    config = {**group_bench.FORMAL, "cores": cores}
    assert config["run_seed"] == 45 and config["matched_roots"] == 64
    mw = {"stopped": None, "speedup": 1.3}
    runtime = {"host": "swarmer02", "image_digest": group_bench.h2h.REGISTERED_IMAGE, "cpuset": f"0-{cores - 1}", "affinity_cores": cores,
               "torch_threads": 1, "torch_interop_threads": 1, "deterministic": True}  # fmt: skip
    return config, cells, mw, runtime


def test_group_bench_verdict_ratios_and_formal_adoption():
    config, cells, mw, runtime = formal_inputs()
    assert group_bench.applicability(config, "small", cells, mw, runtime) == []
    for i, (k, w, t) in enumerate(group_bench.declared_cells(32, [4, 8, 16])):
        if (k, w, t) == ("group", 32, 8):
            cells[i] = c(k, w, t, 125.0)
        if (k, w, t) == ("group", 1, 8):
            cells[i] = c(k, w, t, 200.0)
    v = group_bench.verdict(cells, 32, 7, group_bench.applicability(config, "small", cells, mw, runtime))
    assert v["ratio"] == 1.25 and v["formal"] and v["adopted"] is True
    assert v["diagnostic_ratio_w1"] == pytest.approx(2.0)
    i8 = group_bench.declared_cells(32, [4, 8, 16]).index(("group", 32, 8))
    cells[i8] = c("group", 32, 8, 119.0)
    assert group_bench.verdict(cells, 32, 7, [])["adopted"] is False


@pytest.mark.parametrize(
    "damage",
    ["one-kept-round", "no-warmup", "trees", "sims", "run-seed", "matched-roots", "net", "any-size", "allocation", "threads", "image",
     "other-host", "missing-host", "missing-cell", "stopped-cell", "matched-missing", "matched-stopped"],
)  # fmt: skip
def test_group_bench_shortened_or_altered_run_is_diagnostic_only(damage):
    config, cells, mw, runtime = formal_inputs()
    size = "small"
    if damage == "one-kept-round":
        config["repeats"] = 1
        cells = [{**x, "rounds_kept": 1} for x in cells]
    elif damage == "no-warmup":
        config["warmup"] = 0
    elif damage == "trees":
        config["trees"] = [8]
    elif damage == "sims":
        config["sims"] = 50
    elif damage == "run-seed":
        config["run_seed"] = 46
    elif damage == "matched-roots":
        config["matched_roots"] = 8
    elif damage == "net":
        size = "medium"
    elif damage == "any-size":
        config["any_size"] = True
    elif damage == "allocation":
        runtime["affinity_cores"] = 8
    elif damage == "threads":
        runtime["torch_threads"] = 4
    elif damage == "image":
        runtime["image_digest"] = None
    elif damage == "other-host":
        runtime["host"] = "xaskar"
    elif damage == "missing-host":
        del runtime["host"]
    elif damage == "missing-cell":
        cells = cells[1:]
    elif damage == "stopped-cell":
        cells[0] = {**cells[0], "stopped": "budget"}
    elif damage == "matched-missing":
        mw = {}
    else:
        mw = {"stopped": "budget"}
    # Ratios that would adopt, on a run that may not.
    cells = [{**x, "games_per_hour_per_allocated_core": {"median": 200.0 if x["kind"] == "group" else 100.0}} for x in cells]
    why = group_bench.applicability(config, size, cells, mw, runtime)
    v = group_bench.verdict(cells, 32, config["repeats"], why)
    assert why and v["formal"] is False and v["adopted"] is None


def test_group_bench_every_cell_gives_each_worker_the_same_games_per_round():
    n = group_bench.games_per_worker([4, 8, 16], 2)
    assert n == 32
    for kind, w, t in group_bench.declared_cells(32, [4, 8, 16]):
        tasks = group_bench.worker_tasks(kind, w, t, 0, n)
        assert len(tasks) == w
        assert all(sum(len(b) for b in batches) == 32 and all(len(b) == t for b in batches) for _, batches in tasks)
        assert sorted(g for _, batches in tasks for b in batches for g in b) == list(range(w * 32))
    with pytest.raises(SystemExit):
        group_bench.games_per_worker([3, 8], 1)


def test_group_bench_matched_work_keeps_only_warm_samples():
    """A scripted pass and clock: warmup passes are never timed into the
    comparison, and the kept rounds alternate the order."""
    calls, t = [], [0.0]
    cost = {"single": 10.0, "group": 5.0}

    def run(kind):
        calls.append(kind)
        # The first pass of each mode pays a startup cost.
        t[0] += cost[kind] + (1000.0 if calls.count(kind) == 1 else 0.0)
        return {"root_searches": 4, "leaves": 1}

    r = group_bench.matched_rounds(run, lambda: t[0], warmup=2, repeats=3)
    assert calls[:4] == ["single", "group", "single", "group"]
    assert calls[4:] == ["single", "group", "group", "single", "single", "group"]
    assert r["samples_s"] == {"single": [10.0] * 3, "group": [5.0] * 3}


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
    assert {"boot_id", "cpu_model", "mem_total", "host"} <= set(report["runtime"])
    v = report["verdict"]
    assert v["formal"] is False and v["adopted"] is None and v["not_formal_because"]
    mw = report["matched_work"]
    assert mw["stopped"] is None and len(mw["samples_s"]["single"]) == 1 and "speedup" in mw
    # The layout reconstructs each group's active schedule, and a game's
    # seeds never depend on its slot or group.
    seeds_by_cell = {}
    for cell in report["cells"]:
        for rnd in cell["layout"]:
            for w in rnd["workers"]:
                assert w["busy_s"] > 0
                for batch in w["batches"]:
                    if cell["kind"] == "group":
                        sets = group_bench.active_sets(batch)
                        assert [len(x) for x in sets] == batch[0]["active_schedule"]
                    for g in batch:
                        seeds_by_cell.setdefault(g["g"], set()).add((g["env_seed"], g["search_seed"]))
    assert all(len(v) == 1 for v in seeds_by_cell.values())
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
