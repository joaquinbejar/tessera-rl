# SPDX-License-Identifier: MIT OR Apache-2.0
"""The benchmark harness (ADR-0025): statistics, report shape and a quick
smoke run of every configuration kind. The numbers are not checked."""

import importlib.util
import pathlib

import numpy as np
import pytest

_TOOL = pathlib.Path(__file__).resolve().parents[2] / "tools" / "bench" / "bench.py"
_spec = importlib.util.spec_from_file_location("bench", _TOOL)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_bench_summary_median_and_bootstrap_ci():
    data = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.0]
    a = bench.summarize(data, resamples=2000, seed=0)
    b = bench.summarize(data, resamples=2000, seed=0)
    assert a == b  # seeded
    assert a["median"] == float(np.median(data))
    lo, hi = a["ci95"]
    assert lo <= a["median"] <= hi
    assert min(data) <= lo and hi <= max(data)
    assert a["n"] == 10


def test_bench_quick_run_covers_every_kind():
    report = bench.run(quick=True, profile="debug")
    assert report["schema"] == "tessera-bench/2"
    assert report["valid_for_claims"] is False  # quick runs never are
    names = {c["name"] for c in bench.tomllib.loads(bench.CONFIGS.read_text())["config"]}
    assert set(report["results"]) == names
    for name, r in report["results"].items():
        metric = r["config"].get("metric", "throughput")
        if metric == "memory":
            assert r["record_bytes_per_instance"] > 0
        else:
            stats = [v for v in r.values() if isinstance(v, dict) and "median" in v]
            assert stats and all(s["n"] == bench.QUICK["reps"] for s in stats), name
    assert "python_overhead/cartpole/n1024/borrowed" in report["derived"]
    assert report["environment"]["software"]["gymnasium"]


def test_bench_claims_need_a_reference_machine():
    # No machine is characterized yet: even the full method is indicative.
    assert all(m.get("status") != "reference" for m in bench.machines().values())


def test_bench_runs_mode_aggregates_process_medians():
    report = bench.run(only="rust/cartpole/n1024/snapshot", quick=True, profile="debug", runs=3)
    r = report["results"]["rust/cartpole/n1024/snapshot"]
    assert len(r["runs"]) == 3 and r["snapshot_all_copy"]["n"] == 3
    assert "process medians" in r["statistic"]
    assert len(report["run_orders"]) == 3
    # Copy-only costs are reported against ADR-0010's budget.
    d = report["derived"]["rust/cartpole/n1024/snapshot/copy_ns_per_kib"]
    assert d["budget_ns_per_kib"] == 1000.0 and d["value"] > 0
    for key in ("restore_with_reobserve", "fork_with_reobserve", "import_with_validation_and_reobserve"):
        assert key in r


def test_bench_pinning_uses_the_machine_cpu_set(monkeypatch):
    monkeypatch.setattr(bench, "machines", lambda: {"m": {"bench_cpu_set": [2, 4, 6, 8]}})
    monkeypatch.setattr(bench.shutil, "which", lambda tool: "/usr/bin/taskset")
    pin = bench.pinning("m")
    assert pin["applied"] and pin["cpu_set"] == [2, 4, 6, 8]
    assert bench._cpus_for({"name": "inline", "workers": 0}, pin) == [2]
    assert bench._cpus_for({"name": "sync", "workers": 4}, pin) == [2, 4, 6, 8]
    assert bench._prefix(pin, [2]) == ["taskset", "-c", "2"]
    with pytest.raises(ValueError):
        bench._cpus_for({"name": "sync8", "workers": 8}, pin)
    monkeypatch.setattr(bench.shutil, "which", lambda tool: None)
    assert not bench.pinning("m")["applied"]  # recorded as unpinned


def test_bench_ref_x86_pins_distinct_physical_cores():
    cpu_set = bench.machines()["ref-x86"]["bench_cpu_set"]
    assert 0 not in cpu_set
    # Hyper-threaded P-cores are numbered in sibling pairs (2k, 2k + 1).
    assert len({c // 2 for c in cpu_set}) == len(cpu_set)
