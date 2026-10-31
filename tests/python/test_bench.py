# SPDX-License-Identifier: MIT OR Apache-2.0
"""The benchmark harness (ADR-0025): statistics, report shape and a quick
smoke run of every configuration kind. The numbers are not checked."""

import importlib.util
import pathlib

import numpy as np

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
    assert report["schema"] == "tessera-bench/1"
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
