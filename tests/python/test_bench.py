# SPDX-License-Identifier: MIT OR Apache-2.0
"""The benchmark harness (ADR-0025): statistics, the shared action plan,
warm-up, report shape, pinning and the claim qualification policy. The
numbers themselves are not checked."""

import copy
import importlib.util
import pathlib
import subprocess

import numpy as np
import pytest

_TOOL = pathlib.Path(__file__).resolve().parents[2] / "tools" / "bench" / "bench.py"
_spec = importlib.util.spec_from_file_location("bench", _TOOL)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


@pytest.fixture(scope="module")
def quick_report():
    return bench.run(quick=True, profile="debug")


# -- statistics -----------------------------------------------------------------


def test_bench_summary_median_and_bootstrap_ci():
    data = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.0]
    a = bench.summarize(data, resamples=2000, seed=0)
    assert a == bench.summarize(data, resamples=2000, seed=0)  # seeded
    assert a["median"] == float(np.median(data))
    lo, hi = a["ci95"]
    assert min(data) <= lo <= a["median"] <= hi <= max(data)


def test_bench_ratio_ci_is_reproducible_and_propagates_either_operand():
    rng = np.random.default_rng(3)
    num = 100 + rng.normal(0, 1, 30)
    den = 50 + rng.normal(0, 1, 30)
    base = bench.ratio_ci(num, den, resamples=4000, seed=1)
    assert base == bench.ratio_ci(num, den, resamples=4000, seed=1)
    assert base["ci95"][0] <= base["value"] <= base["ci95"][1]
    width = lambda r: r["ci95"][1] - r["ci95"][0]  # noqa: E731
    noisy_num = 100 + rng.normal(0, 10, 30)
    noisy_den = 50 + rng.normal(0, 5, 30)
    assert width(bench.ratio_ci(noisy_num, den, resamples=4000, seed=1)) > 3 * width(base)
    assert width(bench.ratio_ci(num, noisy_den, resamples=4000, seed=1)) > 3 * width(base)
    assert "independent" in base["method"]


# -- the shared plan and the method ---------------------------------------------


def test_bench_python_plan_matches_the_runner_plan(quick_report):
    rust = quick_report["results"]["rust/cartpole/inline/n1024"]
    py = quick_report["results"]["py/native/cartpole/n1024/borrowed"]
    assert rust["plan_fingerprint"] == py["plan_fingerprint"] == bench.fingerprint(bench.lcg_plan(1024, 2))
    assert rust["options"] == py["options"]
    d = quick_report["derived"]["python_overhead/cartpole/n1024/borrowed"]
    assert d["same_work"] and d["ci95"][0] <= d["value"] <= d["ci95"][1]


def test_bench_ratios_of_different_work_are_refused(quick_report):
    results = copy.deepcopy(quick_report["results"])
    results["py/native/cartpole/n1024/borrowed"]["options"]["max_episode_steps"] = 0
    spec, method = bench.load(quick=True)
    derived = bench.derive(results, spec["derived"], method)
    assert derived["python_overhead/cartpole/n1024/borrowed"]["value"] is None


def test_bench_snapshot_costs_are_warmed_up_and_labelled(quick_report):
    r = quick_report["results"]["rust/cartpole/n1024/snapshot"]
    assert r["method"]["warmup_s"] == bench.QUICK["warmup_s"]
    assert r["method"]["warmup_iterations"] > 0
    for key in ("snapshot_copy", "snapshot_all_copy", "restore_with_reobserve", "fork_with_reobserve"):
        assert key in r
    d = quick_report["derived"]["rust/cartpole/n1024/snapshot/copy_ns_per_kib"]
    assert d["budget_ns_per_kib"] == 1000.0 and d["value"] > 0


def test_bench_quick_run_covers_every_kind(quick_report):
    names = {c["name"] for c in bench.tomllib.loads(bench.CONFIGS.read_text())["config"]}
    assert set(quick_report["results"]) == names
    for name, r in quick_report["results"].items():
        if r["config"].get("metric") == "memory":
            assert r["record_bytes_per_instance"] > 0
        else:
            stats = [v for v in r.values() if isinstance(v, dict) and "median" in v]
            assert stats and all(s["n"] == bench.QUICK["reps"] for s in stats), name
        assert "method" in r or r["config"].get("metric") == "memory"


def test_bench_runs_mode_aggregates_process_medians():
    report = bench.run(only="rust/cartpole/n1024/snapshot", quick=True, profile="debug", runs=3)
    r = report["results"]["rust/cartpole/n1024/snapshot"]
    assert len(r["runs"]) == 3 and r["snapshot_all_copy"]["n"] == 3
    assert "process medians" in r["statistic"]
    assert len(report["run_orders"]) == 3


def test_bench_empty_selection_is_refused():
    with pytest.raises(ValueError):
        bench.run(only="no-such-configuration", quick=True, profile="debug")


def test_bench_runner_fails_loudly_on_bad_input():
    runner = bench.runner_path("debug")
    bad = subprocess.run([str(runner), "throughput", "cartpole", "0", "0", "0", "1", "0.01"], capture_output=True, text=True)
    assert bad.returncode != 0 and "num_envs" in bad.stderr


# -- pinning --------------------------------------------------------------------


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
    assert not bench.pinning("m")["applied"]


def test_bench_ref_x86_pins_distinct_physical_cores():
    cpu_set = bench.machines()["ref-x86"]["bench_cpu_set"]
    assert 0 not in cpu_set
    # Hyper-threaded P-cores are numbered in sibling pairs (2k, 2k + 1).
    assert len({c // 2 for c in cpu_set}) == len(cpu_set)


# -- claim qualification policy -------------------------------------------------

REF = {"status": "reference", "cpu_id": "i7-12700H", "bench_cpu_set": [2, 4, 6, 8]}
QUAL = {"machine": "ref-x86", "exclusive_window": True, "operator": "test"}


def qualified_report():
    """A synthetic report that meets every condition: Rust and Python
    configurations, every process record qualified."""
    method = dict(bench.load(quick=False)[1])
    pin = {"cpu_set": [2, 4, 6, 8], "applied": True, "tool": "taskset", "note": None}
    n = method["runs_for_claims"]
    stats = {"median": 1.0, "ci95": [1.0, 1.0], "n": n, "samples": [1.0] * n}
    options = {"seed": 1, "auto_reset": "next_step", "max_episode_steps": 500, "action_batches": 64}

    run_stats = {"median": 1.0, "ci95": [1.0, 1.0], "n": method["reps"], "samples": [1.0] * method["reps"]}

    def record(affinity):
        return {
            "method": {"warmup_s": method["warmup_s"], "reps": method["reps"], "rep_s": method["rep_s"]},
            "profile": "release",
            "cpu": "12th Gen Intel(R) Core(TM) i7-12700H",
            "pinning": {"process_affinity": affinity},
            "options": dict(options),
            "plan_fingerprint": "45cd2a7ac0333cd5",
            "samples": run_stats,
        }

    def result(name, kind, workers, affinity):
        runs = [record(affinity) for _ in range(n)]
        return {"config": {"name": name, "kind": kind, "workers": workers}, **runs[0], "runs": runs, "samples": stats}

    results = {
        "rust/x/inline": result("rust/x/inline", "rust", 0, "2"),
        "rust/x/sync4": result("rust/x/sync4", "rust", 4, "2,4,6,8"),
        "py/x/borrowed": result("py/x/borrowed", "python", 0, "2"),
    }
    env = {
        "machine": "ref-x86",
        "hardware": {"cpu": "12th Gen Intel(R) Core(TM) i7-12700H"},
        "software": {"native_build_profile": "release"},
    }
    return {"quick": False, "runs": n, "method": method, "pinning": pin, "environment": env, "results": results}


@pytest.fixture
def reference_machine(monkeypatch):
    monkeypatch.setattr(bench, "machines", lambda: {"ref-x86": dict(REF)})


def test_bench_qualified_report_is_valid(reference_machine):
    assert bench.qualify(qualified_report(), QUAL) == []


def test_bench_mislabelled_hardware_is_not_valid(reference_machine):
    report = qualified_report()
    report["environment"]["hardware"]["cpu"] = "Apple M5 Max"
    assert any("does not match" in r for r in bench.qualify(report, QUAL))


def test_bench_missing_qualification_is_not_valid(reference_machine):
    assert any("qualification" in r for r in bench.qualify(qualified_report(), None))
    other = dict(QUAL, exclusive_window=False)
    assert any("exclusive" in r for r in bench.qualify(qualified_report(), other))


def test_bench_debug_builds_are_not_valid(reference_machine):
    report = qualified_report()
    report["results"]["rust/x/inline"]["runs"][0]["profile"] = "debug"
    assert any("debug" in r for r in bench.qualify(report, QUAL))
    report = qualified_report()
    report["environment"]["software"]["native_build_profile"] = "debug"
    assert any("release" in r for r in bench.qualify(report, QUAL))


def test_bench_unobserved_pinning_is_not_valid(reference_machine):
    report = qualified_report()
    report["results"]["rust/x/sync4"]["runs"][0]["pinning"]["process_affinity"] = "0-19"
    assert any("affinity" in r for r in bench.qualify(report, QUAL))


def test_bench_empty_or_short_reports_are_not_valid(reference_machine):
    report = qualified_report()
    report["results"] = {}
    assert "no results" in bench.qualify(report, QUAL)
    report = qualified_report()
    report["runs"] = 1
    assert any("process runs" in r for r in bench.qualify(report, QUAL))


def test_bench_non_reference_machines_are_not_valid(monkeypatch):
    monkeypatch.setattr(bench, "machines", lambda: {"ref-x86": dict(REF, status="candidate")})
    assert any("not 'reference'" in r for r in bench.qualify(qualified_report(), QUAL))


def test_bench_quick_reports_explain_why_they_are_indicative(quick_report):
    assert quick_report["valid_for_claims"] is False
    assert "quick run" in quick_report["claim_check"]["reasons"]


def test_bench_a_bad_later_run_is_not_valid(reference_machine):
    # codex R-PR24 R2: every process record qualifies, not only the first.
    for field, value, reason in (
        ("profile", "debug", "build"),
        ("cpu", "Apple M5 Max", "does not match"),
        ("plan_fingerprint", "0000000000000000", "different plans"),
    ):
        report = qualified_report()
        report["results"]["py/x/borrowed"]["runs"][3][field] = value
        assert any(reason in r and ("run 3" in r or "process runs" in r) for r in bench.qualify(report, QUAL)), field
    report = qualified_report()
    report["results"]["py/x/borrowed"]["runs"][5]["pinning"]["process_affinity"] = "0-19"
    assert any("run 5" in r and "affinity" in r for r in bench.qualify(report, QUAL))


def test_bench_a_missing_run_is_not_valid(reference_machine):
    report = qualified_report()
    report["results"]["rust/x/inline"]["runs"].pop()
    assert any("process records" in r for r in bench.qualify(report, QUAL))


def test_bench_a_python_configuration_qualifies_with_observed_affinity(reference_machine):
    report = qualified_report()
    assert bench.qualify(report, QUAL) == []
    report["results"]["py/x/borrowed"]["runs"][0]["pinning"] = {}
    assert any("py/x/borrowed run 0" in r for r in bench.qualify(report, QUAL))


def test_bench_python_children_record_their_affinity(quick_report):
    py = quick_report["results"]["py/native/cartpole/n1024/borrowed"]
    assert all("process_affinity" in run["pinning"] for run in py["runs"])


def test_bench_ratio_refused_when_a_later_run_differs():
    a = {"runs": [{"options": {"x": 1}, "plan_fingerprint": "f"}] * 3}
    b = {"runs": [{"options": {"x": 1}, "plan_fingerprint": "f"}, {"options": {"x": 1}, "plan_fingerprint": "g"}]}
    assert bench._same_work(a, a)
    assert not bench._same_work(a, b)


def test_bench_affinity_lists_parse_ranges():
    assert bench.parse_affinity("2,4") == {2, 4}
    assert bench.parse_affinity("0-3,8") == {0, 1, 2, 3, 8}
    assert bench.parse_affinity("not exposed by this OS") is None


def test_bench_a_later_run_with_another_method_or_fewer_samples_is_not_valid(reference_machine):
    report = qualified_report()
    report["results"]["rust/x/sync4"]["runs"][7]["method"]["reps"] = 3
    assert any("run 7" in r and "method" in r for r in bench.qualify(report, QUAL))
    report = qualified_report()
    report["results"]["rust/x/sync4"]["runs"][2]["samples"] = {"median": 1.0, "ci95": [1, 1], "n": 4, "samples": [1.0] * 4}
    assert any("run 2" in r and "samples" in r for r in bench.qualify(report, QUAL))
    report = qualified_report()
    del report["results"]["rust/x/sync4"]["runs"][1]["method"]
    assert any("run 1" in r for r in bench.qualify(report, QUAL))  # missing metadata stays indicative


def test_bench_copy_cost_per_kib_carries_a_ci(quick_report):
    d = quick_report["derived"]["rust/cartpole/n1024/snapshot/copy_ns_per_kib"]
    assert d["ci95"][0] <= d["value"] <= d["ci95"][1]
