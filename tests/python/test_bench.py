# SPDX-License-Identifier: MIT OR Apache-2.0
"""The benchmark harness (ADR-0025): statistics, the shared action plan,
warm-up, report shape, pinning and the claim qualification policy. The
numbers themselves are not checked."""

import copy
import hashlib
import importlib.util
import json
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


def test_bench_continuous_plan_matches_the_runner_plan(quick_report):
    rust = quick_report["results"]["rust/pendulum/inline/n1024"]
    py = quick_report["results"]["py/native/pendulum/n1024/borrowed"]
    plan = bench.lcg_continuous(1024, [(-2.0, 2.0)])
    assert plan.min() >= -2.0 and plan.max() < 2.0
    empty = np.empty((bench.ACTION_BATCHES, 0), dtype=np.int32)
    assert rust["plan_fingerprint"] == py["plan_fingerprint"] == bench.fingerprint(empty, plan)
    assert rust["options"] == py["options"]


def test_bench_configs_cover_every_catalog_env():
    spec, _ = bench.load(quick=True)
    rust_envs = {c["env"] for c in spec["config"] if c["kind"] == "rust" and c["metric"] == "throughput"}
    assert rust_envs == {
        "acrobot", "brickwall", "cartpole", "connect4", "go9", "grid", "keydoor", "lanterns", "leduc",
        "mountaincar", "othello", "paddleduel", "pendulum", "rts", "snake", "tag", "lob",
        "jobshop", "jobshop-64x32", "vrp", "vrp-500", "survival-16", "survival-128",
    }  # fmt: skip
    envpool = {c["env"] for c in spec["config"] if c.get("api") == "envpool"}
    assert envpool == set(bench.PY_ENVS)
    for env in envpool:
        assert f"speedup/native-vs-envpool/{env}/n1024" in {d["name"] for d in spec["derived"]}


def test_bench_legal_actions_option_reaches_the_runner(quick_report):
    assert quick_report["results"]["rust/connect4/inline/n1024"]["options"]["legal_actions"] is True
    assert "legal_actions" not in quick_report["results"]["rust/cartpole/inline/n1024"]["options"]
    assert quick_report["results"]["rust/connect4/inline/n1024"]["action_policy"] == "legal_actions"
    assert quick_report["results"]["rust/cartpole/inline/n1024"]["action_policy"] == "planned"


def test_bench_schedule_and_full_workers_reach_the_runner(quick_report):
    full = quick_report["results"]["rust/rts/syncfull/n1024"]
    assert full["options"]["schedule"] == "dynamic" and full["schedule"] == "dynamic"
    assert full["workers"] == bench.workers_for(full["config"], quick_report["pinning"]) >= 1
    sync4 = quick_report["results"]["rust/lanterns/sync4/n1024"]
    assert (sync4["workers"], sync4["schedule"], sync4["options"]["schedule"]) == (4, "static", "static")
    inline = quick_report["results"]["rust/lanterns/inline/n1024"]
    assert "schedule" not in inline["options"] and inline["schedule"] is None
    assert inline["action_policy"] == "legal_actions"
    d = quick_report["derived"]["scaling_efficiency/rts/syncfull/n1024"]
    assert d["from"] == ["rust/rts/syncfull/n1024", "rust/rts/inline/n1024"] and d["value"] > 0


def test_bench_chunk_size_reaches_the_runner_and_is_recorded(quick_report):
    sync4 = quick_report["results"]["rust/survival-16/sync4/n256"]
    assert sync4["options"]["chunk_size"] == 16 and sync4["chunk_size"] == 16
    inline = quick_report["results"]["rust/survival-16/inline/n256"]
    # Not given: the engine default, kept out of the options.
    assert "chunk_size" not in inline["options"] and inline["chunk_size"] == 64


def test_bench_derive_missing_multi_is_an_error_on_a_full_run(quick_report):
    spec, method = bench.load(quick=True)
    results = copy.deepcopy(quick_report["results"])
    del results["rust/survival-16/syncfull/n256"]
    with pytest.raises(ValueError, match="full run"):
        bench.derive(results, spec["derived"], method)
    derived = bench.derive(results, spec["derived"], method, partial=True)
    assert "scaling_efficiency/survival-16/syncfull/n256" not in derived
    assert "scaling_efficiency/survival-16/sync4/n256" in derived


def test_bench_only_run_skips_scaling_rows_it_excluded():
    report = bench.run(only="rust/survival-16/sync4/", quick=True, profile="debug")
    assert set(report["results"]) == {"rust/survival-16/sync4/n256"}
    assert not any(k.startswith("scaling_efficiency/") for k in report["derived"])


def test_bench_full_workers_resolve_to_the_cpu_set():
    pin = {"applied": True, "cpu_set": [2, 4, 6, 8, 10]}
    assert bench.workers_for({"name": "f", "workers": "full"}, pin) == 5
    assert bench._cpus_for({"name": "f", "workers": "full"}, pin) == [2, 4, 6, 8, 10]
    with pytest.raises(ValueError, match="not a count"):
        bench.workers_for({"name": "bad", "workers": "many"}, pin)


def test_bench_runner_refuses_a_schedule_without_workers():
    runner = bench.runner_path("debug")
    bad = subprocess.run([str(runner), "throughput", "cartpole", "4", "0", "0", "1", "0.01", "--schedule", "dynamic"], capture_output=True, text=True)
    assert bad.returncode != 0 and "Sync mode" in bad.stderr


def test_bench_envpool_speedups_carry_the_output_asymmetry_note():
    spec, method = bench.load(quick=True)
    fake = {"samples": {"samples": [1.0, 1.0, 1.0]}, "options": {}, "plan_fingerprint": "f"}
    results = {}
    for d in spec["derived"]:
        if d["name"].startswith("speedup/native-vs-envpool/"):
            results[d["numerator"]] = fake
            results[d["denominator"]] = fake
    derived = bench.derive(results, spec["derived"], method, partial=True)
    speedups = [v for k, v in derived.items() if k.startswith("speedup/native-vs-envpool/")]
    assert len(speedups) == len(bench.PY_ENVS)
    assert all("info dict" in v["note"] for v in speedups)


def test_bench_configs_needing_a_missing_package_are_skipped(monkeypatch):
    real = bench.importlib.util.find_spec
    monkeypatch.setattr(bench.importlib.util, "find_spec", lambda name: None if name == "envpool" else real(name))
    with pytest.raises(ValueError, match="missing package"):
        bench.run(only="py/envpool/", quick=True, profile="debug")
    report = bench.run(only="py/", quick=True, profile="debug")
    assert set(report["skipped"]) == {f"py/envpool/{env}/n1024" for env in bench.PY_ENVS}
    assert not any(name.startswith("py/envpool/") for name in report["results"])
    assert not any(name.startswith("speedup/native-vs-envpool/") for name in report["derived"])


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
    # Configurations needing a package not installed here (EnvPool) are
    # skipped and listed, never silently dropped.
    assert set(quick_report["results"]) | set(quick_report["skipped"]) == names
    assert not set(quick_report["results"]) & set(quick_report["skipped"])
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


@pytest.mark.parametrize(
    ("env", "mask_bytes"),
    [
        # Mask bytes per instance and step: JobShop one per job, Vrp one per
        # node (customers + depot) in each of its 8 slots (#566).
        ("jobshop", 6),
        ("jobshop-64x32", 64),
        ("vrp", 8 * 21),
        ("vrp-500", 8 * 501),
        # Survival: 12 mask rows (move 5, act 7) in each of its 128 slots.
        ("survival-16", 128 * 12),
        ("survival-128", 128 * 12),
    ],
)
def test_bench_ops_entries_run_at_their_sizes(env, mask_bytes):
    runner = bench.runner_path("debug")
    out = subprocess.run([str(runner), "memory", env, "2"], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout)["mask_bytes_per_step"] == 2 * mask_bytes


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


@pytest.mark.parametrize("machine", ["ref-x86", "ref-x86-alt"])
def test_bench_ref_x86_pins_distinct_physical_cores(machine):
    cpu_set = bench.machines()[machine]["bench_cpu_set"]
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
    load = {"before": [0.5, 0.4, 0.3], "after": [0.6, 0.4, 0.3]}
    return {"quick": False, "runs": n, "method": method, "pinning": pin, "environment": env, "load_average": load, "results": results}


@pytest.fixture
def reference_machine(monkeypatch):
    monkeypatch.setattr(bench, "machines", lambda: {"ref-x86": dict(REF)})


def test_bench_qualified_report_is_valid(reference_machine):
    assert bench.qualify(qualified_report(), QUAL) == []


def test_bench_load_average_unavailable_is_not_valid(reference_machine, monkeypatch):
    # Windows has no os.getloadavg: the run is recorded, never valid for claims.
    monkeypatch.delattr(bench.os, "getloadavg", raising=False)
    assert bench.load_average() is None
    report = qualified_report()
    report["load_average"] = {"before": None, "after": None}
    assert "load average unavailable on this OS" in bench.qualify(report, QUAL)


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


# -- the measurement load rule (#640) -------------------------------------------


class FakeLoads:
    """Load readings served in order (the last one repeats), and the sleeps
    taken, so no test waits."""

    def __init__(self, monkeypatch, load5s):
        self.load5s, self.sleeps = list(load5s), []
        monkeypatch.setattr(bench, "load_average", self.read)
        monkeypatch.setattr(bench, "_sleep", self.sleeps.append)

    def read(self):
        value = self.load5s.pop(0) if len(self.load5s) > 1 else self.load5s[0]
        return [value + 0.5, value, 0.25]


def test_bench_load_gate_passes_below_the_bound_without_waiting(monkeypatch):
    fake = FakeLoads(monkeypatch, [1.9])
    gate = bench.LoadGate(2.0, 0.0)
    assert gate.before() == [2.4, 1.9, 0.25] and fake.sleeps == []


def test_bench_load_gate_waits_at_the_bound_until_below(monkeypatch):
    fake = FakeLoads(monkeypatch, [2.0, 3.0, 1.5])
    gate = bench.LoadGate(2.0, 7200.0)
    assert gate.before()[1] == 1.5
    assert fake.sleeps == [bench.LOAD_POLL_S] * 2 and gate.waited_s == 60


def test_bench_load_gate_fails_when_the_budget_is_spent(monkeypatch):
    fake = FakeLoads(monkeypatch, [5.0])
    gate = bench.LoadGate(2.0, 60.0)
    with pytest.raises(bench.LoadBudgetExceeded):
        gate.before()
    # Two polls fit the 60 s budget; the third would exceed it.
    assert fake.sleeps == [bench.LOAD_POLL_S] * 2


def test_bench_load_gate_budget_is_cumulative(monkeypatch):
    FakeLoads(monkeypatch, [3.0, 1.0, 3.0, 3.0])
    gate = bench.LoadGate(2.0, 45.0)
    gate.before()
    with pytest.raises(bench.LoadBudgetExceeded):
        gate.before()


@pytest.mark.parametrize(("k", "after", "void"), [(1, 2.9, False), (1, 3.0, True), (4, 5.9, False), (4, 6.0, True), (4, 2.0, False)])
def test_bench_load_void_bound_is_gate_plus_k_run(monkeypatch, k, after, void):
    FakeLoads(monkeypatch, [after])
    gate = bench.LoadGate(2.0, 0.0)
    gate.after("c", 0, k, [1.0, 1.0, 1.0])
    (r,) = gate.runs
    assert (r["void"], r["gate_bound"], r["void_bound"], r["k_run"]) == (void, 2.0, 2.0 + k, k)
    assert r["before"] == {"load1": 1.0, "load5": 1.0} and r["after"] == {"load1": after + 0.5, "load5": after}
    assert gate.record()["load_void"] is void
    assert gate.record()["void_runs"] == ([{"config": "c", "run": 0}] if void else [])


def test_bench_k_run_counts_workers():
    pin = {"applied": True, "cpu_set": [2, 4, 6, 8]}
    assert bench.k_run({"kind": "rust", "metric": "throughput", "workers": 0}, pin) == 1
    assert bench.k_run({"kind": "rust", "metric": "throughput", "workers": 4}, pin) == 4
    assert bench.k_run({"kind": "rust", "metric": "throughput", "workers": "full"}, pin) == 4
    assert bench.k_run({"kind": "rust", "metric": "snapshot"}, pin) == 1
    assert bench.k_run({"kind": "rust", "metric": "memory"}, pin) == 1
    assert bench.k_run({"kind": "python", "workers": 4}, pin) == 1


def test_bench_report_records_loads_of_every_process_run(monkeypatch):
    FakeLoads(monkeypatch, [1.0])
    name = "rust/cartpole/n1024/snapshot"
    report = bench.run(only=name, quick=True, profile="debug", runs=2, max_load5=2.0, load_wait_s=0.0)
    rule = report["load_rule"]
    assert (rule["max_load5"], rule["load_wait_s"], rule["load_void"], report["load_void"]) == (2.0, 0.0, False, False)
    assert [(r["config"], r["run"], r["void_bound"]) for r in rule["runs"]] == [(name, 0, 3.0), (name, 1, 3.0)]


def test_bench_report_marks_a_void_run(monkeypatch):
    # Readings: the invocation's start, the run's gate, the run's end.
    FakeLoads(monkeypatch, [1.0, 1.0, 3.0])
    name = "rust/cartpole/n1024/snapshot"
    report = bench.run(only=name, quick=True, profile="debug", max_load5=2.0, load_wait_s=0.0)
    assert report["load_void"] is True and report["load_rule"]["void_runs"] == [{"config": name, "run": 0}]


def test_bench_without_load_options_is_unchanged(monkeypatch):
    report = bench.run(only="rust/cartpole/n1024/snapshot", quick=True, profile="debug")
    assert "load_rule" not in report and "load_void" not in report and "containers" not in report


FOREIGN = [("tessera-bench-1", "sha256:aaa"), ("payroll-db", "sha256:bbb"), ("acme/web", "sha256:ccc")]


def test_bench_report_records_containers_at_start_and_end(monkeypatch):
    FakeLoads(monkeypatch, [1.0])
    seen = iter([FOREIGN[:2], FOREIGN])
    monkeypatch.setattr(bench, "_docker_containers", lambda: next(seen))
    report = bench.run(only="rust/cartpole/n1024/snapshot", quick=True, profile="debug", max_load5=2.0, load_wait_s=0.0)
    start, end = report["containers"]["start"], report["containers"]["end"]
    assert (start["count"], start["ours"], len(start["others"])) == (2, ["tessera-bench-1"], 1)
    assert (end["count"], len(end["others"])) == (3, 2) and set(start["others"]) < set(end["others"])


def test_bench_containers_never_name_another_service(monkeypatch, capsys):
    monkeypatch.setattr(bench, "_docker_containers", lambda: FOREIGN)
    record = bench.containers(b"k" * 16)
    text = json.dumps(record)
    for name, image in FOREIGN[1:]:
        assert name not in text and image not in text
    assert all(len(d) == 12 for d in record["others"])
    # The raw list goes to the run log (stderr), not the report.
    assert "payroll-db" in capsys.readouterr().err


def test_bench_containers_without_docker_is_none(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(bench.subprocess, "run", missing)
    assert bench.containers(b"k") is None


def test_bench_containers_with_none_running_is_empty(monkeypatch):
    class Done:
        stdout = "\n"

    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: Done())
    assert bench.containers(b"k") == {"count": 0, "ours": [], "others": []}


def test_bench_container_digests_depend_on_the_salt(monkeypatch):
    monkeypatch.setattr(bench, "_docker_containers", lambda: FOREIGN)
    a, b = bench.containers(b"a" * 16), bench.containers(b"b" * 16)
    assert a["others"] == bench.containers(b"a" * 16)["others"] and set(a["others"]).isdisjoint(b["others"])
    unsalted = hashlib.sha256(b"payroll-db\0sha256:bbb").hexdigest()[:12]
    assert unsalted not in a["others"]


def test_bench_two_runs_give_different_digests_for_the_same_container(monkeypatch):
    FakeLoads(monkeypatch, [1.0])
    monkeypatch.setattr(bench, "_docker_containers", lambda: FOREIGN)
    kw = dict(only="rust/cartpole/n1024/snapshot", quick=True, profile="debug", max_load5=2.0, load_wait_s=0.0)
    a, b = bench.run(**kw)["containers"], bench.run(**kw)["containers"]
    assert a["start"]["others"] == a["end"]["others"]
    assert set(a["start"]["others"]).isdisjoint(b["start"]["others"])


def test_bench_container_salt_goes_only_to_the_run_log(monkeypatch, capsys):
    FakeLoads(monkeypatch, [1.0])
    monkeypatch.setattr(bench, "_docker_containers", lambda: FOREIGN)
    report = bench.run(only="rust/cartpole/n1024/snapshot", quick=True, profile="debug", max_load5=2.0, load_wait_s=0.0)
    salt = next(l.split()[-1] for l in capsys.readouterr().err.splitlines() if "digest salt" in l)
    assert salt not in json.dumps(report)


def test_bench_spent_budget_writes_no_report(monkeypatch, tmp_path):
    FakeLoads(monkeypatch, [9.0])
    out = tmp_path / "r.json"
    argv = ["bench.py", "--quick", "--only", "rust/cartpole/n1024/snapshot", "--max-load5", "2", "--load-wait", "30", "--out", str(out)]
    monkeypatch.setattr(bench.sys, "argv", argv)
    monkeypatch.setattr(bench, "runner_path", lambda profile="release": bench.ROOT)
    assert bench.main() == 1 and not out.exists()


def test_bench_failing_runner_writes_its_stderr_before_raising(monkeypatch, capsys):
    # #640/#728: a failing runner's own message must reach the log, not
    # only CalledProcessError's argv.
    def fake(argv, **_):
        return subprocess.CompletedProcess(argv, 1, stdout="partial\n", stderr="engine: record 1: boom\n")

    monkeypatch.setattr(bench.subprocess, "run", fake)
    with pytest.raises(subprocess.CalledProcessError) as err:
        bench._run_runner(["runner", "snapshot", "lanterns"])
    assert err.value.returncode == 1
    assert err.value.stderr == "engine: record 1: boom\n"
    out = capsys.readouterr().err
    assert "rc=1: runner snapshot lanterns" in out
    assert "runner stderr:\nengine: record 1: boom\n" in out
    assert "runner stdout:\npartial\n" in out


def test_bench_successful_runner_output_is_returned_unchanged(monkeypatch, capsys):
    def fake(argv, **_):
        return subprocess.CompletedProcess(argv, 0, stdout="{}", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake)
    assert bench._run_runner(["runner"]).stdout == "{}"
    assert capsys.readouterr().err == ""


# -- source commit (#640) -------------------------------------------------------

_A = "a" * 40
_B = "b" * 40


def _fake_git(monkeypatch, tmp_path, head=None, rev="unknown", status="unknown", top=None):
    """``top`` is the work tree's top level: the source root when ``rev`` is a
    commit, unless given."""
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    if head is not None:
        (tmp_path / ".head").write_text(head + "\n", encoding="utf-8")
    if top is None:
        top = str(tmp_path) if rev != "unknown" else "unknown"
    answers = {("rev-parse", "--show-toplevel"): top, ("rev-parse", "HEAD"): rev, ("status", "--porcelain"): status}
    monkeypatch.setattr(bench, "_run", lambda cmd: answers.get(tuple(cmd[1:]), "unknown"))


def test_source_archive_with_head_returns_head_clean(monkeypatch, tmp_path):
    _fake_git(monkeypatch, tmp_path, head=_A)
    assert bench._source() == (_A, False, ".head")


def test_source_no_checkout_no_head_raises(monkeypatch, tmp_path):
    _fake_git(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="commit is unknown"):
        bench._source()


@pytest.mark.parametrize("bad", ["", "abc", "A" * 40, _A + "0", "g" * 40])
def test_source_malformed_head_raises(monkeypatch, tmp_path, bad):
    _fake_git(monkeypatch, tmp_path, head=bad, rev=_A, status="")
    with pytest.raises(RuntimeError, match="40-hex"):
        bench._source()


def test_source_checkout_returns_commit_and_dirty(monkeypatch, tmp_path):
    _fake_git(monkeypatch, tmp_path, rev=_A, status="")
    assert bench._source() == (_A, False, "checkout")
    _fake_git(monkeypatch, tmp_path, rev=_A, status=" M x")
    assert bench._source() == (_A, True, "checkout")


def test_source_head_disagreeing_with_checkout_raises(monkeypatch, tmp_path):
    _fake_git(monkeypatch, tmp_path, head=_B, rev=_A, status="")
    with pytest.raises(RuntimeError, match="says"):
        bench._source()


def test_source_checkout_status_failure_raises(monkeypatch, tmp_path):
    _fake_git(monkeypatch, tmp_path, rev=_A, status="unknown")
    with pytest.raises(RuntimeError, match="git status failed"):
        bench._source()


def test_source_environment_records_git_source(monkeypatch, tmp_path):
    _fake_git(monkeypatch, tmp_path, head=_A)
    software = bench.environment(None)["software"]
    assert (software["git"], software["git_dirty"], software["git_source"]) == (_A, False, ".head")


def test_run_unknown_commit_raises_before_any_runner_call(monkeypatch, tmp_path):
    _fake_git(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(bench, "runner_path", lambda *a, **k: calls.append("build") or tmp_path / "runner")
    monkeypatch.setattr(bench, "_run_runner", lambda argv: calls.append("run"))
    monkeypatch.setattr(bench, "load", lambda *a, **k: calls.append("load") or ({"config": []}, {}))
    with pytest.raises(RuntimeError, match="commit is unknown"):
        bench.run(quick=True, profile="debug")
    assert calls == []


def test_source_inside_outer_work_tree_without_head_raises(monkeypatch, tmp_path):
    _fake_git(monkeypatch, tmp_path, rev=_A, status="", top=str(tmp_path.parent))
    with pytest.raises(RuntimeError, match="no git checkout rooted"):
        bench._source()


def test_source_inside_outer_work_tree_takes_head_not_outer_commit(monkeypatch, tmp_path):
    _fake_git(monkeypatch, tmp_path, head=_B, rev=_A, status=" M x", top=str(tmp_path.parent))
    assert bench._source() == (_B, False, ".head")
