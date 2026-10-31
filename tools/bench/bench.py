#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Benchmark harness (ADR-0025, SPEC 19): runs the named configurations of
``configs.toml`` and writes one JSON report.

Method: at least 2 s of warm-up and 10 repetitions per configuration;
median and 95% bootstrap confidence interval of the median (seeded); fixed
seeds and action streams. ``--runs N`` repeats every configuration in N
separate processes, in a seeded shuffled order, and the claim statistic is
the median and CI over the process medians; one run (the default) gives
in-process statistics, which are indicative only. Benchmarks are pinned to
the machine's ``bench_cpu_set`` with ``taskset`` where available. Rust configurations run ``tessera-rl-bench-runner``
(raw samples); Python ones run in this interpreter. The report records the
machine, the hardware, the CPU frequency policy, the load, and the
software versions, and marks whether the run is valid for claims
(``--quick`` runs never are).

Usage::

    python tools/bench/bench.py --machine ref-x86 --runs 10 [--only PREFIX] [--out FILE]
    python tools/bench/bench.py --quick     # smoke run, not for claims
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time
import tomllib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIGS = pathlib.Path(__file__).resolve().parent / "configs.toml"
QUICK = {"warmup_s": 0.1, "reps": 3, "rep_s": 0.1, "batches": 50}


# -- statistics ---------------------------------------------------------------


def summarize(samples, resamples: int = 10000, seed: int = 0) -> dict:
    """Median and 95% bootstrap confidence interval of the median."""
    x = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(x, size=(resamples, len(x)), replace=True), axis=1)
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return {"median": float(np.median(x)), "ci95": [float(lo), float(hi)], "n": len(x), "samples": x.tolist()}


# -- environment description --------------------------------------------------


def _run(cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cpu_model() -> str:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _frequency_policy() -> str:
    gov = pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov.exists():
        return f"governor={gov.read_text().strip()}"
    if sys.platform == "darwin":
        return "macOS: no user-visible governor (frequency managed by the OS)"
    return "unknown"


def machines() -> dict:
    """The reference machines of compat/machines.toml, by name."""
    spec = tomllib.loads((ROOT / "compat" / "machines.toml").read_text())
    return {k: v for k, v in spec.items() if isinstance(v, dict)}


def environment(machine: str | None) -> dict:
    import gymnasium

    import tessera_rl

    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "machine": machine,
        "machine_record": machines().get(machine) if machine else None,
        "hardware": {
            "cpu": _cpu_model(),
            "arch": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "os": platform.platform(),
            "frequency_policy": _frequency_policy(),
            "affinity": affinity,
        },
        "software": {
            "tessera_rl": tessera_rl.__version__,
            "git": _run(["git", "rev-parse", "HEAD"]),
            "git_dirty": bool(_run(["git", "status", "--porcelain"]) not in ("", "unknown")),
            "rustc": _run(["rustc", "--version"]),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
        },
    }


# -- Rust configurations ------------------------------------------------------


def runner_path(profile: str = "release") -> pathlib.Path:
    args = ["cargo", "build", "-q", "-p", "tessera-rl-bench-runner", "--locked"]
    if profile == "release":
        args.append("--release")
    subprocess.run(args, check=True, cwd=ROOT)
    return ROOT / "target" / profile / "tessera-rl-bench-runner"


# -- pinning ------------------------------------------------------------------


def pinning(machine: str | None) -> dict:
    """The CPUs benchmarks run on: the machine's ``bench_cpu_set`` (distinct
    physical cores, CPU 0 avoided, SMT siblings idle), applied with
    ``taskset`` where it exists (Linux). Elsewhere runs are unpinned, and
    the report says so."""
    record = machines().get(machine, {}) if machine else {}
    cpu_set = record.get("bench_cpu_set")
    taskset = shutil.which("taskset")
    return {
        "cpu_set": cpu_set,
        "applied": bool(cpu_set and taskset),
        "tool": "taskset" if taskset else None,
        "note": None if cpu_set and taskset else "unpinned: no bench_cpu_set for this machine or no taskset on this OS",
    }


def _prefix(pin: dict, cpus) -> list[str]:
    if not pin["applied"] or not cpus:
        return []
    return ["taskset", "-c", ",".join(str(c) for c in cpus)]


def _cpus_for(cfg: dict, pin: dict):
    """Process CPUs for a configuration: one core for Inline and Python
    runs, one per worker for Sync runs."""
    if not pin["applied"]:
        return None
    k = max(1, int(cfg.get("workers", 0)))
    if len(pin["cpu_set"]) < k:
        raise ValueError(f"{cfg['name']}: bench_cpu_set has {len(pin['cpu_set'])} CPUs, needs {k}")
    return pin["cpu_set"][:k]


# -- one measurement (one process) ---------------------------------------------


def run_rust(runner: pathlib.Path, cfg: dict, method: dict, pin: dict) -> dict:
    env, n = cfg["env"], str(cfg["num_envs"])
    metric = cfg["metric"]
    if metric == "throughput":
        args = [metric, env, n, str(cfg["workers"]), str(method["warmup_s"]), str(method["reps"]), str(method["rep_s"])]
    elif metric == "latency":
        args = [metric, env, n, str(cfg["workers"]), str(method["warmup_s"]), str(method["reps"]), str(method["batches"])]
    elif metric == "snapshot":
        args = [metric, env, n, str(method["reps"])]
    elif metric == "memory":
        args = [metric, env, n]
    else:
        raise ValueError(f"unknown metric {metric!r}")
    cpus = _cpus_for(cfg, pin)
    if cpus and cfg.get("workers", 0) > 0:
        args += ["--cpus", ",".join(str(c) for c in cpus)]
    proc = subprocess.run([*_prefix(pin, cpus), str(runner), *args], capture_output=True, text=True, check=True)
    raw = json.loads(proc.stdout)
    stats = {k: summarize(v, method["bootstrap_resamples"], method["bootstrap_seed"]) for k, v in raw.items() if isinstance(v, list)}
    return {**{k: v for k, v in raw.items() if not isinstance(v, list)}, **stats}


def _python_env(cfg: dict):
    """(step function, steps per call, close) for a Python configuration."""
    n = cfg["num_envs"]
    api = cfg["api"]
    rng = np.random.default_rng(1)
    plan = rng.integers(2, size=(64, n))
    if api == "native":
        import tessera_rl

        env = tessera_rl.make("tessera/CartPole-v1", num_envs=n, seed=1, outputs=cfg["outputs"])
        env.reset()
        acts = [p.reshape(n, 1).astype(np.int32) for p in plan]
        return (lambda k: env.step(acts[k % 64])), n, env.close
    if api == "gym":
        import tessera_rl.gym as tg

        env = tg.make("tessera/CartPole-v1", num_envs=n, seed=1, outputs=cfg["outputs"])
        env.reset(seed=1)
        return (lambda k: env.step(plan[k % 64])), n, env.close
    if api == "gymnasium":
        import gymnasium

        env = gymnasium.make_vec("CartPole-v1", num_envs=n, vectorization_mode="sync")
        env.reset(seed=1)
        return (lambda k: env.step(plan[k % 64])), n, env.close
    raise ValueError(f"unknown api {api!r}")


def measure_python(cfg: dict, method: dict) -> dict:
    """One Python measurement in this process."""
    step, per_call, close = _python_env(cfg)

    def run_for(seconds):
        start = time.perf_counter()
        k = 0
        while time.perf_counter() - start < seconds:
            step(k)
            k += 1
        return k * per_call / (time.perf_counter() - start)

    run_for(method["warmup_s"])
    samples = [run_for(method["rep_s"]) for _ in range(method["reps"])]
    close()
    return {"metric": "throughput", "unit": "steps/s", "samples": summarize(samples, method["bootstrap_resamples"], method["bootstrap_seed"])}


def run_python(cfg: dict, method: dict, pin: dict, quick: bool) -> dict:
    """One Python measurement in a fresh interpreter (``--single``)."""
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--single", cfg["name"]]
    if quick:
        cmd.append("--quick")
    proc = subprocess.run([*_prefix(pin, _cpus_for(cfg, pin)), *cmd], capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


# -- aggregation --------------------------------------------------------------


def combine(per_run: list[dict], method: dict) -> dict:
    """With one run, its in-process summary. With several, the median and
    95% bootstrap CI over the run medians (the claim statistic)."""
    first = per_run[0]
    if len(per_run) == 1:
        return {**first, "statistic": "in-process repetitions (indicative)"}
    out = {k: v for k, v in first.items() if not (isinstance(v, dict) and "median" in v)}
    for key, value in first.items():
        if isinstance(value, dict) and "median" in value:
            medians = [r[key]["median"] for r in per_run]
            out[key] = summarize(medians, method["bootstrap_resamples"], method["bootstrap_seed"])
    out["statistic"] = f"median and 95% bootstrap CI over {len(per_run)} process medians"
    out["runs"] = per_run
    return out


def _median(results: dict, name: str) -> float | None:
    r = results.get(name)
    return None if r is None else r["samples"]["median"]


def derive(results: dict, derived: list[dict]) -> dict:
    out = {}
    for d in derived:
        if d["kind"] == "scaling":
            multi, single = _median(results, d["multi"]), _median(results, d["single"])
            if multi is not None and single:
                out[d["name"]] = {"value": multi / (d["workers"] * single), "from": [d["multi"], d["single"]]}
        elif d["kind"] == "ratio":
            num, den = _median(results, d["numerator"]), _median(results, d["denominator"])
            if num is not None and den:
                out[d["name"]] = {"value": num / den, "from": [d["numerator"], d["denominator"]]}
    for name, r in results.items():
        if r.get("metric") == "snapshot":
            kib = r["record_bytes"] / 1024
            out[f"{name}/copy_ns_per_kib"] = {
                "value": r["snapshot_all_copy"]["median"] / kib,
                "budget_ns_per_kib": 1000.0,  # ADR-0010: at most 1 us/KB
                "from": [name],
            }
    return out


# -- main ---------------------------------------------------------------------


def load(quick: bool):
    spec = tomllib.loads(CONFIGS.read_text())
    method = dict(spec["method"])
    if quick:
        method.update(QUICK)
    return spec, method


def run(
    only: str | None = None,
    quick: bool = False,
    machine: str | None = None,
    profile: str = "release",
    runs: int = 1,
) -> dict:
    spec, method = load(quick)
    configs = [c for c in spec["config"] if only is None or c["name"].startswith(only)]
    pin = pinning(machine)
    load_before = os.getloadavg()
    runner = runner_path(profile) if any(c["kind"] == "rust" for c in configs) else None
    order_rng = np.random.default_rng(method["bootstrap_seed"])
    per_run: dict[str, list] = {c["name"]: [] for c in configs}
    orders = []
    for _ in range(runs):
        order = [configs[i] for i in order_rng.permutation(len(configs))]
        orders.append([c["name"] for c in order])
        for cfg in order:
            if cfg["kind"] == "rust":
                per_run[cfg["name"]].append(run_rust(runner, cfg, method, pin))
            else:
                per_run[cfg["name"]].append(run_python(cfg, method, pin, quick))
    results = {c["name"]: {"config": c, **combine(per_run[c["name"]], method)} for c in configs}
    claims = (
        not quick
        and method["warmup_s"] >= 2.0
        and method["reps"] >= 10
        and runs >= method["runs_for_claims"]
        and pin["applied"]
        and machine is not None
        and machines().get(machine, {}).get("status") == "reference"
    )
    return {
        "schema": "tessera-bench/2",
        # A claim needs the full method, process-level repetitions, pinning
        # and a characterized reference machine (ADR-0025); anything else
        # gives indicative numbers.
        "valid_for_claims": claims,
        "quick": quick,
        "runs": runs,
        "run_orders": orders,
        "method": method,
        "pinning": pin,
        "environment": environment(machine),
        "load_average": {"before": list(load_before), "after": list(os.getloadavg())},
        "results": results,
        "derived": derive(results, spec.get("derived", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--machine", help="reference machine name from compat/machines.toml (required for claims)")
    parser.add_argument("--only", help="run only configurations whose name starts with this prefix")
    parser.add_argument("--runs", type=int, default=1, help="separate processes per configuration, in shuffled order")
    parser.add_argument("--quick", action="store_true", help="short smoke run; never valid for claims")
    parser.add_argument("--out", type=pathlib.Path, help="write the JSON report here (default: stdout)")
    parser.add_argument("--single", help=argparse.SUPPRESS)  # one Python measurement, internal
    args = parser.parse_args()
    if args.single:
        spec, method = load(args.quick)
        cfg = next(c for c in spec["config"] if c["name"] == args.single)
        sys.stdout.write(json.dumps(measure_python(cfg, method)) + "\n")
        return 0
    if args.machine is not None and args.machine not in machines():
        parser.error(f"unknown machine {args.machine!r}; known: {sorted(machines())}")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    report = run(args.only, args.quick, args.machine, runs=args.runs)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
