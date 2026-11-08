#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Benchmark harness (ADR-0025, SPEC 19): runs the named configurations of
``configs.toml`` and writes one JSON report.

Method: at least 2 s of warm-up and 10 repetitions per configuration;
median and 95% bootstrap confidence interval of the median (seeded). Every
configuration, Rust or Python, steps the same seeded action plan (a 64-bit
LCG, fingerprinted with FNV-1a) under the same pool options, which the
report records. ``--runs N`` repeats every configuration in N separate
processes, in a seeded shuffled order, and the statistic is then the median
and CI over the process medians; one run (the default) gives in-process
statistics, which are indicative only. Benchmarks are pinned to the
machine's ``bench_cpu_set`` with ``taskset`` where available. Derived
ratios get a bootstrap CI that resamples each operand independently.

A report is valid for claims only if ``qualify`` finds nothing missing:
the full method, process runs, release builds, applied and observed
pinning, an observed CPU matching the machine record, a reference-status
machine, and a qualification record (``--qualification``) for what cannot
be checked automatically, such as an exclusive window. Otherwise it is
indicative, and ``claim_check.reasons`` says why.

Usage::

    python tools/bench/bench.py --machine ref-x86 --runs 10 \\
        --qualification window.toml [--only PREFIX] [--out FILE]
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from qualification import _cpus_for, _work, parse_affinity  # noqa: E402
from qualification import qualify as _qualify  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIGS = pathlib.Path(__file__).resolve().parent / "configs.toml"
QUICK = {"warmup_s": 0.1, "reps": 3, "rep_s": 0.1, "batches": 50}
ACTION_BATCHES = 64


# -- statistics ---------------------------------------------------------------


def summarize(samples, resamples: int = 10000, seed: int = 0) -> dict:
    """Median and 95% bootstrap confidence interval of the median."""
    x = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(x, size=(resamples, len(x)), replace=True), axis=1)
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return {"median": float(np.median(x)), "ci95": [float(lo), float(hi)], "n": len(x), "samples": x.tolist()}


def ratio_ci(numerator, denominator, scale: float = 1.0, resamples: int = 10000, seed: int = 0) -> dict:
    """``median(numerator) / (scale * median(denominator))`` with a 95%
    bootstrap CI that resamples the two operands independently (they are
    separate measurements, not pairs)."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_med = np.median(rng.choice(num, size=(resamples, len(num)), replace=True), axis=1)
    d_med = np.median(rng.choice(den, size=(resamples, len(den)), replace=True), axis=1)
    lo, hi = np.percentile(n_med / (scale * d_med), [2.5, 97.5])
    return {
        "value": float(np.median(num) / (scale * np.median(den))),
        "ci95": [float(lo), float(hi)],
        "method": "independent bootstrap of each operand's median",
    }


# -- the shared action plan ---------------------------------------------------


def lcg_plan(rows: int, choices: int, batches: int = ACTION_BATCHES) -> np.ndarray:
    """The runner's action plan: batch ``b`` is ``rows`` draws of the LCG
    seeded with ``b``, in ``0..choices``."""
    mask = (1 << 64) - 1
    plan = np.empty((batches, rows), dtype=np.int32)
    for b in range(batches):
        state = b
        for i in range(rows):
            state = (state * 6364136223846793005 + 1442695040888963407) & mask
            plan[b, i] = (state >> 33) % choices
    return plan


def fingerprint(plan: np.ndarray) -> str:
    """FNV-1a over the little-endian int32 bytes, as the runner computes it."""
    h = 0xCBF29CE484222325
    for byte in plan.astype("<i4").tobytes():
        h = ((h ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return f"{h:016x}"


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
    from tessera_rl import _native

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
            "native_build_profile": _native.build_profile(),
            "git": _run(["git", "rev-parse", "HEAD"]),
            "git_dirty": bool(_run(["git", "status", "--porcelain"]) not in ("", "unknown")),
            "rustc": _run(["rustc", "--version"]),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
        },
    }


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


def _observed_affinity() -> str:
    """This process's CPU affinity, in the runner's format."""
    if hasattr(os, "sched_getaffinity"):
        return ",".join(str(c) for c in sorted(os.sched_getaffinity(0)))
    return "not exposed by this OS"


def _prefix(pin: dict, cpus) -> list[str]:
    if not pin["applied"] or not cpus:
        return []
    return ["taskset", "-c", ",".join(str(c) for c in cpus)]


# -- one measurement (one process) ---------------------------------------------


def runner_path(profile: str = "release") -> pathlib.Path:
    args = ["cargo", "build", "-q", "-p", "tessera-rl-bench-runner", "--locked"]
    if profile == "release":
        args.append("--release")
    subprocess.run(args, check=True, cwd=ROOT)
    return ROOT / "target" / profile / "tessera-rl-bench-runner"


def _limit(cfg: dict, method: dict) -> int:
    return int(cfg.get("max_episode_steps", method["max_episode_steps"]))


def run_rust(runner: pathlib.Path, cfg: dict, method: dict, pin: dict) -> dict:
    env, n = cfg["env"], str(cfg["num_envs"])
    metric = cfg["metric"]
    w, r = str(method["warmup_s"]), str(method["reps"])
    if metric == "throughput":
        args = [metric, env, n, str(cfg["workers"]), w, r, str(method["rep_s"])]
    elif metric == "latency":
        args = [metric, env, n, str(cfg["workers"]), w, r, str(method["batches"])]
    elif metric == "snapshot":
        args = [metric, env, n, w, r]
    elif metric == "memory":
        args = [metric, env, n]
    else:
        raise ValueError(f"unknown metric {metric!r}")
    args += ["--max-episode-steps", str(_limit(cfg, method))]
    cpus = _cpus_for(cfg, pin)
    if cpus and cfg.get("workers", 0) > 0:
        args += ["--cpus", ",".join(str(c) for c in cpus)]
    proc = subprocess.run([*_prefix(pin, cpus), str(runner), *args], capture_output=True, text=True, check=True)
    raw = json.loads(proc.stdout)
    stats = {k: summarize(v, method["bootstrap_resamples"], method["bootstrap_seed"]) for k, v in raw.items() if isinstance(v, list)}
    # The runner shares this host; its CPU model is the harness's.
    return {**{k: v for k, v in raw.items() if not isinstance(v, list)}, **stats, "cpu": _cpu_model()}


def _python_env(cfg: dict, method: dict):
    """(step function, steps per call, close, plan) for a Python
    configuration, stepping the runner's plan under the same options."""
    n = cfg["num_envs"]
    api = cfg["api"]
    limit = _limit(cfg, method)
    plan = lcg_plan(n, 2)
    if api == "native":
        import tessera_rl

        env = tessera_rl.make(
            "tessera/CartPole-v1", num_envs=n, seed=1, outputs=cfg["outputs"], auto_reset="next_step", max_episode_steps=limit
        )
        env.reset()
        acts = [p.reshape(n, 1) for p in plan]
        return (lambda k: env.step(acts[k % ACTION_BATCHES])), n, env.close, plan
    if api == "gym":
        import tessera_rl.gym as tg

        env = tg.make("tessera/CartPole-v1", num_envs=n, seed=1, outputs=cfg["outputs"], max_episode_steps=limit)
        env.reset(seed=1)
        return (lambda k: env.step(plan[k % ACTION_BATCHES])), n, env.close, plan
    if api == "gymnasium":
        import gymnasium

        env = gymnasium.make_vec("CartPole-v1", num_envs=n, vectorization_mode="sync", max_episode_steps=limit)
        env.reset(seed=1)
        return (lambda k: env.step(plan[k % ACTION_BATCHES])), n, env.close, plan
    raise ValueError(f"unknown api {api!r}")


def measure_python(cfg: dict, method: dict) -> dict:
    """One Python measurement in this process."""
    from tessera_rl import _native

    step, per_call, close, plan = _python_env(cfg, method)

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
    return {
        "metric": "throughput",
        "unit": "steps/s",
        "method": {"warmup_s": method["warmup_s"], "reps": method["reps"], "rep_s": method["rep_s"]},
        "options": {"seed": 1, "auto_reset": "next_step", "max_episode_steps": _limit(cfg, method), "action_batches": ACTION_BATCHES},
        "plan_fingerprint": fingerprint(plan),
        "profile": _native.build_profile(),
        "cpu": _cpu_model(),
        "pinning": {"process_affinity": _observed_affinity()},
        "samples": summarize(samples, method["bootstrap_resamples"], method["bootstrap_seed"]),
    }


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
        return {**first, "statistic": "in-process repetitions (indicative)", "runs": per_run}
    out = {k: v for k, v in first.items() if not (isinstance(v, dict) and "median" in v)}
    for key, value in first.items():
        if isinstance(value, dict) and "median" in value:
            medians = [r[key]["median"] for r in per_run]
            out[key] = summarize(medians, method["bootstrap_resamples"], method["bootstrap_seed"])
    out["statistic"] = f"median and 95% bootstrap CI over {len(per_run)} process medians"
    out["runs"] = per_run
    return out


def _samples(results: dict, name: str):
    r = results.get(name)
    return None if r is None else r["samples"]["samples"]


def _same_work(a: dict, b: dict) -> bool:
    """Every run on both sides ran one and the same plan and options."""
    work_a, work_b = _work(a), _work(b)
    return len(work_a) == 1 and work_a == work_b


def derive(results: dict, derived: list[dict], method: dict) -> dict:
    out = {}
    boot = {"resamples": method["bootstrap_resamples"], "seed": method["bootstrap_seed"]}
    for d in derived:
        if d["kind"] == "scaling":
            a, b = d["multi"], d["single"]
            scale = float(d["workers"])
        elif d["kind"] == "ratio":
            a, b = d["numerator"], d["denominator"]
            scale = 1.0
        else:
            raise ValueError(f"unknown derived kind {d['kind']!r}")
        num, den = _samples(results, a), _samples(results, b)
        if num is None or den is None:
            continue
        entry = {**ratio_ci(num, den, scale, **boot), "from": [a, b]}
        entry["same_work"] = _same_work(results[a], results[b])
        if d.get("require_same_work") and not entry["same_work"]:
            entry = {"value": None, "from": [a, b], "invalid": "the operands ran different plans or options"}
        out[d["name"]] = entry
    for name, r in results.items():
        if r.get("metric") == "snapshot":
            kib = r["record_bytes"] / 1024
            copy = r["snapshot_all_copy"]
            out[f"{name}/copy_ns_per_kib"] = {
                "value": copy["median"] / kib,
                # A linear transform of snapshot_all_copy, so its CI scales.
                "ci95": [copy["ci95"][0] / kib, copy["ci95"][1] / kib],
                "budget_ns_per_kib": 1000.0,  # ADR-0010: at most 1 us/KB
                "from": [name],
            }
    return out


# -- qualification ------------------------------------------------------------


def load_average() -> list[float] | None:
    """The 1, 5 and 15 minute load averages, or None where the OS has none
    (Windows): a run without it is never valid for claims."""
    try:
        return list(os.getloadavg())
    except (AttributeError, OSError):
        return None


def qualify(report: dict, qualification: dict | None) -> list[str]:
    """Reasons the report is NOT valid for claims, against the current record."""
    machine = report["environment"]["machine"]
    return _qualify(report, qualification, machines().get(machine) if machine else None)


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
    qualification: dict | None = None,
) -> dict:
    spec, method = load(quick)
    configs = [c for c in spec["config"] if only is None or c["name"].startswith(only)]
    if not configs:
        raise ValueError(f"no configuration matches {only!r}")
    pin = pinning(machine)
    load_before = load_average()
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
    report = {
        "schema": "tessera-bench/3",
        "quick": quick,
        "runs": runs,
        "run_orders": orders,
        "method": method,
        "pinning": pin,
        "qualification": qualification,
        "environment": environment(machine),
        "load_average": {"before": load_before, "after": load_average()},
        "results": results,
        "derived": derive(results, spec.get("derived", []), method),
    }
    reasons = qualify(report, qualification)
    report["valid_for_claims"] = not reasons
    report["claim_check"] = {"valid": not reasons, "reasons": reasons}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--machine", help="reference machine name from compat/machines.toml (required for claims)")
    parser.add_argument("--only", help="run only configurations whose name starts with this prefix")
    parser.add_argument("--runs", type=int, default=1, help="separate processes per configuration, in shuffled order")
    parser.add_argument("--qualification", type=pathlib.Path, help="TOML record of the run's conditions (machine, exclusive_window, operator, notes)")
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
    qualification = tomllib.loads(args.qualification.read_text()) if args.qualification else None
    try:
        report = run(args.only, args.quick, args.machine, runs=args.runs, qualification=qualification)
    except ValueError as e:
        parser.error(str(e))
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
