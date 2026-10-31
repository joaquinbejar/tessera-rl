#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Benchmark harness (ADR-0025, SPEC 19): runs the named configurations of
``configs.toml`` and writes one JSON report.

Method: at least 2 s of warm-up and 10 repetitions per configuration;
median and 95% bootstrap confidence interval of the median (seeded); fixed
seeds and action streams. Rust configurations run ``tessera-rl-bench-runner``
(raw samples); Python ones run in this interpreter. The report records the
machine, the hardware, the CPU frequency policy, the load, and the
software versions, and marks whether the run is valid for claims
(``--quick`` runs never are).

Usage::

    python tools/bench/bench.py --machine ref-mac [--only PREFIX] [--out FILE]
    python tools/bench/bench.py --quick     # smoke run, not for claims
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
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


def run_rust(runner: pathlib.Path, cfg: dict, method: dict) -> dict:
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
    raw = json.loads(subprocess.run([str(runner), *args], capture_output=True, text=True, check=True).stdout)
    stats = {k: summarize(v, method["bootstrap_resamples"], method["bootstrap_seed"]) for k, v in raw.items() if isinstance(v, list)}
    scalars = {k: v for k, v in raw.items() if not isinstance(v, list)}
    return {**scalars, **stats}


# -- Python configurations ----------------------------------------------------


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


def run_python(cfg: dict, method: dict) -> dict:
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


# -- derived metrics ----------------------------------------------------------


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
    return out


# -- main ---------------------------------------------------------------------


def run(only: str | None = None, quick: bool = False, machine: str | None = None, profile: str = "release") -> dict:
    spec = tomllib.loads(CONFIGS.read_text())
    method = dict(spec["method"])
    if quick:
        method.update(QUICK)
    configs = [c for c in spec["config"] if only is None or c["name"].startswith(only)]
    load_before = os.getloadavg()
    runner = runner_path(profile) if any(c["kind"] == "rust" for c in configs) else None
    results = {}
    for cfg in configs:
        result = run_rust(runner, cfg, method) if cfg["kind"] == "rust" else run_python(cfg, method)
        results[cfg["name"]] = {"config": cfg, **result}
    return {
        "schema": "tessera-bench/1",
        # A claim needs the full method on a characterized reference machine
        # (ADR-0025): candidates and pending machines give indicative numbers.
        "valid_for_claims": (
            not quick
            and method["warmup_s"] >= 2.0
            and method["reps"] >= 10
            and machine is not None
            and machines().get(machine, {}).get("status") == "reference"
        ),
        "quick": quick,
        "method": method,
        "environment": environment(machine),
        "load_average": {"before": list(load_before), "after": list(os.getloadavg())},
        "results": results,
        "derived": derive(results, spec.get("derived", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--machine", help="reference machine name from compat/machines.toml (required for claims)")
    parser.add_argument("--only", help="run only configurations whose name starts with this prefix")
    parser.add_argument("--quick", action="store_true", help="short smoke run; never valid for claims")
    parser.add_argument("--out", type=pathlib.Path, help="write the JSON report here (default: stdout)")
    args = parser.parse_args()
    if args.machine is not None and args.machine not in machines():
        parser.error(f"unknown machine {args.machine!r}; known: {sorted(machines())}")
    report = run(args.only, args.quick, args.machine)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
