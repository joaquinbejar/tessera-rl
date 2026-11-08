# SPDX-License-Identifier: MIT OR Apache-2.0
"""The claim rules of ADR-0025 over a benchmark report (standard library
only: ``tools/dashboard`` applies the same rules to stored reports)."""

from __future__ import annotations

import json


def parse_affinity(text) -> set[int] | None:
    """A CPU list such as ``2,4`` or ``0-3,8`` (Linux ``Cpus_allowed_list``
    syntax) as a set; ``None`` when the OS does not expose it."""
    if not isinstance(text, str) or not text or not text[0].isdigit():
        return None
    cpus = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _cpus_for(cfg: dict, pin: dict):
    """Process CPUs for a configuration: one core for Inline and Python
    runs, one per worker for Sync runs."""
    if not pin["applied"]:
        return None
    k = max(1, int(cfg.get("workers", 0)))
    if len(pin["cpu_set"]) < k:
        raise ValueError(f"{cfg['name']}: bench_cpu_set has {len(pin['cpu_set'])} CPUs, needs {k}")
    return pin["cpu_set"][:k]


def _work(result: dict) -> set:
    """The distinct (options, plan) pairs over every process run."""
    return {
        (json.dumps(r.get("options"), sort_keys=True), r.get("plan_fingerprint"))
        for r in result.get("runs", [result])
    }


def qualify(report: dict, qualification: dict | None, record: dict | None) -> list[str]:
    """Reasons the report is NOT valid for claims; empty means valid.

    ``record`` is the machine's record: the caller says which one, so a
    stored report can be judged against the record it embeds."""
    reasons = []
    method = report["method"]
    env = report["environment"]
    machine = env["machine"]
    if report["quick"]:
        reasons.append("quick run")
    if method["warmup_s"] < 2.0 or method["reps"] < 10:
        reasons.append("method below 2 s warm-up and 10 repetitions")
    if report["runs"] < method["runs_for_claims"]:
        reasons.append(f"fewer than {method['runs_for_claims']} process runs")
    if not report["results"]:
        reasons.append("no results")
    if record is None:
        reasons.append("no machine named")
    else:
        if record.get("status") != "reference":
            reasons.append(f"machine status is {record.get('status')!r}, not 'reference'")
        cpu_id = record.get("cpu_id")
        if not cpu_id or cpu_id not in env["hardware"]["cpu"]:
            reasons.append(f"observed CPU {env['hardware']['cpu']!r} does not match the record's cpu_id {cpu_id!r}")
    if env["software"]["native_build_profile"] != "release":
        reasons.append("the native module is not a release build")
    if not report["pinning"]["applied"]:
        reasons.append("pinning not applied")
    load = report.get("load_average") or {}
    if load.get("before") is None or load.get("after") is None:
        reasons.append("load average unavailable on this OS")
    cpu_id = record.get("cpu_id") if record else None
    for name, r in report["results"].items():
        runs = r.get("runs", [])
        if len(runs) != report["runs"]:
            reasons.append(f"{name}: {len(runs)} process records, expected {report['runs']}")
        if len(_work(r)) != 1:
            reasons.append(f"{name}: the process runs ran different plans or options")
        cpus = _cpus_for(r["config"], report["pinning"]) if report["pinning"]["applied"] else None
        # Every process record must qualify, not only the first.
        for k, run in enumerate(runs):
            if run.get("profile") != "release":
                reasons.append(f"{name} run {k}: {run.get('profile')!r} build")
            observed = parse_affinity(run.get("pinning", {}).get("process_affinity"))
            if cpus and observed != set(cpus):
                reasons.append(f"{name} run {k}: observed affinity {observed} differs from {cpus}")
            if not cpu_id or cpu_id not in str(run.get("cpu", "")):
                reasons.append(f"{name} run {k}: CPU {run.get('cpu')!r} does not match {cpu_id!r}")
            if r["config"].get("metric") != "memory":
                eff = run.get("method") or {}
                if eff.get("warmup_s") != method["warmup_s"] or eff.get("reps") != method["reps"]:
                    reasons.append(f"{name} run {k}: effective method {eff} differs from the report's")
                for key, value in run.items():
                    if isinstance(value, dict) and "n" in value and value["n"] != method["reps"]:
                        reasons.append(f"{name} run {k}: {key} has {value['n']} samples, expected {method['reps']}")
        for key, value in r.items():
            if isinstance(value, dict) and "n" in value and value["n"] < report["runs"]:
                reasons.append(f"{name}: {key} has {value['n']} samples")
    if qualification is None:
        reasons.append("no qualification record (an exclusive window cannot be verified automatically)")
    elif qualification.get("machine") != machine or not qualification.get("exclusive_window"):
        reasons.append("the qualification record does not cover this machine with an exclusive window")
    return reasons
