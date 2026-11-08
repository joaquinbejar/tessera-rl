#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Measures manifest scenarios under Callgrind, per object, by difference
(docs/reviews/pgo-wheels-design.md r4, section 5).

Refuses to run without `--approved-window`, the coordinator's approval of a host
window, and only on Linux. This has never run against Valgrind: command lines and
the handling of its output are unproven until the feasibility gate.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evidence  # noqa: E402

WORKLOAD = pathlib.Path(__file__).resolve().parent / "workload.py"
CONTROL = "control/interpreter-only"
AA_RUNS = 5


def command(python: str, manifest: str, scenario: str, steps: int, out: str, separate_threads: bool = False, extra: tuple[str, ...] = ()) -> list[str]:
    """The Callgrind command of one run. Threads are aggregated unless the
    feasibility control needs one file per thread."""
    return [
        "valgrind",
        "--tool=callgrind",
        "--cache-sim=yes",
        "--branch-sim=yes",
        f"--separate-threads={'yes' if separate_threads else 'no'}",
        f"--callgrind-out-file={out}",
        python,
        "-I",
        str(WORKLOAD),
        "--manifest",
        manifest,
        "--scenario",
        scenario,
        "--steps",
        str(steps),
        *extra,
    ]


def measure(python: str, manifest: str, scenario: str, steps: int, out_dir: pathlib.Path, runner=subprocess.run, tag: str = "", extra: tuple[str, ...] = ()) -> evidence.Cost:
    """The native cost of one run; a failing run raises."""
    safe = scenario.replace("/", "_")
    out = out_dir / f"{safe}.{steps}{tag}.callgrind"
    runner(command(python, manifest, scenario, steps, str(out), extra=extra), check=True)
    return evidence.native_cost(evidence.parse_callgrind(out.read_text(encoding="utf-8")))


def cost_of(python: str, manifest: str, s: evidence.Scenario, out_dir, runner=subprocess.run, tag: str = "", extra: tuple[str, ...] = ()) -> evidence.Cost:
    """The scenario's cost with setup removed: the run at 2N minus the run at N."""
    short = measure(python, manifest, s.name, s.steps, out_dir, runner, tag, extra)
    return measure(python, manifest, s.name, 2 * s.steps, out_dir, runner, tag, extra) - short


def thread_files(out: pathlib.Path) -> list[pathlib.Path]:
    """The `<out>-<tid>` files of a separate-threads run, by numeric thread id:
    tid 1 is the caller, and "-10" must not sort before "-2"."""
    found = []
    for path in out.parent.glob(out.name + "-*"):
        suffix = path.name[len(out.name) + 1 :]
        if suffix.isdigit():
            found.append((int(suffix), path))
    return [path for _, path in sorted(found)]


def worker_control(python: str, manifest: str, s: evidence.Scenario, out_dir, runner=subprocess.run) -> dict:
    """Section 5 item 4. Sync threaded scenarios against the same steps at 0
    workers; modes that need workers from the per-thread files."""
    if s.mode == "sync":
        threaded = cost_of(python, manifest, s, out_dir, runner, ".workers").ir
        inline = cost_of(python, manifest, s, out_dir, runner, ".inline", ("--workers", "0")).ir
        return {"kind": "against-inline", "passed": evidence.workers_collected(threaded, inline)}
    out = out_dir / f"{s.name.replace('/', '_')}.threads.callgrind"
    runner(command(python, manifest, s.name, s.steps, str(out), separate_threads=True), check=True)
    texts = [p.read_text(encoding="utf-8") for p in thread_files(out)]
    others, share = evidence.non_caller_native(texts)
    return {"kind": "per-thread", "passed": others > 0, "non_caller_ir": others, "share": share}


def feasibility(python: str, manifest_path: str, manifest: dict, out_dir: pathlib.Path, runner=subprocess.run) -> dict:
    """The gate of section 5, on W1. Every check is recorded; `passed` needs all."""
    all_scenarios = evidence.scenarios(manifest)
    control = measure(python, manifest_path, CONTROL, 1, out_dir, runner).ir
    report: dict = {"control_native_ir": control, "scenarios": {}}
    aa: dict[str, list[evidence.Cost]] = {}
    ok = True
    for s in all_scenarios:
        entry: dict = {}
        aa[s.name] = [cost_of(python, manifest_path, s, out_dir, runner, f".aa{i}") for i in range(AA_RUNS)]
        entry["aa_ir"] = [c.ir for c in aa[s.name]]
        entry["aa_ir_spread"] = evidence.spread(entry["aa_ir"])
        median_ir = int(statistics.median(entry["aa_ir"]))
        entry["interpreter_excluded"] = evidence.interpreter_excluded(control, median_ir)
        ok &= entry["interpreter_excluded"]
        if s.single_threaded:
            irs = [measure(python, manifest_path, s.name, k * s.steps, out_dir, runner, ".linear").ir for k in (1, 2, 3)]
            entry["linear"] = evidence.linear(*irs)
            ok &= entry["linear"]
        else:
            entry["workers"] = worker_control(python, manifest_path, s, out_dir, runner)
            ok &= entry["workers"]["passed"]
        report["scenarios"][s.name] = entry
    sets = evidence.gating_sets(all_scenarios, aa)
    report["gating"] = list(sets.gating)
    report["mispredict"] = list(sets.mispredict)
    report["mispredict_available"] = sets.mispredict_available
    report["usable"] = sets.usable
    report["passed"] = bool(ok and sets.usable)
    return report


def main(argv=None, platform=sys.platform) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approved-window", default=None)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--feasibility", action="store_true", help="the gate of section 5, on W1")
    args = parser.parse_args(argv)
    if not args.approved_window:
        print("refused: Callgrind runs need --approved-window (design section 8)", file=sys.stderr)
        return 2
    if platform != "linux":
        print("refused: Valgrind needs Linux", file=sys.stderr)
        return 2
    manifest = evidence.load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.feasibility:
        report = feasibility(args.python, args.manifest, manifest, args.out_dir)
        report["approved_window"] = args.approved_window
        (args.out_dir / "feasibility.json").write_text(json.dumps(report, indent=1))
        return 0 if report["passed"] else 1
    costs = {
        s.name: cost_of(args.python, args.manifest, s, args.out_dir).events
        for s in evidence.scenarios(manifest)
    }
    (args.out_dir / "costs.json").write_text(json.dumps({"approved_window": args.approved_window, "costs": costs}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
