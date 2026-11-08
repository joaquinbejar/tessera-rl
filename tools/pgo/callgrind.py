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
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evidence  # noqa: E402

WORKLOAD = pathlib.Path(__file__).resolve().parent / "workload.py"


def command(python: str, manifest: str, scenario: str, steps: int, out: str, separate_threads: bool = False) -> list[str]:
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
    ]


def measure(python: str, manifest: str, scenario: str, steps: int, out_dir: pathlib.Path, runner=subprocess.run) -> evidence.Cost:
    """The native cost of one run; a failing run raises."""
    safe = scenario.replace("/", "_")
    out = out_dir / f"{safe}.{steps}.callgrind"
    runner(command(python, manifest, scenario, steps, str(out)), check=True)
    return evidence.native_cost(evidence.parse_callgrind(out.read_text(encoding="utf-8")))


def cost_of(python: str, manifest: str, s: evidence.Scenario, out_dir, runner=subprocess.run) -> evidence.Cost:
    """The scenario's cost with setup removed: the run at 2N minus the run at N."""
    short = measure(python, manifest, s.name, s.steps, out_dir, runner)
    return measure(python, manifest, s.name, 2 * s.steps, out_dir, runner) - short


def main(argv=None, platform=sys.platform) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approved-window", default=None)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    if not args.approved_window:
        print("refused: Callgrind runs need --approved-window (design section 8)", file=sys.stderr)
        return 2
    if platform != "linux":
        print("refused: Valgrind needs Linux", file=sys.stderr)
        return 2
    manifest = evidence.load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    costs = {
        s.name: cost_of(args.python, args.manifest, s, args.out_dir).events
        for s in evidence.scenarios(manifest)
    }
    (args.out_dir / "costs.json").write_text(json.dumps({"approved_window": args.approved_window, "costs": costs}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
