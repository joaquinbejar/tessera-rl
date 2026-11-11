#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The cross-window control of a ladder window (pgo-ladder-plan.md r4): a probe
of W1 on the gating scenarios must give exactly the native Ir the feasibility
gate measured for the same N-step runs, since its A/A spread was exactly 0.

    control.py expected --gate-out OUT --feasibility F --manifest M --out expected.json
    control.py check PROBE_DIR expected.json

`expected` reads the gate's own finished `aa0` N-step files (no Valgrind) and
pins, per gating scenario, their native Ir, with the gate's stamp. `check` reads
the probe reports a window wrote in PROBE_DIR (callgrind.py --probe) and exits 1
on any scenario that differs by even one instruction, is missing, was probed
twice, was probed on another wheel, or under a stamp whose other fields
(manifest, interpreter, Valgrind, CPU) differ from the gate's. `kernels`, which
the gate's older stamp carries, is ignored as in decide.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evidence  # noqa: E402


def _rest(stamp: dict) -> dict:
    return {k: v for k, v in stamp.items() if k not in ("wheel", "kernels")}


def expected(gate_out: pathlib.Path, feasibility: dict, manifest: dict) -> dict:
    by_name = {s.name: s for s in evidence.scenarios(manifest)}
    values = {}
    for name in feasibility["gating"]:
        s = by_name[name]
        path = gate_out / f"{name.replace('/', '_')}.{s.steps}.aa0.callgrind"
        values[name] = evidence.native_cost(evidence.parse_callgrind(path.read_text(encoding="utf-8"))).ir
    if not values:
        raise ValueError("the gate names no gating scenario")
    return {"wheel": feasibility["sha256"]["wheel"], "stamp": _rest(feasibility["sha256"]), "native_ir": values}


def check(probe_dir: pathlib.Path, want: dict) -> list[str]:
    """The differences; empty when the control holds."""
    got, wheels, problems = {}, set(), []
    for p in sorted(probe_dir.glob("feasibility.probe.*.json")):
        report = json.loads(p.read_text(encoding="utf-8"))
        stamp = report.get("sha256") or {}
        wheels.add(stamp.get("wheel"))
        if _rest(stamp) != want["stamp"]:
            diff = sorted(k for k in set(want["stamp"]) | set(_rest(stamp)) if want["stamp"].get(k) != stamp.get(k))
            problems.append(f"{p.name}: stamp differs from the gate's in {diff}")
        for name, r in report["probe"].items():
            if name in got:
                problems.append(f"{name}: probed twice")
            got[name] = r["ir"]
    if wheels != {want["wheel"]}:
        problems.append(f"probed wheels {sorted(map(str, wheels))}, not W1 {want['wheel']}")
    for name, ir in want["native_ir"].items():
        if name not in got:
            problems.append(f"{name}: not probed")
        elif got[name] != ir:
            problems.append(f"{name}: {got[name]} != gate {ir} ({got[name] - ir:+d})")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("expected")
    for name in ("--gate-out", "--feasibility", "--manifest", "--out"):
        e.add_argument(name, required=True)
    c = sub.add_parser("check")
    c.add_argument("probe_dir")
    c.add_argument("expected")
    args = parser.parse_args(argv)
    if args.cmd == "expected":
        feas = json.loads(pathlib.Path(args.feasibility).read_text(encoding="utf-8"))
        try:
            result = expected(pathlib.Path(args.gate_out), feas, evidence.load_manifest(args.manifest))
        except (OSError, KeyError, ValueError) as e:
            print(f"control: refused: {e!r}", file=sys.stderr)
            return 2
        pathlib.Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
        return 0
    want = json.loads(pathlib.Path(args.expected).read_text(encoding="utf-8"))
    problems = check(pathlib.Path(args.probe_dir), want)
    for line in problems:
        print(f"control: {line}", file=sys.stderr)
    if problems:
        return 1
    print(f"control: W1 equal to the gate on {len(want['native_ir'])} scenarios")
    return 0


if __name__ == "__main__":
    sys.exit(main())
