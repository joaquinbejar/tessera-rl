#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The layout of a wheel's costs measurement over N containers (pgo-ladder-plan.md
r4): from the wheel's own probe reports, the A/A units of every scenario with a
modelled cost (aa0 2t, its N run being the probe; aa1 to aa4 3t; t the probe's
seconds), longest first onto the least loaded part. Ties broken by unit name, so
the layout is deterministic. Stamps are not checked here beyond one wheel: the
layout only places units on CPUs and decides nothing; the measured files' stamps
are enforced by `callgrind.py --merge` and `decide.py`.

    layout_costs.py PROBE_DIR MANIFEST N BUDGET_S > layout.json

Exit 1 when the largest part projects past BUDGET_S (the window does not start
the parts), 2 when a scenario has no probe, is probed twice, or the probe
reports name more than one wheel or none.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import callgrind  # noqa: E402
import evidence  # noqa: E402


def layout(probe: dict, manifest: dict, n: int) -> dict:
    names = [s.name for s in evidence.scenarios(manifest)]
    missing = [x for x in names if x not in probe]
    if missing:
        raise ValueError(f"no probe for {missing}")
    units = []
    for u in callgrind.aa_units(manifest):
        name, tag = u.split("#")
        t = probe[name]["seconds"]
        units.append((u, 2 * t if tag == "aa0" else 3 * t))
    parts = [{"part": i, "pairs": [], "modelled_s": 0.0} for i in range(n)]
    for u, c in sorted(units, key=lambda x: (-x[1], x[0])):
        b = min(parts, key=lambda b: (b["modelled_s"], b["part"]))
        b["pairs"].append(u)
        b["modelled_s"] = round(b["modelled_s"] + c, 1)
    return {"units": len(units), "unit_s": {u: round(c, 1) for u, c in units},
            "total_s": round(sum(c for _, c in units), 1), "parts": parts,
            "largest_part_s": max(b["modelled_s"] for b in parts)}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        probe_dir, manifest_path, n, budget = pathlib.Path(argv[0]), argv[1], int(argv[2]), float(argv[3])
        if len(argv) != 4 or n < 1:
            raise ValueError
    except (IndexError, ValueError):
        print("usage: layout_costs.py PROBE_DIR MANIFEST N BUDGET_S (N >= 1)", file=sys.stderr)
        return 2
    probe, wheels = {}, set()
    try:
        for p in sorted(probe_dir.glob("feasibility.probe.*.json")):
            report = json.loads(p.read_text(encoding="utf-8"))
            wheels.add((report.get("sha256") or {}).get("wheel"))
            for name, r in report["probe"].items():
                if name in probe:
                    raise ValueError(f"{name} probed twice")
                probe[name] = r
        if len(wheels) != 1 or None in wheels:
            raise ValueError(f"probe reports name wheels {sorted(map(str, wheels))}, not exactly one")
        result = layout(probe, evidence.load_manifest(manifest_path), n)
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    result["budget_s"] = budget
    result["stop"] = result["largest_part_s"] > budget
    print(json.dumps(result, indent=1))
    return 1 if result["stop"] else 0


if __name__ == "__main__":
    sys.exit(main())
