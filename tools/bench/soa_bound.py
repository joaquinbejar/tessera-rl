#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Lane-wise snapshot and fork cost against ADR-0010's bound (#735, ADR-0090 4.4).

    soa_bound.py <criterion dir> --group soa_snapshot_8192 --instances 8192

Reads Criterion's results for one benchmark group
(`<criterion dir>/<group>/<case>/new/{benchmark,estimates}.json`). Each case
declares its throughput as `size_of::<Record<E>>() * instances` bytes, so the
record size is recovered exactly as throughput / instances. For every case it
prints the record bytes, the per-instance median and 95% interval in ns, and
the bound of 1 us per KB per instance (record bytes / 1024 us). A case passes
when its median is at or below the bound. Exit status 0 if every bounded case
passes, 1 if one fails, 2 on bad input. Report-only operations (`snapshot_one`,
`record_hash`, `state_hash`) are printed without a verdict.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

BOUNDED = ("snapshot_all", "fork", "fork_branched", "restore")


class InputError(Exception):
    """Missing or malformed Criterion output."""


def bound_ns(record_bytes: int) -> float:
    """ADR-0010: at most 1 us per KB (1024 bytes) per instance, in ns."""
    return record_bytes * 1000.0 / 1024.0


def read_case(case_dir: pathlib.Path, instances: int) -> dict:
    try:
        bench = json.loads((case_dir / "new" / "benchmark.json").read_text())
        est = json.loads((case_dir / "new" / "estimates.json").read_text())
    except (OSError, ValueError) as e:
        raise InputError(f"{case_dir}: {e}") from e
    tp = bench.get("throughput") or {}
    total = tp.get("Bytes") if isinstance(tp, dict) else None
    if not isinstance(total, int) or total <= 0 or total % instances:
        raise InputError(f"{case_dir}: throughput {tp!r} is not a multiple of {instances} bytes")
    fid = bench.get("function_id") or ""
    if "/" not in fid:
        raise InputError(f"{case_dir}: function_id {fid!r} is not env/op")
    env, op = fid.split("/", 1)
    med = est["median"]
    return {
        "env": env,
        "op": op,
        "record_bytes": total // instances,
        "median_ns": med["point_estimate"] / instances,
        "lo_ns": med["confidence_interval"]["lower_bound"] / instances,
        "hi_ns": med["confidence_interval"]["upper_bound"] / instances,
    }


def report(criterion: pathlib.Path, group: str, instances: int) -> tuple[list[str], bool]:
    gdir = criterion / group
    if not gdir.is_dir():
        raise InputError(f"no Criterion group {gdir}")
    rows = sorted(
        (read_case(d, instances) for d in gdir.iterdir() if (d / "new" / "benchmark.json").is_file()),
        key=lambda r: (r["env"], r["op"]),
    )
    if not rows:
        raise InputError(f"{gdir}: no cases")
    ok = True
    out = [f"group {group}, {instances} instances; bound = record bytes / 1024 us per instance (ADR-0010)"]
    for r in rows:
        b = bound_ns(r["record_bytes"])
        if r["op"] in BOUNDED:
            verdict = "PASS" if r["median_ns"] <= b else "FAIL"
            ok &= verdict == "PASS"
        else:
            verdict = "report"
        out.append(
            f"{r['env']}/{r['op']}: record {r['record_bytes']} B, median {r['median_ns']:.1f} ns"
            f" [{r['lo_ns']:.1f}, {r['hi_ns']:.1f}], bound {b:.1f} ns, {verdict}"
        )
    return out, ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("criterion", type=pathlib.Path)
    ap.add_argument("--group", required=True)
    ap.add_argument("--instances", type=int, required=True)
    a = ap.parse_args(argv)
    if a.instances <= 0:
        print("soa_bound: --instances must be positive", file=sys.stderr)
        return 2
    try:
        lines, ok = report(a.criterion, a.group, a.instances)
    except InputError as e:
        print(f"soa_bound: {e}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
