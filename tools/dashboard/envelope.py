# SPDX-License-Identifier: MIT OR Apache-2.0
"""Wraps an unchanged ``tools/bench/bench.py`` report in a result envelope
(``tessera-dashboard-result/1``) for ``benchmarks/results``.

The envelope carries the report's hash, the admitting PR and approvers,
the measurement instant and, at top level, ``loads``: the invocation's
start and end load averages (load1, load5, load15) and every process run's
loads with its gate and void bounds, and ``containers``: the running
containers at the invocation's start and end, both copied from the report.
A report
marked ``load_void`` is refused: it is never admitted.

Usage::

    python tools/dashboard/envelope.py REPORT --pr N --approved-by NAME \\
        --measured-at 2026-01-01T00:00:00Z [--supersedes FILE] --out FILE
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import schema


def wrap(report: dict, pr: int, approved_by: list[str], measured_at: str, supersedes: str | None = None) -> dict:
    """The envelope for ``report``. ``ValueError`` if it cannot be admitted."""
    if report.get("load_void") is True:
        raise ValueError("the report is load_void; run the flagship again")
    doc = {
        "schema": schema.ENVELOPE,
        "kind": "wall",
        "measured_at": measured_at,
        "admitted_by": {"pr": pr, "approved_by": list(approved_by)},
        "report_sha256": schema.report_hash(report),
        "report": report,
    }
    load = report.get("load_average") or {}
    rule = report.get("load_rule")
    doc["loads"] = {"start": load.get("before"), "end": load.get("after"), "runs": rule["runs"] if rule else []}
    if "containers" in report:
        doc["containers"] = report["containers"]
    if supersedes:
        doc["supersedes"] = supersedes
    reasons = schema.envelope_reasons(doc)
    if reasons:
        raise ValueError("; ".join(reasons))
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=pathlib.Path)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--approved-by", action="append", required=True)
    parser.add_argument("--measured-at", required=True, help="UTC instant YYYY-MM-DDTHH:MM:SSZ")
    parser.add_argument("--supersedes")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    try:
        doc = wrap(report, args.pr, args.approved_by, args.measured_at, args.supersedes)
    except ValueError as e:
        parser.error(str(e))
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
