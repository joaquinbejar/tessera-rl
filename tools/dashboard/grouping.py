# SPDX-License-Identifier: MIT OR Apache-2.0
"""Which numbers may sit side by side: one comparison key, one table."""

from __future__ import annotations

import json

# `tessera_rl` tracks the commit, which is the axis of a trend.
SOFTWARE_OUTSIDE_KEY = frozenset({"git", "git_dirty", "tessera_rl"})


def comparison_key(report: dict, name: str, stat: str) -> tuple:
    """Every field that must match. Commit is left out: it is a trend's axis."""
    env = report["environment"]
    hardware = env["hardware"]
    method = report["method"]
    software = {k: v for k, v in env["software"].items() if k not in SOFTWARE_OUTSIDE_KEY}
    result = report["results"][name]
    work = sorted({(json.dumps(r.get("options"), sort_keys=True), str(r.get("plan_fingerprint"))) for r in result["runs"]})
    record = env["machine_record"] or {}
    return (
        ("kind", "wall"),
        ("machine", env["machine"]),
        ("cpu", hardware["cpu"]),
        ("cpu_id", record.get("cpu_id")),
        ("arch", hardware["arch"]),
        ("os", hardware["os"]),
        ("frequency_policy", hardware["frequency_policy"]),
        ("software", json.dumps(software, sort_keys=True)),
        ("cpu_set", json.dumps(report["pinning"]["cpu_set"])),
        ("warmup_s", method["warmup_s"]),
        ("reps", method["reps"]),
        ("rep_s", method["rep_s"]),
        ("batches", method["batches"]),
        ("runs", report["runs"]),
        ("configuration", name),
        ("config", json.dumps(result["config"], sort_keys=True)),
        ("work", json.dumps(work)),
        ("statistic", stat),
    )


def differing(a: tuple, b: tuple) -> list[str]:
    """Names of the fields where two keys differ."""
    return [x[0] for x, y in zip(a, b, strict=True) if x != y]
