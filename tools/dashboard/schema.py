# SPDX-License-Identifier: MIT OR Apache-2.0
"""The result file of the benchmark dashboard, and its checks."""

from __future__ import annotations

import datetime
import hashlib
import json
import re

ENVELOPE = "tessera-dashboard-result/1"
REPORT = "tessera-bench/3"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_INSTANT = "%Y-%m-%dT%H:%M:%SZ"


def canonical(report) -> bytes:
    """The bytes ``report_sha256`` covers. ``ValueError`` on NaN or Infinity."""
    text = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return text.encode("utf-8")


def report_hash(report) -> str:
    return hashlib.sha256(canonical(report)).hexdigest()


def _is_instant(value) -> bool:
    try:
        return datetime.datetime.strptime(value, _INSTANT).strftime(_INSTANT) == value
    except (TypeError, ValueError):
        return False


def _exact_positive(value) -> bool:
    return type(value) is int and value > 0


def _three_loads(value) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(type(v) in (int, float) for v in value)


def loads_reasons(doc: dict, report: dict) -> list[str]:
    """The envelope's ``loads`` (#640): the invocation's start and end load1,
    load5 and load15 at top level, and every process run's loads with its
    bounds, copied from the report. Required when the report was made
    under the load rule (``load_rule``); files from before it have neither."""
    rule = report.get("load_rule")
    if "loads" not in doc:
        return ["the report was made under the load rule, but the envelope has no loads"] if rule is not None else []
    loads = doc["loads"]
    if not isinstance(loads, dict):
        return ["loads is not an object"]
    reasons = []
    measured = report.get("load_average") if isinstance(report.get("load_average"), dict) else {}
    for key, source in (("start", "before"), ("end", "after")):
        if not _three_loads(loads.get(key)):
            reasons.append(f"loads.{key} is not load1, load5 and load15")
        elif loads[key] != measured.get(source):
            reasons.append(f"loads.{key} differs from the report's load_average.{source}")
    runs = rule.get("runs") if isinstance(rule, dict) else []
    if loads.get("runs") != runs:
        reasons.append("loads.runs differs from the report's per-run loads")
    return reasons


def envelope_reasons(doc) -> list[str]:
    """Why the file's envelope and hash cannot be admitted; empty if they can."""
    if not isinstance(doc, dict):
        return ["the file is not a JSON object"]
    if doc.get("schema") != ENVELOPE:
        return [f"unknown schema {doc.get('schema')!r}"]
    reasons = []
    if "fixture" in doc:
        reasons.append("a test fixture is not a result")
    if doc.get("kind") != "wall":
        reasons.append(f"unknown kind {doc.get('kind')!r}")
    if not _is_instant(doc.get("measured_at")):
        reasons.append("measured_at is not a UTC instant YYYY-MM-DDTHH:MM:SSZ")
    admitted = doc.get("admitted_by")
    admitted = admitted if isinstance(admitted, dict) else {}
    if not _exact_positive(admitted.get("pr")):
        reasons.append("admitted_by.pr is not a pull request number")
    approved = admitted.get("approved_by")
    if not (isinstance(approved, list) and approved and all(isinstance(a, str) and a for a in approved)):
        reasons.append("admitted_by.approved_by is empty")
    if "supersedes" in doc and not (isinstance(doc["supersedes"], str) and doc["supersedes"]):
        reasons.append("supersedes is not a file name")
    report = doc.get("report")
    if not isinstance(report, dict):
        reasons.append("no report")
        return reasons
    if report.get("schema") != REPORT:
        reasons.append(f"unknown report schema {report.get('schema')!r}")
    if report.get("load_void") is not None and report.get("load_void") is not False:
        reasons.append("the report is load_void: a process run ended at or above its load bound")
    rule = report.get("load_rule")
    if isinstance(rule, dict) and rule.get("load_void") is not False:
        reasons.append("the report's load rule is void or unrecorded")
    reasons += loads_reasons(doc, report)
    try:
        digest = report_hash(report)
    except ValueError:
        reasons.append("the report holds NaN or Infinity")
    else:
        declared = doc.get("report_sha256")
        if not (isinstance(declared, str) and _SHA256.fullmatch(declared)) or declared != digest:
            reasons.append("report_sha256 does not match the report")
    return reasons
