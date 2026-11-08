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
    try:
        digest = report_hash(report)
    except ValueError:
        reasons.append("the report holds NaN or Infinity")
    else:
        declared = doc.get("report_sha256")
        if not (isinstance(declared, str) and _SHA256.fullmatch(declared)) or declared != digest:
            reasons.append("report_sha256 does not match the report")
    return reasons
