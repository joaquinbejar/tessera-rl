# SPDX-License-Identifier: MIT OR Apache-2.0
"""Admission and status of the committed result files."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bench"))
import schema
from qualification import qualify

VALID, INDICATIVE, INVALID, SUPERSEDED = "valid", "indicative", "invalid", "superseded"
RESULTS = pathlib.PurePosixPath("benchmarks/results")


@dataclasses.dataclass
class Entry:
    """One result file and its verdict."""

    name: str
    status: str
    reasons: list[str]
    doc: dict | None


def _report_reasons(report: dict) -> tuple[list[str], list[str]]:
    """(reasons it is invalid, reasons it is only indicative), recomputed
    over the record the report embeds."""
    software = report["environment"]["software"]
    invalid = []
    if software["git_dirty"] is not False:
        invalid.append("the working tree was dirty")
    commit = software["git"]
    if not isinstance(commit, str) or not commit or commit == "unknown":
        invalid.append("the commit is unknown")
    machine = report["environment"]["machine"]
    record = report["environment"]["machine_record"] if machine else None
    indicative = qualify(report, report["qualification"], record)
    computed = not indicative
    check = report["claim_check"]
    if report["valid_for_claims"] is not computed or check["valid"] is not computed:
        invalid.append("the stored verdict disagrees with the recomputed one")
    return invalid, indicative


def judge(name: str, text: str) -> Entry:
    """The verdict on one file, before supersession."""
    try:
        doc = json.loads(text)
    except ValueError as e:
        return Entry(name, INVALID, [f"not JSON: {e}"], None)
    reasons = schema.envelope_reasons(doc)
    if reasons:
        return Entry(name, INVALID, reasons, doc if isinstance(doc, dict) else None)
    try:
        invalid, indicative = _report_reasons(doc["report"])
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
        return Entry(name, INVALID, [f"malformed report: {type(e).__name__}: {e}"], doc)
    if invalid:
        return Entry(name, INVALID, invalid, doc)
    if indicative:
        return Entry(name, INDICATIVE, indicative, doc)
    return Entry(name, VALID, [], doc)


def _refuse(entry: Entry, reason: str) -> None:
    if entry.status != INVALID:
        entry.status, entry.reasons = INVALID, []
    entry.reasons.append(reason)


def apply_supersession(entries: list[Entry]) -> None:
    """Refuses bad ``supersedes`` links, then marks replaced files."""
    by_name = {e.name: e for e in entries}
    links = {e.name: e.doc["supersedes"] for e in entries if e.doc and isinstance(e.doc.get("supersedes"), str)}
    refused: dict[str, str] = {}
    for name, target in links.items():
        if target == name:
            refused[name] = "the file supersedes itself"
        elif target not in by_name:
            refused[name] = f"supersedes {target!r}, which does not exist"
        elif sum(t == target for t in links.values()) > 1:
            refused[name] = f"another file also supersedes {target!r}"
        else:
            seen, at = {name}, target
            while at in links and at not in seen:
                seen.add(at)
                at = links[at]
            if at == name:
                refused[name] = "supersession cycle"
    for name, reason in refused.items():
        _refuse(by_name[name], reason)
    for name, target in links.items():
        replaced = by_name.get(target)
        if name not in refused and by_name[name].status != INVALID and replaced and replaced.status != INVALID:
            replaced.status, replaced.reasons = SUPERSEDED, [f"replaced by {name}"]


def load(root: pathlib.Path) -> list[Entry]:
    """Every ``*.json`` of ``root/benchmarks/results`` in file-name order."""
    results = root.joinpath(*RESULTS.parts)
    if not results.is_dir():
        return []
    entries = [admit(path, root) for path in sorted(results.glob("*.json"), key=lambda p: p.name)]
    apply_supersession(entries)
    return entries


def admit(path: pathlib.Path, root: pathlib.Path) -> Entry:
    """One file, refused unless it really lies in the results directory."""
    allowed = root.joinpath(*RESULTS.parts).resolve()
    if path.resolve().parent != allowed:
        return Entry(path.name, INVALID, [f"outside {RESULTS}"], None)
    try:
        return judge(path.name, path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        return Entry(path.name, INVALID, [f"unreadable: {type(e).__name__}"], None)
