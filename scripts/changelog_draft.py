#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Changelog drafts and checks from conventional commits (SPEC 20.1, #350).

The CHANGELOG stays hand-written (Keep a Changelog): its entries carry
contract detail that commit subjects do not. This tool reads the local git
history only, never writes a file, and has two modes:

``changelog_draft.py <from> [<to>]``
    Prints a draft ``[Unreleased]`` section for ``<from>..<to>`` (``<to>``
    defaults to ``HEAD``). Conventional subjects ``type(scope)!: text`` map to
    sections: ``feat`` to Added, ``fix`` to Fixed, ``perf`` to Changed,
    ``deprecate`` to Deprecated. ``!`` or a ``BREAKING CHANGE:`` trailer adds
    the commit to a Breaking list as well. A ``Revert "..."`` subject goes to
    Reverted, and the commit it reverts leaves the draft when both are in the
    range. Other types (test, bench, ci, docs, style, build, chore, ...) are
    left out unless ``--all`` is given. Subjects that do not parse are listed
    under "Not conventional".

``changelog_draft.py --check CHANGELOG.md <from> [<to>]``
    Lists every feat, fix, perf and deprecate commit in the range, every
    Breaking one, and every commit whose subject does not parse (it cannot be
    classified, so it might be any of them), that the ``[Unreleased]``
    section does not mention: none of
    the commit's references (``#NNN`` in the subject, or ``Closes``,
    ``Fixes``, ``Resolves`` or ``Part of #NNN`` in the body) appears there.
    Exit status 1 when any is missing, 0 otherwise. A release-checklist step:
    add an entry, or justify the omission in the release PR.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field

SECTIONS = {"feat": "Added", "fix": "Fixed", "perf": "Changed", "deprecate": "Deprecated"}
ORDER = ["Breaking", "Added", "Changed", "Deprecated", "Fixed", "Reverted"]
CHECKED = frozenset(SECTIONS)

SUBJECT = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?: (?P<text>.+)$")
REVERT = re.compile(r'^Revert "(?P<subject>.+)"$')
REF = re.compile(r"#(\d+)")
BODY_REF = re.compile(r"(?im)^\s*(?:closes|fixes|resolves|part of)\s+#(\d+)")
BREAKING = re.compile(r"(?m)^BREAKING[ -]CHANGE:")
REVERTS_SHA = re.compile(r"This reverts commit ([0-9a-f]{7,40})")


@dataclass
class Commit:
    sha: str
    subject: str
    body: str
    type: str | None = None
    scope: str | None = None
    text: str = ""
    breaking: bool = False
    refs: list[str] = field(default_factory=list)


class GitError(RuntimeError):
    pass


def log(frm: str, to: str, cwd: str | None = None) -> list[Commit]:
    """The commits of ``frm..to``, oldest first."""
    fmt = "%H%x00%s%x00%b%x1e"
    try:
        out = subprocess.run(
            ["git", "log", "--reverse", f"--format={fmt}", f"{frm}..{to}"],
            cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        raise GitError(e.stderr.strip() or f"git log {frm}..{to} failed") from e
    commits = []
    for rec in out.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        sha, subject, body = (rec.split("\x00") + ["", ""])[:3]
        commits.append(parse(sha, subject, body))
    return commits


def parse(sha: str, subject: str, body: str) -> Commit:
    c = Commit(sha=sha, subject=subject, body=body)
    m = SUBJECT.match(subject)
    if m:
        c.type, c.scope, c.text = m["type"], m["scope"], m["text"]
        c.breaking = bool(m["bang"]) or bool(BREAKING.search(body))
    elif REVERT.match(subject):
        c.type, c.text = "revert", REVERT.match(subject)["subject"]
    seen: list[str] = []
    for r in REF.findall(subject) + BODY_REF.findall(body):
        if r not in seen:
            seen.append(r)
    c.refs = seen
    return c


def effective(commits: list[Commit]) -> list[Commit]:
    """Drops a commit and its revert when both are in the range."""
    reverted = set()
    for c in commits:
        if c.type == "revert":
            m = REVERTS_SHA.search(c.body)
            if m:
                reverted.add(m[1])
    gone = {c.sha for c in commits if any(c.sha.startswith(r) for r in reverted)}
    keep = []
    for c in commits:
        if c.sha in gone:
            continue
        if c.type == "revert":
            m = REVERTS_SHA.search(c.body)
            if m and any(o.startswith(m[1]) for o in gone):
                continue
        keep.append(c)
    return keep


def draft(commits: list[Commit], include_all: bool = False) -> str:
    groups: dict[str, list[str]] = {}
    other: list[str] = []
    odd: list[str] = []
    for c in effective(commits):
        line = f"- {c.text}" if c.type else f"- {c.subject}"
        if c.type is None:
            odd.append(f"- {c.subject} ({c.sha[:8]})")
            continue
        if c.breaking:
            groups.setdefault("Breaking", []).append(line)
        if c.type == "revert":
            groups.setdefault("Reverted", []).append(line)
        elif c.type in SECTIONS:
            groups.setdefault(SECTIONS[c.type], []).append(line)
        elif include_all:
            other.append(f"- {c.type}: {c.text}")
    out = ["## [Unreleased]", ""]
    for name in ORDER:
        if groups.get(name):
            out += [f"### {name}", "", *groups[name], ""]
    if other:
        out += ["### Other", "", *other, ""]
    if odd:
        out += ["### Not conventional", "", *odd, ""]
    return "\n".join(out).rstrip("\n") + "\n"


def unreleased(changelog: str) -> str:
    """The text of the ``[Unreleased]`` section (empty when absent)."""
    m = re.search(r"(?ms)^## \[Unreleased\][^\n]*\n(.*?)(?=^## \[|\Z)", changelog)
    return m[1] if m else ""


def missing(commits: list[Commit], changelog: str) -> list[Commit]:
    section = unreleased(changelog)
    mentioned = set(REF.findall(section))
    out = []
    for c in effective(commits):
        if c.type in CHECKED or c.breaking or c.type is None:
            if not c.refs or not any(r in mentioned for r in c.refs):
                out.append(c)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", metavar="CHANGELOG", help="check this CHANGELOG's [Unreleased] section instead of drafting")
    p.add_argument("--all", action="store_true", help="also list types that are left out of the draft")
    p.add_argument("--repo", default=None, help="the git repository (default: the current directory)")
    p.add_argument("frm", metavar="from")
    p.add_argument("to", nargs="?", default="HEAD")
    a = p.parse_args(argv)
    try:
        commits = log(a.frm, a.to, a.repo)
    except GitError as e:
        sys.stderr.write(f"changelog_draft: {e}\n")
        return 2
    if a.check is None:
        sys.stdout.write(draft(commits, a.all))
        return 0
    try:
        with open(a.check, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        sys.stderr.write(f"changelog_draft: {e}\n")
        return 2
    gaps = missing(commits, text)
    for c in gaps:
        refs = ", ".join(f"#{r}" for r in c.refs) or "no reference"
        sys.stdout.write(f"missing: {c.sha[:8]} {c.subject} [{refs}]\n")
    sys.stdout.write(f"changelog_draft: {len(gaps)} of the checked commits in {a.frm}..{a.to} are not in [Unreleased]\n")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
