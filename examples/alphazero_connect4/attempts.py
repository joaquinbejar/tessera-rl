# SPDX-License-Identifier: MIT OR Apache-2.0
"""The benchmark's attempts journal (ADR-0048 section 3).

An append-only JSONL file across loop roots. The first line is a header naming
the benchmark, and each line holds the digest of the one before. An attempt is
``claimed``, then gets one terminal ``settled`` event. Anything malformed,
truncated or unknown fails closed.

This is a record against mistakes, not a security boundary: another file escapes
it, so each attempt is also recorded in a shared note outside this tool's control.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid

import artifacts

DISPOSITIONS = ("met-815", "below-815", "consumed-no-verdict")
LABELS = ("formal", "development")


class JournalError(SystemExit):
    """The journal is unusable, or refuses this attempt."""


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _append(path: str, record: dict, prev: bytes) -> None:
    line = artifacts.canonical_json({**record, "prev": artifacts.sha256(prev)}).rstrip(b"\n") + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def _load(path: str, benchmark: str, family: str) -> tuple[list[dict], bytes]:
    """The validated entries, without the header, and the last line's bytes."""
    # The lock is by path: a hard link would split it, so none is allowed.
    if os.stat(path).st_nlink != 1:
        raise JournalError(f"{path}: has another hard link, which would split the lock")
    with open(path, "rb") as f:
        raw = f.read()
    if not raw.endswith(b"\n"):
        raise JournalError(f"{path}: truncated")
    lines = raw.split(b"\n")[:-1]
    entries, settled, claimed = [], set(), set()
    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except ValueError:
            raise JournalError(f"{path}: line {i + 1} is malformed") from None
        if not isinstance(rec, dict) or rec.get("prev") != (artifacts.sha256(lines[i - 1]) if i else None):
            raise JournalError(f"{path}: line {i + 1} breaks the chain")
        if i == 0:
            if rec.get("kind") != "p2-g1-attempts" or rec.get("benchmark") != benchmark or rec.get("rule_family") != family:
                raise JournalError(f"{path}: not the journal of this benchmark")
            continue
        event, attempt = rec.get("event"), rec.get("attempt")
        if event == "claimed" and attempt not in claimed and rec.get("label") in LABELS and isinstance(rec.get("checkpoint"), str):
            claimed.add(attempt)
        elif event == "settled" and attempt in claimed and attempt not in settled and rec.get("disposition") in DISPOSITIONS:
            settled.add(attempt)
        else:
            raise JournalError(f"{path}: line {i + 1} is not a valid {event!r} event")
        entries.append(rec)
    if not lines:
        raise JournalError(f"{path}: no header")
    return entries, lines[-1]


class _Locked:
    """The sidecar lock, held across validating, checking and appending."""

    def __init__(self, path: str):
        self.fd = os.open(path + ".lock", os.O_CREAT | os.O_RDWR, 0o644)

    def __enter__(self):
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        os.close(self.fd)


def admit(entries: list[dict], label: str, checkpoint: str) -> None:
    """ADR-0048 section 3; raises when this attempt may not start."""
    settled = {e["attempt"] for e in entries if e["event"] == "settled"}
    claims = [e for e in entries if e["event"] == "claimed"]
    if label == "formal":
        if claims:
            raise JournalError("the benchmark's one formal attempt is spent or was preempted: a later run is development")
        return
    if any(e["attempt"] not in settled for e in claims):
        raise JournalError("an attempt is active or interrupted with no settled outcome: development is refused")
    if not any(e["label"] == "formal" for e in claims):
        raise JournalError("no formal attempt is settled: the set is still blind, and development is refused")
    if any(e["checkpoint"] == checkpoint for e in claims):
        raise JournalError("this checkpoint was already evaluated: it is not evaluated again")


def claim(path: str, benchmark: str, family: str, create: bool, label: str, root: str, checkpoint: str) -> dict:
    """Admits and records an attempt, durably, before any read of the set.
    Returns the block the claim and report carry."""
    real = os.path.realpath(path)
    with _Locked(real):
        created = False
        if not os.path.exists(real):
            if not create:
                raise JournalError(f"{real}: no journal; a new one needs --create-journal and is a visible departure")
            header = {"kind": "p2-g1-attempts", "benchmark": benchmark, "rule_family": family, "created": _stamp(), "prev": None}
            fd = os.open(real, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, artifacts.canonical_json(header).rstrip(b"\n") + b"\n")
                os.fsync(fd)
            finally:
                os.close(fd)
            _fsync_dir(os.path.dirname(real))
            created = True
        entries, last = _load(real, benchmark, family)
        admit(entries, label, checkpoint)
        attempt = uuid.uuid4().hex
        _append(real, {"event": "claimed", "attempt": attempt, "label": label, "root": os.path.realpath(root),
                       "checkpoint": checkpoint, "time": _stamp()}, last)  # fmt: skip
        return {"id": attempt, "label": label, "journal": real, "journal_sha256": artifacts.sha256_file(real),
                "journal_created": created, "benchmark": benchmark, "rule_family": family}  # fmt: skip


def settlement(block: dict) -> dict | None:
    """The attempt's terminal event, if it has one."""
    with _Locked(block["journal"]):
        entries, _ = _load(block["journal"], block["benchmark"], block["rule_family"])
    return next((e for e in entries if e["event"] == "settled" and e["attempt"] == block["id"]), None)


def open_attempt(path: str, benchmark: str, family: str, root: str, label: str, checkpoint: str | None) -> dict | None:
    """An unsettled claim of this root, for a crash that left no claim file.
    It only finds an existing attempt, and never creates one."""
    real = os.path.realpath(path)
    if not os.path.exists(real):
        raise JournalError(f"{real}: no journal")
    with _Locked(real):
        entries, _ = _load(real, benchmark, family)
    done = {e["attempt"] for e in entries if e["event"] == "settled"}
    for e in entries:
        if (e["event"] == "claimed" and e["attempt"] not in done and e.get("root") == os.path.realpath(root)
                and e["label"] == label and (checkpoint is None or e["checkpoint"] == checkpoint)):  # fmt: skip
            return {"id": e["attempt"], "label": label, "journal": real, "benchmark": benchmark, "rule_family": family}
    return None


def settle(block: dict, disposition: str, digests: dict) -> None:
    """The attempt's one terminal event; the first one stands."""
    real = block["journal"]
    with _Locked(real):
        entries, last = _load(real, block["benchmark"], block["rule_family"])
        if any(e["event"] == "settled" and e["attempt"] == block["id"] for e in entries):
            return
        if not any(e["attempt"] == block["id"] for e in entries):
            raise JournalError(f"{real}: attempt {block['id']} is not in this journal")
        _append(real, {"event": "settled", "attempt": block["id"], "disposition": disposition, **digests, "time": _stamp()}, last)
