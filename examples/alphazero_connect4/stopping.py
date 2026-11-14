# SPDX-License-Identifier: MIT OR Apache-2.0
"""Stopping a loop root: the STOP file and the recorded process identities.

**STOP.** ``<root>/STOP`` is written durably by ``loop.py stop``. Once it
exists:
- ``loop.py`` allocates no further attempt;
- ``remote.py run`` dispatches nothing;
- ``remote.py reconcile`` stops a still-running container by its recorded
  name.

A dispatch already allocated when the stop began is therefore refused at
the transport, and nothing new can start.

**Identities.** ``loop.py run`` records its PID, PGID and argv in
``<root>/driver.json``. Each ``remote.py run`` records its own identity in
its dispatch file. Before any signal, ``loop.py stop`` checks the recorded
identity against the observed process (``ps``): the same PGID, and the
command still the recorded one. A mismatch refuses the kill. Nothing is
ever killed by pattern.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time

import artifacts

STOP = "STOP"
DRIVER = "driver.json"


def stop_path(root: str) -> str:
    return os.path.join(root, STOP)


def requested(root: str) -> bool:
    return os.path.exists(stop_path(root))


def request(root: str, reason: str) -> dict:
    """Writes the STOP file durably (idempotent); returns its content."""
    path = stop_path(root)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    doc = {"reason": reason, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    artifacts.write_atomic(path, artifacts.canonical_json(doc))
    return doc


def identity(argv: list[str]) -> dict:
    """This process's identity, to be recorded."""
    return {"pid": os.getpid(), "pgid": os.getpgid(0), "argv": list(argv)}


def observed(pid: int) -> dict | None:
    """The live process's PGID and command line, or None if it is gone."""
    r = subprocess.run(["ps", "-o", "pgid=,args=", "-p", str(pid)], capture_output=True, text=True, check=False)
    line = r.stdout.strip()
    if r.returncode != 0 or not line:
        return None
    pgid, _, args = line.partition(" ")
    return {"pgid": int(pgid), "args": args.strip()}


def matches(rec: dict, obs: dict | None) -> bool:
    """The observed process is the recorded one: the same PGID, and its
    command line ends with the recorded argv (the interpreter path may
    differ in form)."""
    if obs is None:
        return False
    tail = " ".join(rec["argv"])
    return obs["pgid"] == rec["pgid"] and obs["args"].endswith(tail)


def kill_recorded(rec: dict, group: bool, sig: int = signal.SIGTERM) -> str:
    """Signals a recorded process (or its recorded group) after checking
    its identity. Returns what happened: "gone", "refused" or "signalled"."""
    obs = observed(rec["pid"])
    if obs is None:
        return "gone"
    if not matches(rec, obs):
        return "refused"
    if group:
        if rec["pgid"] != rec["pid"]:
            # Only a group the recorded process leads is signalled whole.
            return "refused"
        os.killpg(rec["pgid"], sig)
    else:
        os.kill(rec["pid"], sig)
    return "signalled"
