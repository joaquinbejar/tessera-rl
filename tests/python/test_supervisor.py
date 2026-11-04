# SPDX-License-Identifier: MIT OR Apache-2.0
"""The supervisor's group kill when the OS answers EPERM.

On macOS, ``killpg`` returns EPERM for a process group whose remaining
members are exiting (zombies), which happens under load right after the
child returns. The supervisor retries until the group is gone and never
settles ok, nor as a clean deadline kill, while a member may still run: it
fails closed. ``os.killpg`` is replaced to inject EPERM, so these run on
every platform. The child spawns a descendant in its group, as self-play's
multiprocessing pool does.
"""

import json
import os
import pathlib
import signal
import sys
import time

EXAMPLE = pathlib.Path(__file__).parents[2] / "examples" / "alphazero_connect4"
sys.path.insert(0, str(EXAMPLE))

import supervisor  # noqa: E402

REAL_KILLPG = os.killpg

CHILD = """
import pathlib, subprocess, sys, time
sys.path.insert(0, {example!r})
import supervisor

def work(clock):
    d = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    pathlib.Path({pidfile!r}).write_text(str(d.pid))
    {body}
    return 0

sys.exit(supervisor.run(__file__, sys.argv[1:], work, ledger="", budget_core_hours=1.0, cores=1,
                        wall_cap_s=1.0, out="", name="child"))
"""


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers; ps tells them apart.
    stat = os.popen(f"ps -o stat= -p {pid}").read().strip()
    return bool(stat) and not stat.startswith("Z")


def run(tmp_path, monkeypatch, *, refusals, body="pass", wall_cap_s=30.0):
    """Runs a supervised child whose group kill is refused with EPERM
    ``refusals`` times (then really killed); returns rc, the ledger
    entries and the descendant's pid."""
    pidfile = tmp_path / "descendant.pid"
    script = tmp_path / "child.py"
    script.write_text(CHILD.format(example=str(EXAMPLE), pidfile=str(pidfile), body=body))
    left = [refusals]

    def killpg(pgid, sig):
        if left[0] != 0:
            left[0] -= 1
            raise PermissionError(1, "Operation not permitted")
        REAL_KILLPG(pgid, sig)

    monkeypatch.setattr(supervisor.os, "killpg", killpg)
    monkeypatch.setattr(supervisor, "KILL_CONFIRM_S", 0.5)
    ledger = tmp_path / "ledger.json"
    rc = supervisor.run(str(script), [], None, ledger=str(ledger), budget_core_hours=1.0, cores=1,
                        wall_cap_s=wall_cap_s, out=str(tmp_path / "out"), name="t")  # fmt: skip
    return rc, json.loads(ledger.read_text())["entries"], int(pidfile.read_text())


def cleanup(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_supervisor_transient_eperm_after_return_kills_the_descendant_and_settles_ok(tmp_path, monkeypatch):
    rc, entries, pid = run(tmp_path, monkeypatch, refusals=3)
    time.sleep(0.2)
    assert rc == 0 and [e["status"] for e in entries] == ["ok"]
    assert not alive(pid), "the retried group kill reached the descendant"


def test_supervisor_persistent_eperm_after_return_fails_closed(tmp_path, monkeypatch):
    rc, entries, pid = run(tmp_path, monkeypatch, refusals=-1)
    try:
        assert rc == 1
        assert entries[0]["status"].startswith("failed: the process group was not confirmed gone")
    finally:
        cleanup(pid)


def test_supervisor_transient_eperm_at_the_deadline_kills_child_and_descendant(tmp_path, monkeypatch):
    rc, entries, pid = run(tmp_path, monkeypatch, refusals=3, body="time.sleep(60)", wall_cap_s=3.0)
    time.sleep(0.2)
    assert rc == supervisor.EXIT_DEADLINE
    assert entries[0]["status"].startswith("killed at the deadline") and entries[0]["wall_s"] < 30
    assert not alive(pid)


def test_supervisor_persistent_eperm_at_the_deadline_fails_closed(tmp_path, monkeypatch):
    rc, entries, pid = run(tmp_path, monkeypatch, refusals=-1, body="time.sleep(60)", wall_cap_s=3.0)
    try:
        assert rc == 1
        assert entries[0]["status"].startswith("failed: the process group was not confirmed gone")
        assert entries[0]["wall_s"] < 30, "the leader is still killed directly"
    finally:
        cleanup(pid)
