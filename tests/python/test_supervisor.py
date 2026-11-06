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
import subprocess
import sys
import time

EXAMPLE = pathlib.Path(__file__).parents[2] / "examples" / "alphazero_connect4"
sys.path.insert(0, str(EXAMPLE))

import supervisor  # noqa: E402

REAL_KILLPG = os.killpg

# The child's own watchdog kills its group at the deadline; with killpg
# neutralised in the child, only the supervisor's direct kill of the leader
# can end it.
NEUTRALISED = "import os; os.killpg = lambda *a: None; time.sleep(60)"

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
    """Whether ``pid`` runs; a zombie does not. Fails if ``ps`` cannot
    inspect a process that exists, rather than reading that as dead."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers the signal check; ps tells them apart.
    r = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    stat = r.stdout.strip()
    if r.returncode != 0 or not stat:
        raise AssertionError(f"cannot inspect pid {pid}: ps exited {r.returncode}: {r.stderr.strip()}")
    return not stat.startswith("Z")


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
    rc, entries, pid = run(tmp_path, monkeypatch, refusals=3, body=NEUTRALISED, wall_cap_s=3.0)
    time.sleep(0.2)
    assert rc == supervisor.EXIT_DEADLINE
    assert entries[0]["status"].startswith("killed at the deadline") and entries[0]["wall_s"] < 30
    assert not alive(pid)


def test_supervisor_persistent_eperm_at_the_deadline_fails_closed(tmp_path, monkeypatch):
    rc, entries, pid = run(tmp_path, monkeypatch, refusals=-1, body=NEUTRALISED, wall_cap_s=3.0)
    try:
        assert rc == 1
        assert entries[0]["status"].startswith("failed: the process group was not confirmed gone")
        assert entries[0]["wall_s"] < 30, "the leader is still killed directly"
    finally:
        cleanup(pid)


SUPERVISED = """
import os, sys
sys.path.insert(0, {example!r})
import supervisor

def eperm(pgid, sig):
    raise PermissionError(1, "Operation not permitted")

os.killpg = eperm
supervisor.KILL_CONFIRM_S = 0.3
sys.exit(supervisor.run({child!r}, [], None, ledger={ledger!r}, budget_core_hours=1.0, cores=1,
                        wall_cap_s=30.0, out={out!r}, name="t"))
"""


def test_supervisor_sigterm_with_a_refused_group_kill_says_so_in_the_status(tmp_path):
    """on_term with a persistent EPERM: the settled status keeps the
    accepted "interrupted by signal" prefix and names the refusal."""
    pidfile = tmp_path / "descendant.pid"
    child = tmp_path / "child.py"
    child.write_text(CHILD.format(example=str(EXAMPLE), pidfile=str(pidfile), body=NEUTRALISED))
    ledger = tmp_path / "ledger.json"
    sup = tmp_path / "supervised.py"
    sup.write_text(SUPERVISED.format(example=str(EXAMPLE), child=str(child), ledger=str(ledger),
                                     out=str(tmp_path / "out")))  # fmt: skip
    p = subprocess.Popen([sys.executable, str(sup)])
    end = time.monotonic() + 30
    while not pidfile.exists() and time.monotonic() < end:
        time.sleep(0.05)
    p.send_signal(signal.SIGTERM)
    try:
        assert p.wait(timeout=30) == 128 + signal.SIGTERM
        status = json.loads(ledger.read_text())["entries"][0]["status"]
        assert status == f"interrupted by signal {signal.SIGTERM} ({supervisor.NOT_CONFIRMED})"
    finally:
        cleanup(int(pidfile.read_text()))


def test_supervisor_spawn_record_names_the_child_group_and_starts_empty(tmp_path, monkeypatch):
    """An empty record first, then the group: a parent dying between them
    leaves a group that a recovery cannot account for, never a false ESRCH."""
    record = tmp_path / "spawn.json"
    seen = []
    real_popen = supervisor.subprocess.Popen

    def popen(*a, **k):
        seen.append(json.loads(record.read_text()))
        return real_popen(*a, **k)

    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    script = tmp_path / "child.py"
    script.write_text("import sys\nsys.exit(0)\n")
    rc = supervisor.run(str(script), [], None, ledger=str(tmp_path / "l.json"), budget_core_hours=1.0, cores=1,
                        wall_cap_s=30.0, out=str(tmp_path / "out"), name="t", spawn_record=str(record))  # fmt: skip
    assert rc == 0 and seen[0]["pgid"] is None
    assert json.loads(record.read_text())["pgid"] > 0
