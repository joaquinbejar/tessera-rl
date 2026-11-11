# SPDX-License-Identifier: MIT OR Apache-2.0
"""The supervised-run harness of the example's tools (the pilot's rules,
PILOT.md "Budget and stop rules", shared by self-play, training and the
head-to-head).

A tool's ``main`` calls ``run(argv, work, ...)``. Without
``--child-deadline-epoch`` this process is the **supervisor**:
- it reserves the run's allowance in the core-hour ledger before any work;
- it starts the same command as a child in a new process group;
- on its own monotonic clock, ``CLEANUP_GRACE_S`` before the reservation
  ends, it kills the whole group (exit 124);
- it settles the ledger to the measured wall time, never clamped, and
  writes a run log.

The child runs ``work(args, clock)``, whose return value is its exit code.
A watchdog thread in the child kills the group when the supervisor is gone
(its pid, passed as ``--supervisor-pid``, is no longer the child's parent:
see orphan.py) or the deadline passes. The tools run that check before
their heavy imports and ``run`` again when it arms the watchdog, so a
supervisor that died while the child was starting is seen at once. Native
code holding the GIL can starve the thread, so the supervisor's own kill
is the real limit.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import artifacts
import orphan
from ledger import BudgetExceeded, Ledger

EXIT_DEADLINE = 124
CLEANUP_GRACE_S = 2.0
# How long a group kill refused with EPERM is retried before the run is
# failed closed (see `kill_group` in `run`); inside CLEANUP_GRACE_S.
KILL_CONFIRM_S = orphan.KILL_CONFIRM_S
NOT_CONFIRMED = orphan.NOT_CONFIRMED


class Clock:
    """The child's view of the deadline, to skip work that would not fit."""

    def __init__(self, deadline_epoch: float):
        self.deadline_epoch = deadline_epoch

    def left(self) -> float:
        return self.deadline_epoch - time.time()

    def check(self, projected_s: float, what: str) -> None:
        if projected_s > self.left():
            raise BudgetExceeded(f"{what}: {projected_s:.1f} s projected, {self.left():.1f} s left")


def _watchdog(deadline_epoch: float, supervisor_pid: int) -> None:
    while True:
        if time.time() >= deadline_epoch:
            os.killpg(0, signal.SIGKILL)
        orphan.check(supervisor_pid)
        time.sleep(orphan.POLL_S)


def child_flag(argv: list[str]) -> float | None:
    if "--child-deadline-epoch" in argv:
        return float(argv[argv.index("--child-deadline-epoch") + 1])
    return None


def strip_child_flag(argv: list[str]) -> list[str]:
    """``argv`` without the internal flags the supervisor adds for its child."""
    for flag in ("--child-deadline-epoch", orphan.FLAG):
        if flag in argv:
            i = argv.index(flag)
            argv = argv[:i] + argv[i + 2 :]
    return argv


def run(script: str, argv: list[str], work, *, ledger: str, budget_core_hours: float, cores: int,
        wall_cap_s: float, out: str, name: str, exit_names: dict[int, str] | None = None,
        fresh_out: bool = True, spawn_record: str | None = None) -> int:
    """Runs ``work`` supervised (see the module docs); returns the exit
    code. ``spawn_record`` holds the child's process group: written with no
    group before the start and with it after, so a recovery can tell a child
    that never started from one it cannot account for."""
    deadline = child_flag(argv)
    if deadline is not None:
        supervisor_pid = orphan.supervisor_pid(argv)
        if supervisor_pid is None:
            raise SystemExit("supervisor: the child needs --supervisor-pid")
        # A supervisor SIGKILLed while the child was importing is seen here,
        # not at the work deadline.
        orphan.check(supervisor_pid)
        threading.Thread(target=_watchdog, args=(deadline, supervisor_pid), daemon=True).start()
        return work(Clock(deadline))
    if fresh_out:
        # Never reuse an earlier run's output (checked before anything is
        # reserved or written).
        artifacts.fresh_dir(out)
    else:
        os.makedirs(out, exist_ok=True)
    led = Ledger(ledger, budget_core_hours)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    start = time.monotonic()
    index, granted = led.reserve(f"{name} {stamp}", cores, wall_cap_s)
    work_s = granted - CLEANUP_GRACE_S
    if work_s <= 0:
        led.settle(index, time.monotonic() - start, "refused: the budget left is under the cleanup grace")
        return EXIT_DEADLINE
    work_deadline = start + work_s
    deadline_epoch = time.time() + (work_deadline - time.monotonic())
    host = socket.gethostname()
    if spawn_record:
        artifacts.write_atomic(spawn_record, artifacts.canonical_json({"host": host, "pgid": None}))
    proc = subprocess.Popen(
        [
            sys.executable,
            os.path.abspath(script),
            *argv,
            "--child-deadline-epoch",
            repr(deadline_epoch),
            orphan.FLAG,
            str(os.getpid()),
        ],
        start_new_session=True,
    )
    if spawn_record:
        artifacts.write_atomic(spawn_record, artifacts.canonical_json({"host": host, "pgid": proc.pid}))

    def kill_group() -> bool:
        return orphan.kill_group(proc, KILL_CONFIRM_S)

    def on_term(signum, _frame):
        gone = kill_group()
        proc.wait()
        status = f"interrupted by signal {signum}" + ("" if gone else f" ({NOT_CONFIRMED})")
        led.settle(index, time.monotonic() - start, status)
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    try:
        rc = proc.wait(timeout=max(0.0, work_deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        rc = None
    gone = kill_group()
    proc.wait()
    wall = time.monotonic() - start
    names = exit_names or {}
    if not gone:
        # Fail closed: never settle ok, nor as a clean deadline kill, while a
        # member of the group may still be running.
        status, code = f"failed: {NOT_CONFIRMED}", 1
    elif rc is None or rc == -signal.SIGKILL:
        status, code = f"killed at the deadline ({work_s:.1f} s of work)", EXIT_DEADLINE
    elif rc == 0:
        status, code = "ok", 0
    elif rc in names:
        status, code = f"failed: {names[rc]}", rc
    else:
        status, code = f"failed: child exit {rc}", rc if rc > 0 else 1
    entry = led.settle(index, wall, status)
    log = {"run": name, "host": socket.gethostname(), "status": status, "ledger": entry}
    path = os.path.join(out, f"run-{socket.gethostname()}-{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    print(json.dumps({"status": status, "log": path}))
    return code
