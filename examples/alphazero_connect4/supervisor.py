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
or the deadline passes. Native code holding the GIL can starve that
thread, so the supervisor's own kill is the real limit.
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
from ledger import BudgetExceeded, Ledger

EXIT_DEADLINE = 124
CLEANUP_GRACE_S = 2.0
# How long a group kill refused with EPERM is retried before the run is
# failed closed (see `kill_group` in `run`).
KILL_CONFIRM_S = 5.0


class Clock:
    """The child's view of the deadline, to skip work that would not fit."""

    def __init__(self, deadline_epoch: float):
        self.deadline_epoch = deadline_epoch

    def left(self) -> float:
        return self.deadline_epoch - time.time()

    def check(self, projected_s: float, what: str) -> None:
        if projected_s > self.left():
            raise BudgetExceeded(f"{what}: {projected_s:.1f} s projected, {self.left():.1f} s left")


def _watchdog(deadline_epoch: float, parent: int) -> None:
    while True:
        if time.time() >= deadline_epoch or os.getppid() != parent:
            os.killpg(0, signal.SIGKILL)
        time.sleep(0.2)


def child_flag(argv: list[str]) -> float | None:
    if "--child-deadline-epoch" in argv:
        return float(argv[argv.index("--child-deadline-epoch") + 1])
    return None


def strip_child_flag(argv: list[str]) -> list[str]:
    if "--child-deadline-epoch" not in argv:
        return argv
    i = argv.index("--child-deadline-epoch")
    return argv[:i] + argv[i + 2 :]


def run(script: str, argv: list[str], work, *, ledger: str, budget_core_hours: float, cores: int,
        wall_cap_s: float, out: str, name: str, exit_names: dict[int, str] | None = None,
        fresh_out: bool = True) -> int:
    """Runs ``work`` supervised (see the module docs); returns the exit
    code."""
    deadline = child_flag(argv)
    if deadline is not None:
        threading.Thread(target=_watchdog, args=(deadline, os.getppid()), daemon=True).start()
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
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(script), *argv, "--child-deadline-epoch", repr(deadline_epoch)],
        start_new_session=True,
    )

    def kill_group() -> bool:
        """Kills the child's group; returns False when the kill was refused
        until ``KILL_CONFIRM_S`` passed, so a member may still be alive."""
        end = time.monotonic() + KILL_CONFIRM_S
        while True:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                return True
            except ProcessLookupError:
                return True
            except PermissionError:
                # macOS answers EPERM for a group whose remaining members
                # are exiting (zombies), under load right after the child
                # returns. Retried until the group is gone (ESRCH) or the
                # kill goes through; the leader is killed directly if it
                # still runs.
                if proc.poll() is None:
                    proc.kill()
                if time.monotonic() >= end:
                    return False
                time.sleep(0.05)

    def on_term(signum, _frame):
        kill_group()
        proc.wait()
        led.settle(index, time.monotonic() - start, f"interrupted by signal {signum}")
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
        status, code = "failed: the process group was not confirmed gone (killpg EPERM)", 1
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
