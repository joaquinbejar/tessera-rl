# SPDX-License-Identifier: MIT OR Apache-2.0
"""The supervised child's check that its supervisor is still alive (shared
by ``supervisor.py`` and ``pilot.py``; standard library only, so a script
can run it before its heavy imports).

The supervisor passes its own pid as ``--supervisor-pid``. The child is
orphaned once ``os.getppid()``, its real current parent, differs from it:
a child re-parented (to pid 1 or to a subreaper) before it got to look
sees the mismatch at once, so a supervisor SIGKILLed while the child was
still importing never leaves it running until the work deadline. The
supervisor's pid cannot come back as the parent of an already re-parented
process, so pid reuse cannot hide the death. The check runs first thing in
the child, again when its watchdog arms, and on every watchdog poll.
"""

from __future__ import annotations

import os
import signal
import sys
import time

FLAG = "--supervisor-pid"
# How long a group kill that macOS refused with EPERM is retried before the
# group is reported as not confirmed gone.
KILL_CONFIRM_S = 1.5
NOT_CONFIRMED = "the process group was not confirmed gone (killpg EPERM)"
# The watchdog's poll interval: an orphan is seen within one poll.
POLL_S = 0.2


def supervisor_pid(argv: list[str]) -> int | None:
    if FLAG in argv:
        return int(argv[argv.index(FLAG) + 1])
    return None


def check(pid: int) -> None:
    """Kills this child's whole process group (its workers and the bot
    included) if its parent is no longer the supervisor ``pid``.

    The group is killed only when this process leads it, as a child the
    supervisor started in a new session does (issue #207). A process started
    with the flag by hand, or by a caller that forgot the new session, shares
    its caller's group; it then kills only itself, never its caller."""
    if os.getppid() != pid:
        if os.getpgid(0) == os.getpid():
            os.killpg(0, signal.SIGKILL)
        os.kill(os.getpid(), signal.SIGKILL)


def kill_group(proc, confirm_s: float = KILL_CONFIRM_S) -> bool:
    """Kills the group led by the child ``proc`` (a ``subprocess.Popen``
    started in a new session). Returns False only when an EPERM could not be
    followed by ESRCH within ``confirm_s``, so a member may still be alive.

    macOS answers EPERM for a group whose remaining members are exiting
    (zombies), under load right after the child returns (issue #202). Only
    ESRCH proves the group is empty, so the kill is retried until it does,
    killing and reaping the leader directly meanwhile."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        pass
    end = time.monotonic() + confirm_s
    while time.monotonic() < end:
        if proc.poll() is None:
            proc.kill()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        time.sleep(0.05)
    return False


def check_argv(argv: list[str] | None = None) -> None:
    """``check`` for a process started as a supervised child; a no-op
    without the flag (a supervisor, a test). Call it from the script's
    ``__main__`` only: a spawned pool worker inherits its parent's argv,
    and its parent is the child, not the supervisor."""
    pid = supervisor_pid(sys.argv if argv is None else argv)
    if pid is not None:
        check(pid)
