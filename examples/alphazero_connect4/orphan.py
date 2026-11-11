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

FLAG = "--supervisor-pid"
# The watchdog's poll interval: an orphan is seen within one poll.
POLL_S = 0.2


def supervisor_pid(argv: list[str]) -> int | None:
    if FLAG in argv:
        return int(argv[argv.index(FLAG) + 1])
    return None


def check(pid: int) -> None:
    """Kills this child's whole process group (its workers and the bot
    included) if its parent is no longer the supervisor ``pid``."""
    if os.getppid() != pid:
        os.killpg(0, signal.SIGKILL)


def check_argv(argv: list[str] | None = None) -> None:
    """``check`` for a process started as a supervised child; a no-op
    without the flag (a supervisor, a test). Call it from the script's
    ``__main__`` only: a spawned pool worker inherits its parent's argv,
    and its parent is the child, not the supervisor."""
    pid = supervisor_pid(sys.argv if argv is None else argv)
    if pid is not None:
        check(pid)
