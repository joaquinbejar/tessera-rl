# SPDX-License-Identifier: MIT OR Apache-2.0
"""The resource ledger of the example: allocated core-hours, cumulative
across runs, with a durable reservation per run.

A core-hour is one allocated CPU core for one hour of wall time, whatever
the cores did: a 90 minute run on 16 allocated cores spends 24.

One run at a time uses a ledger file: a reservation is refused while
another entry is open. A run **reserves** its whole allowance (its deadline times its allocated
cores) in the ledger file before any work starts, under an exclusive file
lock, so concurrent runs cannot spend the same allowance. On normal
completion, failure or a deadline kill the supervisor settles the entry to
the wall time actually used. If the supervisor itself dies (SIGKILL, a
lost host), the entry stays ``running`` and the next run settles it as
``interrupted`` at its full reservation: the conservative charge.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import socket
import time


class BudgetExceeded(RuntimeError):
    """The next unit of work does not fit the remaining budget."""


class LedgerBusy(RuntimeError):
    """Another run holds an open reservation: one run at a time per ledger."""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Ledger:
    def __init__(self, path: str, budget_core_hours: float):
        if not (math.isfinite(budget_core_hours) and budget_core_hours > 0):
            raise ValueError("budget_core_hours must be positive and finite")
        self.path = path
        self.budget_core_hours = budget_core_hours

    @contextlib.contextmanager
    def _locked(self):
        """The entries under an exclusive lock; writes them back on exit."""
        with open(self.path + ".lock", "a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            entries = []
            if os.path.exists(self.path):
                with open(self.path, encoding="utf-8") as f:
                    entries = json.load(f)["entries"]
            self._recover(entries)
            yield entries
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "budget_core_hours": self.budget_core_hours,
                        "spent_core_hours": round(sum(e["core_hours"] for e in entries), 6),
                        "entries": entries,
                    },
                    f,
                    indent=1,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)

    @staticmethod
    def _recover(entries: list[dict]) -> None:
        """Settles the reservations of dead supervisors of this host at
        their full value. A pid can only be checked on its own host (and in
        its own container, whose hostname is its own), so an entry of
        another host is left open."""
        here = socket.gethostname()
        for e in entries:
            if e["status"] == "running" and e["host"] == here and not _alive(e["pid"]):
                e["status"] = "interrupted (settled at the full reservation)"

    def entries(self) -> list[dict]:
        with self._locked() as entries:
            return [dict(e) for e in entries]

    def spent_core_hours(self) -> float:
        """Settled spend plus every open reservation."""
        return sum(e["core_hours"] for e in self.entries())

    def reserve(self, run: str, cores: int, deadline_s: float) -> tuple[int, float]:
        """Reserves ``cores x deadline_s`` (capped to what is left) and
        returns the entry index and the granted deadline in seconds."""
        if cores < 1:
            raise ValueError("cores must be at least 1")
        if not (math.isfinite(deadline_s) and deadline_s > 0):
            raise ValueError("the deadline must be positive and finite")
        with self._locked() as entries:
            for e in entries:
                if e["status"] == "running":
                    raise LedgerBusy(
                        f"{run}: {e['run']!r} (host {e['host']}, pid {e['pid']}) holds an open reservation; "
                        "one run at a time per ledger file. Settle a dead entry of another host by hand."
                    )
            left = self.budget_core_hours - sum(e["core_hours"] for e in entries)
            granted = min(deadline_s, left * 3600 / cores)
            if granted <= 0:
                raise BudgetExceeded(f"{run}: {left:.3f} core-hours left")
            entries.append(
                {
                    "run": run,
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "cores": cores,
                    "deadline_s": round(granted, 3),
                    "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "wall_s": None,
                    "core_hours": round(granted * cores / 3600, 6),
                    "status": "running",
                }
            )
            return len(entries) - 1, granted

    def settle(self, index: int, wall_s: float, status: str) -> dict:
        """Charges the wall time actually used (never more than reserved)."""
        with self._locked() as entries:
            e = entries[index]
            wall = min(wall_s, e["deadline_s"])
            e["wall_s"] = round(wall, 3)
            e["core_hours"] = round(wall * e["cores"] / 3600, 6)
            e["status"] = status
            return dict(e)
