# SPDX-License-Identifier: MIT OR Apache-2.0
"""The resource ledger of the example: allocated core-hours and wall time,
cumulative across runs, with stop rules.

A core-hour is one allocated CPU core for one hour of wall time, whatever
the cores did: a 90 minute run on 16 allocated cores spends 24. Every run
is entered, failed and interrupted ones included. A run (or a measurement
cell inside it) starts only if its projected cost fits what is left of
both the core-hour budget and the wall ceiling.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    """The next unit of work does not fit the remaining budget."""


@dataclass
class Ledger:
    path: str
    budget_core_hours: float
    wall_cap_s: float
    cores: int

    def __post_init__(self):
        self.entries: list[dict] = []
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.entries = json.load(f)["entries"]
        self.start = time.monotonic()

    def spent_core_hours(self) -> float:
        return sum(e["core_hours"] for e in self.entries)

    def elapsed_s(self) -> float:
        return time.monotonic() - self.start

    def remaining_core_hours(self) -> float:
        """Left after the earlier runs and this one so far."""
        return self.budget_core_hours - self.spent_core_hours() - self.elapsed_s() * self.cores / 3600

    def remaining_wall_s(self) -> float:
        return self.wall_cap_s - self.elapsed_s()

    def check(self, projected_s: float, what: str) -> None:
        """Raises ``BudgetExceeded`` if ``projected_s`` more seconds on the
        allocated cores would exceed either limit."""
        if projected_s > self.remaining_wall_s():
            raise BudgetExceeded(f"{what}: {projected_s:.0f} s projected, {self.remaining_wall_s():.0f} s of wall left")
        need = projected_s * self.cores / 3600
        if need > self.remaining_core_hours():
            raise BudgetExceeded(
                f"{what}: {need:.3f} core-hours projected, {self.remaining_core_hours():.3f} left"
            )

    def close(self, run: str, host: str, status: str) -> dict:
        """Enters this run and writes the ledger."""
        wall = self.elapsed_s()
        entry = {
            "run": run,
            "host": host,
            "cores": self.cores,
            "wall_s": round(wall, 3),
            "core_hours": round(wall * self.cores / 3600, 6),
            "status": status,
        }
        self.entries.append(entry)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "budget_core_hours": self.budget_core_hours,
                    "spent_core_hours": round(self.spent_core_hours(), 6),
                    "entries": self.entries,
                },
                f,
                indent=1,
            )
        os.replace(tmp, self.path)
        return entry
