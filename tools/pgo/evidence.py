#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The analysis of the PGO evidence (docs/reviews/pgo-wheels-design.md r4,
sections 4 to 6): the manifest, Callgrind's per-object costs, the feasibility
gate, the gating sets and the shipment ladder.

Pure: it reads text and numbers, and runs nothing. Valgrind has not been run
with it, so whether real Callgrind output meets the gate is UNPROVEN until a
host window exists. The tests use hand-written samples of the file format.
"""

from __future__ import annotations

import math
import os
import statistics
import tomllib
from dataclasses import dataclass, field

MAX_INTERPRETER_SHARE = 0.001
MAX_NONLINEARITY = 0.01
MIN_WORKER_SHARE = 0.90
MAX_IR_SPREAD = 0.001
MAX_MISPREDICT_SPREAD = 0.01
MIN_MISPREDICTS = 1000
MIN_TRAINED_GATING = 8
MAX_REGRESSION = 1.02
PGO_IR_GAIN = 0.97
PGO_MISPREDICT_GAIN = 0.90
LTO_IR_GAIN = 0.98
NATIVE_OBJECT = "_native"


# --------------------------------------------------------------- manifest


@dataclass(frozen=True)
class Scenario:
    name: str
    env_id: str
    config: tuple
    num_envs: int
    steps: int
    workers: int
    mode: str
    outputs: str
    kind: str
    holdout: bool

    @property
    def single_threaded(self) -> bool:
        return self.mode == "sync" and self.workers == 0


def scenarios(manifest: dict) -> list[Scenario]:
    out = []
    for g in manifest["group"]:
        for env in g["envs"]:
            config = env.get("config", {})
            suffix = "".join(f"-{v}" for v in config.values())
            out.append(
                Scenario(
                    name=f"{g['name']}/{env['id'].split('/')[1]}{suffix}",
                    env_id=env["id"],
                    config=tuple(sorted(config.items())),
                    num_envs=env.get("num_envs", g.get("num_envs", 0)),
                    steps=g["steps"],
                    workers=g.get("workers", 0),
                    mode=g.get("mode", "sync"),
                    outputs=g.get("outputs", "borrowed"),
                    kind=g.get("kind", "step"),
                    holdout=g.get("holdout", False),
                )
            )
    names = [s.name for s in out]
    if len(names) != len(set(names)):
        raise ValueError("duplicate scenario names")
    if any(s.num_envs <= 0 or s.steps <= 0 for s in out):
        raise ValueError("every scenario needs positive num_envs and steps")
    return out


def load_manifest(path: str) -> dict:
    with open(path, "rb") as f:
        manifest = tomllib.load(f)
    if not isinstance(manifest.get("version"), int):
        raise ValueError("the manifest has no integer version")
    scenarios(manifest)
    return manifest


def training_set(all_scenarios: list[Scenario], holdout_build: bool) -> list[Scenario]:
    """W2H trains without the holdout; W2F on everything."""
    return [s for s in all_scenarios if not (holdout_build and s.holdout)]


# --------------------------------------------------------------- callgrind


@dataclass
class Cost:
    events: dict[str, int] = field(default_factory=dict)

    @property
    def ir(self) -> int:
        return self.events["Ir"]

    @property
    def mispredicts(self) -> int:
        return self.events["Bcm"] + self.events["Bim"]

    def __sub__(self, other: Cost) -> Cost:
        return Cost({k: v - other.events[k] for k, v in self.events.items()})


REQUIRED_EVENTS = ("Ir", "Bcm", "Bim", "I1mr", "D1mr", "D1mw", "ILmr", "DLmr", "DLmw")


def parse_callgrind(text: str) -> dict[str, Cost]:
    """Exclusive cost per object file. The inclusive line after each `calls=`
    is another function's cost and is skipped, or everything would count twice."""
    positions = 1
    events: list[str] | None = None
    names: dict[str, str] = {}
    current: str | None = None
    totals: dict[str, list[int]] = {}
    skip_next = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("positions:"):
            positions = len(line.split(":", 1)[1].split())
        elif line.startswith("events:"):
            events = line.split(":", 1)[1].split()
            missing = [e for e in REQUIRED_EVENTS if e not in events]
            if missing:
                raise ValueError(f"callgrind events missing: {missing}")
        elif line.startswith("cob="):
            # Names are compressed across ob= and cob=: the first mention of an
            # id may be here. The cost of a call stays with its caller's object.
            key, _, name = line[4:].strip().partition(" ")
            if key.startswith("(") and name:
                names[key] = name
        elif line.startswith("ob="):
            value = line[3:].strip()
            key, _, name = value.partition(" ")
            if key.startswith("("):
                if name:
                    names[key] = name
                if key not in names:
                    raise ValueError(f"unknown object id {key}")
                current = names[key]
            else:
                current = value
        elif line.startswith("calls="):
            skip_next = True
        elif line[0] in "0123456789+-*":
            if skip_next:
                skip_next = False
                continue
            if events is None or current is None:
                raise ValueError("a cost line before the events or an object")
            values = line.split()[positions:]
            row = totals.setdefault(current, [0] * len(events))
            for i, v in enumerate(values):
                row[i] += int(v)
    if events is None:
        raise ValueError("no events line: not a callgrind file")
    return {k: Cost(dict(zip(events, v, strict=True))) for k, v in totals.items()}


def native_cost(objects: dict[str, Cost]) -> Cost:
    """The cost inside the `_native` shared object; exactly one must exist."""
    found = [c for name, c in objects.items() if os.path.basename(name).startswith(NATIVE_OBJECT)]
    if len(found) != 1:
        raise ValueError(f"expected one {NATIVE_OBJECT} object, found {len(found)}")
    return found[0]


# ---------------------------------------------------------- feasibility gate


def interpreter_excluded(control_ir: int, scenario_ir: int) -> bool:
    return scenario_ir > 0 and control_ir < MAX_INTERPRETER_SHARE * scenario_ir


def linear(n: int, n2: int, n3: int) -> bool:
    """Whether 2N-N is within 1% of 3N-2N."""
    first, second = n2 - n, n3 - n2
    return first > 0 and second > 0 and abs(first / second - 1.0) <= MAX_NONLINEARITY


def workers_collected(with_workers_ir: int, inline_ir: int) -> bool:
    return inline_ir > 0 and with_workers_ir >= MIN_WORKER_SHARE * inline_ir


def spread(values: list[int]) -> float:
    median = statistics.median(values)
    if median <= 0:
        return math.inf
    return (max(values) - min(values)) / median


@dataclass(frozen=True)
class Sets:
    gating: tuple[str, ...]
    mispredict: tuple[str, ...]
    usable: bool
    mispredict_available: bool


def gating_sets(all_scenarios: list[Scenario], aa: dict[str, list[Cost]]) -> Sets:
    """Section 5, from the A/A runs of W1 only, before any W2 exists."""
    if set(aa) != {s.name for s in all_scenarios}:
        raise ValueError("the A/A must cover exactly the manifest's scenarios")
    gating = [s for s in all_scenarios if s.single_threaded and spread([c.ir for c in aa[s.name]]) <= MAX_IR_SPREAD]
    trained = [s for s in gating if not s.holdout]
    holdout_all = {s.name for s in all_scenarios if s.holdout and s.single_threaded}
    usable = len(trained) >= MIN_TRAINED_GATING and holdout_all <= {s.name for s in gating}
    mispredict = [
        s
        for s in gating
        if spread([c.mispredicts for c in aa[s.name]]) <= MAX_MISPREDICT_SPREAD
        and statistics.median(c.mispredicts for c in aa[s.name]) >= MIN_MISPREDICTS
    ]
    available = sum(not s.holdout for s in mispredict) >= MIN_TRAINED_GATING
    return Sets(tuple(s.name for s in gating), tuple(s.name for s in mispredict), usable, available)


# ------------------------------------------------------------------ ladder


def geomean(ratios: list[float]) -> float:
    if not ratios or any(r <= 0 for r in ratios):
        raise ValueError("a geometric mean needs positive ratios")
    return math.exp(sum(map(math.log, ratios)) / len(ratios))


def _ratios(new: dict[str, Cost], base: dict[str, Cost], names, metric) -> list[float]:
    return [metric(new[n]) / metric(base[n]) for n in names]


def _ir(c: Cost) -> int:
    return c.ir


def _mispredicts(c: Cost) -> int:
    return c.mispredicts


def accept_w1(w1: dict, w0: dict, gating, w1_deterministic: bool) -> bool:
    ratios = _ratios(w1, w0, gating, _ir)
    return w1_deterministic and all(r <= MAX_REGRESSION for r in ratios) and geomean(ratios) <= LTO_IR_GAIN


def accept_w2f(w2f: dict, w2h: dict, w1: dict, all_scenarios, sets: Sets, w1_accepted: bool, w2f_deterministic: bool) -> bool:
    if not w1_accepted or not w2f_deterministic or not sets.usable:
        return False
    holdout = {s.name for s in all_scenarios if s.holdout}
    trained = [n for n in sets.gating if n not in holdout]
    held = [n for n in sets.gating if n in holdout]
    if any(r > MAX_REGRESSION for r in _ratios(w2f, w1, sets.gating, _ir)):
        return False
    gain = geomean(_ratios(w2f, w1, trained, _ir)) <= PGO_IR_GAIN
    if not gain and sets.mispredict_available:
        names = [n for n in sets.mispredict if n not in holdout]
        gain = geomean(_ratios(w2f, w1, names, _mispredicts)) <= PGO_MISPREDICT_GAIN
    return gain and geomean(_ratios(w2h, w1, held, _ir)) <= 1.0


def ship(w1_accepted: bool, w2f_accepted: bool) -> str:
    """W0 by default; W2F only on an accepted W1; a failing PGO falls back to W1."""
    if not w1_accepted:
        return "W0"
    return "W2F" if w2f_accepted else "W1"
