#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The M4 wall-time protocol r7 (docs/reviews/wall-protocol-m4.md).

    wall_r7.py aa BIN OUT_DIR
    wall_r7.py compare BASE_BIN HEAD_BIN OUT_DIR

Runs the frozen four cases of `per_instance_wall`, 20 pairs each, under r7's
quiet checks, foreign-CPU sampling and thermal sampling. `aa` compares a binary
with itself and applies the host gate. A case that is refused or invalid is
reported and never retried. The pairing and bootstrap are those of
tools/bench/per_instance_pairs.py.

It has run only against fakes: never on the M4.
"""

from __future__ import annotations

import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

CASES = ("cartpole_inline", "cartpole_w2", "tag_inline", "connect4_inline")
PAIRS = 20
ORDER_SEED = 20261106
BOOT_SEED = 20261107
RESAMPLES = 10_000
#: Frozen in r7; the sudoers rule pins exactly this.
POWERMETRICS = ["/usr/bin/powermetrics", "-i", "2000", "-n", "10800", "--samplers", "cpu_power,thermal"]
SAMPLE_S = 2.0
LOAD_LIMIT = 1.0
QUIET_PROCESS_LIMIT = 0.25
MEAN_FOREIGN_LIMIT = 0.5
INTERVAL_FOREIGN_LIMIT = 1.5
AA_HALF_WIDTH_LIMIT = 0.015
RUNNER_NAMES = ("Runner.Listener", "Runner.Worker")


def parse_cputime(text: str) -> float:
    """Seconds from `ps` cputime: `[[DD-]HH:]MM:SS[.cc]`."""
    days = 0
    if "-" in text:
        head, text = text.split("-", 1)
        days = int(head)
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return days * 86400 + seconds


def parse_ps(text: str) -> dict[int, tuple[int, float]]:
    """`pid -> (ppid, cpu seconds)` from `ps -A -o pid=,ppid=,cputime=`."""
    rows = {}
    for line in text.splitlines():
        if line.strip():
            pid, ppid, cpu = line.split()
            rows[int(pid)] = (int(ppid), parse_cputime(cpu))
    return rows


def own_tree(rows: dict[int, tuple[int, float]], root: int) -> set[int]:
    """`root` and its descendants among `rows`."""
    tree = {root}
    grew = True
    while grew:
        grew = False
        for pid, (ppid, _) in rows.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                grew = True
    return tree


def foreign_per_process(previous, current, seconds: float, own: set[int]) -> dict[int, float]:
    """CPUs used by each process outside `own` over the interval. One that
    appears counts from zero, as does a reused PID whose time went back."""
    out = {}
    for pid, (_, cpu) in current.items():
        if pid in own:
            continue
        before = previous[pid][1] if pid in previous else 0.0
        out[pid] = (cpu - before if cpu >= before else cpu) / seconds
    return out


def foreign_invalid(intervals: list[float]) -> str | None:
    if not intervals:
        return "no foreign-CPU interval"
    mean = sum(intervals) / len(intervals)
    if mean > MEAN_FOREIGN_LIMIT:
        return f"mean foreign CPU {mean:.3f} over {MEAN_FOREIGN_LIMIT}"
    worst = max(intervals)
    if worst > INTERVAL_FOREIGN_LIMIT:
        return f"foreign CPU {worst:.3f} in one interval, over {INTERVAL_FOREIGN_LIMIT}"
    return None


def pmset_invalid(text: str) -> str | None:
    """Only a numeric limit below 100 invalidates: a missing field proves nothing."""
    match = re.search(r"CPU_Speed_Limit\s*=?\s*(\d+)", text)
    if match and int(match.group(1)) < 100:
        return f"CPU_Speed_Limit {match.group(1)}"
    return None


def thermal_levels(lines) -> list[str]:
    return [m.group(1).strip() for m in (re.search(r"Current pressure level:\s*(.+)", x) for x in lines) if m]


def thermal_invalid(levels: list[str]) -> str | None:
    """No level at all invalidates: without one, throttling is not detected."""
    if not levels:
        return "no thermal pressure level was read"
    bad = sorted({x for x in levels if x != "Nominal"})
    return f"thermal pressure {bad}" if bad else None


def interval(changes: list[float], level: float, seed: int = BOOT_SEED) -> tuple[float, float]:
    """The central `level` bootstrap interval of the median, as the pairs tool."""
    boot = random.Random(seed)
    n = len(changes)
    medians = sorted(statistics.median(boot.choices(changes, k=n)) for _ in range(RESAMPLES))
    tail = (1 - level) / 2
    return medians[int(tail * RESAMPLES)], medians[int((1 - tail) * RESAMPLES) - 1]


def aa_verdict(changes: list[float]) -> dict:
    """r7's gate for one case: 98.75% contains 0, 95% half-width at most 1.5%."""
    lo98, hi98 = interval(changes, 0.9875)
    lo95, hi95 = interval(changes, 0.95)
    half = (hi95 - lo95) / 2
    contains = lo98 <= 0.0 <= hi98
    return {
        "ci98_75": [lo98, hi98],
        "ci95": [lo95, hi95],
        "half_width": half,
        "contains_zero": contains,
        "passed": contains and half <= AA_HALF_WIDTH_LIMIT,
    }


class Sampler:
    """The resident powermetrics of one case. Being root's, it is ended by
    closing the read end of its pipe; end of output before `stop` invalidates."""

    def __init__(self, command: list[str]):
        self.lines: list[str] = []
        self.stopping = False
        self.ended_early = False
        self.first = threading.Event()
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, errors="replace")
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        try:
            for line in self.process.stdout:
                self.lines.append(line)
                if "Current pressure level" in line:
                    self.first.set()
        except (ValueError, OSError):
            return
        if not self.stopping:
            self.ended_early = True
        self.first.set()

    def stop(self, timeout: float) -> bool:
        """Closes the pipe and waits for the end; False if it is still running."""
        self.stopping = True
        os.close(self.process.stdout.fileno())
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            return False
        return True


@dataclass
class Host:
    """Everything the driver reads from the machine, replaced in tests."""

    powermetrics: list[str] = field(default_factory=lambda: ["sudo", "-n", *POWERMETRICS])
    ps: list[str] = field(default_factory=lambda: ["ps", "-A", "-o", "pid=,ppid=,cputime="])
    pmset: list[str] = field(default_factory=lambda: ["pmset", "-g", "therm"])
    root: int = field(default_factory=os.getpid)
    sample_s: float = SAMPLE_S
    quiet_gap_s: float = 1.0
    stop_timeout_s: float = 2 * SAMPLE_S + 3.0
    first_sample_timeout_s: float = 30.0

    def read(self, command: list[str]) -> str:
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout

    def load(self) -> float:
        return os.getloadavg()[0]

    def runners(self) -> list[str]:
        out = self.read(["ps", "-A", "-o", "comm="])
        return sorted({x.strip() for x in out.splitlines() if x.strip().rsplit("/", 1)[-1] in RUNNER_NAMES})


def quiet_refusal(host: Host) -> str | None:
    """Why a case may not start, if any."""
    load = host.load()
    if load >= LOAD_LIMIT:
        return f"load {load:.2f}"
    runners = host.runners()
    if runners:
        return f"runner processes present: {runners}"
    previous = parse_ps(host.read(host.ps))
    start = time.monotonic()
    for _ in range(3):
        time.sleep(host.quiet_gap_s)
        current = parse_ps(host.read(host.ps))
        now = time.monotonic()
        own = own_tree(current, host.root)
        busy = {p: c for p, c in foreign_per_process(previous, current, now - start, own).items() if c > QUIET_PROCESS_LIMIT}
        if busy:
            return f"foreign processes busy: {busy}"
        previous, start = current, now
    return None


class Monitor(threading.Thread):
    """Samples foreign CPU and pmset during a case. A read failure invalidates."""

    def __init__(self, host: Host):
        super().__init__(daemon=True)
        self.host = host
        self.halt = threading.Event()
        self.intervals: list[float] = []
        self.problems: list[str] = []
        self.pmset: list[str] = []

    def run(self) -> None:
        host = self.host
        try:
            previous = parse_ps(host.read(host.ps))
            start = time.monotonic()
            while not self.halt.wait(host.sample_s):
                current = parse_ps(host.read(host.ps))
                now = time.monotonic()
                own = own_tree(current, host.root)
                self.intervals.append(sum(foreign_per_process(previous, current, now - start, own).values()))
                previous, start = current, now
                text = host.read(host.pmset)
                self.pmset.append(text)
                problem = pmset_invalid(text)
                if problem:
                    self.problems.append(problem)
        except (subprocess.CalledProcessError, OSError, ValueError) as e:
            self.problems.append(f"read failure: {e}")


def once(binary: str, case: str) -> dict:
    out = subprocess.run([binary, case], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def run_pairs(base: str, head: str, case: str, log) -> list[float] | str:
    """The per-pair relative changes, or why they could not be measured."""
    order = random.Random(ORDER_SEED)
    changes = []
    for k in range(PAIRS):
        pair = {}
        for name in order.sample(["base", "head"], 2):
            r = once(base if name == "base" else head, case)
            pair[name] = r
            log.write(json.dumps({"pair": k, "build": name, **r}) + "\n")
        if pair["base"]["digest"] != pair["head"]["digest"]:
            return f"pair {k}: digests differ"
        changes.append(pair["head"]["ns"] / pair["base"]["ns"] - 1)
    return changes


def run_case(host: Host, base: str, head: str, case: str, out_dir: str) -> dict:
    """One case: refused, invalid or measured. Never retried."""
    record: dict = {"case": case, "status": "refused"}
    refusal = quiet_refusal(host)
    if refusal:
        record["reason"] = refusal
        return record
    sampler = Sampler(host.powermetrics)
    monitor = Monitor(host)
    problems = []
    try:
        if not sampler.first.wait(host.first_sample_timeout_s) or sampler.ended_early:
            record["reason"] = "the sampler gave no first sample"
            return record
        monitor.start()
        with open(os.path.join(out_dir, f"{case}.jsonl"), "w", encoding="utf-8") as log:
            try:
                changes = run_pairs(base, head, case, log)
            except (subprocess.CalledProcessError, OSError, ValueError, KeyError) as e:
                changes = f"the harness failed: {e}"
        monitor.halt.set()
        monitor.join()
    finally:
        monitor.halt.set()
        ended_early = sampler.ended_early
        gone = sampler.stop(host.stop_timeout_s)
        sampler.thread.join(5)
    if ended_early:
        problems.append("the sampler's output ended during the case")
    if not gone:
        problems.append("the sampler did not end after its pipe closed")
    problems += monitor.problems
    foreign = foreign_invalid(monitor.intervals)
    if foreign:
        problems.append(foreign)
    thermal = thermal_invalid(thermal_levels(sampler.lines))
    if thermal:
        problems.append(thermal)
    if isinstance(changes, str):
        problems.append(changes)
    record["foreign_cpu"] = monitor.intervals
    record["pmset_samples"] = len(monitor.pmset)
    record["thermal_levels"] = sorted(set(thermal_levels(sampler.lines)))
    if problems:
        record.update(status="invalid", problems=problems)
        return record
    record.update(status="measured", median_change=statistics.median(changes), changes=changes)
    return record


def aa(host: Host, binary: str, out_dir: str) -> dict:
    rows = []
    for case in CASES:
        row = run_case(host, binary, binary, case, out_dir)
        if row["status"] == "measured":
            row["aa"] = aa_verdict(row["changes"])
        rows.append(row)
    usable = all(r["status"] == "measured" and r["aa"]["passed"] for r in rows)
    return {"mode": "aa", "host_usable": usable, "cases": rows}


def compare(host: Host, base: str, head: str, out_dir: str) -> dict:
    rows = []
    for case in CASES:
        row = run_case(host, base, head, case, out_dir)
        if row["status"] == "measured":
            row["ci95"] = interval(row["changes"], 0.95)
        rows.append(row)
    return {"mode": "compare", "cases": rows}


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) == 3 and args[0] == "aa":
        os.makedirs(args[2], exist_ok=True)
        result = aa(Host(), args[1], args[2])
        print(json.dumps(result))
        return 0 if result["host_usable"] else 1
    if len(args) == 4 and args[0] == "compare":
        os.makedirs(args[3], exist_ok=True)
        result = compare(Host(), args[1], args[2], args[3])
        print(json.dumps(result))
        return 0 if all(r["status"] == "measured" for r in result["cases"]) else 1
    sys.stderr.write(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
