#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Nightly wall-clock gate (SPEC 19.4, #374; design docs/reviews/nightly-374).

Each night compares today's head with the **accepted base**, a commit kept
in the nightly state (the previous night's artifact), in the same night:
paired, interleaved process runs of both runner binaries (base, head, head,
base, ...), one pair per repetition. Per case, the change is
``base_throughput / head_throughput - 1`` (positive is slower), and its 95%
interval is a seeded bootstrap of the median over the pairs. A case is
**flagged** when that interval lies entirely above +3% (SPEC 19.4).

The accepted base moves to the head only on a qualifying night with no flag.
A flagged night holds it, so a lasting regression is compared again the next
night instead of being absorbed. A flag clears when a later head is no longer
flagged against the held base, or when the owner acknowledges a flagged head
in ``compat/bench/nightly-ack.toml``. The acknowledgement is a reviewed
commit that lands after the flagged head; once it is in tonight's history it
moves the base to the acknowledged commit, is recorded as consumed, and
tonight is measured against it (``resolve_acks``).

**Red** needs the same case flagged on two consecutive *qualifying* nights.
A non-qualifying night (load over the limit, another job on the host during
the window, or an A/A half-width over 1.5%) neither flags, nor advances the
base, nor counts for the two-night rule. When the head equals the accepted
base, the night runs an A/A instead, which is also the noise check.

The gate binds only on a machine whose ``compat/machines.toml`` status is
``reference``; elsewhere the job reports what it would have done and stays
green.

**Slow drift.** Each night compares against the accepted base, which
advances after every clean night, so a regression smaller than the flag
threshold each night can accumulate unseen. To bound it, ``weekly_base``
holds the base accepted at least seven qualifying nights earlier, and the
report adds the change of the head against it (reported, never flagged).

Subcommands:

- ``base``: print tonight's base (the accepted base after acknowledgements),
  so the workflow builds the right base runner.
- ``night``: measure (``--base-runner``, ``--head-runner``), decide against
  ``--state`` (the previous state, or none), write ``--out``; exit 1 only
  when red on a binding machine.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tomllib
from collections.abc import Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
ACKS = ROOT / "compat" / "bench" / "nightly-ack.toml"

#: The gated cases: single-core Rust throughput (configs.toml names).
CASES = (
    "rust/cartpole/inline/n1024",
    "rust/tag/inline/n256",
    "rust/connect4/inline/n1024",
    "rust/go9/inline/n1024",
    "rust/rts/inline/n1024",
)
PAIRS = 10
FLAG_CHANGE = 0.03
AA_HALF_WIDTH_LIMIT = 0.015
LOAD_LIMIT = 4.0
WEEKLY_NIGHTS = 7
RESAMPLES = 10_000
BOOT_SEED = 20261124


# -- the change and its interval ----------------------------------------------


def paired_change(base: list[float], head: list[float], resamples: int = RESAMPLES, seed: int = BOOT_SEED) -> dict:
    """The change of a head against a base from paired throughputs (one
    pair per repetition): per pair ``base / head - 1`` (positive is slower),
    its median, and a seeded 95% bootstrap interval of the median over the
    pairs."""
    b = np.asarray(base, dtype=np.float64)
    h = np.asarray(head, dtype=np.float64)
    if b.shape != h.shape or b.size < 2 or not (np.all(b > 0) and np.all(h > 0)):
        raise ValueError("paired_change needs at least two pairs of positive throughputs")
    per_pair = b / h - 1.0
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(per_pair, size=(resamples, per_pair.size), replace=True), axis=1)
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return {
        "change": float(np.median(per_pair)),
        "ci95": [float(lo), float(hi)],
        "half_width": float((hi - lo) / 2),
        "pairs": per_pair.tolist(),
    }


def flagged(change: dict) -> bool:
    """SPEC 19.4: the 95% interval lies entirely above +3%."""
    return change["ci95"][0] > FLAG_CHANGE


# -- measuring ------------------------------------------------------------------


def measure(run: Callable[[str, str], float], cases: tuple[str, ...], pairs: int = PAIRS) -> dict:
    """Paired, interleaved runs: for each pair, ``run(side, case)`` on
    ``"base"`` and ``"head"``, in the order base-head on even pairs and
    head-base on odd ones, so a drift during the window hits both sides
    alike. ``run`` returns one process's throughput."""
    out = {}
    for case in cases:
        base, head = [], []
        for p in range(pairs):
            order = ("base", "head") if p % 2 == 0 else ("head", "base")
            got = {side: run(side, case) for side in order}
            base.append(got["base"])
            head.append(got["head"])
        out[case] = {"base": base, "head": head}
    return out


# -- qualification ----------------------------------------------------------------


def qualify(load: list[float | None], overlap: list[str], aa: dict | None, load_limit: float = LOAD_LIMIT) -> list[str]:
    """Why the night is NOT qualifying; empty means it qualifies.

    ``load``: the 1-minute load sampled before, during and after the runs.
    ``overlap``: other jobs seen on the host during the window (CI workers,
    containers that started or stopped). ``aa``: the A/A change per case,
    when the night ran one."""
    reasons = []
    if any(v is None for v in load):
        reasons.append("no load average on this host")
    elif any(v > load_limit for v in load):
        reasons.append(f"load over {load_limit}: {load}")
    if overlap:
        reasons.append(f"other jobs on the host during the window: {sorted(overlap)}")
    for case, c in (aa or {}).items():
        if c["half_width"] > AA_HALF_WIDTH_LIMIT:
            reasons.append(f"A/A half-width {c['half_width']:.4f} over {AA_HALF_WIDTH_LIMIT} for {case}")
    return reasons


# -- the decision ------------------------------------------------------------------


def initial_state(base: str) -> dict:
    return {
        "accepted_base": base,
        "weekly_base": base,
        "qualifying_nights": 0,
        "flag_streak": {},
        "flagged_heads": [],
        "consumed_acks": [],
        "history": [],
    }


def resolve_acks(state: dict, acks: list[dict], head: str, is_ancestor: Callable[[str, str], bool]) -> tuple[dict, list[dict]]:
    """Applies the owner's acknowledgements before tonight is measured.

    An acknowledgement lands as a commit after the flagged head, so it names
    a head that **was** flagged (``state["flagged_heads"]``), never tonight's.
    It applies when that commit was flagged, has not been consumed, and is an
    ancestor of tonight's head (``is_ancestor(commit, head)``). It moves the
    accepted base to the acknowledged commit (not to tonight's head, so a
    regression added after it is still compared), clears the flag streaks,
    and is recorded as consumed. The latest applicable acknowledgement wins.
    Acknowledgements apply to every case: the base is one commit."""
    new = json.loads(json.dumps(state))
    new.setdefault("flagged_heads", [])
    new.setdefault("consumed_acks", [])
    applied = []
    flagged_heads = new["flagged_heads"]
    for ack in acks:
        commit = ack.get("commit")
        if commit in new["consumed_acks"] or commit not in flagged_heads or not is_ancestor(commit, head):
            continue
        applied.append(ack)
    if applied:
        latest = max(applied, key=lambda a: flagged_heads.index(a["commit"]))
        new["accepted_base"] = latest["commit"]
        new["flag_streak"] = {}
        new["consumed_acks"] = [*new["consumed_acks"], *(a["commit"] for a in applied)]
    return new, applied


def decide(state: dict, night: dict, binding: bool, applied: list[dict] | None = None) -> tuple[dict, dict]:
    """The new state and the verdict of one night.

    ``night``: ``head``, ``base`` (the accepted base it was measured against),
    ``reasons`` (non-qualifying reasons, empty when it qualifies) and
    ``cases`` (case -> change dict, as ``paired_change``). ``state`` is the
    state after ``resolve_acks``; ``applied`` lists the acknowledgements it
    applied, which the verdict reports."""
    new = json.loads(json.dumps(state))
    head = night["head"]
    reasons = list(night["reasons"])
    flags = sorted(case for case, c in night["cases"].items() if flagged(c))
    verdict = {"head": head, "base": night["base"], "qualifying": not reasons, "reasons": reasons, "flags": flags}
    if applied:
        verdict["acknowledged"] = [a["commit"] for a in applied]
    if reasons:
        # Neither flags, nor advances the base, nor counts for the streaks.
        verdict.update(status="non-qualifying", red=False)
    elif head == night["base"]:
        verdict.update(status="a/a", red=False)
        new["qualifying_nights"] += 1
    elif not flags:
        new["accepted_base"] = head
        new["flag_streak"] = {}
        new["qualifying_nights"] += 1
        verdict.update(status="passed", red=False)
    else:
        streak = {case: new["flag_streak"].get(case, 0) + 1 for case in flags}
        new["flag_streak"] = streak
        new["flagged_heads"] = [*new.get("flagged_heads", []), head][-60:]
        new["qualifying_nights"] += 1
        red = any(n >= 2 for n in streak.values())
        verdict.update(status="flagged", red=red)
    if not reasons and new["qualifying_nights"] and new["qualifying_nights"] % WEEKLY_NIGHTS == 0:
        new["weekly_base"] = new["accepted_base"]
    if not binding and verdict["red"]:
        verdict.update(red=False, would_be_red=True)
    new["history"] = [*state["history"], verdict][-60:]
    return new, verdict


# -- the night ----------------------------------------------------------------------


def load_acks(path: pathlib.Path = ACKS) -> list[dict]:
    if not path.exists():
        return []
    acks = list(tomllib.loads(path.read_text()).get("ack", []))
    for ack in acks:
        if set(ack) != {"commit", "reason"}:
            raise ValueError(f"an acknowledgement has exactly commit and reason, not {sorted(ack)}")
    return acks


def git_is_ancestor(commit: str, head: str) -> bool:
    import subprocess  # noqa: PLC0415

    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], cwd=ROOT, check=False).returncode == 0


def _runner_call(paths: dict[str, pathlib.Path], configs: dict, method: dict, pin: dict) -> Callable[[str, str], float]:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import bench  # noqa: PLC0415

    def run(side: str, case: str) -> float:
        result = bench.run_rust(paths[side], configs[case], method, pin)
        return float(result["samples"]["median"])

    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("base", help="print tonight's base: the accepted base after acknowledgements")
    b.add_argument("--state", type=pathlib.Path)
    b.add_argument("--head", required=True)
    n = sub.add_parser("night")
    n.add_argument("--state", type=pathlib.Path, help="the previous night's state (absent on the first night)")
    n.add_argument("--out", type=pathlib.Path, required=True)
    n.add_argument("--head", required=True, help="the head commit")
    n.add_argument("--base-runner", type=pathlib.Path, required=True)
    n.add_argument("--head-runner", type=pathlib.Path, required=True)
    n.add_argument("--machine", default="ref-x86")
    n.add_argument("--overlap", default="", help="comma-separated jobs seen on the host during the window")
    n.add_argument("--pairs", type=int, default=PAIRS)
    n.add_argument("--quick", action="store_true", help="the harness's quick method: a smoke run, never valid for a gate")
    n.add_argument("--load-limit", type=float, default=LOAD_LIMIT, help="1-minute load above which the night does not qualify")
    args = parser.parse_args(argv)

    previous = json.loads(args.state.read_text()) if args.state and args.state.exists() else None
    state, applied = resolve_acks(previous or initial_state(args.head), load_acks(), args.head, git_is_ancestor)
    base = state["accepted_base"]
    if args.cmd == "base":
        print(base)
        return 0

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import bench  # noqa: PLC0415

    spec, method = bench.load(quick=args.quick)
    configs = {c["name"]: c for c in spec["config"]}
    pin = bench.pinning(args.machine)
    run = _runner_call({"base": args.base_runner, "head": args.head_runner}, configs, method, pin)
    load = [bench.load_average()[0] if bench.load_average() else None]
    raw = measure(run, CASES, args.pairs)
    load.append(bench.load_average()[0] if bench.load_average() else None)
    cases = {case: paired_change(r["base"], r["head"]) for case, r in raw.items()}
    aa = cases if base == args.head else None
    overlap = [j for j in args.overlap.split(",") if j]
    reasons = qualify(load, overlap, aa, args.load_limit)
    if args.quick:
        reasons.append("quick method: a smoke run")
    night = {"head": args.head, "base": base, "reasons": reasons, "cases": cases}
    binding = bench.machines().get(args.machine, {}).get("status") == "reference"
    new, verdict = decide(state, night, binding, applied)
    report = {
        "schema": "tessera-nightly/1",
        "machine": args.machine,
        "binding": binding,
        "environment": bench.environment(args.machine),
        "pinning": pin,
        "load": load,
        "verdict": verdict,
        "cases": cases,
        "raw": raw,
        "weekly_base": state.get("weekly_base"),
        "state": new,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(verdict, indent=2))
    return 1 if verdict["red"] else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    sys.exit(main())
