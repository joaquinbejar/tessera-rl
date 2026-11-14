# SPDX-License-Identifier: MIT OR Apache-2.0
"""Drives the PuffeRL split runs (``docs/reviews/gpu-split-plan.md`` r5).

Every (case, mode) pair runs once per round, interleaved mode by mode, with
the case order rotated by one position per round (ADR-0061 item 5), each in a fresh temporary working directory (importing
PufferLib creates ``resources`` in the cwd) and in its own process group.

- A run is killed at ``--timeout`` seconds. The session's first CUDA trainer
  run (mode ``e2e`` or ``sync`` on ``cuda``) gets ``--first-timeout``, as it
  includes CUDA context creation and cuDNN autotune; env-only runs never
  create a CUDA context and never get it.
- After every run, timed out or not, the whole process group is killed, and
  the driver checks that no process of that group remains; one that does is
  recorded and stops the session.
- Only the run's JSON line is kept, appended to ``--out`` with the round, the
  order, the exit code and the group check. The working directory is
  removed.

Usage::

    python run.py --python PY --out results.jsonl --device cuda [--rounds 5] [--cases ...] [--modes ...]
    python run.py --python PY --out results.jsonl --device cuda \
        --manifest manifest.json --manifest-sha256 HEX --wheel PATH

With ``--manifest`` (the gate attempt of ADR-0061), the order comes from the
manifest (``manifest.py``), whose sha256 must be the declared one, and the
wheel file must hash to the manifest's ``wheel_sha256``; a mismatch refuses
before any run (exit 7). Every row carries the manifest sha256. Once the first
run starts, the session is the attempt (item 8, A1): the last line of
``--out`` is a summary row with per-cell (median, min, max) of ``sps``, and
``attempt`` is ``complete`` only when every run succeeded, else
``failed, no result``. ``gate_eligible`` also needs the device ``cuda`` and the
CUDA advantage kernel on every trainer run (item 6).

``PY`` is the interpreter (or command) that runs ``split.py``, for example
``uv run --locked --python 3.11 --project compat/pufferlib --with WHEEL python -I``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(HERE, "split.py")
CASES = ("ocean-serial", "ocean-mp", "tessera-w1", "tessera-w2")
MODES = ("e2e", "sync", "env")
EXIT_REFUSED = 7

sys.path.insert(0, HERE)
import manifest as manifestmod  # noqa: E402


def group_alive(pgid: int) -> list[int]:
    """PIDs still running in process group ``pgid`` (``ps``, POSIX). Zombies
    are not counted: they hold no resources and only wait to be reaped."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,stat="], capture_output=True, text=True, check=True).stdout
    alive = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and int(parts[1]) == pgid and not parts[2].startswith("Z"):
            alive.append(int(parts[0]))
    return sorted(alive)


def kill_group(pgid: int) -> list[int]:
    """Kills the whole group and returns what is still running after a short
    wait, which should be nothing. The caller reaps the leader first."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for _ in range(50):
        left = group_alive(pgid)
        if not left:
            return []
        time.sleep(0.1)
    return group_alive(pgid)


def one_run(command: list[str], timeout: float) -> dict:
    work = tempfile.mkdtemp(prefix="gpu-split-run-")
    log = os.path.join(work, "out.log")
    try:
        with open(log, "w") as f:
            p = subprocess.Popen(command, cwd=work, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            timed_out = False
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                rc = None
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if rc is None:
                rc = p.wait()
            left = kill_group(p.pid)
        with open(log, errors="replace") as f:
            text = f.read()
        lines = [x[x.index('{"case"') :] for x in text.splitlines() if '{"case"' in x]
        result = json.loads(lines[-1]) if lines else None
        return {"rc": rc, "timed_out": timed_out, "group_left": left, "result": result, "tail": None if result else text[-2000:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def plan(cases, modes, rounds: int) -> list[tuple[int, int, str, str]]:
    """(round, position, case, mode) for every run, rotated per round."""
    return from_order(manifestmod.order(cases, modes, rounds))


def from_order(order) -> list[tuple[int, int, str, str]]:
    return [(r, i, c, m) for r, pairs in enumerate(order) for i, (c, m) in enumerate(pairs)]


def summary(rows: list[dict], device: str, complete: bool, manifest_sha256: str | None) -> dict:
    """The attempt's last row: per (case, mode) cell the median, min and max of
    ``sps`` over its successful runs, and whether the attempt can be a gate
    result."""
    cells: dict[str, list[float]] = {}
    for row in rows:
        res = row.get("result") or {}
        if row["rc"] == 0 and "sps" in res:
            cells.setdefault(f"{row['case']}/{row['mode']}", []).append(res["sps"])
    backends = sorted({(row.get("result") or {}).get("advantage_backend") for row in rows if row["mode"] != "env"} - {None})
    return {
        "summary": True,
        "manifest_sha256": manifest_sha256,
        "attempt": "complete" if complete else "failed, no result",
        "advantage_backends": backends,
        "gate_eligible": complete and manifest_sha256 is not None and device == "cuda" and backends == ["cuda"],
        "cells": {k: {"n": len(v), "median": statistics.median(v), "min": min(v), "max": max(v)} for k, v in sorted(cells.items())},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--python", required=True, help="command prefix that runs split.py")
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--cases", default=",".join(CASES))
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--timeout", type=float, default=100.0)
    p.add_argument("--first-timeout", type=float, default=300.0)
    p.add_argument("--split-args", default="", help="extra arguments for split.py (tests only)")
    p.add_argument("--manifest", help="ADR-0061 manifest; its order replaces --rounds, --cases and --modes")
    p.add_argument("--manifest-sha256", help="the manifest sha256 sent to the reviewer")
    p.add_argument("--wheel", help="the Tessera wheel file, checked against the manifest")
    args = p.parse_args(argv)

    cases = args.cases.split(",")
    modes = args.modes.split(",")
    for c in cases:
        if c not in CASES:
            p.error(f"unknown case {c}")
    for m in modes:
        if m not in MODES:
            p.error(f"unknown mode {m}")
    msha = None
    if args.manifest or args.manifest_sha256 or args.wheel:
        if not (args.manifest and args.manifest_sha256 and args.wheel):
            p.error("--manifest, --manifest-sha256 and --wheel go together")
        try:
            m = manifestmod.load(args.manifest, args.manifest_sha256)
            if manifestmod.file_sha256(args.wheel) != m["wheel_sha256"]:
                raise ValueError("the wheel file does not hash to the manifest's wheel_sha256")
        except (ValueError, OSError, KeyError) as e:
            print(f"REFUSED: {e}", flush=True)
            return EXIT_REFUSED
        msha = args.manifest_sha256
        runs = from_order(m["order"])
    else:
        runs = plan(cases, modes, args.rounds)
    status = 0
    rows: list[dict] = []
    first_cuda_pending = args.device == "cuda"
    try:
        status = drive(args, runs, rows, msha, first_cuda_pending)
    finally:
        complete = status == 0 and len(rows) == len(runs)
        with open(args.out, "a") as f:
            f.write(json.dumps(summary(rows, args.device, complete, msha)) + "\n")
    return status


def drive(args, runs, rows: list[dict], msha, first_cuda_pending: bool) -> int:
    status = 0
    for r, i, case, mode in runs:
        command = [*shlex.split(args.python), SPLIT, case, mode, args.device, *shlex.split(args.split_args)]
        first = first_cuda_pending and mode != "env"
        if first:
            first_cuda_pending = False
        out = one_run(command, args.first_timeout if first else args.timeout)
        out["timeout_s"] = args.first_timeout if first else args.timeout
        row = {"round": r, "position": i, "case": case, "mode": mode, "manifest_sha256": msha, **out}
        rows.append(row)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{r} {i} {case} {mode} rc={out['rc']} timed_out={out['timed_out']} left={out['group_left']}", flush=True)
        if out["group_left"]:
            print("STOP: processes of the run's group remain", flush=True)
            return 6
        if out["rc"] != 0:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
