# SPDX-License-Identifier: MIT OR Apache-2.0
"""Drives the native Windows split (``docs/reviews/gpu-windows-native-plan.md``).

Every (case, mode) pair runs once per round, in a fixed shuffle, each in its
own process and fresh working directory. A run is killed at ``--timeout``
seconds; the session's first CUDA trainer run gets ``--first-timeout``.

Windows has no process groups as POSIX has, so the driver watches the run's
descendants with ``psutil`` while it lives, kills all it saw, and stops the
session if any remains. Only the run's JSON line is kept, appended to ``--out``.

Usage::

    python run_native.py --out results.jsonl --device cuda [--rounds 3]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(HERE, "split_native.py")
CASES = ("tessera-w1", "tessera-w2")
MODES = ("e2e", "sync", "env")
SHUFFLE_SEED = 1
EXIT_LEFTOVER = 6
POLL_S = 0.05


def plan(cases=CASES, modes=MODES, rounds: int = 3, seed: int = SHUFFLE_SEED) -> list[tuple[int, int, str, str]]:
    """(round, position, case, mode) for every run, shuffled per round."""
    rng = random.Random(seed)
    out = []
    for r in range(rounds):
        pairs = [(c, m) for c in cases for m in modes]
        rng.shuffle(pairs)
        out.extend((r, i, c, m) for i, (c, m) in enumerate(pairs))
    return out


def _kill(processes: dict[tuple[int, float], psutil.Process]) -> list[int]:
    """Kills every process seen, and returns the PIDs still running."""
    alive = []
    for p in processes.values():
        try:
            p.kill()
            alive.append(p)
        except psutil.NoSuchProcess:
            pass
    _, left = psutil.wait_procs(alive, timeout=5)
    return sorted(p.pid for p in left if p.is_running() and p.status() != psutil.STATUS_ZOMBIE)


def one_run(command: list[str], timeout: float) -> dict:
    work = tempfile.mkdtemp(prefix="gpu-windows-run-")
    log = os.path.join(work, "out.log")
    seen: dict[tuple[int, float], psutil.Process] = {}
    try:
        with open(log, "w") as f:
            p = subprocess.Popen(command, cwd=work, stdout=f, stderr=subprocess.STDOUT)
            root = psutil.Process(p.pid)
            deadline = time.monotonic() + timeout
            timed_out = False
            while p.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                try:
                    for child in root.children(recursive=True):
                        seen[(child.pid, child.create_time())] = child
                except psutil.NoSuchProcess:
                    pass
                time.sleep(POLL_S)
            if p.poll() is None:
                try:
                    for child in root.children(recursive=True):
                        seen[(child.pid, child.create_time())] = child
                except psutil.NoSuchProcess:
                    pass
                p.kill()
            rc = p.wait()
            left = _kill(seen)
        with open(log, errors="replace") as f:
            text = f.read()
        lines = [x[x.index('{"case"') :] for x in text.splitlines() if '{"case"' in x]
        result = json.loads(lines[-1]) if lines else None
        return {"rc": rc, "timed_out": timed_out, "left": left, "result": result, "tail": None if result else text[-2000:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--timeout", type=float, default=100.0)
    p.add_argument("--first-timeout", type=float, default=300.0)
    p.add_argument("--split-args", default="", help="extra arguments for split_native.py (tests only)")
    args = p.parse_args(argv)
    status = 0
    first_cuda_pending = args.device == "cuda"
    for r, i, case, mode in plan(rounds=args.rounds):
        command = [sys.executable, SPLIT, case, mode, args.device, *shlex.split(args.split_args)]
        first = first_cuda_pending and mode != "env"
        if first:
            first_cuda_pending = False
        timeout = args.first_timeout if first else args.timeout
        out = one_run(command, timeout)
        row = {"round": r, "position": i, "case": case, "mode": mode, "timeout_s": timeout, **out}
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{r} {i} {case} {mode} rc={out['rc']} timed_out={out['timed_out']} left={out['left']}", flush=True)
        if out["left"]:
            print("STOP: processes of the run remain", flush=True)
            return EXIT_LEFTOVER
        if out["rc"] != 0:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
