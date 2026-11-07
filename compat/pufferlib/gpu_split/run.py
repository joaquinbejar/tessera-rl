# SPDX-License-Identifier: MIT OR Apache-2.0
"""Drives the PuffeRL split runs (``docs/reviews/gpu-split-plan.md`` r5).

Every (case, mode) pair runs once per round, in an order shuffled per round
with a fixed seed, each in a fresh temporary working directory (importing
PufferLib creates ``resources`` in the cwd) and in its own process group.

- A run is killed at ``--timeout`` seconds, the first run of the session at
  ``--first-timeout`` (it includes CUDA context creation and cuDNN autotune).
- After every run, timed out or not, the whole process group is killed, and
  the driver checks that no process of that group remains; one that does is
  recorded and stops the session.
- Only the run's JSON line is kept, appended to ``--out`` with the round, the
  order, the exit code and the group check. The working directory is
  removed.

Usage::

    python run.py --python PY --out results.jsonl --device cuda [--rounds 3] [--cases ...] [--modes ...]

``PY`` is the interpreter (or command) that runs ``split.py``, for example
``uv run --locked --python 3.11 --project compat/pufferlib --with WHEEL python -I``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
SHUFFLE_SEED = 20261107


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


def plan(cases, modes, rounds: int, seed: int = SHUFFLE_SEED) -> list[tuple[int, int, str, str]]:
    """(round, position, case, mode) for every run, shuffled per round."""
    rng = random.Random(seed)
    out = []
    for r in range(rounds):
        pairs = [(c, m) for c in cases for m in modes]
        rng.shuffle(pairs)
        out.extend((r, i, c, m) for i, (c, m) in enumerate(pairs))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--python", required=True, help="command prefix that runs split.py")
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--cases", default=",".join(CASES))
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--timeout", type=float, default=100.0)
    p.add_argument("--first-timeout", type=float, default=300.0)
    p.add_argument("--split-args", default="", help="extra arguments for split.py (tests only)")
    args = p.parse_args(argv)

    cases = args.cases.split(",")
    modes = args.modes.split(",")
    for c in cases:
        if c not in CASES:
            p.error(f"unknown case {c}")
    for m in modes:
        if m not in MODES:
            p.error(f"unknown mode {m}")
    status = 0
    for n, (r, i, case, mode) in enumerate(plan(cases, modes, args.rounds)):
        command = [*shlex.split(args.python), SPLIT, case, mode, args.device, *shlex.split(args.split_args)]
        out = one_run(command, args.first_timeout if n == 0 else args.timeout)
        row = {"round": r, "position": i, "case": case, "mode": mode, **out}
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
