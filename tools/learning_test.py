#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""SPEC 18 learning test (nightly): the reference PPO of
``reference_training.py`` on each environment, over 5 seeds.

A night fails when the median final score is below the threshold. The
threshold is the 5th percentile of at least 40 calibration seeds. An alert
(a red job) requires two consecutive failing nights. The logic is in
``learning_gate.py``.

Subcommands:

- ``scores``: train on the given seeds and write their scores (calibration
  runs in parallel shards: one single-threaded process per shard).
- ``calibrate``: merge score files of the calibration seeds and write the
  calibration (``compat/learning/<env>.json``). A re-calibration is
  reviewed like a gungraun re-bless.
- ``night``: train on the nightly seeds, compare with the calibration and
  with the previous night's result; exit 1 only when red.

Calibration (``compat/learning/cartpole.json``): seeds 101-140 on Linux
x86-64 (the CI platform, image ``rust:1.99.1``), in parallel shards of
single-threaded processes (a seed scores the same alone and in a shard).
The budget is 150k steps: at 500k and 200k the greedy evaluation is 500
on most seeds, so the 5th percentile degenerates to the maximum (or to a
single outlier); at 150k the scores still spread below 500. The nightly
seeds 1-5 are disjoint from the calibration seeds.

With fixed seeds, one thread and deterministic algorithms, consecutive
nights on the same commit and hardware give identical scores: there is no
per-night randomness. The two-night rule filters noise across commits and
hardware (another runner CPU, a changed dependency), not a random draw.
The setup records a fingerprint (the PPO hyperparameters, the trainer's
source hash and the torch, NumPy and Gymnasium versions); a night whose
fingerprint differs from the calibration's fails until a reviewed
re-calibration.

The final score is the greedy evaluation mean (``--eval-episodes``
episodes, seeds ``EVAL_SEED_BASE + i``) after ``--total-steps`` training
steps; the calibration records the budget it was made with, and a night
with another setup fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from learning_gate import alert, calibrate, night  # noqa: E402

ENV_ID = "tessera/CartPole-v1"


def _git(*args: str) -> str:
    # A source tree without .git (a Docker copy) names its commit here.
    if args == ("rev-parse", "HEAD") and os.environ.get("TESSERA_GIT_SHA"):
        return os.environ["TESSERA_GIT_SHA"]
    if args[0] == "status" and os.environ.get("TESSERA_GIT_SHA"):
        # A "+worktree" suffix marks uncommitted changes on that commit.
        return "dirty" if os.environ["TESSERA_GIT_SHA"].endswith("+worktree") else ""
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _environment() -> dict:
    import gymnasium
    import numpy
    import torch

    import tessera_rl

    return {
        "git": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain") not in ("", "unknown"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "arch": platform.machine(),
        "cpu": _cpu(),
        # The Docker image (TESSERA_IMAGE) or the GitHub runner image.
        "image": os.environ.get("TESSERA_IMAGE")
        or " ".join(filter(None, (os.environ.get("ImageOS"), os.environ.get("ImageVersion"))))
        or "unknown",
        "tessera_rl": tessera_rl.__version__,
        "gymnasium": gymnasium.__version__,
        "torch": torch.__version__,
        "numpy": numpy.__version__,
    }


def _fingerprint() -> str:
    """What determines a score besides the seed: the PPO hyperparameters,
    the trainer's source and the numeric libraries. A change fails every
    night until a reviewed re-calibration."""
    from dataclasses import asdict

    import gymnasium
    import numpy
    import torch

    import reference_training as rt

    source = pathlib.Path(rt.__file__).read_bytes()
    payload = json.dumps(
        {
            "hyper": asdict(rt.Hyper()),
            "trainer_sha256": hashlib.sha256(source).hexdigest(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "gymnasium": gymnasium.__version__,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _setup(args) -> dict:
    return {
        "env": ENV_ID,
        "total_steps": args.total_steps,
        "eval_episodes": args.eval_episodes,
        "fingerprint": _fingerprint(),
    }


def _cpu() -> str:
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or "unknown"


def _train(args, seeds: list[int]) -> list[dict]:
    import torch

    import reference_training as rt

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    h = rt.Hyper(total_steps=args.total_steps)
    records = []
    for seed in seeds:
        r = rt.train("tessera", seed, h, args.eval_episodes)
        records.append({"seed": seed, "score": r["eval_mean"], "eval_returns": r["eval_returns"], "train_seconds": r["train_seconds"]})
        sys.stderr.write(f"seed {seed}: score {r['eval_mean']:.2f} in {r['train_seconds']}s\n")
    return records


def _write(path: str, data: dict) -> None:
    pathlib.Path(path).write_text(json.dumps(data, indent=2) + "\n")


def cmd_scores(args) -> int:
    _write(args.out, {"setup": _setup(args), "environment": _environment(), "records": _train(args, args.seeds)})
    return 0


def cmd_calibrate(args) -> int:
    records, setups = [], []
    for f in args.scores:
        data = json.loads(pathlib.Path(f).read_text())
        setups.append(data.get("setup"))
        records += data.get("records", [])
    out = calibrate(records, args.seeds)
    if any(s != setups[0] for s in setups) or not setups:
        out.setdefault("errors", []).append("the score files have different setups")
    out["setup"] = setups[0] if setups else None
    out["environment"] = json.loads(pathlib.Path(args.scores[0]).read_text()).get("environment") if args.scores else None
    out["records"] = sorted(({"seed": r["seed"], "score": r["score"]} for r in records), key=lambda r: r["seed"])
    _write(args.out, out)
    sys.stderr.write(json.dumps({k: out.get(k) for k in ("threshold", "median", "min", "max", "errors")}) + "\n")
    return 1 if out.get("errors") else 0


def cmd_night(args) -> int:
    calibration = json.loads(pathlib.Path(args.calibration).read_text())
    # The night uses the calibration's budget unless told otherwise (a
    # different one then fails the setup check).
    setup = calibration.get("setup") or {}
    if args.total_steps is None:
        args.total_steps = setup.get("total_steps")
    if args.eval_episodes is None:
        args.eval_episodes = setup.get("eval_episodes")
    if not isinstance(args.total_steps, int) or not isinstance(args.eval_episodes, int):
        sys.stderr.write("the calibration has no valid setup\n")
        return 1
    records = _train(args, args.seeds)
    tonight = night(records, args.seeds, calibration, _setup(args))
    previous = None
    if args.previous and pathlib.Path(args.previous).exists():
        try:
            previous = json.loads(pathlib.Path(args.previous).read_text()).get("night")
        except (OSError, ValueError):
            previous = None
    verdict = alert(tonight, previous)
    _write(args.out, {"setup": _setup(args), "environment": _environment(), "night": tonight, "alert": verdict, "records": records})
    sys.stderr.write(json.dumps({"night": {k: tonight.get(k) for k in ("median", "threshold", "failed", "errors")}, "alert": verdict}) + "\n")
    return 1 if verdict["red"] else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("scores", "night"):
        q = sub.add_parser(name)
        q.add_argument("--total-steps", type=int, required=name == "scores")
        q.add_argument("--eval-episodes", type=int, default=20 if name == "scores" else None)
        q.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
        q.add_argument("--out", required=True)
    sub.choices["night"].add_argument("--calibration", required=True)
    sub.choices["night"].add_argument("--previous")
    q = sub.add_parser("calibrate")
    q.add_argument("--seeds", type=int, nargs="+", default=list(range(101, 141)))
    q.add_argument("--out", required=True)
    q.add_argument("scores", nargs="+")
    args = p.parse_args()
    return {"scores": cmd_scores, "calibrate": cmd_calibrate, "night": cmd_night}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
