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

Sensitivity (measured by Codex at 150k on xaskar): an adapter that gives
each instance its neighbour's observation drops the median to 9.7 and
fails (red after two nights); an adapter with terminated and truncated
swapped keeps the median at 488.05 and passes, although the mean drops
from 493 to 393. This test detects learning degradation, not every
adapter error: the flag swap is caught by the negative controls of the
reference training (``reference_training.py``), which complement it.

Invalid data or setup (missing, repeated or non-finite scores, another
fingerprint, a calibration whose threshold is not its records') is red at
once; only a valid night below the threshold waits for the second night.
The calibration is for Linux x86-64; elsewhere ``night`` stops with that
message (exit 1) instead of training and comparing.

The final score is the greedy evaluation mean (``--eval-episodes``
episodes, seeds ``EVAL_SEED_BASE + i``) after ``--total-steps`` training
steps; the calibration records the budget it was made with, and a night
with another setup fails.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from learning_gate import CALIBRATION_SEEDS, NIGHT_SEEDS, alert, calibrate, check_shards, night  # noqa: E402

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
    shards = [json.loads(pathlib.Path(f).read_text()) for f in args.scores]
    records = [r for shard in shards for r in shard.get("records", [])]
    out = calibrate(records, args.seeds)
    errors = out.get("errors", []) + check_shards(shards)
    if errors:
        out["errors"] = errors
    out["setup"] = shards[0].get("setup") if shards else None
    out["environment"] = shards[0].get("environment") if shards else None
    out["records"] = sorted(({"seed": r.get("seed"), "score": r.get("score")} for r in records), key=lambda r: str(r["seed"]))
    sys.stderr.write(json.dumps({k: out.get(k) for k in ("threshold", "median", "min", "max", "errors")}) + "\n")
    if errors:
        # A failed calibration never replaces the committed one.
        return 1
    _write(args.out, out)
    return 0


def _platform_errors(calibration: dict) -> list[str]:
    env = calibration.get("environment") or {}
    here = (platform.system(), platform.machine())
    there = (str(env.get("os", "")).split("-", 1)[0], env.get("arch"))
    if here != there:
        return [
            f"the calibration is for {there[0]} {there[1]} (the CI platform); this is {here[0]} {here[1]}: "
            "scores here would only be indicative, so the learning test does not run (run it on Linux x86-64)"
        ]
    return []


def cmd_night(args) -> int:
    path = pathlib.Path(args.calibration)
    calibration = json.loads(path.read_text())
    identity = {
        "event": os.environ.get("GITHUB_EVENT_NAME", "local"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "calibration": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    wrong_platform = _platform_errors(calibration)
    if wrong_platform:
        sys.stderr.write(wrong_platform[0] + "\n")
        return 1
    # The night uses the calibration's budget unless told otherwise (a
    # different one then fails the setup check).
    setup = calibration.get("setup") or {}
    if args.total_steps is None:
        args.total_steps = setup.get("total_steps")
    if args.eval_episodes is None:
        args.eval_episodes = setup.get("eval_episodes")
    if isinstance(args.total_steps, int) and isinstance(args.eval_episodes, int):
        records = _train(args, args.seeds)
        tonight = night(records, args.seeds, calibration, _setup(args))
    else:
        records = []
        tonight = {"seeds": args.seeds, "errors": ["the calibration has no valid setup"], "failed": True}
    previous = None
    if args.previous and pathlib.Path(args.previous).exists():
        try:
            previous = json.loads(pathlib.Path(args.previous).read_text())
        except (OSError, ValueError):
            previous = {}
    verdict = alert(tonight, identity, previous)
    _write(
        args.out,
        {"identity": identity, "setup": setup, "environment": _environment(), "night": tonight, "alert": verdict, "records": records},
    )
    sys.stderr.write(json.dumps({"night": {k: tonight.get(k) for k in ("median", "threshold", "failed", "errors")}, "alert": verdict}) + "\n")
    return 1 if verdict["red"] else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("scores", "night"):
        q = sub.add_parser(name)
        q.add_argument("--total-steps", type=int, required=name == "scores")
        q.add_argument("--eval-episodes", type=int, default=20 if name == "scores" else None)
        q.add_argument("--seeds", type=int, nargs="+", default=NIGHT_SEEDS)
        q.add_argument("--out", required=True)
    sub.choices["night"].add_argument("--calibration", required=True)
    sub.choices["night"].add_argument("--previous")
    q = sub.add_parser("calibrate")
    q.add_argument("--seeds", type=int, nargs="+", default=CALIBRATION_SEEDS)
    q.add_argument("--out", required=True)
    q.add_argument("scores", nargs="+")
    args = p.parse_args()
    return {"scores": cmd_scores, "calibrate": cmd_calibrate, "night": cmd_night}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
