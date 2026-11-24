#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Reference training with Stable-Baselines3 (SPEC 14.2, #368): SB3's own
PPO on ``tessera/CartPole-v1`` through the Gymnasium adapter, with
Gymnasium's ``CartPole-v1`` as the control. It validates the adapter with an
unmodified third-party trainer; it is not the SPEC 18 learning test.

Each backend runs SB3's ``DummyVecEnv`` over ``gymnasium.make`` single
environments (``tessera_rl.gym.TesseraEnv`` for Tessera), with the RL Zoo's
CartPole-v1 PPO hyperparameters, for ``--total-steps`` steps per seed, then a
greedy evaluation of ``--eval-episodes`` episodes on a fresh environment
(seeds ``EVAL_SEED_BASE + i``, as tools/reference_training.py).

Rule (report-only until 7 green nights, then a gate; #368 review): the
numbers of tools/reference_training.py's defaults, ``--threshold 475``
(CartPole-v1's reward threshold) and ``--margin 20`` (derived there from the
seed spread of its PPO):

- the median across seeds of the Tessera greedy evaluation mean is at least
  475;
- |mean over seeds of the Tessera training-curve mean minus the same for
  Gymnasium| is at most 20, where a seed's curve mean is the mean, over
  updates, of the mean return of the episodes that ended in that update.

The report records SB3, Gymnasium, PyTorch and NumPy versions, the Tessera
version and build profile, the torch thread count, the CPU model, the git
commit and the CPU seconds of every run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import tessera_rl
import tessera_rl.gym  # noqa: F401  (registers tessera/* with Gymnasium)

EVAL_SEED_BASE = 10_000
BACKENDS = {"tessera": "tessera/CartPole-v1", "gymnasium": "CartPole-v1"}
# RL Zoo (rl-baselines3-zoo, hyperparams/ppo.yml) CartPole-v1.
N_ENVS = 8
HYPER = {
    "n_steps": 32,
    "batch_size": 256,
    "gae_lambda": 0.8,
    "gamma": 0.98,
    "n_epochs": 20,
    "ent_coef": 0.0,
    "learning_rate": 0.001,
    "clip_range": 0.2,
}


def _cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _curve_mean(monitors: list[Monitor], update_steps: int) -> float:
    """Mean over updates of the mean return of the episodes ended in them."""
    per_update: dict[int, list[float]] = {}
    for m in monitors:
        t = 0
        for ret, length in zip(m.get_episode_rewards(), m.get_episode_lengths(), strict=True):
            t += length
            per_update.setdefault(t * N_ENVS // update_steps, []).append(float(ret))
    means = [statistics.fmean(v) for _, v in sorted(per_update.items())]
    return statistics.fmean(means) if means else 0.0


def run(backend: str, seed: int, total_steps: int, eval_episodes: int) -> dict:
    env_id = BACKENDS[backend]
    monitors: list[Monitor] = []

    def factory(rank: int):
        def make():
            m = Monitor(gymnasium.make(env_id))
            m.reset(seed=seed * 1000 + rank)
            monitors.append(m)
            return m

        return make

    vec = DummyVecEnv([factory(r) for r in range(N_ENVS)])
    model = PPO("MlpPolicy", vec, seed=seed, device="cpu", verbose=0, **HYPER)
    cpu0 = time.process_time()
    model.learn(total_timesteps=total_steps)
    cpu = time.process_time() - cpu0
    curve = _curve_mean(monitors, HYPER["n_steps"] * N_ENVS)
    vec.close()
    env = gymnasium.make(env_id)
    returns = []
    for i in range(eval_episodes):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + i)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            total += float(r)
            done = term or trunc
        returns.append(total)
    env.close()
    return {"backend": backend, "seed": seed, "eval_mean": statistics.fmean(returns), "eval_returns": returns, "curve_mean": curve, "train_cpu_s": round(cpu, 2)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--threshold", type=float, default=475.0)
    p.add_argument("--margin", type=float, default=20.0)
    p.add_argument("--out", default="reference-sb3.json")
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results = [run(b, s, args.total_steps, args.eval_episodes) for s in args.seeds for b in BACKENDS]
    tess = [r for r in results if r["backend"] == "tessera"]
    ctrl = [r for r in results if r["backend"] == "gymnasium"]
    median_eval = statistics.median(r["eval_mean"] for r in tess)
    diff = statistics.fmean(r["curve_mean"] for r in tess) - statistics.fmean(r["curve_mean"] for r in ctrl)
    passed = median_eval >= args.threshold and abs(diff) <= args.margin
    report = {
        "library": "stable-baselines3",
        "versions": {
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gymnasium.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "tessera_rl": tessera_rl.version(),
            "tessera_build_profile": tessera_rl._native.build_profile(),
            "python": platform.python_version(),
        },
        "torch_threads": torch.get_num_threads(),
        "cpu": _cpu_model(),
        "commit": _commit(),
        "hyperparameters": {"n_envs": N_ENVS, **HYPER, "total_steps": args.total_steps},
        "rule": {"threshold": args.threshold, "margin": args.margin, "source": "tools/reference_training.py defaults"},
        "median_tessera_eval": median_eval,
        "curve_mean_difference": diff,
        "passed": passed,
        "runs": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for r in results:
        print(f"sb3 {r['backend']:9} seed {r['seed']}: eval {r['eval_mean']:.1f} curve {r['curve_mean']:.1f} cpu {r['train_cpu_s']} s")
    print(f"sb3: median tessera eval {median_eval:.1f} (>= {args.threshold}), curve diff {diff:+.1f} (|.| <= {args.margin}): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
