#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Reference training with RLlib (SPEC 14.2, #447): RLlib's own PPO on
``tessera/CartPole-v1`` through the Gymnasium adapter, with Gymnasium's
``CartPole-v1`` as the control. It validates the adapter with an
unmodified third-party trainer; it is not the SPEC 18 learning test.

Scope: RLlib pins ``gymnasium==1.2.2``, so this runs in its own environment
(compat/training/rllib, ``[env.rllib]``). Tessera is *validated against
Gymnasium 1.2.2 (RLlib's pin)* here; it is not a supported Gymnasium version.

The configuration is RLlib's own example
``rllib/examples/algorithms/ppo/cartpole_ppo.py`` at tag ``ray-2.59.1``
(sha256 c9009d8e1080c77fb193d2ba0342dedc8f527df71373d92a76de7507748f38e2):
``lr=0.0003``, ``num_epochs=6``, ``vf_loss_coeff=0.01``, a ``[32]`` linear
network with shared value layers, and RLlib's defaults for everything else
(including its env runners), for its default budget of 300k steps. The
upstream example stops early at a return of 450; this script trains the
fixed budget for the comparison. The env is ``gymnasium.make(id)`` through
``tune.register_env``; the seed goes through ``.debugging(seed=...)``.

After training, the greedy policy (the argmax of the module's action
logits) is evaluated on ``--eval-episodes`` episodes of a fresh environment
(seeds ``EVAL_SEED_BASE + i``, as in tools/reference_training.py). A seed's
curve mean is the mean, over training iterations (RLlib's unit of update),
of ``env_runners/episode_return_mean``.

The rule is the same as in tools/reference/sb3_cartpole.py: the defaults
of tools/reference_training.py, with a median Tessera evaluation of at
least 475 and an absolute curve difference of at most 20. It is
report-only (#501). Ray spreads the work over processes, so the report
records the wall clock as well as the driver's CPU time.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time

import gymnasium
import numpy as np
import ray
import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

import tessera_rl
import tessera_rl.gym  # noqa: F401  (registers tessera/* with Gymnasium)

EVAL_SEED_BASE = 10_000
BACKENDS = {"tessera": "tessera/CartPole-v1", "gymnasium": "CartPole-v1"}
UPSTREAM = "rllib/examples/algorithms/ppo/cartpole_ppo.py@ray-2.59.1"
HYPER = {"lr": 0.0003, "num_epochs": 6, "vf_loss_coeff": 0.01, "fcnet_hiddens": [32], "fcnet_activation": "linear", "vf_share_layers": True}


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


def _make(env_id: str):
    """Runs in Ray's env-runner processes, which have not imported
    tessera_rl.gym: import it there so ``tessera/*`` is registered."""
    import tessera_rl.gym  # noqa: F401, PLC0415

    return gymnasium.make(env_id)


def _config(backend: str, seed: int) -> PPOConfig:
    name = f"ref_{backend}"
    env_id = BACKENDS[backend]
    tune.register_env(name, lambda _cfg, env_id=env_id: _make(env_id))
    return (
        PPOConfig()
        .environment(name)
        .training(lr=HYPER["lr"], num_epochs=HYPER["num_epochs"], vf_loss_coeff=HYPER["vf_loss_coeff"])
        .rl_module(
            model_config=DefaultModelConfig(
                fcnet_hiddens=HYPER["fcnet_hiddens"],
                fcnet_activation=HYPER["fcnet_activation"],
                vf_share_layers=HYPER["vf_share_layers"],
            )
        )
        .debugging(seed=seed)
    )


def run(backend: str, seed: int, total_steps: int, eval_episodes: int) -> dict:
    algo = _config(backend, seed).build_algo()
    curve: list[float] = []
    sampled = 0
    wall0, cpu0 = time.perf_counter(), time.process_time()
    while sampled < total_steps:
        result = algo.train()
        sampled = int(result["num_env_steps_sampled_lifetime"])
        ret = result.get("env_runners", {}).get("episode_return_mean")
        if ret is not None and np.isfinite(ret):
            curve.append(float(ret))
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    module = algo.get_module()
    env = gymnasium.make(BACKENDS[backend])
    returns = []
    with torch.no_grad():
        for i in range(eval_episodes):
            obs, _ = env.reset(seed=EVAL_SEED_BASE + i)
            done, total = False, 0.0
            while not done:
                out = module.forward_inference({"obs": torch.as_tensor(np.asarray(obs, dtype=np.float32)[None])})
                obs, r, term, trunc, _ = env.step(int(out["action_dist_inputs"].argmax(-1)[0]))
                total += float(r)
                done = term or trunc
            returns.append(total)
    env.close()
    algo.stop()
    return {
        "backend": backend,
        "seed": seed,
        "eval_mean": statistics.fmean(returns),
        "eval_returns": returns,
        "curve_mean": statistics.fmean(curve) if curve else 0.0,
        "iterations": len(curve),
        "env_steps": sampled,
        "train_wall_s": round(wall, 2),
        "train_driver_cpu_s": round(cpu, 2),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--threshold", type=float, default=475.0)
    p.add_argument("--margin", type=float, default=20.0)
    p.add_argument("--num-cpus", type=int, default=3)
    p.add_argument("--out", default="reference-rllib.json")
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    ray.init(num_cpus=args.num_cpus, include_dashboard=False, log_to_driver=False)
    try:
        results = [run(b, s, args.total_steps, args.eval_episodes) for s in args.seeds for b in BACKENDS]
    finally:
        ray.shutdown()
    tess = [r for r in results if r["backend"] == "tessera"]
    ctrl = [r for r in results if r["backend"] == "gymnasium"]
    median_eval = statistics.median(r["eval_mean"] for r in tess)
    diff = statistics.fmean(r["curve_mean"] for r in tess) - statistics.fmean(r["curve_mean"] for r in ctrl)
    passed = median_eval >= args.threshold and abs(diff) <= args.margin
    report = {
        "library": "rllib",
        "scope": "validated against Gymnasium 1.2.2 (RLlib's pin); not a supported Gymnasium version",
        "versions": {
            "ray": ray.__version__,
            "gymnasium": gymnasium.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "tessera_rl": tessera_rl.version(),
            "tessera_build_profile": tessera_rl._native.build_profile(),
            "python": platform.python_version(),
        },
        "torch_threads": torch.get_num_threads(),
        "ray_num_cpus": args.num_cpus,
        "cpu": _cpu_model(),
        "commit": _commit(),
        "upstream": UPSTREAM,
        "hyperparameters": {**HYPER, "total_steps": args.total_steps, "other": "RLlib PPOConfig defaults"},
        "rule": {"threshold": args.threshold, "margin": args.margin, "source": "tools/reference_training.py defaults"},
        "median_tessera_eval": median_eval,
        "curve_mean_difference": diff,
        "passed": passed,
        "runs": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for r in results:
        print(f"rllib {r['backend']:9} seed {r['seed']}: eval {r['eval_mean']:.1f} curve {r['curve_mean']:.1f} iters {r['iterations']} wall {r['train_wall_s']} s")
    print(f"rllib: median tessera eval {median_eval:.1f} (>= {args.threshold}), curve diff {diff:+.1f} (|.| <= {args.margin}): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
