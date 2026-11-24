#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Reference training with Tianshou (SPEC 14.2, #368): Tianshou's own PPO
on ``tessera/CartPole-v1`` through the Gymnasium adapter, with Gymnasium's
``CartPole-v1`` as the control. It validates the adapter with an
unmodified third-party trainer; it is not the SPEC 18 learning test.

Each backend runs Tianshou's ``DummyVectorEnv`` over ``gymnasium.make``
single environments, with the CartPole PPO settings of Tianshou's own test
``test/discrete/test_ppo_discrete.py`` at tag v2.0.1 (its argparse defaults
and model construction: one shared ``Net`` for actor and critic, a softmax
actor with ``Categorical``, no advantage normalization). The upstream test
stops at a reward of 195; this script trains a fixed budget instead, for ``--total-steps`` steps per seed. It then evaluates the
greedy policy (argmax of the actor's probabilities) on ``--eval-episodes``
episodes of a fresh environment (seeds ``EVAL_SEED_BASE + i``, as in
tools/reference_training.py).

The rule is the same as in tools/reference/sb3_cartpole.py: the defaults
of tools/reference_training.py (``--threshold 475``, ``--margin 20``). The
median Tessera evaluation must be at least 475, and the Tessera minus
Gymnasium difference of the training-curve means must be at most 20 in
absolute value. It is report-only until 7 green nights.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time

import gymnasium
import numpy as np
import tianshou
import torch
from tianshou.algorithm import PPO
from tianshou.algorithm.modelfree.reinforce import DiscreteActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic

import tessera_rl
import tessera_rl.gym  # noqa: F401  (registers tessera/* with Gymnasium)

EVAL_SEED_BASE = 10_000
BACKENDS = {"tessera": "tessera/CartPole-v1", "gymnasium": "CartPole-v1"}
# thu-ml/tianshou test/discrete/test_ppo_discrete.py at v2.0.1 (argparse defaults).
N_ENVS = 20
HIDDEN = (64, 64)
HYPER = {
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "eps_clip": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.0,
    "max_grad_norm": 0.5,
    "collection_step_num_env_steps": 2000,
    "update_step_num_repetitions": 10,
    "batch_size": 64,
    "buffer_size": 20_000,
    "epoch_num_steps": 50_000,
    "advantage_normalization": False,
    "return_scaling": False,
    "recompute_advantage": False,
    "value_clip": False,
    "dual_clip": None,
}


class Episodes(gymnasium.Wrapper):
    """Records each finished episode's return and length; the first reset
    uses the given seed."""

    def __init__(self, env, seed: int):
        super().__init__(env)
        self._seed: int | None = seed
        self.returns: list[float] = []
        self.lengths: list[int] = []
        self._ret, self._len = 0.0, 0

    def reset(self, *, seed=None, options=None):
        if seed is None and self._seed is not None:
            seed, self._seed = self._seed, None
        self._ret, self._len = 0.0, 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._ret += float(r)
        self._len += 1
        if term or trunc:
            self.returns.append(self._ret)
            self.lengths.append(self._len)
        return obs, r, term, trunc, info


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


def _curve_mean(wrappers: list[Episodes], update_steps: int) -> float:
    """Mean over updates of the mean return of the episodes ended in them."""
    per_update: dict[int, list[float]] = {}
    for w in wrappers:
        t = 0
        for ret, length in zip(w.returns, w.lengths, strict=True):
            t += length
            per_update.setdefault(t * N_ENVS // update_steps, []).append(ret)
    means = [statistics.fmean(v) for _, v in sorted(per_update.items())]
    return statistics.fmean(means) if means else 0.0


def run(backend: str, seed: int, total_steps: int, eval_episodes: int) -> dict:
    env_id = BACKENDS[backend]
    torch.manual_seed(seed)
    np.random.seed(seed)
    wrappers: list[Episodes] = []

    def factory(rank: int):
        def make():
            w = Episodes(gymnasium.make(env_id), seed * 1000 + rank)
            wrappers.append(w)
            return w

        return make

    envs = DummyVectorEnv([factory(r) for r in range(N_ENVS)])
    probe = gymnasium.make(env_id)
    obs_shape = probe.observation_space.shape
    n_actions = int(probe.action_space.n)
    net = Net(state_shape=obs_shape, hidden_sizes=HIDDEN)  # shared, as upstream
    actor = DiscreteActor(preprocess_net=net, action_shape=n_actions)
    critic = DiscreteCritic(preprocess_net=net)
    policy = DiscreteActorPolicy(actor=actor, dist_fn=torch.distributions.Categorical, action_space=probe.action_space, deterministic_eval=True)
    probe.close()
    algo = PPO(
        policy=policy,
        critic=critic,
        optim=AdamOptimizerFactory(lr=HYPER["lr"]),
        eps_clip=HYPER["eps_clip"],
        vf_coef=HYPER["vf_coef"],
        ent_coef=HYPER["ent_coef"],
        max_grad_norm=HYPER["max_grad_norm"],
        gae_lambda=HYPER["gae_lambda"],
        gamma=HYPER["gamma"],
        return_scaling=HYPER["return_scaling"],
        dual_clip=HYPER["dual_clip"],
        value_clip=HYPER["value_clip"],
        advantage_normalization=HYPER["advantage_normalization"],
        recompute_advantage=HYPER["recompute_advantage"],
    )
    collect = HYPER["collection_step_num_env_steps"]
    collector = Collector(algo, envs, VectorReplayBuffer(HYPER["buffer_size"], N_ENVS))
    collector.reset()
    epoch_steps = HYPER["epoch_num_steps"]
    cpu0 = time.process_time()
    algo.run_training(
        OnPolicyTrainerParams(
            training_collector=collector,
            max_epochs=math.ceil(total_steps / epoch_steps),
            epoch_num_steps=epoch_steps,
            collection_step_num_env_steps=collect,
            update_step_num_repetitions=HYPER["update_step_num_repetitions"],
            batch_size=HYPER["batch_size"],
            test_collector=None,
            verbose=False,
            show_progress=False,
        )
    )
    cpu = time.process_time() - cpu0
    curve = _curve_mean(wrappers, collect)
    envs.close()
    env = gymnasium.make(env_id)
    returns = []
    actor.eval()
    with torch.no_grad():
        for i in range(eval_episodes):
            obs, _ = env.reset(seed=EVAL_SEED_BASE + i)
            done, total = False, 0.0
            while not done:
                probs, _ = actor(np.asarray(obs, dtype=np.float32)[None])
                obs, r, term, trunc, _ = env.step(int(torch.as_tensor(probs).argmax(-1)[0]))
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
    p.add_argument("--out", default="reference-tianshou.json")
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
        "library": "tianshou",
        "versions": {
            "tianshou": tianshou.__version__,
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
        "hyperparameters": {"n_envs": N_ENVS, "hidden": list(HIDDEN), **HYPER, "total_steps": args.total_steps},
        "rule": {"threshold": args.threshold, "margin": args.margin, "source": "tools/reference_training.py defaults"},
        "median_tessera_eval": median_eval,
        "curve_mean_difference": diff,
        "passed": passed,
        "runs": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for r in results:
        print(f"tianshou {r['backend']:9} seed {r['seed']}: eval {r['eval_mean']:.1f} curve {r['curve_mean']:.1f} cpu {r['train_cpu_s']} s")
    print(f"tianshou: median tessera eval {median_eval:.1f} (>= {args.threshold}), curve diff {diff:+.1f} (|.| <= {args.margin}): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
