#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""RLlib multi-agent smoke run on ``tessera/Tag-v1`` through the PettingZoo
adapter (SPEC 14.2, #447 head 2). It checks wiring only and makes no
learning claim: a multi-agent learning claim needs its own protocol.

Steps:

1. PettingZoo's own ``parallel_api_test`` on ``tessera_rl.pz.parallel_env``.
2. RLlib's ``ParallelPettingZooEnv`` wrapper over the same env, PPO with one
   policy per agent (RLlib's defaults otherwise), for ``--iterations``
   training iterations.
3. The run passes if every iteration completes, sampled steps grow, episodes
   end, and every per-agent episode return and every per-module loss
   (``LOSS_KEYS``) is finite. It fails (exit 1) otherwise.

On a capture the evader gets -1 and each pursuer on its cell +1
(tag/mod.rs), so RLlib's summed ``episode_return_mean`` mixes the two sides
(0 when one pursuer captures). The report records the per-agent returns. RLlib's
bookkeeping counters (``num_trainable_parameters``) come back as NaN after
the first iteration; they are not checked.

Like rllib_cartpole.py, it runs in compat/training/rllib (Gymnasium 1.2.2,
RLlib's pin). Like it, the script sets ``RAY_ENABLE_UV_RUN_RUNTIME_ENV=0``
itself before importing Ray, and puts Ray's session in a private temporary
directory that it removes afterwards.

Tag's observation is a Dict (``action_mask`` and a uint8 ``window`` image),
for which RLlib has no default encoder. A small ParallelEnv wrapper in this
script, not the adapter, flattens each agent's ``window`` to a float32 Box and
leaves out the mask. That is enough for a wiring smoke; masked training is
out of scope here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time

# Before Ray is imported (see rllib_cartpole.py).
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import gymnasium  # noqa: E402
import numpy as np
import pettingzoo
import ray
import torch
from pettingzoo.test import parallel_api_test
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec

import tessera_rl
from tessera_rl import pz

ENV_ID = "tessera/Tag-v1"
MAX_STEPS = 200


class _FlatWindow(pettingzoo.utils.BaseParallelWrapper):
    """Each agent's observation becomes its ``window`` as a flat float32
    vector in [0, 1]; the action mask is left out (smoke only)."""

    def __init__(self, env):
        super().__init__(env)
        self._spaces = {}
        for a in env.possible_agents:
            w = env.observation_space(a)["window"]
            self._spaces[a] = gymnasium.spaces.Box(0.0, 1.0, (int(np.prod(w.shape)),), np.float32)

    def observation_space(self, agent):
        return self._spaces[agent]

    def _flat(self, obs: dict) -> dict:
        return {a: (np.asarray(o["window"], dtype=np.float32).reshape(-1) / 255.0) for a, o in obs.items()}

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._flat(obs), info

    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)
        return self._flat(obs), rew, term, trunc, info


def _make(_cfg=None):
    from tessera_rl import pz as _pz  # noqa: PLC0415  (Ray's env-runner processes)

    return ParallelPettingZooEnv(_FlatWindow(_pz.parallel_env(ENV_ID, max_episode_steps=MAX_STEPS)))


LOSS_KEYS = ("policy_loss", "vf_loss", "total_loss", "entropy")


def _checked(runners: dict, learners: dict, agents: list[str]) -> tuple[bool, dict, dict]:
    """Per-agent returns and per-module losses, and whether all are finite
    and present."""
    returns = {a: float(v) for a, v in (runners.get("agent_episode_returns_mean") or {}).items()}
    losses = {a: {k: float(learners[a][k]) for k in LOSS_KEYS if k in learners.get(a, {})} for a in agents}
    ok = set(returns) == set(agents) and all(math.isfinite(v) for v in returns.values())
    ok &= all(len(l) == len(LOSS_KEYS) and all(math.isfinite(v) for v in l.values()) for l in losses.values())
    return ok, returns, losses


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-cpus", type=int, default=3)
    p.add_argument("--out", default="reference-rllib-tag.json")
    args = p.parse_args(argv)
    torch.set_num_threads(1)

    parallel_api_test(pz.parallel_env(ENV_ID, seed=args.seed, max_episode_steps=MAX_STEPS), num_cycles=200)
    print("rllib-tag: parallel_api_test ok")

    probe = pz.parallel_env(ENV_ID, seed=args.seed)
    agents = list(probe.possible_agents)
    probe.close()
    tune.register_env("tessera_tag", _make)
    config = (
        PPOConfig()
        .environment("tessera_tag")
        .multi_agent(
            policies={a: PolicySpec() for a in agents},
            policy_mapping_fn=lambda agent_id, *_a, **_k: agent_id,
        )
        .debugging(seed=args.seed)
    )
    iters = []
    ray_tmp = tempfile.mkdtemp(prefix="tessera-ray-")
    ok = True
    try:
        ray.init(num_cpus=args.num_cpus, include_dashboard=False, log_to_driver=False, _temp_dir=ray_tmp)
        try:
            algo = config.build_algo()
            last = -1
            t0 = time.perf_counter()
            for i in range(args.iterations):
                r = algo.train()
                steps = int(r["num_env_steps_sampled_lifetime"])
                runners = r.get("env_runners", {})
                learners = r.get("learners", {})
                finite, returns, losses = _checked(runners, learners, agents)
                grew = steps > last
                episodes = int(runners.get("num_episodes", 0) or 0)
                ok &= finite and grew and episodes > 0
                last = steps
                iters.append({"iteration": i + 1, "env_steps": steps, "episodes": episodes, "agent_returns": returns, "losses": losses, "finite": finite, "steps_grew": grew})
                shown = " ".join(f"{a} {v:+.2f}" for a, v in sorted(returns.items()))
                print(f"rllib-tag: iter {i + 1} steps {steps} episodes {episodes} returns [{shown}] finite {finite}")
            wall = time.perf_counter() - t0
            algo.stop()
        finally:
            ray.shutdown()
    finally:
        shutil.rmtree(ray_tmp, ignore_errors=True)
    report = {
        "env": ENV_ID,
        "scope": "wiring smoke only, no learning claim; Gymnasium 1.2.2 (RLlib's pin)",
        "agents": agents,
        "versions": {
            "ray": ray.__version__,
            "pettingzoo": pettingzoo.__version__,
            "gymnasium": gymnasium.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "tessera_rl": tessera_rl.version(),
            "python": platform.python_version(),
        },
        "iterations": iters,
        "wall_s": round(wall, 2),
        "passed": ok,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"rllib-tag: {len(iters)} iterations, {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
