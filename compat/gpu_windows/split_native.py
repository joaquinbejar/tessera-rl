# SPDX-License-Identifier: MIT OR Apache-2.0
"""One run of the native Windows split (``docs/reviews/gpu-windows-native-plan.md``).

Tessera's CartPole with a plain PyTorch PPO-shaped forward and learn. PufferLib
does not build on Windows, so this has its own trainer: its numbers are this
trainer's and say nothing about PuffeRL or Ocean. A run prints one JSON line
and keeps nothing else: no checkpoint, policy or log.

Modes, each a separate run and reported separately:

- ``e2e``: uninstrumented. ``torch.cuda.synchronize()`` runs immediately before
  and after the timed window only, so queued learner work is counted.
- ``sync``: exclusive phase costs, with a synchronize at each phase boundary,
  reconciled to this run's own wall with an unattributed remainder. It is a
  cost breakdown of a different run, not a share of ``e2e``.
- ``env``: the same pool, random actions, no policy, no CUDA call.

The first epoch is a warm-up and is dropped.

Usage::

    python split_native.py CASE MODE DEVICE [--epochs 11] [--threads 4] [--env-steps 2000] [--scale full]

``--scale small`` is for the local tests only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time

CASES = {"tessera-w1": 1, "tessera-w2": 2}
MODES = ("e2e", "sync", "env")
PHASES = ("env", "forward", "copy", "gae", "learn")
AGENTS = {"full": 8192, "small": 64}
MINIBATCH = {"full": 32768, "small": 1024}
BPTT = 64
UPDATE_EPOCHS = 1
HIDDEN = 128
SEED = 1
LEARNING_RATE = 2.5e-4
GAMMA, LAMBDA = 0.99, 0.95
CLIP, VALUE_COEF, ENTROPY_COEF = 0.2, 0.5, 0.01
ENV_WARMUP = 300
EXIT_NO_CUDA = 3


class Phases:
    """Charges each interval between two marks to exactly one phase, so the
    totals are exclusive. Disabled in ``e2e``: no clock and no synchronize."""

    def __init__(self, enabled: bool, synchronize):
        self.enabled = enabled
        self.synchronize = synchronize
        self.reset()

    def reset(self) -> None:
        self.totals = dict.fromkeys(PHASES, 0.0)
        self.current = None
        self.last = None

    def mark(self, name: str | None) -> None:
        if not self.enabled:
            return
        self.synchronize()
        now = time.perf_counter()
        if self.current is not None:
            self.totals[self.current] += now - self.last
        self.current, self.last = name, time.perf_counter()


def make_env(case: str, agents: int):
    from tessera_rl import make

    return make("tessera/CartPole-v1", num_envs=agents, seed=SEED, workers=CASES[case], auto_reset="same_step")


def _network(obs_size: int, actions: int):
    import torch
    from torch import nn

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(obs_size, HIDDEN), nn.ReLU())
            self.policy = nn.Linear(HIDDEN, actions)
            self.value = nn.Linear(HIDDEN, 1)

        def forward(self, obs: torch.Tensor):
            hidden = self.body(obs)
            return self.policy(hidden), self.value(hidden).squeeze(-1)

    return Network()


def _observation(step) -> "object":
    import numpy as np

    return np.array(step.observations["obs"], dtype=np.float32).reshape(step.rewards.shape[0], -1)


class Trainer:
    def __init__(self, env, device: str, minibatch: int, phases: Phases):
        import torch

        self.torch, self.env, self.device, self.minibatch, self.phases = torch, env, device, minibatch, phases
        self.agents = env.num_envs * env.agents
        obs = _observation(env.reset(seed=SEED))
        self.obs = torch.as_tensor(obs, device=device)
        self.net = _network(self.obs.shape[1], env.action_choices).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        size = (BPTT, self.agents)
        self.observations = torch.zeros((*size, self.obs.shape[1]), device=device)
        self.actions = torch.zeros(size, dtype=torch.int64, device=device)
        self.log_probs, self.values = torch.zeros(size, device=device), torch.zeros(size, device=device)
        self.rewards, self.dones = torch.zeros(size, device=device), torch.zeros(size, device=device)
        self.generator = torch.Generator(device=device).manual_seed(SEED)

    def collect(self) -> None:
        torch, mark = self.torch, self.phases.mark
        for t in range(BPTT):
            mark("forward")
            self.observations[t] = self.obs
            with torch.no_grad():
                logits, value = self.net(self.obs)
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
                self.log_probs[t], self.values[t], self.actions[t] = distribution.log_prob(action), value, action
            mark("copy")
            actions = action.cpu().numpy().astype("int32").reshape(self.env.num_envs, self.env.agents)
            mark("env")
            step = self.env.step(actions)
            mark("copy")
            self.obs = torch.as_tensor(_observation(step), device=self.device)
            self.rewards[t] = torch.as_tensor(step.rewards.reshape(-1).copy(), device=self.device)
            done = (step.agent_terminated | step.agent_truncated).reshape(-1).copy()
            self.dones[t] = torch.as_tensor(done, device=self.device).float()
        mark(None)

    def learn(self) -> None:
        torch, mark = self.torch, self.phases.mark
        mark("gae")
        with torch.no_grad():
            _, following = self.net(self.obs)
            advantages = torch.zeros_like(self.rewards)
            last = torch.zeros(self.agents, device=self.device)
            for t in reversed(range(BPTT)):
                keep = 1.0 - self.dones[t]
                delta = self.rewards[t] + GAMMA * following * keep - self.values[t]
                last = delta + GAMMA * LAMBDA * keep * last
                advantages[t], following = last, self.values[t]
            returns = advantages + self.values
        obs, actions = self.observations.flatten(0, 1), self.actions.flatten()
        log_probs, advantages, returns = self.log_probs.flatten(), advantages.flatten(), returns.flatten()
        mark("learn")
        for _ in range(UPDATE_EPOCHS):
            order = torch.randperm(actions.shape[0], generator=self.generator, device=self.device)
            for start in range(0, actions.shape[0] - self.minibatch + 1, self.minibatch):
                index = order[start : start + self.minibatch]
                logits, value = self.net(obs[index])
                distribution = torch.distributions.Categorical(logits=logits)
                ratio = (distribution.log_prob(actions[index]) - log_probs[index]).exp()
                advantage = advantages[index]
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                unclipped, clipped = -advantage * ratio, -advantage * ratio.clamp(1 - CLIP, 1 + CLIP)
                loss = (
                    torch.max(unclipped, clipped).mean()
                    + VALUE_COEF * (value - returns[index]).pow(2).mean()
                    - ENTROPY_COEF * distribution.entropy().mean()
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
        mark(None)


def run_trainer(env, device: str, mode: str, epochs: int, minibatch: int) -> dict:
    import torch

    def synchronize() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    phases = Phases(mode == "sync", synchronize)
    trainer = Trainer(env, device, minibatch, phases)
    trainer.collect()
    trainer.learn()
    phases.reset()
    synchronize()
    start = time.perf_counter()
    timed = 0
    for _ in range(epochs - 1):
        trainer.collect()
        trainer.learn()
        timed += 1
    synchronize()
    wall = time.perf_counter() - start
    steps = timed * BPTT * trainer.agents
    out = {"epochs_timed": timed, "steps": steps, "wall_s": wall, "sps": steps / wall, "batch": BPTT * trainer.agents, "minibatch": minibatch}
    if mode == "sync":
        out["exclusive_s"] = dict(phases.totals)
        out["unattributed_s"] = wall - sum(phases.totals.values())
    return out


def run_env(env, steps: int) -> dict:
    import numpy as np

    rng = np.random.default_rng(SEED)
    shape = (env.num_envs, env.agents)
    actions = [rng.integers(0, env.action_choices, size=shape, dtype=np.int32) for _ in range(256)]
    env.reset(seed=SEED)
    for i in range(ENV_WARMUP):
        env.step(actions[i % 256])
    start = time.perf_counter()
    for i in range(steps):
        env.step(actions[(ENV_WARMUP + i) % 256])
    wall = time.perf_counter() - start
    agents = env.num_envs * env.agents
    return {"steps": steps * agents, "wall_s": wall, "sps": steps * agents / wall}


def native_sha256() -> str:
    import tessera_rl._native as native

    with open(native.__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("case", choices=tuple(CASES))
    p.add_argument("mode", choices=MODES)
    p.add_argument("device", choices=("cpu", "cuda"))
    p.add_argument("--epochs", type=int, default=11)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--env-steps", type=int, default=2000)
    p.add_argument("--scale", choices=tuple(AGENTS), default="full")
    args = p.parse_args(argv)
    if args.epochs < 2:
        p.error("--epochs must be at least 2: the first is dropped")

    import numpy
    import torch

    import tessera_rl

    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    record = {"case": args.case, "mode": args.mode, "device": args.device, "scale": args.scale, "threads": args.threads}
    record["versions"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "tessera_rl": tessera_rl.version(),
    }
    record["native_sha256"] = native_sha256()
    record["host"] = platform.platform()
    if args.mode != "env" and args.device == "cuda":
        if not torch.cuda.is_available():
            print(json.dumps({**record, "error": "torch.cuda.is_available() is false"}), flush=True)
            return EXIT_NO_CUDA
        record["device_name"] = torch.cuda.get_device_name(0)
    env = make_env(args.case, AGENTS[args.scale])
    if args.mode == "env":
        record.update(run_env(env, args.env_steps))
    else:
        record.update(run_trainer(env, args.device, args.mode, args.epochs, MINIBATCH[args.scale]))
    print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
