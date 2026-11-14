# SPDX-License-Identifier: MIT OR Apache-2.0
"""One run of the PuffeRL split (``docs/reviews/gpu-split-plan.md`` r5).

Exploratory, CartPole only, no gate claim. A run prints one JSON line and
keeps nothing else: no checkpoint, policy or log.

Modes, each a separate run and reported separately:

- ``e2e``: end to end, uninstrumented. The profiler never fires inside the
  timed window, and ``torch.cuda.synchronize()`` runs immediately before and
  after the window only, so queued learner work at the end is counted.
  Throughput only.
- ``sync``: synchronised phase costs. The trainer comes from the scratch copy
  of ``pufferl.py`` with its two synchronize blocks enabled (``sync.py``,
  ``pufferl-sync.patch``), profiled every epoch. The report gives exclusive
  child categories only, never the parent ``eval`` or ``train`` added to
  them, reconciled to this run's own wall with an unattributed remainder.
  Synchronising changes overlap and scheduling, so this is a cost breakdown
  of a different run, not a share of ``e2e``.
- ``env``: environment only. The same vector environment and agent count,
  random actions, no policy. CPU work. Random actions change trajectories,
  resets and cache behaviour and drop policy transfers, so this is env
  throughput, not a bound.

Process rules:

- CUDA is never initialised before the vector environment, and so its fork
  workers, exist. That order is checked, and recorded.
- ``try`` and ``finally``: the vector environment is closed and the
  utilization monitor stopped and joined. The run's own child processes are
  listed before and after, terminated and reaped. ``PuffeRL.close()`` is not
  called, as it saves a checkpoint. No ``os._exit``.
- Wall time uses ``time.perf_counter``. PufferLib's own ``Profile`` keeps
  ``time.time``, as published.

Usage::

    python split.py CASE MODE DEVICE [--epochs 31] [--threads 4] [--env-steps 2000] [--scale full]

``--scale small`` shrinks the agent count and batch for the local tests only;
a measurement uses ``full``.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import json
import os
import platform
import shutil
import signal
import sys
import tempfile
import time

CASES = ("ocean-serial", "ocean-mp", "tessera-w1", "tessera-w2")
MODES = ("e2e", "sync", "env")
CHILDREN = ("env", "eval_forward", "eval_copy", "eval_misc", "train_forward", "train_copy", "train_misc", "learn")
PARENTS = ("eval", "train")
# Every name PufferLib's trainer profiles, in the flat timeline.
CATEGORIES = PARENTS + CHILDREN
# Agents per case at full scale, and the small test scale.
AGENTS = {"full": 8192, "small": 1024}
MINIBATCH = {"full": 32768, "small": 8192}
EXIT_NO_CUDA = 3


def config(device: str, scale: str) -> dict:
    """PufferLib's committed cartpole.ini over default.ini, with the run's
    fixed overrides (seed 1, no compile, no RNN, nothing saved)."""
    import pufferlib

    root = os.path.dirname(pufferlib.__file__)
    parser = configparser.ConfigParser()
    parser.read([os.path.join(root, "config", "default.ini"), os.path.join(root, "config", "ocean", "cartpole.ini")])

    def value(v: str):
        try:
            return ast.literal_eval(v.replace("_", "") if v.replace("_", "").replace(".", "").replace("e", "").replace("-", "").isdigit() else v)
        except (ValueError, SyntaxError):
            return v

    cfg = {k: value(v) for k, v in parser["train"].items()}
    cfg.update(
        env="puffer_cartpole",
        device=device,
        seed=1,
        cpu_offload=False,
        compile=False,
        use_rnn=False,
        data_dir="experiments",
        checkpoint_interval=10**9,
        total_timesteps=10**12,
        minibatch_size=MINIBATCH[scale],
    )
    return cfg


def make_vecenv(case: str, agents: int):
    import pufferlib.vector

    if case.startswith("ocean"):
        from pufferlib.ocean import environment as ocean

        creator = ocean.env_creator("puffer_cartpole")
        if case == "ocean-mp":
            return pufferlib.vector.make(
                creator,
                env_kwargs={"num_envs": agents // 2},
                backend=pufferlib.vector.Multiprocessing,
                num_envs=2,
                num_workers=2,
                batch_size=2,
                zero_copy=True,
                seed=1,
            )
        return pufferlib.vector.make(
            creator, env_kwargs={"num_envs": agents}, backend=pufferlib.vector.Serial, num_envs=1, seed=1
        )
    from tessera_rl.puffer import make_puffer

    workers = {"tessera-w1": 1, "tessera-w2": 2}[case]
    return pufferlib.vector.make(make_puffer("tessera/CartPole-v1", num_envs=agents, workers=workers))


def children() -> list[int]:
    import psutil

    return sorted(p.pid for p in psutil.Process().children(recursive=True))


def reap(pids: list[int]) -> list[int]:
    """Terminates and reaps the given child processes; returns those still
    alive afterwards, which should be none."""
    import psutil

    procs = []
    for pid in pids:
        try:
            procs.append(psutil.Process(pid))
        except psutil.NoSuchProcess:
            pass
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=5)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    return sorted(p.pid for p in alive)


def synchronize(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def advantage_backend(module) -> str:
    """The advantage path PufferLib loaded: the CUDA kernel only when it found
    ``nvcc``, else the CPU fallback with copies to and from the device."""
    return "cuda" if getattr(module, "ADVANTAGE_CUDA", False) else "cpu-fallback"


def run_env(vec, steps: int, warmup: int = 300) -> dict:
    """Environment only: random actions through the vector environment."""
    import numpy as np

    rng = np.random.default_rng(1)
    n = vec.num_agents
    space = vec.single_action_space
    actions = [rng.integers(0, space.n, size=n, dtype=np.int32) for _ in range(256)]
    vec.async_reset(seed=1)
    vec.recv()
    for i in range(warmup):
        vec.send(actions[i % 256])
        vec.recv()
    # Nothing is pending here. Each timed step is a send and its matching recv,
    # so the last transition completes inside the window.
    start = time.perf_counter()
    for i in range(steps):
        vec.send(actions[(warmup + i) % 256])
        vec.recv()
    wall = time.perf_counter() - start
    return {"steps": steps * n, "wall_s": wall, "sps": steps * n / wall}


def flat_profile(module):
    """A profiler for mode ``sync`` on top of the synchronised copy's Profile.

    PufferLib's own Profile is not exclusive: ``train_misc`` is pushed with
    ``nest=True`` inside the minibatch loop without popping the entry before
    it, so phases overlap and are counted twice. This subclass keeps every
    call of the parent, including the published synchronize points, and also
    keeps a flat timeline: each interval between two profiler calls is
    charged to exactly one name, the one most recently started, and the
    interval after ``end()`` to none. So the per-name totals are exclusive and
    add up to at most the run's wall; the rest is reported as unattributed.
    """

    class FlatProfile(module.Profile):
        def __init__(self, frequency=1):
            super().__init__(frequency)
            self.flat = {n: 0.0 for n in CATEGORIES}
            self.current = None
            self.mark = None

        def _charge(self):
            now = time.perf_counter()
            if self.current is not None:
                self.flat[self.current] = self.flat.get(self.current, 0.0) + now - self.mark
            self.mark = now

        def __call__(self, name, epoch, nest=False):
            if epoch % self.frequency != 0:
                return
            super().__call__(name, epoch, nest)
            self._charge()
            self.current = name

        def end(self):
            super().end()
            self._charge()
            self.current = None

    return FlatProfile(1)


def run_trainer(module, vec, cfg: dict, mode: str, epochs: int, device: str) -> dict:
    """e2e or sync on one trainer built from ``module``. The first epoch is a
    warm-up and is dropped."""
    import torch
    from pufferlib import models

    policy = models.Default(vec, hidden_size=128).to(device)
    trainer = module.PuffeRL(cfg, vec, policy)
    try:
        if mode == "sync":
            trainer.profile = flat_profile(module)
        else:
            trainer.profile.frequency = 10**9
        trainer.evaluate()
        trainer.train()
        if mode == "sync":
            base = {n: trainer.profile.profiles[n]["elapsed"] for n in CATEGORIES}
            flat0 = dict(trainer.profile.flat)
        synchronize(torch, device)
        step0 = int(trainer.global_step)
        start = time.perf_counter()
        for _ in range(epochs - 1):
            trainer.evaluate()
            trainer.train()
        synchronize(torch, device)
        wall = time.perf_counter() - start
        steps = int(trainer.global_step) - step0
        out = {
            "steps": steps,
            "wall_s": wall,
            "sps": steps / wall,
            "batch": trainer.config["batch_size"],
            "profile_frequency": trainer.profile.frequency,
        }
        if mode == "sync":
            flat = trainer.profile.flat
            delta = {n: flat[n] - flat0.get(n, 0.0) for n in flat}
            out["exclusive_s"] = {n: delta[n] for n in CHILDREN}
            # Time between a parent's start and its first child: overhead,
            # not a phase.
            out["parent_overhead_s"] = {n: delta[n] for n in PARENTS}
            out["unattributed_s"] = wall - sum(delta.values())
            end = {n: trainer.profile.profiles[n]["elapsed"] for n in CATEGORIES}
            out["pufferlib_profile_s_not_additive"] = {n: end[n] - base[n] for n in CATEGORIES}
        return out
    finally:
        stop_monitors()


def stop_monitors() -> list[str]:
    """Stops and joins every PufferLib utilization monitor thread, including
    one left by a trainer whose construction failed after starting it.
    Returns the names of threads still alive afterwards."""
    import threading

    monitors = [t for t in threading.enumerate() if type(t).__name__ == "Utilization"]
    for t in monitors:
        t.stopped = True
    for t in monitors:
        t.join(timeout=10)
    return [t.name for t in monitors if t.is_alive()]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("case", choices=CASES)
    p.add_argument("mode", choices=MODES)
    p.add_argument("device", choices=("cpu", "cuda"))
    # ADR-0061 item 2 (G5): 31 epochs, first dropped, only lengthen the
    # timed window; every cartpole.ini value stays as published.
    p.add_argument("--epochs", type=int, default=31)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--env-steps", type=int, default=2000)
    p.add_argument("--scale", choices=tuple(AGENTS), default="full")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    record = {"case": args.case, "mode": args.mode, "device": args.device, "scale": args.scale, "threads": args.threads}
    # No CUDA call of any kind before the vector environment and its fork
    # workers exist: even torch.cuda.is_available() may initialise the driver.
    # is_initialized() only reads torch's own flag.
    record["cuda_initialized_before_vecenv"] = bool(torch.cuda.is_initialized())
    if record["cuda_initialized_before_vecenv"]:
        print(json.dumps({**record, "error": "CUDA was initialised before the vector environment"}), flush=True)
        return 4

    before = children()
    status = None
    vec = None
    scratch = tempfile.mkdtemp(prefix="gpu-split-")
    try:
        vec = make_vecenv(args.case, AGENTS[args.scale])
        record["workers"] = [pid for pid in children() if pid not in before]
        if args.device == "cuda":
            if not torch.cuda.is_available():
                record["error"] = "torch.cuda.is_available() is false"
                status = EXIT_NO_CUDA
                return status
            record["device_name"] = torch.cuda.get_device_name(0)
        if args.mode == "env":
            result = run_env(vec, args.env_steps)
            import pufferlib.pufferl as module
        else:
            if args.mode == "sync":
                module, digests = __import__("sync").load(scratch)
                record.update(digests)
            else:
                import pufferlib.pufferl as module
            result = run_trainer(module, vec, config(args.device, args.scale), args.mode, args.epochs, args.device)
            record["epochs_timed"] = args.epochs - 1
        record["advantage_backend"] = advantage_backend(module)
        record.update(result)
    finally:
        record["monitors_left"] = stop_monitors()
        if vec is not None:
            vec.close()
        record["children_after_close"] = children()
        record["children_left"] = reap([pid for pid in record["children_after_close"] if pid not in before])
        shutil.rmtree(scratch)
        if status is not None:
            print(json.dumps(record), flush=True)

    import numpy
    import pufferlib

    record["versions"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pufferlib": getattr(pufferlib, "__version__", None),
        "numpy": numpy.__version__,
    }
    record["host"] = platform.machine()
    print(json.dumps(record), flush=True)
    return 0 if not record["children_left"] and not record["monitors_left"] else 5


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
