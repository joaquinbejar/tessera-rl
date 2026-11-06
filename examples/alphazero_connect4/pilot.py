# SPDX-License-Identifier: MIT OR Apache-2.0
"""P2-M3 CPU throughput and budget pilot for the AlphaZero Connect Four
example (PILOT.md has the pre-registered plan and criteria).

Measures, each separately: network inference (``infer``), self-play
(``selfplay``), training steps (``train``) and evaluation against the
ADR-0042 bot with its repeatability check (``eval``); ``all`` runs the
four in that order.

The command runs as a **supervisor**. It reserves the run's whole
allowance in the core-hour ledger (``--ledger``) before any work. Then it
runs the measurements in a child process group and kills that group
(workers and the bot included) at the deadline: the smaller of the wall
cap and what is left of the core-hour budget. It then settles the ledger
to the time used. The child writes ``report.json`` into ``--out`` after
every section, so a killed run keeps the sections it finished; the
supervisor adds the final status. Exit codes:

- 0: ok;
- 3: the repeatability check failed;
- 124: killed at the deadline;
- any other code: the child failed.

``pilot.py ledger-settle --ledger PATH --index N --interrupted`` settles by
hand, at its full reservation, the open entry of a run lost where its pid
cannot be checked (another host or container); until then it blocks the
ledger.

Usage (from the repository root, the ``test`` and ``torch`` groups
installed)::

    python examples/alphazero_connect4/pilot.py all --out DIR \\
        --ledger LEDGER.json --cores 16 --budget-core-hours 24 \\
        --wall-cap-s 5400 --source-sha SHA --bot PATH/tessera-rl-connect4-ref
"""

from __future__ import annotations

import os

# Library thread pools are pinned to 1 before torch is imported (P2-M3:
# one thread, deterministic algorithms); an inherited value is recorded
# and overridden, never kept. Inference cells that use more threads set
# them explicitly with torch.set_num_threads.
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
# The supervisor passes what it inherited to its child.
_INHERITED = os.environ.get("PILOT_INHERITED_THREAD_VARS")
INHERITED_THREAD_VARS = (
    __import__("json").loads(_INHERITED) if _INHERITED else {v: os.environ.get(v) for v in THREAD_VARS}
)
os.environ["PILOT_INHERITED_THREAD_VARS"] = __import__("json").dumps(INHERITED_THREAD_VARS)
for _var in THREAD_VARS:
    os.environ[_var] = "1"

import argparse  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import augment  # noqa: E402
import game  # noqa: E402
import net as netmod  # noqa: E402
import tessera_rl  # noqa: E402
from ledger import BudgetExceeded, Ledger  # noqa: E402
from tessera_rl.search import Search  # noqa: E402

TIE_BASE = 0x7E55E2A000000000
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EVAL_OPENINGS = os.path.join(REPO, "compat", "connect4", "openings-eval.txt")
HELDOUT_OPENINGS = os.path.join(REPO, "compat", "connect4", "openings-heldout.txt")

EXIT_NOT_REPEATABLE = 3
EXIT_DEADLINE = 124
# Reserved for the supervisor's kill and cleanup: the work deadline is
# this much before the end of the reservation.
CLEANUP_GRACE_S = 2.0


class NotRepeatable(RuntimeError):
    """Two completed evaluation passes gave different canonical logs."""


def configure_torch(threads: int = 1) -> None:
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already set in this process: it can be set once only.
        pass
    torch.use_deterministic_algorithms(True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def canonical_hash(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def quantiles(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)

    def q(p):
        return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]

    return {"n": len(s), "p10": q(0.1), "median": statistics.median(s), "p90": q(0.9), "min": s[0], "max": s[-1]}


def machine() -> dict:
    """CPU model, logical and physical cores, governor."""
    info = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cores": os.cpu_count(),
        "affinity_cores": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            text = f.read()
        cores = set()
        phys = core = None
        for line in text.splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key == "model name":
                info["cpu_model"] = value
            elif key == "physical id":
                phys = value
            elif key == "core id":
                core = value
            elif not line.strip() and core is not None:
                cores.add((phys, core))
                phys = core = None
        if core is not None:
            cores.add((phys, core))
        info["physical_cores"] = len(cores) or None
    except OSError:
        info["cpu_model"] = platform.processor()
        info["physical_cores"] = None
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", encoding="utf-8") as f:
            info["governor"] = f.read().strip()
    except OSError:
        info["governor"] = None
    return info


def provenance(args) -> dict:
    return {
        "source_sha": args.source_sha,
        "env_id": game.ENV_ID,
        "image_digest": os.environ.get("PILOT_IMAGE_DIGEST"),
        "cpuset": os.environ.get("PILOT_CPUSET"),
        "bot_sha256": sha256_file(args.bot) if args.bot else None,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "tessera_rl": tessera_rl.version(),
        },
        "threads": {
            "torch_intra_op": torch.get_num_threads(),
            "torch_inter_op": torch.get_num_interop_threads(),
            "env": {v: os.environ.get(v) for v in THREAD_VARS},
            "env_inherited": INHERITED_THREAD_VARS,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "machine": machine(),
        "openings": {"eval": sha256_file(EVAL_OPENINGS), "heldout": sha256_file(HELDOUT_OPENINGS)},
    }


class Clock:
    """The child's view of its deadline (an absolute monotonic time is
    not shared across processes, so the supervisor passes epoch seconds).
    The hard limit is the supervisor's kill; this lets a cell that would
    not fit be skipped and reported as missing instead of being cut."""

    def __init__(self, deadline_epoch: float, skips: bool = True):
        self.deadline_epoch = deadline_epoch
        self.skips = skips

    def left(self) -> float:
        return self.deadline_epoch - time.time()

    def check(self, projected_s: float, what: str) -> None:
        if self.skips and projected_s > self.left():
            raise BudgetExceeded(f"{what}: {projected_s:.1f} s projected, {self.left():.1f} s left")


def repeat_cell(clock: Clock, name: str, warmup: int, repeats: int, allowance_s: float, fn) -> dict:
    """Runs ``fn`` ``warmup`` times (discarded) then ``repeats`` times. The
    first repetition needs ``allowance_s`` left; each later one, the last
    one's duration. Returns the samples and, if stopped, why."""
    samples: list[float] = []
    projected = allowance_s
    try:
        for i in range(warmup + repeats):
            clock.check(projected, f"{name} #{i}")
            start = time.perf_counter()
            fn()
            projected = time.perf_counter() - start
            if i >= warmup:
                samples.append(projected)
        stopped = None
    except BudgetExceeded as e:
        stopped = str(e)
    return {"warmup": warmup, "repeats": repeats, "samples_s": samples, "stopped": stopped}


# ---------------------------------------------------------------- infer


def _forward(model, x, legal, inner):
    with torch.inference_mode():
        for _ in range(inner):
            logits, _ = model(x)
            netmod.masked_priors(logits, legal)


def run_infer(args, clock: Clock) -> dict:
    cells = []
    for size in args.sizes:
        model = netmod.build(size, seed=args.seed)
        for threads in sorted({1, args.cores}):
            torch.set_num_threads(threads)
            for batch in args.infer_batches:
                x = torch.zeros((batch, 2, 6, 7))
                legal = np.ones((batch, 7), dtype=bool)
                fn = functools.partial(_forward, model, x, legal, args.infer_inner)
                cell = repeat_cell(
                    clock, f"infer {size} t{threads} b{batch}", args.warmup, args.repeats, args.cell_allowance_s, fn
                )
                per_sample = [s / (args.infer_inner * batch) for s in cell["samples_s"]]
                cells.append(
                    {
                        "size": size,
                        "params": netmod.parameters(model),
                        "threads": threads,
                        "batch": batch,
                        "inner": args.infer_inner,
                        **cell,
                        "per_sample_s": quantiles(per_sample),
                        "per_batch_s": quantiles([s / args.infer_inner for s in cell["samples_s"]]),
                    }
                )
    torch.set_num_threads(1)
    return {"cells": cells}


# ---------------------------------------------------------------- selfplay

# Per worker process: built once by the pool initializer.
_WORKER: dict = {}


def _selfplay_init(cfg: dict, barrier, ready) -> None:
    start = time.time()
    configure_torch(1)
    if cfg["evaluator"] == "uniform":
        evaluator = netmod.Uniform()
    else:
        evaluator = netmod.Evaluator(netmod.build(cfg["size"], seed=cfg["net_seed"]))
    _WORKER.update(
        evaluator=evaluator,
        cfg=cfg,
        pid=os.getpid(),
    )
    ready.put({"pid": os.getpid(), "init_s": time.time() - start, "ready_epoch": time.time()})
    # Every worker is built before any measured work starts.
    barrier.wait()


def _selfplay_round(task: dict) -> dict:
    cfg = _WORKER["cfg"]
    evaluator = _WORKER["evaluator"]
    search = Search(
        simulations=cfg["sims"],
        batch_size=cfg["batch_size"],
        capacity=cfg["capacity"],
        noise_alpha=1.0,
        noise_epsilon=0.25,
        seed=task["seed"],
    )
    games = []
    for g in range(task["games"]):
        r = game.self_play(search, evaluator, seed=task["seed"] + g, temperature_moves=cfg["temperature_moves"])
        games.append(
            {
                "wall_s": r.wall_ns / 1e9,
                "moves": r.timing.moves,
                "search_s": r.timing.search_ns / 1e9,
                "net_s": r.timing.net_ns / 1e9,
                "batches": np.bincount(r.timing.batches, minlength=cfg["batch_size"] + 1).tolist(),
            }
        )
    return {"pid": os.getpid(), "games": games}


def _selfplay_cell(args, clock: Clock, kind: str, size, sims: int, workers: int) -> dict:
    cell: dict = {"evaluator": kind, "size": size, "sims": sims, "workers": workers, "allocated_cores": args.cores}
    try:
        clock.check(args.cell_allowance_s, f"selfplay {kind}:{size} sims{sims} W{workers} startup")
    except BudgetExceeded as e:
        return {**cell, "stopped": str(e)}
    cfg = {
        "evaluator": kind,
        "size": size,
        "net_seed": args.seed,
        "sims": sims,
        "batch_size": args.batch_size,
        "capacity": args.capacity,
        "temperature_moves": args.temperature_moves,
    }
    ctx = mp.get_context("spawn")
    start = time.time()
    barrier = ctx.Barrier(workers + 1)
    ready = ctx.Queue()
    with ctx.Pool(workers, initializer=_selfplay_init, initargs=(cfg, barrier, ready)) as pool:
        # Startup ends when every worker has imported torch and built its
        # net and search evaluator: the barrier releases them together.
        barrier.wait()
        seen = {}
        for _ in range(workers):
            p = ready.get()
            seen[p["pid"]] = p
        startup = max(p["ready_epoch"] for p in seen.values()) - start
        rounds = []
        projected = args.cell_allowance_s
        stopped = None
        for i in range(args.warmup + args.repeats):
            try:
                clock.check(projected, f"selfplay {kind}:{size} sims{sims} W{workers} #{i}")
            except BudgetExceeded as e:
                stopped = str(e)
                break
            tasks = [{"seed": args.seed * 1000 + i * 100 + w, "games": args.games_per_worker} for w in range(workers)]
            t0 = time.perf_counter()
            results = pool.map(_selfplay_round, tasks, chunksize=1)
            projected = time.perf_counter() - t0
            if i >= args.warmup:
                rounds.append({"wall_s": projected, "games": [g for r in results for g in r["games"]]})
    total = time.time() - start
    games = [g for r in rounds for g in r["games"]]
    search = sum(g["search_s"] for g in games)
    netsec = sum(g["net_s"] for g in games)
    steady_wall = sum(r["wall_s"] for r in rounds)
    per_round = [len(r["games"]) / r["wall_s"] * 3600 for r in rounds if r["wall_s"] > 0]
    return {
        **cell,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rounds_kept": len(rounds),
        "workers_seen_ready": len(seen),
        "worker_init_s": quantiles([p["init_s"] for p in seen.values()]),
        "startup_to_all_ready_s": startup,
        "cell_wall_s": total,
        "games": len(games),
        "game_wall_s": quantiles([g["wall_s"] for g in games]),
        "moves_per_game": quantiles([g["moves"] for g in games]),
        "search_per_move_s": quantiles([g["search_s"] / g["moves"] for g in games if g["moves"]]),
        "net_share_of_search": netsec / search if search else None,
        "non_net_share_of_search": 1 - netsec / search if search else None,
        "leaf_batch_histogram": np.sum([g["batches"] for g in games], axis=0).tolist() if games else [],
        # Steady state: the kept rounds only, per round.
        "steady_games_per_hour": quantiles(per_round),
        "steady_games_per_hour_per_allocated_core": quantiles([x / args.cores for x in per_round]),
        "steady_games_per_hour_per_worker": quantiles([x / workers for x in per_round]),
        # End to end: the whole cell, startup and warmup included.
        "end_to_end_games_per_hour_per_allocated_core": (len(games) / total * 3600 / args.cores) if total else None,
        "steady_wall_s": steady_wall,
        "raw_rounds": rounds,
        "stopped": stopped,
    }


def run_selfplay(args, clock: Clock) -> dict:
    cells = []
    evaluators = [("uniform", None)] + [("net", s) for s in args.sizes]
    for kind, size in evaluators:
        for sims in args.sims:
            for workers in sorted({1, min(8, args.cores), args.cores}):
                cells.append(_selfplay_cell(args, clock, kind, size, sims, workers))
    return {"cells": cells}


# ---------------------------------------------------------------- train


def _train_step(model, opt, x, tp, tv):
    logits, values = model(x)
    loss = -(tp * torch.log_softmax(logits, dim=1)).sum(dim=1).mean() + ((values - tv) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()


def run_train(args, clock: Clock) -> dict:
    hold = os.environ.get("PILOT_TEST_HOLD_GIL_S")
    if hold and args.source_sha == "test":
        # Test hook only (with --source-sha test): native work that holds
        # the GIL, which starves the child's watchdog thread.
        import ctypes

        ctypes.PyDLL(None).sleep(int(hold))
    cells = []
    gen = np.random.default_rng(args.seed)
    for size in args.sizes:
        model = netmod.build(size, seed=args.seed).train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        for batch in args.train_batches:
            # Synthetic samples of the right shapes, mirrored: timing only.
            half = batch // 2
            planes = (gen.random((half, 2, 6, 7)) < 0.3).astype(np.float32)
            policy = gen.dirichlet(np.ones(7), size=half)
            value = gen.uniform(-1, 1, size=half)
            p, pol, v = augment.augment(planes, policy, value)
            x, tp, tv = torch.from_numpy(p), torch.from_numpy(pol).float(), torch.from_numpy(v).float()
            fn = functools.partial(_train_step, model, opt, x, tp, tv)
            cell = repeat_cell(clock, f"train {size} b{batch}", args.warmup, args.repeats, args.cell_allowance_s, fn)
            cells.append(
                {"size": size, "batch": 2 * half, "device": "cpu", "threads": 1, **cell, "step_s": quantiles(cell["samples_s"])}
            )
    return {"cells": cells, "gpu": "not measured (no Windows session approved)"}


# ---------------------------------------------------------------- eval


def _eval_pass(ckpt: bytes, openings: list[str], args) -> dict:
    """One evaluation pass from the serialized checkpoint with a fresh net,
    fresh searches and a fresh bot process."""
    model = netmod.Net(*netmod.SIZES[args.eval_size])
    model.load_state_dict(torch.load(io.BytesIO(ckpt), weights_only=True))
    model.eval()
    evaluator = netmod.Evaluator(model)
    bot = game.BotClient(args.bot)
    games = []
    try:
        for j, opening in enumerate(openings):
            for first in (True, False):
                # No root noise (no noise options), greedy moves, a fixed seed.
                search = Search(
                    simulations=args.eval_sims, batch_size=args.batch_size, capacity=args.capacity, seed=args.seed
                )
                start = time.perf_counter_ns()
                g = game.versus_bot(search, evaluator, bot, j, opening, first, TIE_BASE + j, seed=args.seed)
                games.append({"g": g, "wall_ns": time.perf_counter_ns() - start})
    finally:
        bot.close()
    log = [[x["g"].opening, x["g"].candidate_first, x["g"].result, x["g"].log] for x in games]
    if os.environ.get("PILOT_TEST_PERTURB_EVAL") == "1" and args.source_sha == "test":
        # Test hook only (with --source-sha test): makes each pass's log
        # differ, to exercise the repeatability failure path.
        log.append(["perturbed", time.time_ns()])
    return {
        "log": log,
        "log_sha256": canonical_hash(log),
        "results": {
            "win": sum(1 for x in games if x["g"].result > 0),
            "draw": sum(1 for x in games if x["g"].result == 0),
            "loss": sum(1 for x in games if x["g"].result < 0),
        },
        "distinct_games": len({tuple(c for _, c, _ in x["g"].log) for x in games}),
        "game_wall_s": quantiles([x["wall_ns"] / 1e9 for x in games]),
        "bot_share": sum(x["g"].bot_ns for x in games) / max(1, sum(x["wall_ns"] for x in games)),
        "candidate_moves": sum(x["g"].timing.moves for x in games),
        "games": len(games),
    }


def first_difference(a: list, b: list):
    """The index of the first game whose canonical log differs, or None."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if canonical(x) != canonical(y):
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def run_eval(args, clock: Clock) -> dict:
    """Timing and repeatability only: a frozen random-init checkpoint on
    the fixed ADR-0042 protocol. Not a P2-G1 result, not a selection
    input. Raises ``NotRepeatable`` (after writing the logs) if two
    completed passes differ."""
    configure_torch(1)
    model = netmod.build(args.eval_size, seed=args.seed)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    ckpt = buf.getvalue()
    out = {
        "checkpoint_sha256": hashlib.sha256(ckpt).hexdigest(),
        "size": args.eval_size,
        "sims": args.eval_sims,
        "passes": {},
    }
    mismatches = []
    for name, path in (("heldout", HELDOUT_OPENINGS), ("eval", EVAL_OPENINGS)):
        available = game.read_openings(path)
        openings = available[: args.eval_openings]
        passes = []
        for rep in range(2):
            try:
                clock.check(args.eval_allowance_s, f"eval {name} pass {rep}")
            except BudgetExceeded as e:
                passes.append({"stopped": str(e)})
                continue
            start = time.perf_counter()
            r = _eval_pass(ckpt, openings, args)
            r["wall_s"] = time.perf_counter() - start
            # The full canonical log goes to its own file for diagnosis.
            log_path = os.path.join(args.out, f"eval-{name}-pass{rep}.log.json")
            write_json(log_path, r.pop("log"))
            r["log_file"] = os.path.basename(log_path)
            passes.append(r)
        complete = all("log_sha256" in p for p in passes)
        verdict: dict = {
            "openings_in_file": len(available),
            "openings_played": len(openings),
            "games_per_pass": 2 * len(openings),
            "runs": passes,
        }
        if not complete:
            verdict["repeatable"] = "incomplete"
        elif passes[0]["log_sha256"] == passes[1]["log_sha256"]:
            verdict["repeatable"] = True
        else:
            with open(os.path.join(args.out, passes[0]["log_file"]), encoding="utf-8") as f:
                a = json.load(f)
            with open(os.path.join(args.out, passes[1]["log_file"]), encoding="utf-8") as f:
                b = json.load(f)
            verdict["repeatable"] = False
            verdict["first_differing_game"] = first_difference(a, b)
            mismatches.append(name)
        out["passes"][name] = verdict
    out["mismatches"] = mismatches
    return out


# ---------------------------------------------------------------- report


def starvation(report: dict) -> dict:
    """The pre-registered criterion of PILOT.md."""
    verdict: dict = {}
    infer = (report.get("infer") or {}).get("cells", [])

    def per_sample(size, batch):
        for c in infer:
            if c["size"] == size and c["threads"] == 1 and c["batch"] == batch and c["per_sample_s"].get("n"):
                return c["per_sample_s"]["median"]
        return None

    a, b = per_sample("small", 8), per_sample("small", 64)
    verdict["infer_ratio_b8_over_b64"] = a / b if a and b else None
    share = None
    for c in (report.get("selfplay") or {}).get("cells", []):
        if c.get("evaluator") == "net" and c.get("size") == "small" and c.get("sims") == 100 and c.get("workers") == 1:
            share = c.get("non_net_share_of_search")
    verdict["non_net_share_small_100"] = share
    r1 = verdict["infer_ratio_b8_over_b64"]
    fired = [r1 is not None and r1 > 2.0, share is not None and share > 0.30]
    verdict["criterion_1_fired"], verdict["criterion_2_fired"] = fired
    verdict["complete"] = r1 is not None and share is not None
    verdict["starved"] = any(fired) if verdict["complete"] else "incomplete"
    return verdict


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["infer", "selfplay", "train", "eval", "all"])
    p.add_argument("--out", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--cores", type=int, required=True, help="allocated cores (the container's cpuset)")
    p.add_argument("--budget-core-hours", type=float, required=True)
    p.add_argument("--wall-cap-s", type=float, required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--bot", help="path to the tessera-rl-connect4-ref binary (eval)")
    p.add_argument("--sizes", nargs="+", default=["small", "medium", "large"], choices=list(netmod.SIZES))
    p.add_argument("--sims", nargs="+", type=int, default=[50, 100, 200])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--capacity", type=int, default=8192)
    p.add_argument("--infer-batches", nargs="+", type=int, default=[1, 8, 64, 256])
    p.add_argument("--infer-inner", type=int, default=50)
    p.add_argument("--train-batches", nargs="+", type=int, default=[256, 1024])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--games-per-worker", type=int, default=2)
    p.add_argument("--temperature-moves", type=int, default=8)
    p.add_argument("--cell-allowance-s", type=float, default=60, help="time a cell's first unit needs left")
    p.add_argument("--eval-size", default="small", choices=list(netmod.SIZES))
    p.add_argument("--eval-sims", type=int, default=100)
    p.add_argument("--eval-openings", type=int, default=500)
    p.add_argument("--eval-allowance-s", type=float, default=900)
    p.add_argument("--seed", type=int, default=20261102)
    # Internal: set by the supervisor for the child.
    p.add_argument("--child-deadline-epoch", type=float, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.command in ("eval", "all") and not args.bot:
        p.error("eval needs --bot")
    for name in ("budget_core_hours", "wall_cap_s", "cell_allowance_s", "eval_allowance_s"):
        v = getattr(args, name)
        if not (math.isfinite(v) and v > 0):
            p.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("cores", "batch_size", "capacity", "infer_inner", "repeats", "games_per_worker", "eval_sims", "eval_openings"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.warmup < 0 or args.temperature_moves < 0:
        p.error("--warmup and --temperature-moves must be at least 0")
    if any(v < 1 for v in args.sims + args.infer_batches + args.train_batches):
        p.error("--sims, --infer-batches and --train-batches must be at least 1")
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    if args.cores > (affinity or 1):
        p.error(f"--cores {args.cores} exceeds the {affinity} cores this process may use")
    return args


def _watchdog(deadline_epoch: float, parent: int) -> None:
    """Kills the whole process group at the deadline, or as soon as the
    supervisor is gone (the child's own guard: a SIGKILLed supervisor
    cannot clean up)."""
    while True:
        if time.time() >= deadline_epoch or os.getppid() != parent:
            os.killpg(0, signal.SIGKILL)
        time.sleep(0.2)


def child(args) -> int:
    threading.Thread(target=_watchdog, args=(args.child_deadline_epoch, os.getppid()), daemon=True).start()
    configure_torch(1)
    # A test that needs the kill must not depend on how long startup took.
    no_skip = os.environ.get("PILOT_TEST_NO_SKIP") == "1" and args.source_sha == "test"
    clock = Clock(args.child_deadline_epoch, skips=not no_skip)
    params = {k: v for k, v in vars(args).items() if k != "child_deadline_epoch"}
    report: dict = {"command": args.command, "params": params, "config_sha256": canonical_hash(params)}
    report["provenance"] = provenance(args)
    path = os.path.join(args.out, "report.json")
    order = ["infer", "selfplay", "train", "eval"] if args.command == "all" else [args.command]
    runners = {"infer": run_infer, "selfplay": run_selfplay, "train": run_train, "eval": run_eval}
    report["sections_done"] = []
    write_json(path, report)
    for name in order:
        report[name] = runners[name](args, clock)
        report["sections_done"].append(name)
        report["starvation"] = starvation(report)
        write_json(path, report)
    if report.get("eval", {}).get("mismatches"):
        return EXIT_NOT_REPEATABLE
    return 0


def supervise(args, argv) -> int:
    os.makedirs(args.out, exist_ok=True)
    ledger = Ledger(args.ledger, args.budget_core_hours)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run = f"pilot {args.command} {stamp}"
    start = time.monotonic()
    index, granted = ledger.reserve(run, args.cores, args.wall_cap_s)
    work_s = granted - CLEANUP_GRACE_S
    if work_s <= 0:
        ledger.settle(index, time.monotonic() - start, "refused: the budget left is under the cleanup grace")
        print(f"pilot: {granted:.1f} s left, under the {CLEANUP_GRACE_S} s cleanup grace", file=sys.stderr)
        return EXIT_DEADLINE
    # The supervisor enforces the work deadline itself, on its own
    # monotonic clock: the child's watchdog is only a second guard (native
    # code holding the GIL can starve it).
    work_deadline = start + work_s
    deadline_epoch = time.time() + (work_deadline - time.monotonic())
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), *argv, "--child-deadline-epoch", repr(deadline_epoch)],
        start_new_session=True,
    )

    def kill_group():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def on_term(signum, _frame):
        kill_group()
        proc.wait()
        ledger.settle(index, time.monotonic() - start, f"interrupted by signal {signum}")
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    try:
        rc = proc.wait(timeout=max(0.0, work_deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        rc = None
    # Whatever happened, no descendant outlives the run.
    kill_group()
    proc.wait()
    wall = time.monotonic() - start
    if rc is None or rc == -signal.SIGKILL:
        status, code = f"killed at the deadline ({work_s:.1f} s of work)", EXIT_DEADLINE
    elif rc == 0:
        status, code = "ok", 0
    elif rc == EXIT_NOT_REPEATABLE:
        status, code = "failed: evaluation not repeatable", EXIT_NOT_REPEATABLE
    else:
        status, code = f"failed: child exit {rc}", rc if rc > 0 else 1
    entry = ledger.settle(index, wall, status)
    path = os.path.join(args.out, "report.json")
    report = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
    order = ["infer", "selfplay", "train", "eval"] if args.command == "all" else [args.command]
    report["missing_sections"] = [s for s in order if s not in report.get("sections_done", [])]
    report["status"] = status
    report["ledger"] = entry
    write_json(path, report)
    final = os.path.join(args.out, f"pilot-{socket.gethostname()}-{stamp}.json")
    os.replace(path, final)
    print(final)
    return code


def ledger_settle(argv) -> int:
    """``ledger-settle --ledger PATH --index N --interrupted``: settles a
    lost run's open entry at its full reservation."""
    p = argparse.ArgumentParser(prog="pilot.py ledger-settle", description=ledger_settle.__doc__)
    p.add_argument("--ledger", required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--interrupted", action="store_true", required=True)
    args = p.parse_args(argv)
    with open(args.ledger, encoding="utf-8") as f:
        budget = json.load(f)["budget_core_hours"]
    entry = Ledger(args.ledger, budget).settle_interrupted(args.index)
    print(json.dumps(entry))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["ledger-settle"]:
        return ledger_settle(argv[1:])
    args = parse(argv)
    if args.child_deadline_epoch is not None:
        return child(args)
    return supervise(args, argv)


if __name__ == "__main__":
    sys.exit(main())
