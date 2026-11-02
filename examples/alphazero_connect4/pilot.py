# SPDX-License-Identifier: MIT OR Apache-2.0
"""P2-M3 CPU throughput and budget pilot for the AlphaZero Connect Four
example (PILOT.md has the pre-registered plan and criteria).

Measures, each separately: network inference (``infer``), self-play
(``selfplay``), training steps (``train``) and evaluation against the
ADR-0042 bot with its repeatability check (``eval``); ``all`` runs the
four in that order. Every run writes ``pilot-<host>-<time>.json`` into
``--out`` with the provenance, every parameter, the raw samples and the
derived rates, and enters itself in the core-hour ledger
(``--ledger``). A cell that does not fit the remaining budget is skipped
and reported as missing, never extrapolated.

Usage (from the repository root, the ``test`` and ``torch`` groups
installed)::

    python examples/alphazero_connect4/pilot.py all --out DIR \\
        --ledger DIR/ledger.json --cores 16 --budget-core-hours 24 \\
        --wall-cap-s 5400 --source-sha SHA --bot PATH/tessera-rl-connect4-ref
"""

from __future__ import annotations

import os

# Library thread pools are pinned before torch is imported (ADR-0043,
# P2-M3: one thread, deterministic algorithms). Inference cells that use
# more threads set them explicitly with torch.set_num_threads.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import platform
import socket
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import augment  # noqa: E402
import game  # noqa: E402
import net as netmod  # noqa: E402
from ledger import BudgetExceeded, Ledger  # noqa: E402

import tessera_rl  # noqa: E402
from tessera_rl.search import Search  # noqa: E402

TIE_BASE = 0x7E55E2A000000000
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EVAL_OPENINGS = os.path.join(REPO, "compat", "connect4", "openings-eval.txt")
HELDOUT_OPENINGS = os.path.join(REPO, "compat", "connect4", "openings-heldout.txt")


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


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "tessera_rl": tessera_rl.version(),
        },
        "threads": {
            "torch_intra_op": torch.get_num_threads(),
            "torch_inter_op": torch.get_num_interop_threads(),
            "env": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "machine": machine(),
        "openings": {"eval": sha256_file(EVAL_OPENINGS), "heldout": sha256_file(HELDOUT_OPENINGS)},
    }


def repeat_cell(ledger: Ledger, name: str, warmup: int, repeats: int, fn) -> dict:
    """Runs ``fn`` ``warmup`` times (discarded) then ``repeats`` times,
    each only if the last duration fits the remaining budget. Returns the
    samples and, if the budget stopped it, why."""
    samples: list[float] = []
    last = 0.0
    try:
        for i in range(warmup + repeats):
            ledger.check(last, f"{name} #{i}")
            start = time.perf_counter()
            fn()
            last = time.perf_counter() - start
            if i >= warmup:
                samples.append(last)
        stopped = None
    except BudgetExceeded as e:
        stopped = str(e)
    return {"warmup": warmup, "repeats": repeats, "samples_s": samples, "stopped": stopped}


# ---------------------------------------------------------------- infer


def run_infer(args, ledger: Ledger) -> dict:
    cells = []
    for size in args.sizes:
        model = netmod.build(size, seed=args.seed)
        for threads in sorted({1, args.cores}):
            torch.set_num_threads(threads)
            for batch in args.infer_batches:
                x = torch.zeros((batch, 2, 6, 7))
                legal = np.ones((batch, 7), dtype=bool)

                def forward():
                    with torch.inference_mode():
                        for _ in range(args.infer_inner):
                            logits, _ = model(x)
                            netmod.masked_priors(logits, legal)

                cell = repeat_cell(ledger, f"infer {size} t{threads} b{batch}", args.warmup, args.repeats, forward)
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


def _selfplay_worker(task: dict) -> dict:
    configure_torch(1)
    if task["evaluator"] == "uniform":
        evaluator = netmod.Uniform()
    else:
        evaluator = netmod.Evaluator(netmod.build(task["size"], seed=task["net_seed"]))
    search = Search(
        simulations=task["sims"],
        batch_size=task["batch_size"],
        capacity=task["capacity"],
        noise_alpha=1.0,
        noise_epsilon=0.25,
        seed=task["seed"],
    )
    games = []
    deadline = task["deadline"]
    for g in range(task["games"]):
        if time.time() > deadline:
            break
        r = game.self_play(search, evaluator, seed=task["seed"] + g, temperature_moves=task["temperature_moves"])
        games.append(
            {
                "wall_s": r.wall_ns / 1e9,
                "moves": r.timing.moves,
                "search_s": r.timing.search_ns / 1e9,
                "net_s": r.timing.net_ns / 1e9,
                "batches": np.bincount(r.timing.batches, minlength=task["batch_size"] + 1).tolist(),
            }
        )
    return {"games": games}


def run_selfplay(args, ledger: Ledger) -> dict:
    cells = []
    ctx = mp.get_context("spawn")
    evaluators = [("uniform", None)] + [("net", s) for s in args.sizes]
    for kind, size in evaluators:
        for sims in args.sims:
            for workers in sorted({1, min(8, args.cores), args.cores}):
                name = f"selfplay {kind}:{size} sims{sims} W{workers}"
                # The projection is the last cell's wall time at least.
                projected = args.selfplay_cell_s
                try:
                    ledger.check(projected, name)
                except BudgetExceeded as e:
                    cells.append({"evaluator": kind, "size": size, "sims": sims, "workers": workers, "stopped": str(e)})
                    continue
                tasks = [
                    {
                        "evaluator": kind,
                        "size": size,
                        "net_seed": args.seed,
                        "sims": sims,
                        "batch_size": args.batch_size,
                        "capacity": args.capacity,
                        "seed": args.seed * 1000 + w * 100,
                        "games": args.games_per_worker,
                        "temperature_moves": args.temperature_moves,
                        "deadline": time.time() + args.selfplay_cell_s,
                    }
                    for w in range(workers)
                ]
                start = time.perf_counter()
                with ctx.Pool(workers) as pool:
                    startup = time.perf_counter() - start
                    results = pool.map(_selfplay_worker, tasks)
                wall = time.perf_counter() - start
                games = [g for r in results for g in r["games"]]
                batches = np.sum([g["batches"] for g in games], axis=0).tolist() if games else []
                search = sum(g["search_s"] for g in games)
                netsec = sum(g["net_s"] for g in games)
                per_move = [g["search_s"] / g["moves"] for g in games if g["moves"]]
                cells.append(
                    {
                        "evaluator": kind,
                        "size": size,
                        "sims": sims,
                        "workers": workers,
                        "games": len(games),
                        "wall_s": wall,
                        "pool_startup_s": startup,
                        "game_wall_s": quantiles([g["wall_s"] for g in games]),
                        "moves_per_game": quantiles([g["moves"] for g in games]),
                        "search_per_move_s": quantiles(per_move),
                        "net_share_of_search": netsec / search if search else None,
                        "non_net_share_of_search": 1 - netsec / search if search else None,
                        "leaf_batch_histogram": batches,
                        "games_per_hour": len(games) / wall * 3600 if wall else None,
                        "games_per_hour_per_allocated_core": len(games) / wall * 3600 / workers if wall else None,
                        "raw_games": games,
                        "stopped": None,
                    }
                )
    return {"cells": cells}


# ---------------------------------------------------------------- train


def run_train(args, ledger: Ledger) -> dict:
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

            def step():
                logits, values = model(x)
                loss = -(tp * torch.log_softmax(logits, dim=1)).sum(dim=1).mean() + ((values - tv) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            cell = repeat_cell(ledger, f"train {size} b{batch}", args.warmup, args.repeats, step)
            cells.append(
                {"size": size, "batch": 2 * half, "device": "cpu", "threads": 1, **cell, "step_s": quantiles(cell["samples_s"])}
            )
    return {"cells": cells, "gpu": "not measured (no Windows session approved)"}


# ---------------------------------------------------------------- eval


def _eval_pass(ckpt: bytes, size: str, sims: int, openings: list[str], bot_path: str, args) -> dict:
    """One evaluation pass from the serialized checkpoint with fresh nets,
    searches and bot process."""
    model = netmod.Net(*netmod.SIZES[size])
    model.load_state_dict(torch.load(io.BytesIO(ckpt), weights_only=True))
    model.eval()
    evaluator = netmod.Evaluator(model)
    bot = game.BotClient(bot_path)
    games = []
    try:
        for j, opening in enumerate(openings):
            for first in (True, False):
                # No root noise (no noise options), greedy moves, a fixed seed.
                search = Search(simulations=sims, batch_size=args.batch_size, capacity=args.capacity, seed=args.seed)
                start = time.perf_counter_ns()
                g = game.versus_bot(search, evaluator, bot, j, opening, first, TIE_BASE + j, seed=args.seed)
                games.append({"g": g, "wall_ns": time.perf_counter_ns() - start})
    finally:
        bot.close()
    log = [[x["g"].opening, x["g"].candidate_first, x["g"].result, x["g"].log] for x in games]
    return {
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


def run_eval(args, ledger: Ledger) -> dict:
    """Timing and repeatability only: a frozen random-init checkpoint on
    the fixed ADR-0042 protocol. Not a P2-G1 result, not a selection
    input."""
    configure_torch(1)
    model = netmod.build(args.eval_size, seed=args.seed)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    ckpt = buf.getvalue()
    digest = hashlib.sha256(ckpt).hexdigest()
    sets = {
        "heldout": game.read_openings(HELDOUT_OPENINGS)[: args.eval_openings],
        "eval": game.read_openings(EVAL_OPENINGS)[: args.eval_openings],
    }
    out = {"checkpoint_sha256": digest, "size": args.eval_size, "sims": args.eval_sims, "passes": {}}
    for name, openings in sets.items():
        passes = []
        for rep in range(2):
            label = f"eval {name} pass {rep}"
            try:
                ledger.check(args.eval_cell_s, label)
            except BudgetExceeded as e:
                passes.append({"stopped": str(e)})
                continue
            start = time.perf_counter()
            r = _eval_pass(ckpt, args.eval_size, args.eval_sims, openings, args.bot, args)
            r["wall_s"] = time.perf_counter() - start
            passes.append(r)
        hashes = [p.get("log_sha256") for p in passes]
        out["passes"][name] = {
            "openings": len(openings),
            "runs": passes,
            "repeatable": None if None in hashes else hashes[0] == hashes[1],
        }
    return out


# ---------------------------------------------------------------- report


def starvation(results: dict) -> dict:
    """The pre-registered criterion of PILOT.md."""
    verdict: dict = {}
    infer = results.get("infer", {}).get("cells", [])

    def per_sample(size, batch):
        for c in infer:
            if c["size"] == size and c["threads"] == 1 and c["batch"] == batch and c["per_sample_s"].get("n"):
                return c["per_sample_s"]["median"]
        return None

    a, b = per_sample("small", 8), per_sample("small", 64)
    verdict["infer_ratio_b8_over_b64"] = a / b if a and b else None
    share = None
    for c in results.get("selfplay", {}).get("cells", []):
        if c.get("evaluator") == "net" and c.get("size") == "small" and c.get("sims") == 100 and c.get("workers") == 1:
            share = c.get("non_net_share_of_search")
    verdict["non_net_share_small_100"] = share
    r1 = verdict["infer_ratio_b8_over_b64"]
    fired = [r1 is not None and r1 > 2.0, share is not None and share > 0.30]
    verdict["criterion_1_fired"], verdict["criterion_2_fired"] = fired
    verdict["complete"] = r1 is not None and share is not None
    verdict["starved"] = any(fired) if verdict["complete"] else None
    return verdict


def main(argv=None) -> int:
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
    p.add_argument("--games-per-worker", type=int, default=8)
    p.add_argument("--temperature-moves", type=int, default=8)
    p.add_argument("--selfplay-cell-s", type=float, default=240)
    p.add_argument("--eval-size", default="small", choices=list(netmod.SIZES))
    p.add_argument("--eval-sims", type=int, default=100)
    p.add_argument("--eval-openings", type=int, default=500)
    p.add_argument("--eval-cell-s", type=float, default=900)
    p.add_argument("--seed", type=int, default=20261102)
    args = p.parse_args(argv)
    if args.command in ("eval", "all") and not args.bot:
        p.error("eval needs --bot")

    configure_torch(1)
    os.makedirs(args.out, exist_ok=True)
    ledger = Ledger(args.ledger, args.budget_core_hours, args.wall_cap_s, args.cores)
    host = socket.gethostname()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report: dict = {"command": args.command, "params": vars(args).copy(), "provenance": provenance(args)}
    report["config_sha256"] = canonical_hash(report["params"])
    order = ["infer", "selfplay", "train", "eval"] if args.command == "all" else [args.command]
    runners = {"infer": run_infer, "selfplay": run_selfplay, "train": run_train, "eval": run_eval}
    status = "ok"
    try:
        for name in order:
            report[name] = runners[name](args, ledger)
    except BaseException as e:
        status = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        report["starvation"] = starvation(report)
        report["ledger"] = ledger.close(f"pilot {args.command} {stamp}", host, status)
        path = os.path.join(args.out, f"pilot-{host}-{stamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
