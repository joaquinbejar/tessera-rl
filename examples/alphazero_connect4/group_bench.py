# SPDX-License-Identifier: MIT OR Apache-2.0
"""The ADR-0045 §5 measurement: SearchGroup against independent single
trees for self-play, supervised.

All cells share the settings below. Self-play is as in PR 2: root noise
alpha 1.0 and eps 0.25, and temperature 1 for the first 8 plies.

| Setting | Value |
| --- | --- |
| Run | one run, one cpuset, one source and runtime |
| Checkpoint | the frozen ``small`` checkpoint, by SHA-256 |
| Search | 100 simulations, batch 8, capacity 8192 per tree |
| Threads | one torch thread per worker |

Game ``g`` of a cell is always the same logical game: its environment seed
is ``seeds.env(run_seed, g)``, its search seed ``seeds.search(run_seed,
g)`` and its slot state fresh. A group plays ``T`` games at once. Each
round builds a new ``SearchGroup`` of ``T`` slots, plays its games until all
end (a finished slot goes inactive), and runs ``games_per_slot`` such
batches. Membership never reseeds a game.

The cells (pre-registered in ADR-0045 §5) are:
- baseline single-tree at ``W`` in ``{1, 8, allocated}``;
- the group with ``T`` in ``{4, 8, 16}`` at ``W = allocated``;
- the diagnostic group at ``W = 1``, ``T = 8``.

Every cell runs 2 warmup rounds and 7 kept rounds with 2 games per slot
per round.

The verdict reads only the named cells. **Adopted** if the median steady
games per hour per allocated core of the group at ``T = 8`` and
``W = allocated`` is at least 1.2x the single-tree ``W = allocated``
median. The ``W = 1`` ratio is a diagnostic only. A missing named cell
makes the verdict ``incomplete``.

Companion evidence, per cell:
- moves per game and per hour;
- simulations per hour;
- evaluated leaves per hour;
- the fixed-root matched-work cell (the same roots searched for the same
  simulations, single against group), which separates the scheduler's
  effect from game-length changes.

Usage::

    python examples/alphazero_connect4/group_bench.py --checkpoint SMALL.pt --out DIR \\
        --source-sha SHA --ledger L.json --cores 32 --budget-core-hours 48 --wall-cap-s 5300
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import artifacts  # noqa: E402
import ckpt  # noqa: E402
import game  # noqa: E402
import h2h  # noqa: E402
import net as netmod  # noqa: E402
import provenance  # noqa: E402
import seeds  # noqa: E402
import supervisor  # noqa: E402
from ledger import BudgetExceeded  # noqa: E402
from tessera_rl.search import Search, SearchGroup  # noqa: E402

NOISE = {"noise_alpha": 1.0, "noise_epsilon": 0.25}
TEMPERATURE_PLIES = 8
ADOPTION_RATIO = 1.2


def configure_torch() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


class Counting(netmod.Evaluator):
    """The network evaluator, also counting evaluated leaves."""

    def __init__(self, model):
        super().__init__(model)
        self.leaves = 0

    def __call__(self, obs, legal, hashes):
        self.leaves += len(hashes)
        return super().__call__(obs, legal, hashes)


_W: dict = {}


def _init(path: str, cfg: dict) -> None:
    configure_torch()
    model, _ = ckpt.load(path)
    _W.update(evaluator=Counting(model), cfg=cfg)


def single_games(gids: list[int]) -> list[dict]:
    """Games one after another, each with a fresh single-tree search."""
    cfg, ev = _W["cfg"], _W["evaluator"]
    out = []
    for g in gids:
        search = Search(simulations=cfg["sims"], batch_size=cfg["batch_size"], capacity=cfg["capacity"],
                        seed=seeds.search(cfg["run_seed"], g), **NOISE)  # fmt: skip
        board = game.Board(seeds.env(cfg["run_seed"], g))
        timing = game.Timing()
        leaves0 = ev.leaves
        try:
            while not board.over:
                t = 1.0 if len(board.moves) < TEMPERATURE_PLIES else 0.0
                col, _ = game.search_move(search, board, ev, timing, t)
                board.play(col)
            out.append({"g": g, "moves": len(board.moves), "leaves": ev.leaves - leaves0})
        finally:
            board.close()
    return out


def group_games(gids: list[int]) -> list[dict]:
    """``len(gids)`` games at once in a fresh SearchGroup, one slot each."""
    cfg, ev = _W["cfg"], _W["evaluator"]
    t_ = len(gids)
    group = SearchGroup(
        t_, simulations=cfg["sims"], batch_size=cfg["batch_size"], capacity=cfg["capacity"],
        slot_options=[{"seed": seeds.search(cfg["run_seed"], g)} for g in gids], **NOISE,
    )  # fmt: skip
    boards = [game.Board(seeds.env(cfg["run_seed"], g)) for g in gids]
    leaves0 = ev.leaves
    try:
        while True:
            live = [i for i in range(t_) if not boards[i].over]
            if not live:
                break
            for i in range(t_):
                group.set_active(i, i in live)
                if i in live:
                    group.set_root(i, boards[i].record())
            with torch.inference_mode():
                group.run(ev)
            for i in live:
                t = 1.0 if len(boards[i].moves) < TEMPERATURE_PLIES else 0.0
                boards[i].play(group.select_action(i, t))
        moves = [len(b.moves) for b in boards]
    finally:
        for b in boards:
            b.close()
    # Leaves are shared by the group's batches: attributed evenly.
    leaves = ev.leaves - leaves0
    return [{"g": g, "moves": m, "leaves": leaves / t_} for g, m in zip(gids, moves, strict=True)]


def _round(task: tuple[str, list[list[int]]]) -> list[dict]:
    kind, batches = task
    out = []
    for gids in batches:
        out += single_games(gids) if kind == "single" else group_games(gids)
    return out


def roots(n: int, run_seed: int) -> list[bytes]:
    """Fixed roots for matched work: 6 seeded random legal plies each."""
    out = []
    rng = np.random.Generator(np.random.PCG64(seeds.env(run_seed, 10**9)))
    for r in range(n):
        board = game.Board(seeds.env(run_seed, 10**9 + 1 + r))
        try:
            for _ in range(6):
                legal = np.flatnonzero(board.step.masks[0, board.mover])
                board.play(int(rng.choice(legal)))
            out.append(board.record())
        finally:
            board.close()
    return out


def _matched(task: tuple[str, int, list[bytes]]) -> dict:
    """Searches every root once for the fixed simulations; the time."""
    kind, t_, recs = task
    cfg, ev = _W["cfg"], _W["evaluator"]
    start = time.perf_counter()
    leaves0 = ev.leaves
    with torch.inference_mode():
        if kind == "single":
            for i, rec in enumerate(recs):
                s = Search(simulations=cfg["sims"], batch_size=cfg["batch_size"], capacity=cfg["capacity"], seed=i)
                s.set_root(rec)
                s.run(ev)
        else:
            for at in range(0, len(recs), t_):
                chunk = recs[at : at + t_]
                g = SearchGroup(len(chunk), simulations=cfg["sims"], batch_size=cfg["batch_size"], capacity=cfg["capacity"])
                for i, rec in enumerate(chunk):
                    g.set_root(i, rec)
                g.run(ev)
    return {"wall_s": time.perf_counter() - start, "roots": len(recs), "leaves": ev.leaves - leaves0}


def quant(xs):
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {"n": len(s), "median": statistics.median(s), "min": s[0], "max": s[-1]}


def cell(args, clock, kind: str, workers: int, t_: int) -> dict:
    name = f"{kind} W{workers} T{t_}"
    info = {"kind": kind, "workers": workers, "trees": t_, "allocated_cores": args.cores}
    try:
        clock.check(args.cell_allowance_s, name)
    except BudgetExceeded as e:
        return {**info, "stopped": str(e)}
    cfg = {"run_seed": args.run_seed, "sims": args.sims, "batch_size": args.batch_size, "capacity": args.capacity}
    per_round = workers * t_ * args.games_per_slot
    rounds, next_g = [], 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(args.checkpoint, cfg)) as pool:
        last = args.cell_allowance_s
        for r in range(args.warmup + args.repeats):
            try:
                clock.check(last, name)
            except BudgetExceeded as e:
                info["stopped"] = str(e)
                break
            # Game ids of this round, dealt g mod W, in batches of T.
            gids = list(range(next_g, next_g + per_round))
            next_g += per_round
            tasks = []
            for w in range(workers):
                mine = gids[w::workers]
                tasks.append((kind, [mine[i : i + t_] for i in range(0, len(mine), t_)]))
            t0 = time.perf_counter()
            results = pool.map(_round, tasks, chunksize=1)
            wall = time.perf_counter() - t0
            last = wall
            games = [g for res in results for g in res]
            if r >= args.warmup:
                rounds.append({"wall_s": wall, "games": len(games), "moves": sum(g["moves"] for g in games),
                               "leaves": sum(g["leaves"] for g in games)})  # fmt: skip
    rate = [x["games"] / x["wall_s"] * 3600 / args.cores for x in rounds]
    return {
        **info,
        "rounds_kept": len(rounds),
        "games_per_hour_per_allocated_core": quant(rate),
        "moves_per_game": quant([x["moves"] / x["games"] for x in rounds]),
        "moves_per_hour": quant([x["moves"] / x["wall_s"] * 3600 for x in rounds]),
        "simulations_per_hour": quant([x["moves"] * args.sims / x["wall_s"] * 3600 for x in rounds]),
        "leaves_per_hour": quant([x["leaves"] / x["wall_s"] * 3600 for x in rounds]),
        "raw_rounds": rounds,
        "stopped": info.get("stopped"),
    }


def matched(args, clock, t_: int) -> dict:
    recs = roots(args.matched_roots, args.run_seed)
    cfg = {"run_seed": args.run_seed, "sims": args.sims, "batch_size": args.batch_size, "capacity": args.capacity}
    out = {"roots": len(recs), "trees": t_, "workers": args.cores}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.cores, initializer=_init, initargs=(args.checkpoint, cfg)) as pool:
        for kind in ("single", "group"):
            try:
                clock.check(args.cell_allowance_s, f"matched {kind}")
            except BudgetExceeded as e:
                out[kind] = {"stopped": str(e)}
                continue
            t0 = time.perf_counter()
            res = pool.map(_matched, [(kind, t_, recs)] * args.cores, chunksize=1)
            wall = time.perf_counter() - t0
            out[kind] = {
                "wall_s": wall,
                "root_searches_per_hour_per_allocated_core": len(res) * len(recs) / wall * 3600 / args.cores,
                "leaves": sum(x["leaves"] for x in res),
            }
    if "wall_s" in out.get("single", {}) and "wall_s" in out.get("group", {}):
        out["speedup"] = out["group"]["root_searches_per_hour_per_allocated_core"] / out["single"][
            "root_searches_per_hour_per_allocated_core"
        ]
    return out


def verdict(cells: list[dict], cores: int, repeats: int) -> dict:
    """The pre-registered verdict from the named cells. A cell counts only
    if it kept all its rounds."""

    def median(kind, w, t_):
        for c in cells:
            if (c["kind"], c["workers"], c["trees"]) == (kind, w, t_) and c.get("rounds_kept") == repeats:
                return c["games_per_hour_per_allocated_core"].get("median")
        return None

    base, group = median("single", cores, 1), median("group", cores, 8)
    d_base, d_group = median("single", 1, 1), median("group", 1, 8)
    complete = None not in (base, group, d_base, d_group)
    ratio = group / base if base and group else None
    return {
        "criterion": f"group T=8 W={cores} median >= {ADOPTION_RATIO}x single-tree W={cores} median, "
        "games per hour per allocated core",
        "ratio": ratio,
        "diagnostic_ratio_w1": d_group / d_base if d_base and d_group else None,
        "complete": complete,
        "adopted": (ratio >= ADOPTION_RATIO) if complete else "incomplete",
    }


def work(args, clock) -> int:
    configure_torch()
    _, cmeta = ckpt.load(args.checkpoint)
    if cmeta["size"] != "small" and not args.any_size:
        raise SystemExit("the §5 measurement uses the small net")
    plan = [("single", w, 1) for w in sorted({1, min(8, args.cores), args.cores})]
    plan += [("group", args.cores, t_) for t_ in args.trees] + [("group", 1, 8)]
    cells = [cell(args, clock, kind, w, t_) for kind, w, t_ in plan]
    mw = matched(args, clock, 8)
    v = verdict(cells, args.cores, args.repeats)
    memory = {str(t_): SearchGroup(t_, capacity=args.capacity, batch_size=args.batch_size).memory_bytes for t_ in args.trees}
    config = {k: v for k, v in vars(args).items() if k not in ("out", "ledger", "child_deadline_epoch")}
    report = {
        "kind": "adr-0045-s5",
        "checkpoint": cmeta["sha256"],
        "cells": cells,
        "matched_work": mw,
        "memory_bytes_per_worker": memory,
        "verdict": v,
        "runtime": h2h.runtime(),
        "note": "games/h is end-to-end evidence; the matched-work cell is the fixed-work comparison",
        "provenance": provenance.block(args.source_sha, config),
    }
    artifacts.write_atomic(os.path.join(args.out, "group-bench.json"), artifacts.canonical_json(report))
    print(f"ratio {v['ratio']} adopted {v['adopted']}")
    return 0


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--run-seed", type=int, default=45)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--capacity", type=int, default=8192)
    p.add_argument("--trees", nargs="+", type=int, default=[4, 8, 16])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--games-per-slot", type=int, default=2)
    p.add_argument("--matched-roots", type=int, default=64)
    p.add_argument("--cell-allowance-s", type=float, default=120)
    p.add_argument("--any-size", action="store_true", help="tests only")
    p.add_argument("--attempt", default="group-bench")
    p.add_argument("--ledger", required=True)
    p.add_argument("--cores", type=int, required=True)
    p.add_argument("--budget-core-hours", type=float, required=True)
    p.add_argument("--wall-cap-s", type=float, required=True)
    return p.parse_args(supervisor.strip_child_flag(argv))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse(argv)
    return supervisor.run(
        __file__, argv, lambda clock: work(args, clock), ledger=args.ledger,
        budget_core_hours=args.budget_core_hours, cores=args.cores, wall_cap_s=args.wall_cap_s,
        out=args.out, name=args.attempt,
    )  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
