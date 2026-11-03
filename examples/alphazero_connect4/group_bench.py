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
median. The ``W = 1`` ratio is a diagnostic only.

The ratios are always reported, but the verdict is **formal** only on the
frozen protocol (``FORMAL``), the small net, every declared cell kept in
full, complete matched-work evidence and the observed runtime matching the
declared allocation (host swarmer02, pinned image, cpuset and affinity of ``--cores``,
torch 1/1 threads, deterministic). Otherwise ``adopted`` is null and
``not_formal_because`` says why.

Companion evidence, per cell:
- moves per game and per hour;
- simulations per hour;
- evaluated leaves per hour;
- the fixed-root matched-work cell (the same roots, by SHA-256, searched
  for the same simulations, single against group, both warmed, then
  ``--repeats`` alternating kept rounds with raw samples and medians),
  which separates the scheduler's effect from game-length changes;
- the layout of every round: worker, batch, slot, logical game, its length
  and seeds, and each group's active schedule (ADR-0045 §4).

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
            out.append({"g": g, "moves": len(board.moves), "leaves": ev.leaves - leaves0, **slot_seeds(cfg, g)})
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
    active = []
    try:
        while True:
            live = [i for i in range(t_) if not boards[i].over]
            if not live:
                break
            active.append(len(live))
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
    games = [{"g": g, "slot": i, "moves": m, "leaves": leaves / t_, **slot_seeds(cfg, g)}
             for i, (g, m) in enumerate(zip(gids, moves, strict=True))]  # fmt: skip
    # The active schedule: live slots per group call. Slot i is live at call
    # j exactly while j < its game's moves, so it is reconstructible.
    games[0]["active_schedule"] = active
    return games


def slot_seeds(cfg: dict, g: int) -> dict:
    """A logical game's seeds: a function of the run seed and ``g`` only,
    never of its worker, batch or slot."""
    return {"env_seed": seeds.env(cfg["run_seed"], g), "search_seed": seeds.search(cfg["run_seed"], g)}


def active_sets(batch: list[dict]) -> list[list[int]]:
    """The live slots of each group call, from the recorded game lengths."""
    longest = max(x["moves"] for x in batch)
    return [[x["slot"] for x in batch if j < x["moves"]] for j in range(longest)]


def _round(task: tuple[str, list[list[int]]]) -> list[list[dict]]:
    """A worker's batches in order, each a list of its games."""
    kind, batches = task
    return [single_games(gids) if kind == "single" else group_games(gids) for gids in batches]


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
    rounds, layout, next_g = [], [], 0
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
            games = [g for res in results for batch in res for g in batch]
            # Logical worker -> batch -> slot -> game, with each game's length
            # and seeds (ADR-0045 §4), warmup rounds included.
            layout.append({"round": r, "warmup": r < args.warmup,
                           "workers": [{"worker": w, "batches": res} for w, res in enumerate(results)]})  # fmt: skip
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
        "layout": layout,
        "stopped": info.get("stopped"),
    }


def matched_rounds(run, timer, warmup: int, repeats: int) -> dict:
    """Warms both modes, then times ``repeats`` kept rounds of each,
    alternating. ``run(kind)`` does one full pass over the fixed roots and
    returns its work. Only kept samples enter the comparison."""
    for _ in range(warmup):
        for kind in ("single", "group"):
            run(kind)
    samples = {"single": [], "group": []}
    work = {"single": [], "group": []}
    for r in range(repeats):
        for kind in ("single", "group") if r % 2 == 0 else ("group", "single"):
            t0 = timer()
            work[kind].append(run(kind))
            samples[kind].append(timer() - t0)
    return {"samples_s": samples, "work": work}


def matched(args, clock, t_: int) -> dict:
    recs = roots(args.matched_roots, args.run_seed)
    cfg = {"run_seed": args.run_seed, "sims": args.sims, "batch_size": args.batch_size, "capacity": args.capacity}
    out = {"roots": len(recs), "roots_sha256": artifacts.sha256(b"".join(recs)), "trees": t_, "workers": args.cores,
           "simulations": args.sims, "warmup": args.matched_warmup, "repeats": args.repeats}  # fmt: skip
    projected = args.cell_allowance_s * 2 * (args.matched_warmup + args.repeats)
    try:
        clock.check(projected, "matched work")
    except BudgetExceeded as e:
        return {**out, "stopped": str(e)}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.cores, initializer=_init, initargs=(args.checkpoint, cfg)) as pool:

        def run(kind):
            res = pool.map(_matched, [(kind, t_, recs)] * args.cores, chunksize=1)
            return {"root_searches": sum(x["roots"] for x in res), "leaves": sum(x["leaves"] for x in res)}

        r = matched_rounds(run, time.perf_counter, args.matched_warmup, args.repeats)
    out.update(r)
    per = {k: args.cores * len(recs) for k in ("single", "group")}
    if any(w["root_searches"] != per[k] for k in per for w in r["work"][k]):
        return {**out, "stopped": "a pass did not search every root"}
    med = {k: statistics.median(r["samples_s"][k]) for k in per}
    out["median_s"] = med
    out["root_searches_per_hour_per_allocated_core"] = {k: per[k] / med[k] * 3600 / args.cores for k in per}
    out["speedup"] = med["single"] / med["group"]
    out["stopped"] = None
    return out


# The pre-registered protocol (ADR-0045 §5). Only a run on exactly this,
# with every cell and the matched-work evidence complete on the declared
# allocation, may adopt or reject; anything else is diagnostic.
FORMAL = {"warmup": 2, "repeats": 7, "games_per_slot": 2, "sims": 100, "batch_size": 8, "capacity": 8192,
          "trees": [4, 8, 16], "matched_warmup": 2, "any_size": False}  # fmt: skip


# ADR-0045 §5 fixes the measurement host. The container runs with
# ``--hostname swarmer02`` so the observed host is the machine's name.
REGISTERED_HOST = "swarmer02"


def declared_cells(cores: int, trees: list[int]) -> list[tuple[str, int, int]]:
    plan = [("single", w, 1) for w in sorted({1, min(8, cores), cores})]
    return plan + [("group", cores, t_) for t_ in trees] + [("group", 1, 8)]


def applicability(config: dict, size: str, cells: list[dict], mw: dict, runtime: dict) -> list[str]:
    """Why this run cannot give a formal verdict; empty when it can."""
    cores = config["cores"]
    why = [f"{k} is {config.get(k)!r}, not {v!r}" for k, v in FORMAL.items() if config.get(k) != v]
    if size != "small":
        why.append(f"the net is {size}, not small")
    got = {(c["kind"], c["workers"], c["trees"]): c for c in cells}
    for key in declared_cells(cores, FORMAL["trees"]):
        c = got.get(key)
        if c is None or c.get("stopped") or c.get("rounds_kept") != FORMAL["repeats"]:
            why.append(f"cell {key} is missing, stopped or short")
    if mw.get("stopped") is not None or "speedup" not in mw:
        why.append("the matched-work evidence is missing or stopped")
    if runtime.get("host") != REGISTERED_HOST:
        why.append(f"the host is {runtime.get('host')!r}, not {REGISTERED_HOST}")
    if runtime.get("image_digest") != h2h.REGISTERED_IMAGE:
        why.append("not the pinned image")
    if h2h.cpuset_size(runtime.get("cpuset")) != cores or runtime.get("affinity_cores") != cores:
        why.append(f"the observed cpuset/affinity is not the declared {cores} cores")
    if (runtime.get("torch_threads"), runtime.get("torch_interop_threads"), runtime.get("deterministic")) != (1, 1, True):
        why.append("torch is not 1/1 threads, deterministic")
    return why


def verdict(cells: list[dict], cores: int, repeats: int, why: list[str] | None = None) -> dict:
    """The ratios from the named cells, and the formal verdict only when
    ``why`` (from ``applicability``) is empty. A cell counts for a ratio
    only if it kept all its rounds."""

    def median(kind, w, t_):
        for c in cells:
            if (c["kind"], c["workers"], c["trees"]) == (kind, w, t_) and c.get("rounds_kept") == repeats:
                return c["games_per_hour_per_allocated_core"].get("median")
        return None

    base, group = median("single", cores, 1), median("group", cores, 8)
    d_base, d_group = median("single", 1, 1), median("group", 1, 8)
    ratio = group / base if base and group else None
    why = ["applicability was not checked"] if why is None else why
    formal = not why and ratio is not None
    return {
        "criterion": f"group T=8 W={cores} median >= {ADOPTION_RATIO}x single-tree W={cores} median, "
        "games per hour per allocated core",
        "ratio": ratio,
        "diagnostic_ratio_w1": d_group / d_base if d_base and d_group else None,
        "formal": formal,
        "not_formal_because": why,
        "adopted": (ratio >= ADOPTION_RATIO) if formal else None,
    }


def work(args, clock) -> int:
    configure_torch()
    _, cmeta = ckpt.load(args.checkpoint)
    if cmeta["size"] != "small" and not args.any_size:
        raise SystemExit("the §5 measurement uses the small net")
    cells = [cell(args, clock, kind, w, t_) for kind, w, t_ in declared_cells(args.cores, args.trees)]
    mw = matched(args, clock, 8)
    runtime = h2h.runtime()
    config = {k: v for k, v in vars(args).items() if k not in ("out", "ledger", "child_deadline_epoch")}
    v = verdict(cells, args.cores, args.repeats, applicability(config, cmeta["size"], cells, mw, runtime))
    memory = {str(t_): SearchGroup(t_, capacity=args.capacity, batch_size=args.batch_size).memory_bytes for t_ in args.trees}
    report = {
        "kind": "adr-0045-s5",
        "checkpoint": cmeta["sha256"],
        "cells": cells,
        "matched_work": mw,
        "memory_bytes_per_worker": memory,
        "verdict": v,
        "runtime": runtime,
        "note": "games/h is end-to-end evidence; the matched-work cell is the fixed-work comparison",
        "provenance": provenance.block(args.source_sha, config),
    }
    artifacts.write_atomic(os.path.join(args.out, "group-bench.json"), artifacts.canonical_json(report))
    print(f"ratio {v['ratio']} formal {v['formal']} adopted {v['adopted']}")
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
    p.add_argument("--matched-warmup", type=int, default=2)
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
