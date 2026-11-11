# SPDX-License-Identifier: MIT OR Apache-2.0
"""Self-play data generation of the P2-M3 baseline (PR 2), supervised.

Usage (from the repository root)::

    python examples/alphazero_connect4/selfplay.py --checkpoint C0.pt \\
        --games 10000 --workers 32 --run-seed 1 --out DIR \\
        --ledger LEDGER.json --cores 32 --budget-core-hours 12 --wall-cap-s 1200

The run has ``W`` worker processes, each with 1 torch thread and the frozen
checkpoint (checked by SHA-256) in eval mode. Game ``g`` goes to worker
``g mod W``.

Every game is played from a fresh environment reset with
``seeds.env(run_seed, g)`` and searched by a fresh ``Search`` seeded with
``seeds.search(run_seed, g)``, so no tree, counter or stream carries over.
The root-noise and temperature draws come from the search stream of the
game's root record. A game is therefore a function of the checkpoint,
``run_seed``, ``g`` and the configuration: never of ``W``, the worker or
the timing.

The search runs at 100 simulations and batch 8. The root has Dirichlet
noise (alpha 1.0, eps 0.25). Moves are drawn at temperature 1 for the
first 8 plies, then greedy.

Each worker round writes one shard (``artifacts``) holding, per move:
- the mover's planes;
- the root visits;
- the mover-relative outcome ``z``;
- the game id and the ply.

The manifest is written last. It lists every shard with its SHA-256 and
every game's digest (sorted by ``g``, independent of ``W``), plus the
checkpoint digest, the seed derivation, the configuration, the move
counts and the distinct 8-ply prefixes. A run without it is incomplete.
"""

from __future__ import annotations

import os

import orphan

# A supervised child checks its supervisor before the heavy imports
# (orphan.py); a spawned pool worker re-imports this file as __mp_main__,
# with its parent's argv, and must not.
if __name__ == "__main__":
    orphan.check_argv()

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import artifacts  # noqa: E402
import ckpt  # noqa: E402
import game  # noqa: E402
import net as netmod  # noqa: E402
import provenance  # noqa: E402
import seeds  # noqa: E402
import supervisor  # noqa: E402
from tessera_rl.search import Search  # noqa: E402

NOISE_ALPHA = 1.0
NOISE_EPSILON = 0.25
TEMPERATURE_PLIES = 8
PREFIX_PLIES = 8


def configure_torch() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def play(choose, env_seed: int) -> dict[str, np.ndarray]:
    """One game: ``choose(board) -> (column, root visits)`` at each move.
    Returns the samples with mover-relative outcomes."""
    board = game.Board(env_seed)
    planes, visits, movers = [], [], []
    try:
        while not board.over:
            planes.append(board.planes())
            movers.append(board.mover)
            col, v = choose(board)
            visits.append(v)
            board.play(col)
        rewards = (board.reward(0), board.reward(1))
        moves = list(board.moves)
    finally:
        board.close()
    return {
        "planes": np.stack(planes).astype(np.uint8),
        "visits": np.asarray(visits, dtype=np.uint32),
        "z": np.asarray([rewards[m] for m in movers], dtype=np.int8),
        "moves": np.asarray(moves, dtype=np.uint8),
    }


def search_chooser(evaluator, cfg: dict, g: int):
    """The self-play move rule of game ``g``: a fresh search per game."""
    search = Search(
        simulations=cfg["sims"],
        batch_size=cfg["batch_size"],
        capacity=cfg["capacity"],
        noise_alpha=NOISE_ALPHA,
        noise_epsilon=NOISE_EPSILON,
        seed=seeds.search(cfg["run_seed"], g),
    )
    timing = game.Timing()

    def choose(board):
        temperature = 1.0 if len(board.moves) < TEMPERATURE_PLIES else 0.0
        return game.search_move(search, board, evaluator, timing, temperature)

    return choose


def game_digest(g: int, sample: dict[str, np.ndarray]) -> str:
    return artifacts.sha256(artifacts.encode_shard(sample, {"g": g}))


_WORKER: dict = {}


def _init(path: str, cfg: dict) -> None:
    configure_torch()
    model, _ = ckpt.load(path)
    _WORKER.update(evaluator=netmod.Evaluator(model), cfg=cfg)


def _play_round(gs: list[int]) -> list[tuple[int, dict]]:
    cfg, ev = _WORKER["cfg"], _WORKER["evaluator"]
    out = []
    with torch.inference_mode():
        for g in gs:
            out.append((g, play(search_chooser(ev, cfg, g), seeds.env(cfg["run_seed"], g))))
    return out


def tasks(games: int, workers: int, round_games: int) -> list[tuple[int, int, list[int]]]:
    """(worker, round, game ids): game g to worker g mod W, in rounds."""
    out = []
    for w in range(workers):
        mine = list(range(w, games, workers))
        for r, i in enumerate(range(0, len(mine), round_games)):
            out.append((w, r, mine[i : i + round_games]))
    return out


def work(args, clock) -> int:
    configure_torch()
    _, cmeta = ckpt.load(args.checkpoint)
    cfg = {
        "run_seed": args.run_seed,
        "sims": args.sims,
        "batch_size": args.batch_size,
        "capacity": args.capacity,
    }
    os.makedirs(args.out, exist_ok=True)
    plan = tasks(args.games, args.workers, args.round_games)
    files, game_digests, moves_per_game, prefixes = [], {}, {}, set()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.checkpoint, cfg)) as pool:
        results = pool.imap(_play_round, [gs for _, _, gs in plan], chunksize=1)
        for (w, r, _), played in zip(plan, results, strict=True):
            arrays = {k: [] for k in ("planes", "visits", "z")}
            gcol, ply = [], []
            for g, sample in played:
                game_digests[g] = game_digest(g, sample)
                n = len(sample["z"])
                moves_per_game[g] = n
                prefixes.add(bytes(sample["moves"][:PREFIX_PLIES]))
                for k in arrays:
                    arrays[k].append(sample[k])
                gcol.append(np.full(n, g, dtype=np.uint32))
                ply.append(np.arange(n, dtype=np.uint16))
            shard = {k: np.concatenate(v) for k, v in arrays.items()}
            shard["g"], shard["ply"] = np.concatenate(gcol), np.concatenate(ply)
            name = f"shard-w{w:03d}-r{r:04d}.bin"
            data = artifacts.encode_shard(shard, {"worker": w, "round": r})
            artifacts.write_atomic(os.path.join(args.out, name), data)
            files.append({"name": name, "sha256": artifacts.sha256(data), "samples": int(len(shard["z"]))})
    ordered = [game_digests[g] for g in range(args.games)]
    lengths = [moves_per_game[g] for g in range(args.games)]
    config = {
        **cfg,
        "games": args.games,
        "noise_alpha": NOISE_ALPHA,
        "noise_epsilon": NOISE_EPSILON,
        "temperature_plies": TEMPERATURE_PLIES,
        "checkpoint_sha256": cmeta["sha256"],
    }
    manifest = {
        "kind": "p2-m3-selfplay",
        # W, the round size and the host are recorded but not in the
        # config hash: the games do not depend on them.
        "provenance": provenance.block(args.source_sha, config),
        "checkpoint_sha256": cmeta["sha256"],
        "checkpoint_size": cmeta["size"],
        "games": args.games,
        "workers": args.workers,
        "round_games": args.round_games,
        "config": {
            **cfg,
            "noise_alpha": NOISE_ALPHA,
            "noise_epsilon": NOISE_EPSILON,
            "temperature_plies": TEMPERATURE_PLIES,
            "eval_mode": True,
            "torch_threads": 1,
        },
        "seed_derivation": seeds.DERIVATION,
        "files": files,
        "samples": int(sum(f["samples"] for f in files)),
        # Independent of W: the games in id order.
        "games_digest": artifacts.sha256(artifacts.canonical_json(ordered)),
        "game_digests": ordered,
        "moves_per_game": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)},
        "distinct_prefixes": {"plies": PREFIX_PLIES, "count": len(prefixes)},
    }
    digest = artifacts.write_manifest(args.out, manifest)
    print(f"manifest {digest}")
    return 0


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--run-seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--capacity", type=int, default=8192)
    p.add_argument("--round-games", type=int, default=32)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--attempt", default="selfplay", help="the ledger run name (the loop's attempt id)")
    p.add_argument("--ledger", required=True)
    p.add_argument("--cores", type=int, required=True)
    p.add_argument("--budget-core-hours", type=float, required=True)
    p.add_argument("--wall-cap-s", type=float, required=True)
    args = p.parse_args(supervisor.strip_child_flag(argv))
    if min(args.games, args.workers, args.sims, args.batch_size, args.capacity, args.round_games) < 1:
        p.error("counts must be at least 1")
    if args.workers > args.cores:
        p.error("--workers exceeds --cores")
    return args


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse(argv)
    return supervisor.run(
        __file__,
        argv,
        lambda clock: work(args, clock),
        ledger=args.ledger,
        budget_core_hours=args.budget_core_hours,
        cores=args.cores,
        wall_cap_s=args.wall_cap_s,
        out=args.out,
        name=args.attempt,
    )


if __name__ == "__main__":
    sys.exit(main())
