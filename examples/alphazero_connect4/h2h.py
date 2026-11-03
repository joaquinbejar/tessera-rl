# SPDX-License-Identifier: MIT OR Apache-2.0
"""The C1-against-C0 head-to-head of the P2-M3 baseline (PR 2), supervised.

Each dev opening is played twice, once with each checkpoint moving next
from the opening position. Both sides use a single-tree ``Search`` at 100
simulations and batch 8, with no root noise, greedy moves and a fixed
search seed, and a fresh search per game.

The score of A is p = (W + 0.5 D) / games. The pre-registered criterion
(design revision 3) is that the one-sided Wilson lower bound of p at
z = 1.645 is above 0.5. This is a NOMINAL, descriptive threshold, not
calibrated evidence: draws are fractional and the two games of an opening
are paired.

The report also gives:
- W/D/L, and the splits by colour;
- each opening's pair score (2, 1.5, 1, 0.5 or 0);
- the distinct move sequences;
- the canonical game log and its hash.

The criterion applies only to the frozen dev set
(``compat/connect4/openings-dev.txt``, pinned by SHA-256).

Usage::

    python examples/alphazero_connect4/h2h.py --a C1.pt --b C0.pt --out DIR \\
        --ledger LEDGER.json --cores 1 --budget-core-hours 12 --wall-cap-s 3000
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402

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

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEV_OPENINGS = os.path.join(REPO, "compat", "connect4", "openings-dev.txt")
DEV_SHA256 = "6208bf759e5adf712040f37f05da9a33c469e57ce077b1e6eb8a3270faffebd2"
Z = 1.645
SEARCH_SEED = 20261102
# The frozen protocol: (sims, batch size, capacity, run seed).
FROZEN = (100, 8, 8192, 1)
# The registered runtime (design revision 3): xaskar, the pinned image,
# one allocated core, one torch thread.
REGISTERED_HOST = "xaskar"
REGISTERED_IMAGE = "rust:1.99.1@sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137"


def cpuset_size(spec: str | None) -> int | None:
    """CPUs in a cpuset string such as ``2`` or ``0-3,6``."""
    if not spec:
        return None
    n = 0
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        n += int(hi or lo) - int(lo) + 1
    return n


def runtime() -> dict:
    """The observed runtime, compared with the registered one."""
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    cpuset = os.environ.get("PILOT_CPUSET")
    observed = {
        "host": __import__("socket").gethostname(),
        "image_digest": os.environ.get("PILOT_IMAGE_DIGEST"),
        "cpuset": cpuset,
        "affinity_cores": affinity,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }
    observed["registered"] = (
        observed["host"] == REGISTERED_HOST
        and observed["image_digest"] == REGISTERED_IMAGE
        and cpuset_size(cpuset) == 1
        and affinity == 1
        and observed["torch_threads"] == 1
        and observed["torch_interop_threads"] == 1
        and observed["deterministic"]
    )
    return observed


def wilson_lower(p: float, n: int, z: float = Z) -> float:
    """The lower end of the Wilson score interval of ``p`` over ``n``."""
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def play_pair_game(searchers, opening: str, a_first: bool, env_seed: int):
    """One game from ``opening``; ``searchers`` is ``{"a": (search,
    evaluator), "b": (...)}``. Returns A's result and the move log."""
    board = game.Board(env_seed)
    log = []
    try:
        for ch in opening:
            board.play(int(ch) - 1)
        a_side = board.mover if a_first else 1 - board.mover
        timing = game.Timing()
        while not board.over:
            who = "a" if board.mover == a_side else "b"
            search, ev = searchers[who]
            col, visits = game.search_move(search, board, ev, timing, 0.0)
            log.append([who, col, visits])
            board.play(col)
        return board.reward(a_side), log
    finally:
        board.close()


def work(args, clock) -> int:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    if not args.any_openings and artifacts.sha256_file(args.openings) != DEV_SHA256:
        raise SystemExit("the criterion is defined on the frozen dev set only")
    ma, meta_a = ckpt.load(args.a)
    mb, meta_b = ckpt.load(args.b)
    if args.require_size != "any" and {meta_a["size"], meta_b["size"]} != {args.require_size}:
        raise SystemExit(f"both checkpoints must be {args.require_size}, not {meta_a['size']} and {meta_b['size']}")
    evals = {"a": netmod.Evaluator(ma), "b": netmod.Evaluator(mb)}
    openings = game.read_openings(args.openings)[: args.limit]
    results, pairs, logs, sequences = [], [], [], set()
    by_colour = {"a_first": [0, 0, 0], "b_first": [0, 0, 0]}
    with torch.inference_mode():
        for j, opening in enumerate(openings):
            pair = 0.0
            for a_first in (True, False):
                searchers = {
                    k: (
                        Search(simulations=args.sims, batch_size=args.batch_size, capacity=args.capacity, seed=SEARCH_SEED),
                        evals[k],
                    )
                    for k in ("a", "b")
                }
                # Greedy, noiseless play does not read the root's stream; the
                # env seed is still derived per game for the record identity.
                env_seed = seeds.env(args.run_seed, 2 * j + (0 if a_first else 1))
                r, log = play_pair_game(searchers, opening, a_first, env_seed=env_seed)
                score = 1.0 if r > 0 else 0.5 if r == 0 else 0.0
                pair += score
                results.append(score)
                key = "a_first" if a_first else "b_first"
                by_colour[key][0 if r > 0 else 1 if r == 0 else 2] += 1
                sequences.add(tuple(c for _, c, _ in log))
                logs.append([j, a_first, score, log])
            pairs.append(pair)
    protocol = {
        "a": meta_a["sha256"],
        "b": meta_b["sha256"],
        "openings_sha256": artifacts.sha256_file(args.openings),
        "limit": args.limit,
        "sims": args.sims,
        "batch_size": args.batch_size,
        "capacity": args.capacity,
        "search_seed": SEARCH_SEED,
        "run_seed": args.run_seed,
    }
    observed = runtime()
    applicable = (
        observed["registered"]
        and not args.any_openings
        and protocol["openings_sha256"] == DEV_SHA256
        and len(results) == 400
        and meta_a["size"] == meta_b["size"] == "medium"
        and (args.sims, args.batch_size, args.capacity, args.run_seed) == FROZEN
    )
    n = len(results)
    w = sum(1 for s in results if s == 1.0)
    d = sum(1 for s in results if s == 0.5)
    p = (w + 0.5 * d) / n if n else 0.0
    lower = wilson_lower(p, n)
    log_bytes = artifacts.canonical_json(logs)
    report = {
        "kind": "p2-m3-h2h",
        "a": meta_a["sha256"],
        "b": meta_b["sha256"],
        "sizes": [meta_a["size"], meta_b["size"]],
        "openings_sha256": artifacts.sha256_file(args.openings),
        "openings": len(openings),
        "games": n,
        "settings": {
            "sims": args.sims,
            "batch_size": args.batch_size,
            "noise": None,
            "temperature": 0.0,
            "search_seed": SEARCH_SEED,
            "env_seed": "seeds.env(run_seed, 2j + colour), colour 0 when A moves first",
            "run_seed": args.run_seed,
        },
        "wdl": [w, d, n - w - d],
        "by_colour": by_colour,
        "score": p,
        "wilson_lower_z1645": lower,
        "criterion": "nominal: one-sided Wilson lower bound of the score > 0.5 at z = 1.645",
        "any_openings": args.any_openings,
        # Only the full pre-registered protocol passes or fails; anything
        # else is a diagnostic run with pass = null.
        "runtime": observed,
        "criterion_applicable": applicable,
        "pass": (lower > 0.5) if applicable else None,
        "provenance": provenance.block(args.source_sha, protocol),
        "pair_scores": {str(k): pairs.count(k) for k in (2.0, 1.5, 1.0, 0.5, 0.0)},
        "distinct_sequences": len(sequences),
        "log_sha256": artifacts.sha256(log_bytes),
    }
    artifacts.write_atomic(os.path.join(args.out, "h2h-log.json"), log_bytes)
    artifacts.write_atomic(os.path.join(args.out, "h2h.json"), artifacts.canonical_json(report))
    print(f"score {p:.4f} lower {lower:.4f} criterion_applicable {applicable} pass {report['pass']}")
    return 0


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--a", required=True, help="the candidate (C1)")
    p.add_argument("--b", required=True, help="the baseline (C0)")
    p.add_argument("--openings", default=DEV_OPENINGS)
    p.add_argument("--any-openings", action="store_true", help="tests only: skip the dev-set pin")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--run-seed", type=int, default=1)
    p.add_argument("--require-size", default="medium", help="the net size both sides must have (\"any\" in tests)")
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--capacity", type=int, default=8192)
    p.add_argument("--out", required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--cores", type=int, default=1)
    p.add_argument("--budget-core-hours", type=float, required=True)
    p.add_argument("--wall-cap-s", type=float, required=True)
    return p.parse_args(supervisor.strip_child_flag(argv))


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
        name="h2h",
    )


if __name__ == "__main__":
    sys.exit(main())
