# SPDX-License-Identifier: MIT OR Apache-2.0
"""A held-out look of the P2-M3 loop (PR 3), supervised.

A look plays one checkpoint against the ADR-0042 bot on the 200 held-out
openings, both colours: 400 games. The protocol is frozen and identical
to the PR 4 gate candidate's search:
- a single-tree ``Search`` at 100 simulations, batch 8, capacity 8192;
- greedy moves, no noise, search seed 20261102, and a fresh search per
  game;
- tie seeds ``TIE_BASE + j`` for the bot;
- the registered runtime (``h2h.runtime``).

Wins are wins; draws are non-wins. The report holds:
- W/D/L, the colour splits and the pair counts;
- the canonical game log and its digest;
- the bot binary digest;
- the runtime block, and whether the protocol and runtime are the
  registered ones (``eligible``). Only an eligible look may enter
  selection.

Held-out results are for checkpoint selection only. Up to 30 looks at one
set make the selected score optimistically biased, so it is never an
estimate of gate performance.

Usage::

    python examples/alphazero_connect4/look.py --checkpoint C.pt --bot BOT --out DIR \\
        --source-sha SHA --ledger L.json --budget-core-hours 200 --wall-cap-s 900
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import artifacts  # noqa: E402
import botmatch  # noqa: E402
import ckpt  # noqa: E402
import game  # noqa: E402
import h2h  # noqa: E402
import net as netmod  # noqa: E402
import provenance  # noqa: E402
import supervisor  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
HELDOUT = os.path.join(REPO, "compat", "connect4", "openings-heldout.txt")
HELDOUT_SHA256 = "3276d2d044c911ed29c36f2d42690697f19a4bac0183e3a77b1af3c5af72cc49"
TIE_BASE = botmatch.TIE_BASE
SEARCH_SEED = botmatch.SEARCH_SEED
# The frozen protocol: (sims, batch size, capacity, openings).
FROZEN = (100, 8, 8192, 200)
# The registered bot binaries, per registered image digest: the
# tools/connect4-ref build of the run's source in that image, whose bot.rs,
# board.rs and rng.rs are the ADR-0042 pinned files. An entry is added by
# a reviewed pre-run commit after the build; a look with any other binary
# is never eligible.
REGISTERED_BOTS: dict[str, str] = {}


def is_eligible(observed: dict, bot_sha: str, size: str, openings_sha: str, args, openings: int) -> bool:
    """Whether a look may enter selection: the registered runtime and bot
    binary for that image, a medium net, and the frozen protocol."""
    return bool(
        observed["registered"]
        and REGISTERED_BOTS.get(observed["image_digest"]) == bot_sha
        and size == "medium"
        and not args.any_openings
        and openings_sha == HELDOUT_SHA256
        and (args.sims, args.batch_size, args.capacity, openings) == FROZEN
    )


def work(args, clock) -> int:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    if artifacts.sha256_file(args.openings) != HELDOUT_SHA256 and not args.any_openings:
        raise SystemExit("a look is defined on the pinned held-out set only")
    bot_sha = artifacts.sha256_file(args.bot)
    if bot_sha != args.bot_sha256:
        raise SystemExit(f"the bot binary is {bot_sha}, not the expected {args.bot_sha256}")
    model, meta = ckpt.load(args.checkpoint)
    evaluator = netmod.Evaluator(model)
    openings = game.read_openings(args.openings)[: args.limit]
    bot = game.BotClient(args.bot)
    try:
        played = botmatch.play(evaluator, bot, openings, args.sims, args.batch_size, args.capacity)
    finally:
        bot.close()
    results, logs, pairs, by_colour = (played[k] for k in ("results", "logs", "pairs", "by_colour"))
    observed = h2h.runtime()
    protocol = {
        "checkpoint": meta["sha256"],
        "openings_sha256": artifacts.sha256_file(args.openings),
        "sims": args.sims,
        "batch_size": args.batch_size,
        "capacity": args.capacity,
        "openings": len(openings),
        "search_seed": SEARCH_SEED,
        "tie_base": TIE_BASE,
        "bot_sha256": bot_sha,
        "size": meta["size"],
    }
    eligible = is_eligible(observed, bot_sha, meta["size"], protocol["openings_sha256"], args, len(openings))
    log_bytes = artifacts.canonical_json(logs)
    wins = sum(1 for r in results if r > 0)
    draws = sum(1 for r in results if r == 0)
    report = {
        "kind": "p2-m3-look",
        "checkpoint": meta["sha256"],
        "generation": meta.get("generation"),
        "games": len(results),
        "wins": wins,
        "wdl": [wins, draws, len(results) - wins - draws],
        "by_colour": by_colour,
        "pair_wins": {str(k): pairs.count(k) for k in (2, 1, 0)},
        "log_sha256": artifacts.sha256(log_bytes),
        "runtime": observed,
        "protocol": protocol,
        "eligible": eligible,
        "note": "held-out selection only; biased by repeated looks; never a gate estimate",
        "provenance": provenance.block(args.source_sha, protocol),
    }
    artifacts.write_atomic(os.path.join(args.out, "look-log.json"), log_bytes)
    artifacts.write_atomic(os.path.join(args.out, "look.json"), artifacts.canonical_json({**report, "complete": True}))
    print(f"wins {wins}/{len(results)} eligible {eligible}")
    return 0


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bot", required=True)
    p.add_argument("--bot-sha256", required=True, help="the expected binary digest, checked before the bot starts")
    p.add_argument("--out", required=True)
    p.add_argument("--openings", default=HELDOUT)
    p.add_argument("--any-openings", action="store_true", help="tests only: never eligible")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--capacity", type=int, default=8192)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--attempt", default="look")
    p.add_argument("--ledger", required=True)
    p.add_argument("--cores", type=int, default=1)
    p.add_argument("--budget-core-hours", type=float, required=True)
    p.add_argument("--wall-cap-s", type=float, required=True)
    return p.parse_args(supervisor.strip_child_flag(argv))


def load(directory: str) -> dict:
    """A complete look report; raises ``artifacts.Incomplete``."""
    path = os.path.join(directory, "look.json")
    if not os.path.exists(path):
        raise artifacts.Incomplete(f"{directory}: no look report")
    import json

    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    if report.get("complete") is not True:
        raise artifacts.Incomplete(f"{directory}: look not complete")
    with open(os.path.join(directory, "look-log.json"), "rb") as f:
        if artifacts.sha256(f.read()) != report["log_sha256"]:
            raise artifacts.Incomplete(f"{directory}: look log does not match")
    return report


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
