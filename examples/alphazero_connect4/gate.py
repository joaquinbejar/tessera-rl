# SPDX-License-Identifier: MIT OR Apache-2.0
"""The P2-G1 gate (P2-M3 PR 4, ADR-0042), supervised.

The gate plays the selected checkpoint against the ADR-0042 bot on the
500 pinned evaluation openings, both colours: 1,000 games. It uses the
look's protocol, through the same loop (``botmatch.play``):
- 100 simulations, batch 8, capacity 8192;
- greedy, no noise;
- search seed 20261102;
- tie seeds ``TIE_BASE + j``.

**Pass rule** (ADR-0042 section 5). The gate passes only if both hold:
- at least 900 wins out of 1,000;
- the lower bound of the two-sided Wilson interval at z = 1.96 is at least
  0.90.

Draws are non-wins. The interval is nominal: the games come in pairs per
opening and may repeat.

**Before the bot starts or any evaluation opening is read,** the formal
mode refuses unless all of these hold:
- the committed selection manifest matches ``--selection-sha256``;
- the manifest is formal and has a selected checkpoint;
- a fresh recomputation from the loop root's verified, settled evidence
  agrees with it;
- the checkpoint file is the selected one, and it is a medium net;
- the bot binary is the expected and registered one for the image;
- the runtime is the registered one;
- the ledger has no earlier gate entry. The gate is one shot, and a retry
  needs the owner's decision.

The evaluation openings are read only here, only in the formal mode, and
only after those checks. The diagnostic mode (``--diagnostic-openings``,
tests and dry runs) cannot open the evaluation file, has no pass verdict,
and never names a selection.

Usage::

    python examples/alphazero_connect4/gate.py --root LOOP_ROOT --selection-sha256 SHA \\
        --checkpoint C.pt --bot BOT --bot-sha256 SHA --out DIR --source-sha SHA \\
        --ledger GATE_LEDGER.json --budget-core-hours 1 --wall-cap-s 2400
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import artifacts  # noqa: E402
import botmatch  # noqa: E402
import ckpt  # noqa: E402
import game  # noqa: E402
import h2h  # noqa: E402
import look  # noqa: E402
import loop as loopmod  # noqa: E402
import net as netmod  # noqa: E402
import provenance  # noqa: E402
import supervisor  # noqa: E402
from ledger import Ledger  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EVAL = os.path.join(REPO, "compat", "connect4", "openings-eval.txt")
EVAL_SHA256 = "54a1cea7e6fe70d53a039bc27176486953c8bb6a243353175128526702360475"
# The frozen protocol: (sims, batch size, capacity, openings), the look's
# search on the 500 evaluation openings.
FROZEN = (100, 8, 8192, 500)
GAMES = 1000
WINS_MIN = 900
LB_MIN = 0.90
Z = 1.96  # two-sided 95%; the one-sided 1.645 reading is excluded (ADR-0042)


class Refused(SystemExit):
    """A gate precondition does not hold; nothing was played."""


def passes(wins: int, games: int) -> dict:
    """The ADR-0042 section 5 rule on a complete gate."""
    lb = h2h.wilson_lower(wins / games, games, Z) if games else 0.0
    return {
        "wins": wins,
        "games": games,
        "wilson_lower_z1.96": lb,
        "spec_rule": wins >= WINS_MIN,
        "internal_criterion": lb >= LB_MIN,
        "pass": games == GAMES and wins >= WINS_MIN and lb >= LB_MIN,
    }


def check_selection(root: str, manifest_path: str, manifest_sha: str, checkpoint_sha: str) -> dict:
    """The committed selection, rechecked against the loop's evidence."""
    if artifacts.sha256_file(manifest_path) != manifest_sha:
        raise Refused("the selection manifest is not the committed one")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("kind") != "p2-m3-selection" or manifest.get("formal") is not True:
        raise Refused("the selection is not formal")
    selected = manifest.get("selected")
    if not selected:
        raise Refused("the selection names no checkpoint: there is no gate")
    lp = loopmod.Loop(root)
    if not loopmod.is_formal(lp.cfg):
        raise Refused("the loop root is not a formal run")
    lock = lp.owner()
    try:
        entries = lp.ledger.entries()
        try:
            gens = lp.chain(entries)
        except loopmod.Stop as s:
            raise Refused(f"broken loop history: {s.reason}") from None
        eligible = [e for e in lp.looks(gens, entries) if e["eligible"]]
    finally:
        lock.close()
    if eligible != manifest.get("eligible_looks") or loopmod.choose(eligible) != selected:
        raise Refused("the selection does not match the loop's verified evidence")
    if checkpoint_sha != selected["checkpoint"]:
        raise Refused(f"the checkpoint is {checkpoint_sha}, not the selected {selected['checkpoint']}")
    return manifest


def configure_torch() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def openings_for(args) -> tuple[str, list[str]]:
    """The openings to play. Only the formal mode can name the evaluation
    file; a diagnostic set that is the evaluation file is refused before it
    is read."""
    if args.diagnostic_openings is None:
        return EVAL, game.read_openings(EVAL)
    path = args.diagnostic_openings
    if os.path.exists(EVAL) and os.path.samefile(path, EVAL):
        raise Refused("a diagnostic run never plays the evaluation openings")
    if artifacts.sha256_file(path) == EVAL_SHA256:
        raise Refused("a diagnostic run never plays the evaluation openings")
    return path, game.read_openings(path)[: args.limit]


def work(args, clock) -> int:
    configure_torch()
    formal = args.diagnostic_openings is None
    bot_sha = artifacts.sha256_file(args.bot)
    if bot_sha != args.bot_sha256:
        raise Refused(f"the bot binary is {bot_sha}, not the expected {args.bot_sha256}")
    model, meta = ckpt.load(args.checkpoint)
    observed = h2h.runtime()
    manifest = None
    why = []
    if formal:
        manifest = check_selection(args.root, os.path.join(args.root, "selection-manifest.json"),
                                   args.selection_sha256, meta["sha256"])  # fmt: skip
        if meta["size"] != "medium":
            raise Refused(f"the gate candidate is a {meta['size']} net, not medium")
        if look.REGISTERED_BOTS.get(observed["image_digest"]) != bot_sha:
            raise Refused("the bot binary is not the registered one for this image")
        if not observed["registered"]:
            raise Refused("not the registered runtime")
        if (args.sims, args.batch_size, args.capacity) != FROZEN[:3]:
            raise Refused("not the frozen search protocol")
    else:
        why.append("diagnostic run: not the evaluation openings, no selection")
    path, openings = openings_for(args)
    if formal and (artifacts.sha256_file(path) != EVAL_SHA256 or len(openings) != FROZEN[3]):
        raise Refused("the evaluation openings are not the pinned ones")
    bot = game.BotClient(args.bot)
    try:
        played = botmatch.play(netmod.Evaluator(model), bot, openings, args.sims, args.batch_size, args.capacity)
    finally:
        bot.close()
    results, logs, pairs = played["results"], played["logs"], played["pairs"]
    wins = sum(1 for r in results if r > 0)
    draws = sum(1 for r in results if r == 0)
    distinct = len({artifacts.canonical_json([openings[j], log]) for j, _, _, log in logs})
    log_bytes = artifacts.canonical_json(logs)
    protocol = {
        "checkpoint": meta["sha256"],
        "openings_sha256": artifacts.sha256_file(path),
        "sims": args.sims,
        "batch_size": args.batch_size,
        "capacity": args.capacity,
        "openings": len(openings),
        "search_seed": botmatch.SEARCH_SEED,
        "tie_base": botmatch.TIE_BASE,
        "bot_sha256": bot_sha,
        "size": meta["size"],
        "selection_sha256": args.selection_sha256 if formal else None,
    }
    rule = passes(wins, len(results))
    report = {
        "kind": "p2-g1-gate",
        "formal": formal,
        "not_a_gate_because": why,
        "checkpoint": meta["sha256"],
        "selected": manifest["selected"] if manifest else None,
        "games": len(results),
        "wdl": [wins, draws, len(results) - wins - draws],
        "by_colour": played["by_colour"],
        "pair_wins": {str(k): pairs.count(k) for k in (2, 1, 0)},
        "distinct_games": distinct,
        "openings_covered": len(openings),
        "rule": rule,
        "pass": rule["pass"] if formal else None,
        "interval_note": "nominal two-sided Wilson z=1.96, as if the games were independent Bernoulli trials; "
        "they come in pairs per opening and may repeat, so it is a stricter descriptive criterion, "
        "not calibrated coverage",
        "log_sha256": artifacts.sha256(log_bytes),
        "runtime": observed,
        "protocol": protocol,
        "command": ["gate.py", *supervisor.strip_child_flag(sys.argv[1:])],
        "note": "the exit code and the spend are in this gate's ledger entry",
        "provenance": provenance.block(args.source_sha, protocol),
    }
    # Nothing is written or printed before every game is played.
    artifacts.write_atomic(os.path.join(args.out, "gate-log.json"), log_bytes)
    artifacts.write_atomic(os.path.join(args.out, "gate.json"), artifacts.canonical_json({**report, "complete": True}))
    print(json.dumps({"formal": formal, "games": len(results), "wins": wins, "pass": report["pass"]}))
    return 0


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", help="the formal loop root (formal mode)")
    p.add_argument("--selection-sha256", help="the committed selection-manifest.json digest (formal mode)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bot", required=True)
    p.add_argument("--bot-sha256", required=True, help="the expected binary digest, checked before the bot starts")
    p.add_argument("--out", required=True)
    p.add_argument("--diagnostic-openings", help="tests and dry runs: any set but the evaluation one; never a gate")
    p.add_argument("--limit", type=int, default=500, help="diagnostic mode only")
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--capacity", type=int, default=8192)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--cores", type=int, default=1)
    p.add_argument("--budget-core-hours", type=float, required=True)
    p.add_argument("--wall-cap-s", type=float, required=True)
    args = p.parse_args(supervisor.strip_child_flag(argv))
    if args.diagnostic_openings is None and not (args.root and args.selection_sha256):
        p.error("the formal gate needs --root and --selection-sha256")
    if args.diagnostic_openings is not None and (args.root or args.selection_sha256):
        p.error("a diagnostic run names no selection")
    return args


def one_shot(args) -> None:
    """A formal gate runs once per ledger: any earlier gate entry, whatever
    its status, refuses a new one."""
    if args.diagnostic_openings is not None or not os.path.exists(args.ledger):
        return
    earlier = [e for e in Ledger(args.ledger, args.budget_core_hours).entries() if e["run"].split()[0] == "gate"]
    if earlier:
        raise Refused(f"a gate already ran on this ledger ({earlier[0]['run']}: {earlier[0]['status']}); "
                      "a retry needs the owner's decision")  # fmt: skip


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse(argv)
    if supervisor.child_flag(argv) is None:
        one_shot(args)
    return supervisor.run(
        __file__, argv, lambda clock: work(args, clock), ledger=args.ledger,
        budget_core_hours=args.budget_core_hours, cores=args.cores, wall_cap_s=args.wall_cap_s,
        out=args.out, name="gate" if args.diagnostic_openings is None else "diagnostic-gate",
    )  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
