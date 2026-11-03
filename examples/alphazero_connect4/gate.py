# SPDX-License-Identifier: MIT OR Apache-2.0
"""The P2-G1 gate (P2-M3 PR 4, ADR-0042), supervised and one shot.

The gate plays the selected checkpoint, and only that one (never the final
checkpoint), against the ADR-0042 bot on the 500 pinned evaluation
openings, both colours: 1,000 games. It uses the look's protocol through
the same loop (``botmatch.play``): 100 simulations, batch 8, capacity
8192, greedy, no noise, search seed 20261102, and tie seeds
``TIE_BASE + j``, where ``j`` is the opening's index in the evaluation
file.

**Pass rule** (ADR-0042 section 5). Both must hold:
- at least 900 wins out of 1,000;
- the two-sided Wilson lower bound at z = 1.96 is at least 0.90.

Draws are non-wins. A complete, valid, formal gate records ``pass`` as true
or false. ``pass`` is null only for an incomplete, inapplicable or
diagnostic run. The interval is nominal: the games come in pairs per
opening and may repeat.

**Formal flow.** Everything happens under the loop's owner lock, with the
gate's paths fixed under the loop root: ``<root>/gate/`` holds the claim,
the gate ledger and the run output.

1. Preflight, with nothing of the evaluation set touched yet:
   - the committed selection manifest's digest;
   - formal, and a selected checkpoint;
   - an owner-locked recomputation from the verified, settled evidence
     that equals the manifest;
   - the selected checkpoint file, a medium net;
   - the expected and registered bot binary;
   - the registered runtime;
   - the frozen search;
   - a budget that funds the full wall cap.
2. The claim: an atomic ``mkdir <root>/gate`` and a durable
   ``claim.json``, plus a ``gate-claim`` entry in the loop's audit log. A
   second gate is refused whatever ``--out`` or ledger it names, because
   the directory exists. An interrupted or evaluated attempt stays
   consumed, and a retry needs the owner's decision.
3. The supervised run reserves the budget, then revalidates the claim at
   use (checkpoint, bot, runtime). Only then does it hash, compare and
   parse the evaluation file. Nothing is written before every game is
   played, and the child's report is only provisional.
4. The parent finalizes ``gate.json``. It carries a verdict only with exit
   0, an ok ledger settlement and a complete provisional report; any other
   attempt stays consumed, with ``pass`` null and the reason.

**Diagnostic mode** (``--any-openings --openings FIXTURE``, tests and dry
runs). It never names a selection and has no verdict. It refuses the
evaluation file and any alias of it (the same device and inode, through a
symlink or a hard link) before opening or hashing it, and then refuses a
byte-identical copy.

Usage (formal)::

    python examples/alphazero_connect4/gate.py --root LOOP_ROOT --selection-sha256 SHA \\
        --bot BOT --bot-sha256 SHA --source-sha SHA --budget-core-hours 1 --wall-cap-s 2400
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

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

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
# The evaluation set. Only ``formal_openings`` and ``eval_sha_now`` read
# it; ``diagnostic_openings`` only stats it, to refuse aliases.
EVAL = os.path.join(REPO, "compat", "connect4", "openings-eval.txt")
EVAL_SHA256 = "54a1cea7e6fe70d53a039bc27176486953c8bb6a243353175128526702360475"
# The frozen protocol: (sims, batch size, capacity, openings).
FROZEN = (100, 8, 8192, 500)
GAMES = 1000
WINS_MIN = 900
LB_MIN = 0.90
Z = 1.96  # two-sided 95%; the one-sided 1.645 reading is excluded (ADR-0042)


class Refused(SystemExit):
    """A gate precondition does not hold; nothing of the gate set was read."""


def passes(wins: int, games: int) -> dict:
    """The ADR-0042 section 5 rule. ``pass`` is true or false on exactly
    1,000 games, and null otherwise."""
    lb = h2h.wilson_lower(wins / games, games, Z) if games else 0.0
    return {
        "wins": wins,
        "games": games,
        "wilson_lower_z1.96": lb,
        "wins_at_least_900": wins >= WINS_MIN,
        "lower_bound_at_least_0.90": lb >= LB_MIN,
        # A short or overlong gate is invalid: no verdict, never false.
        "pass": (wins >= WINS_MIN and lb >= LB_MIN) if games == GAMES else None,
        "invalid_because": None if games == GAMES else f"{games} games, not {GAMES}",
    }


def gate_dir(root: str) -> str:
    return os.path.join(root, "gate")


def configure_torch() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------- formal


def snapshot(lp, manifest_path: str, manifest_sha: str) -> tuple[dict, str]:
    """The committed selection, checked against a recomputation from the
    loop's verified evidence; returns it and the selected checkpoint's
    path. The caller holds the owner lock."""
    if artifacts.sha256_file(manifest_path) != manifest_sha:
        raise Refused("the selection manifest is not the committed one")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("kind") != "p2-m3-selection" or manifest.get("formal") is not True:
        raise Refused("the selection is not formal")
    selected = manifest.get("selected")
    if not selected:
        raise Refused("the selection names no checkpoint: there is no gate")
    if not loopmod.is_formal(lp.cfg):
        raise Refused("the loop root is not a formal run")
    entries = lp.ledger.entries()
    try:
        gens = lp.chain(entries)
        eligible = [e for e in lp.looks(gens, entries) if e["eligible"]]
        if eligible != manifest.get("eligible_looks") or loopmod.choose(eligible) != selected:
            raise Refused("the selection does not match the loop's verified evidence")
        path = lp.checkpoint_path(gens, selected["generation"])
    except loopmod.Stop as s:
        raise Refused(f"broken loop history: {s.reason} {json.dumps(s.inputs, sort_keys=True)}") from None
    return manifest, path


def check_candidate(path: str, expected_sha: str, bot: str, bot_sha256: str, args) -> tuple[dict, dict]:
    """The checkpoint, bot, runtime and search checks, run at preflight
    and again at use; returns the checkpoint's metadata and the runtime."""
    _, meta = ckpt.load(path)
    if meta["sha256"] != expected_sha:
        raise Refused(f"the checkpoint is {meta['sha256']}, not the selected {expected_sha}")
    if meta["size"] != "medium":
        raise Refused(f"the gate candidate is a {meta['size']} net, not medium")
    bot_sha = artifacts.sha256_file(bot)
    if bot_sha != bot_sha256:
        raise Refused(f"the bot binary is {bot_sha}, not the expected {bot_sha256}")
    observed = h2h.runtime()
    if look.REGISTERED_BOTS.get(observed["image_digest"]) != bot_sha:
        raise Refused("the bot binary is not the registered one for this image")
    if not observed["registered"]:
        raise Refused("not the registered runtime")
    if (args.sims, args.batch_size, args.capacity) != FROZEN[:3]:
        raise Refused("not the frozen search protocol")
    return meta, observed


def preflight(args, lp) -> dict:
    """Every formal check; the caller holds the owner lock. Returns the
    claim to record."""
    manifest, path = snapshot(lp, os.path.join(args.root, "selection-manifest.json"), args.selection_sha256)
    meta, observed = check_candidate(path, manifest["selected"]["checkpoint"], args.bot, args.bot_sha256, args)
    check_resources(args)
    return {
        "kind": "p2-g1-gate-claim",
        "selection_sha256": args.selection_sha256,
        "selected": manifest["selected"],
        "checkpoint_path": os.path.relpath(path, args.root),
        "checkpoint_sha256": meta["sha256"],
        "bot_sha256": args.bot_sha256,
        "image_digest": observed["image_digest"],
        "source_sha": args.source_sha,
        "claimed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def check_resources(args) -> None:
    """A usable allowance before the claim: one core, a finite positive
    budget that funds the whole wall cap, and a cap above the supervisor's
    cleanup grace."""
    budget, cap = args.budget_core_hours, args.wall_cap_s
    if args.cores != 1:
        raise Refused("the registered gate runs on one core")
    if not (math.isfinite(budget) and budget > 0 and math.isfinite(cap) and cap > supervisor.CLEANUP_GRACE_S):
        raise Refused("the gate budget and wall cap must be finite and positive, above the cleanup grace")
    if budget * 3600 < args.cores * cap:
        raise Refused("the gate budget does not fund the full wall cap")


def fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def claim(args, lp, record: dict) -> None:
    """The durable, irreversible claim: an atomic mkdir, its parent entry
    synced, then the record and the audit entry, all before any read of
    the evaluation set."""
    try:
        os.mkdir(gate_dir(args.root))
    except FileExistsError:
        raise Refused("a gate was already claimed on this loop root; a retry needs the owner's decision") from None
    fsync_dir(args.root)
    artifacts.write_atomic(os.path.join(gate_dir(args.root), "claim.json"), artifacts.canonical_json(record))
    fsync_dir(gate_dir(args.root))
    lp.append_selection({"type": "gate-claim", **record})
    fsync_dir(args.root)


def formal_openings() -> list[str]:
    """The only reader of the evaluation set: hash, compare, then parse."""
    if artifacts.sha256_file(EVAL) != EVAL_SHA256:
        raise Refused("the evaluation openings are not the pinned ones")
    openings = game.read_openings(EVAL)
    if len(openings) != FROZEN[3]:
        raise Refused("the evaluation openings are not the pinned ones")
    return openings


def distinct_games(openings: list[str], logs: list) -> int:
    """Distinct complete move sequences: the opening's columns, then the
    played columns. Roles, results and visit counts are not identity."""
    seqs = {tuple([int(c) - 1 for c in openings[j]] + [entry[1] for entry in log]) for j, _, _, log in logs}
    return len(seqs)


def eval_sha_now() -> str:
    """The evaluation set's digest after the games (formal only)."""
    return artifacts.sha256_file(EVAL)


# ------------------------------------------------------------ diagnostic


def diagnostic_openings(path: str, limit: int) -> list[str]:
    """A fixture that is not the evaluation set: aliases are refused by
    device and inode before the file is opened, copies by digest."""
    real = os.path.realpath(path)
    target = os.stat(real)
    try:
        ev = os.stat(EVAL)
    except FileNotFoundError:
        ev = None
    if real == os.path.realpath(EVAL) or (ev is not None and (target.st_dev, target.st_ino) == (ev.st_dev, ev.st_ino)):
        raise Refused("a diagnostic run never plays the evaluation openings")
    if artifacts.sha256_file(real) == EVAL_SHA256:
        raise Refused("a diagnostic run never plays the evaluation openings")
    return game.read_openings(real)[:limit]


# ------------------------------------------------------------------ work


def work(args, clock) -> int:
    configure_torch()
    formal = not args.any_openings
    if formal:
        with open(os.path.join(gate_dir(args.root), "claim.json"), encoding="utf-8") as f:
            record = json.load(f)
        if record["selection_sha256"] != args.selection_sha256:
            raise Refused("the claim is for another selection")
        path = os.path.join(args.root, record["checkpoint_path"])
        meta, observed = check_candidate(path, record["checkpoint_sha256"], args.bot, args.bot_sha256, args)
        openings = formal_openings()
        openings_sha = EVAL_SHA256
    else:
        record, path = None, args.checkpoint
        _, meta = ckpt.load(path)
        if artifacts.sha256_file(args.bot) != args.bot_sha256:
            raise Refused("the bot binary is not the expected one")
        observed = h2h.runtime()
        openings = diagnostic_openings(args.openings, args.limit)
        openings_sha = artifacts.sha256_file(os.path.realpath(args.openings))
    model, _ = ckpt.load(path)
    bot = game.BotClient(args.bot)
    try:
        played = botmatch.play(netmod.Evaluator(model), bot, openings, args.sims, args.batch_size, args.capacity)
    finally:
        bot.close()
    results, logs, pairs = played["results"], played["logs"], played["pairs"]
    wins = sum(1 for r in results if r > 0)
    draws = sum(1 for r in results if r == 0)
    rule = passes(wins, len(results))
    complete = formal and len(results) == GAMES
    protocol = {
        "checkpoint": meta["sha256"],
        "openings_sha256": openings_sha,
        "sims": args.sims,
        "batch_size": args.batch_size,
        "capacity": args.capacity,
        "openings": len(openings),
        "search_seed": botmatch.SEARCH_SEED,
        "tie_base": botmatch.TIE_BASE,
        "bot_sha256": args.bot_sha256,
        "size": meta["size"],
        "selection_sha256": args.selection_sha256 if formal else None,
    }
    log_bytes = artifacts.canonical_json(logs)
    report = {
        "kind": "p2-g1-gate",
        "gate_run": formal,
        "checkpoint": meta["sha256"],
        "selected": record["selected"] if record else None,
        "games": len(results),
        "wdl": [wins, draws, len(results) - wins - draws],
        "by_colour": played["by_colour"],
        "pair_wins": {str(k): pairs.count(k) for k in (2, 1, 0)},
        "distinct_games": distinct_games(openings, logs),
        "openings_covered": len(openings),
        "rule": rule,
        "pass": rule["pass"] if complete else None,
        "interval_note": "nominal two-sided Wilson z=1.96, as if the games were independent Bernoulli trials; "
        "they come in pairs per opening and may repeat, so it is a stricter descriptive criterion, "
        "not calibrated coverage",
        "log_sha256": artifacts.sha256(log_bytes),
        # Recomputed at the end: the set played is the set pinned.
        "openings_sha256_at_end": eval_sha_now() if formal else openings_sha,
        "runtime": observed,
        "protocol": protocol,
        "command": ["gate.py", *supervisor.strip_child_flag(sys.argv[1:])],
        "note": "the exit code and the spend are in this gate's ledger entry",
        "provenance": provenance.block(args.source_sha, protocol),
    }
    if formal and report["openings_sha256_at_end"] != EVAL_SHA256:
        report["pass"] = None
    # Nothing is written before every game is played, and this is only
    # provisional: the parent finalizes it after the ok settlement.
    artifacts.write_atomic(os.path.join(args.out, "gate-log.json"), log_bytes)
    report["provisional_pass"] = report.pop("pass")
    artifacts.write_atomic(os.path.join(args.out, PROVISIONAL), artifacts.canonical_json(report))
    return 0


PROVISIONAL = "gate.provisional.json"


def settlement(ledger: str, budget: float, name: str) -> dict | None:
    from ledger import Ledger

    if not os.path.exists(ledger):
        return None
    mine = [e for e in Ledger(ledger, budget).entries() if e["run"].split()[0] == name]
    return mine[-1] if mine else None


def finalize(args, rc: int, name: str) -> dict:
    """The gate's result, tied to the supervisor's exit and the actual
    ledger settlement: a verdict only with exit 0, an ok entry and a
    complete provisional report; otherwise pass null with the reason."""
    entry = settlement(args.ledger, args.budget_core_hours, name)
    path = os.path.join(args.out, PROVISIONAL)
    report = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
    why = []
    if rc != 0:
        why.append(f"the supervised run exited {rc}")
    if entry is None or entry["status"] != "ok":
        why.append(f"the ledger entry is {entry['status'] if entry else 'missing'}, not ok")
    if report is None:
        why.append("no complete provisional report")
    final = {
        **(report or {"kind": "p2-g1-gate", "gate_run": not args.any_openings}),
        "pass": None if why or report is None else report["provisional_pass"],
        "attempt_invalid_because": why,
        "settlement": entry,
        "exit_code": rc,
        "complete": True,
    }
    final.pop("provisional_pass", None)
    artifacts.write_atomic(os.path.join(args.out, "gate.json"), artifacts.canonical_json(final))
    print(json.dumps({"gate_run": final["gate_run"], "games": final.get("games"), "pass": final["pass"]}))
    return final


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", help="formal: the formal loop root; the gate's paths are fixed under it")
    p.add_argument("--selection-sha256", help="formal: the committed selection-manifest.json digest")
    p.add_argument("--any-openings", action="store_true", help="diagnostic: never a gate")
    p.add_argument("--openings", help="diagnostic: the fixture openings (required)")
    p.add_argument("--checkpoint", help="diagnostic: the checkpoint (formal: the selected one, from the root)")
    p.add_argument("--out", help="diagnostic: the output directory (formal: fixed)")
    p.add_argument("--ledger", help="diagnostic: the ledger (formal: fixed)")
    p.add_argument("--limit", type=int, default=500, help="diagnostic only")
    p.add_argument("--bot", required=True)
    p.add_argument("--bot-sha256", required=True, help="the expected binary digest, checked before the bot starts")
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--capacity", type=int, default=8192)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--cores", type=int, default=1)
    p.add_argument("--budget-core-hours", type=float, required=True)
    p.add_argument("--wall-cap-s", type=float, required=True)
    args = p.parse_args(supervisor.strip_child_flag(argv))
    if args.any_openings:
        if not (args.openings and args.checkpoint and args.out and args.ledger):
            p.error("a diagnostic run needs --openings, --checkpoint, --out and --ledger")
        if args.root or args.selection_sha256:
            p.error("a diagnostic run names no selection")
    else:
        if not (args.root and args.selection_sha256):
            p.error("the formal gate needs --root and --selection-sha256")
        if args.openings or args.checkpoint or args.out or args.ledger:
            p.error("the formal gate takes no --openings, --checkpoint, --out or --ledger: they are fixed")
        args.out = os.path.join(gate_dir(args.root), "run")
        args.ledger = os.path.join(gate_dir(args.root), "ledger.json")
    return args


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse(argv)

    name = "gate" if not args.any_openings else "diagnostic-gate"

    def supervised() -> int:
        rc = supervisor.run(
            __file__, argv, lambda clock: work(args, clock), ledger=args.ledger,
            budget_core_hours=args.budget_core_hours, cores=args.cores, wall_cap_s=args.wall_cap_s,
            out=args.out, name=name,
        )  # fmt: skip
        if supervisor.child_flag(argv) is None:
            finalize(args, rc, name)
        return rc

    if args.any_openings or supervisor.child_flag(argv) is not None:
        return supervised()
    # Formal parent: preflight, claim and the whole supervised run under
    # the owner lock, so the loop root cannot change underneath.
    configure_torch()
    lp = loopmod.Loop(args.root)
    lock = lp.owner()
    try:
        record = preflight(args, lp)
        claim(args, lp, record)
        return supervised()
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
