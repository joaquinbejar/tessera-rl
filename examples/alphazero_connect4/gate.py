# SPDX-License-Identifier: MIT OR Apache-2.0
"""The P2-G1 gate (P2-M3 PR 4, ADR-0042, rule of ADR-0048), supervised and one shot.

The gate plays the selected checkpoint, and only that one (never the final
checkpoint), against the ADR-0042 bot on the 500 pinned evaluation
openings, both colours: 1,000 games. It uses the look's protocol through
the same loop (``botmatch.play``): 100 simulations, batch 8, capacity
8192, greedy, no noise, search seed 20261102, and tie seeds
``TIE_BASE + j``, where ``j`` is the opening's index in the evaluation
file.

**Pass rule** (ADR-0048, superseding ADR-0042 section 5): at least 815 wins of
1,000, the frozen solver's published score. Draws are non-wins. No interval
decides it: the nominal Wilson bound is descriptive. A complete, valid, formal
attempt records ``pass`` as true or false. A development run has ``pass`` null
with ``meets_815``, as the ADR leaves closing the gate on it to the owner.
``pass`` is also null for an incomplete, inapplicable or diagnostic run. A
report with no ``rule_id`` is under ADR-0042, and keeps that meaning.

**Eligibility and attempts** (ADR-0048 section 3): the selected look won at least
336 of 400, and ``attempts.py`` records one formal attempt across loop roots.

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
   - a budget that funds the full wall cap;
   - a selected look of at least 336 of 400;
   - for a formal attempt, a pinned solver binary, as the report needs the
     comparison.
2. The claim: the journal's admission, an atomic ``mkdir <root>/gate``
   (``gate-dev-<digest>`` for development) and a durable
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

**Comparison** (``compare``): after a valid settled attempt, the solver plays the
evaluation openings against the bot, and ``run/comparison.json`` reports the
discordant slots. It never touches ``gate.json``, and it stays refused until
``SOLVER_SLOTS_SHA256`` is pinned.

Usage (formal)::

    python examples/alphazero_connect4/gate.py --root LOOP_ROOT --selection-sha256 SHA \\
        --attempt formal --attempts-journal JOURNAL --bot BOT --bot-sha256 SHA \\
        --source-sha SHA --budget-core-hours 1 --wall-cap-s 2400
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import artifacts  # noqa: E402
import attempts  # noqa: E402
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
RULE_ID = "adr-0048-wins-815"
RULE_FAMILY = "p2-g1-adr-0048"
WINS_MIN = 815
# Held-out eligibility: the solver's score there (ADR-0048 section 3).
ELIGIBLE_WINS = 336
LOOK_GAMES = 400
LOOK_PROTOCOL = look.FROZEN
Z = 1.96  # descriptive only
INTERVAL_NOTE = ("nominal two-sided Wilson z=1.96, as if the games were independent Bernoulli trials; "
                 "they come in pairs per opening and may repeat; it does not decide the gate")  # fmt: skip
DEVELOPMENT_CLAIM = "fixed-benchmark development result: says nothing on generalization"
# The solver_slots binary built in the registered image; None until an approved
# build. `compare` and formal preflight refuse until then.
SOLVER_SLOTS_SHA256: str | None = None
SOLVER_SOURCES = {
    "tools/connect4-ref/src/bot.rs": "b21c14c26fc77054951c7aa44c139237e0299dc6aa376fa417ae1765bc9ef646",
    "tools/connect4-ref/src/solver.rs": "2927b3f6e49c8defcd5b2348989131cfe2733033cefb44fdce121f2de1e9d19c",
    "tools/connect4-ref/src/board.rs": "664eb16cbbc6d44ddd58a99cfad0e14af6e7dec6c21976f44a255a28bd7a7040",
    "tools/connect4-ref/src/calibrate.rs": "4d97d0cdeb8c013747c81d6ad67b178cfe0d3fb51e924cc2f0755e98697e08ca",
}


class Refused(SystemExit):
    """A gate precondition does not hold; nothing of the gate set was read."""


def passes(wins: int, games: int, label: str | None = "formal") -> dict:
    """The ADR-0048 rule. ``pass`` is true or false only on exactly 1,000 games
    of a formal attempt; a short or overlong gate is invalid, never false."""
    full = games == GAMES
    met = wins >= WINS_MIN if full else None
    return {
        "rule_id": RULE_ID,
        "wins": wins,
        "games": games,
        "wins_at_least_815": met,
        "pass": met if label == "formal" else None,
        "invalid_because": None if full else f"{games} games, not {GAMES}",
    }


def descriptive(wins: int, games: int) -> dict:
    return {"wilson_lower_z1.96": h2h.wilson_lower(wins / games, games, Z) if games else 0.0, "note": INTERVAL_NOTE}


def gate_dir(root: str, development: str | None = None) -> str:
    """The claim's directory: one formal per root, one per development checkpoint."""
    if development is not None and not re.fullmatch(r"[0-9a-f]{64}", development):
        raise Refused("--development-checkpoint is not a sha256 hex digest")
    return os.path.join(root, f"gate-dev-{development[:16]}" if development else "gate")


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


def check_eligible(wins, games) -> None:
    """ADR-0048 section 3, from a look's verified report."""
    if games != LOOK_GAMES or not (isinstance(wins, int) and 0 <= wins <= games):
        raise Refused(f"the qualifying look has {wins} wins of {games} games, not of {LOOK_GAMES}")
    if wins < ELIGIBLE_WINS:
        raise Refused(f"the selected look won {wins} of {games}, under the {ELIGIBLE_WINS} needed")


def look_evidence(lp, entry: dict) -> dict:
    """The eligibility evidence, from the selected look's verified report."""
    gen, _, n = entry["look"].split("-")
    try:
        report = look.load(lp.attempt_dir(int(gen[1:]), "look", int(n)))
    except artifacts.Incomplete as e:
        raise Refused(f"the selected look's report: {e}") from None
    for field, value in (("checkpoint", entry["checkpoint"]), ("wins", entry["wins"]), ("log_sha256", entry["log_sha256"])):
        if report[field] != value:
            raise Refused(f"the selected look's {field} is not the manifest's")
    protocol = report["protocol"]
    if (protocol["sims"], protocol["batch_size"], protocol["capacity"], protocol["openings"]) != LOOK_PROTOCOL:
        raise Refused("the selected look is not on the frozen protocol")
    check_eligible(report["wins"], report["games"])
    return {"look": entry["look"], "wins": report["wins"], "games": report["games"], "log_sha256": report["log_sha256"]}


def preflight(args, lp) -> dict:
    """Every formal check; the caller holds the owner lock. Returns the
    claim to record."""
    manifest, path = snapshot(lp, os.path.join(args.root, "selection-manifest.json"), args.selection_sha256)
    selected = manifest["selected"]
    if args.attempt == "development" and args.development_checkpoint != selected["checkpoint"]:
        raise Refused("the development checkpoint is not the one this selection names")
    meta, observed = check_candidate(path, selected["checkpoint"], args.bot, args.bot_sha256, args)
    check_resources(args)
    entry = next(e for e in manifest["eligible_looks"] if e["checkpoint"] == selected["checkpoint"]
                 and e["generation"] == selected["generation"])  # fmt: skip
    evidence = look_evidence(lp, entry)
    if args.attempt == "formal" and SOLVER_SLOTS_SHA256 is None:
        raise Refused("no solver binary is pinned: the comparison ADR-0048 requires would refuse")
    return {
        "kind": "p2-g1-gate-claim",
        "rule_id": RULE_ID,
        "look_evidence": evidence,
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
    """The durable, irreversible claim: the journal's admission, an atomic
    mkdir, its parent entry synced, then the record and the audit entry, all
    before any read of the evaluation set. A crash after the journal entry
    spends the attempt: nothing could prove the file stayed unread."""
    if os.path.exists(args.gate_dir):
        raise Refused("a gate was already claimed on this loop root; a retry needs the owner's decision")
    block = attempts.claim(args.attempts_journal, EVAL_SHA256, RULE_FAMILY, args.create_journal, args.attempt,
                           args.root, record["checkpoint_sha256"])  # fmt: skip
    record["attempt"] = block
    os.mkdir(args.gate_dir)
    fsync_dir(args.root)
    artifacts.write_atomic(os.path.join(args.gate_dir, "claim.json"), artifacts.canonical_json(record))
    fsync_dir(args.gate_dir)
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
        with open(os.path.join(args.gate_dir, "claim.json"), encoding="utf-8") as f:
            record = json.load(f)
        if record["selection_sha256"] != args.selection_sha256:
            raise Refused("the claim is for another selection")
        path = os.path.join(args.root, record["checkpoint_path"])
        meta, observed = check_candidate(path, record["checkpoint_sha256"], args.bot, args.bot_sha256, args)
        evidence = record["look_evidence"]
        again = look_evidence(loopmod.Loop(args.root), {"look": evidence["look"], "checkpoint": record["checkpoint_sha256"],
                                                        "wins": evidence["wins"], "log_sha256": evidence["log_sha256"]})  # fmt: skip
        if again != evidence:
            raise Refused("the look evidence is not the claim's")
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
    label = record["attempt"]["label"] if record else None
    rule = passes(wins, len(results), label)
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
        "rule_id": RULE_ID,
        "rule": rule,
        "pass": rule["pass"],
        "meets_815": rule["wins_at_least_815"] if formal else None,
        "descriptive": descriptive(wins, len(results)),
        "attempt": record["attempt"] if record else None,
        "look_evidence": record["look_evidence"] if record else None,
        "log_sha256": artifacts.sha256(log_bytes),
        # Recomputed at the end: the set played is the set pinned.
        "openings_sha256_at_end": eval_sha_now() if formal else openings_sha,
        "runtime": observed,
        "protocol": protocol,
        "command": ["gate.py", *supervisor.strip_child_flag(sys.argv[1:])],
        "note": "the exit code and the spend are in this gate's ledger entry",
        "provenance": provenance.block(args.source_sha, protocol),
    }
    if label == "development":
        report["claim"] = DEVELOPMENT_CLAIM
    if formal and report["openings_sha256_at_end"] != EVAL_SHA256:
        report["pass"] = report["meets_815"] = rule["pass"] = rule["wins_at_least_815"] = None
        rule["invalid_because"] = "the evaluation file changed during the games"
    # Nothing is written before every game is played, and this is only
    # provisional: the parent finalizes it after the ok settlement.
    artifacts.write_atomic(os.path.join(args.out, "gate-log.json"), log_bytes)
    report["provisional_pass"] = report.pop("pass")
    artifacts.write_atomic(os.path.join(args.out, PROVISIONAL), artifacts.canonical_json(report))
    return 0


PROVISIONAL = "gate.provisional.json"
SPAWN = "spawn.json"
QUIESCE_WAIT_S = 2.0


def settlement(ledger: str, name: str) -> dict | None:
    """The latest ``name`` entry, read with a plain load: finalizing never
    rewrites the ledger (``Ledger.entries`` would)."""
    if not os.path.exists(ledger):
        return None
    with open(ledger, encoding="utf-8") as f:
        mine = [e for e in json.load(f)["entries"] if e["run"].split()[0] == name]
    return mine[-1] if mine else None


def finalize(args, rc: int, name: str) -> dict:
    """The gate's result, tied to the supervisor's exit and the actual
    ledger settlement: a verdict only with exit 0, an ok entry and a
    complete provisional report; otherwise pass null with the reason."""
    entry = settlement(args.ledger, name)
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
    if not args.any_openings and not quiescent(args.gate_dir, entry):
        if not why:
            # Nothing is wrong yet but the group: no result is written, and
            # `gate.py finalize` rebuilds the verdict once it answers ESRCH.
            final = {"kind": "p2-g1-gate", "pass": None, "interrupted": True,
                     "attempt_invalid_because": ["the process group is not confirmed gone"]}  # fmt: skip
            print(json.dumps(final))
            return final
        why.append("the process group is not confirmed gone")
    final = {
        **(report or {"kind": "p2-g1-gate", "gate_run": not args.any_openings}),
        "pass": None if why or report is None else report["provisional_pass"],
        "attempt_invalid_because": why,
        "settlement": entry,
        "exit_code": rc,
        "complete": True,
    }
    final.pop("provisional_pass", None)
    if why:
        # An invalid attempt has no score to read, at any level.
        if "meets_815" in final:
            final["meets_815"] = None
        # The counts stay only as diagnostics of a run with no result.
        final["provisional_diagnostics"] = {k: final.pop(k) for k in ("wdl", "by_colour", "pair_wins", "distinct_games")
                                            if k in final}  # fmt: skip
        rule = final.get("rule")
        if rule:
            rule["pass"] = None
            if "wins_at_least_815" in rule:
                final["provisional_diagnostics"]["rule_wins"] = rule["wins"]
                rule["wins"] = rule["wins_at_least_815"] = None
            rule["invalid_because"] = "; ".join(why)
    artifacts.write_atomic(os.path.join(args.out, "gate.json"), artifacts.canonical_json(final))
    print(json.dumps({"gate_run": final["gate_run"], "games": final.get("games"), "pass": final["pass"]}))
    if not args.any_openings:
        settle_journal(args.gate_dir, entry, final)
    return final


def quiescent(directory: str, entry: dict | None) -> bool:
    """Whether the supervised run has certainly stopped. A recorded group must
    answer ESRCH, whatever the ledger says, as the supervisor's kill does not
    wait for it; a short poll lets a killed group finish. Unknown is false, and
    a reused group id can only keep it false. With no record no child was ever
    started, as the record is written first; only a running or unconfirmed entry
    then stays unknown."""
    path = os.path.join(directory, SPAWN)
    if not os.path.exists(path):
        return entry is None or (entry["status"] != "running" and supervisor.NOT_CONFIRMED not in entry["status"])
    with open(path, encoding="utf-8") as f:
        spawn = json.load(f)
    if spawn["pgid"] is None or spawn["host"] != socket.gethostname():
        return False
    end = time.monotonic() + QUIESCE_WAIT_S
    while True:
        try:
            os.killpg(spawn["pgid"], 0)
        except ProcessLookupError:
            return True
        except OSError:
            pass
        if time.monotonic() >= end:
            return False
        time.sleep(0.05)


def settle_journal(directory: str, entry: dict | None, final: dict) -> bool:
    """The attempt's terminal journal event, once the run has certainly
    stopped; returns whether it could be written. An attempt with no journal
    block predates ADR-0048 and has nothing to settle."""
    claim_path = os.path.join(directory, "claim.json")
    if not os.path.exists(claim_path):
        return True
    with open(claim_path, encoding="utf-8") as f:
        block = json.load(f).get("attempt")
    if block is None:
        return True
    if not quiescent(directory, entry):
        return False
    met = final.get("meets_815") if not final.get("attempt_invalid_because") else None
    disposition = {True: "met-815", False: "below-815", None: "consumed-no-verdict"}[met]
    ledger = os.path.join(directory, "ledger.json")
    report = os.path.join(directory, "run", "gate.json")
    digest = lambda p: artifacts.sha256_file(p) if os.path.exists(p) else None  # noqa: E731
    attempts.settle(block, disposition, {"claim_sha256": digest(claim_path), "report_sha256": digest(report),
                                         "ledger_sha256": digest(ledger)})  # fmt: skip
    return True


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", help="formal: the formal loop root; the gate's paths are fixed under it")
    p.add_argument("--selection-sha256", help="formal: the committed selection-manifest.json digest")
    p.add_argument("--attempt", choices=attempts.LABELS, help="formal or development: the attempt's label (required)")
    p.add_argument("--attempts-journal", help="formal or development: the benchmark's journal (required)")
    p.add_argument("--create-journal", action="store_true", help="a new journal: a visible departure, recorded")
    p.add_argument("--development-checkpoint", help="development: the digest this selection names (required)")
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
        if args.root or args.selection_sha256 or args.attempt or args.attempts_journal or args.create_journal:
            p.error("a diagnostic run names no selection and no attempt")
    else:
        if not (args.root and args.selection_sha256):
            p.error("the formal gate needs --root and --selection-sha256")
        if args.openings or args.checkpoint or args.out or args.ledger:
            p.error("the formal gate takes no --openings, --checkpoint, --out or --ledger: they are fixed")
        if not (args.attempt and args.attempts_journal):
            p.error("the gate needs --attempt and --attempts-journal")
        if (args.attempt == "development") != bool(args.development_checkpoint):
            p.error("--development-checkpoint goes with a development attempt, and only there")
        args.gate_dir = gate_dir(args.root, args.development_checkpoint)
        args.out = os.path.join(args.gate_dir, "run")
        args.ledger = os.path.join(args.gate_dir, "ledger.json")
    return args


def recover(root: str, development: str | None = None, journal: str | None = None) -> dict:
    """``gate.py finalize --root``: recovery when the parent died after the
    child settled but before it wrote ``gate.json``. It holds the loop's
    owner lock (refused when busy), reads a plain snapshot of the gate
    ledger (never rewriting it or settling a running entry), and rebuilds
    the result from the provisional report under the same rules, which keeps
    an older report's rule. It runs nothing, and settles the journal only once
    the run has certainly stopped; otherwise the attempt stays interrupted. An
    existing ``gate.json`` is returned unchanged. A crash or a failed mkdir
    after the journal entry leaves no claim file: given ``journal``, the entry
    of this root is settled with no verdict, and never claimed afresh."""
    directory = gate_dir(root, development)
    lock = loopmod.Loop(root).owner()
    try:
        run = os.path.join(directory, "run")
        existing = os.path.join(run, "gate.json")
        ledger = os.path.join(directory, "ledger.json")
        entry = settlement(ledger, "gate")
        if os.path.exists(existing):
            with open(existing, encoding="utf-8") as f:
                final = json.load(f)
            settle_journal(directory, entry, final)
            return final
        if not os.path.exists(os.path.join(directory, "claim.json")):
            label = "development" if development else "formal"
            block = journal and attempts.open_attempt(journal, EVAL_SHA256, RULE_FAMILY, root, label, development)
            if not block:
                raise Refused("no gate was claimed on this loop root")
            if not quiescent(directory, entry):
                raise Refused("the process group is not confirmed gone: the attempt stays interrupted")
            attempts.settle(block, "consumed-no-verdict", {"claim_sha256": None, "report_sha256": None, "ledger_sha256": None})
            return {"kind": "p2-g1-gate", "pass": None, "attempt": block,
                    "attempt_invalid_because": ["the claim file was never written"]}  # fmt: skip
        if entry is not None and entry["status"] == "running" and not os.path.exists(os.path.join(directory, SPAWN)):
            raise Refused("the gate has no settled ledger entry yet")
        if not quiescent(directory, entry):
            raise Refused("the gate has no settled ledger entry yet, or its process group is not confirmed gone: "
                          "the attempt stays interrupted")  # fmt: skip
        args = argparse.Namespace(out=run, ledger=ledger, any_openings=False, gate_dir=directory)
        os.makedirs(run, exist_ok=True)
        return finalize(args, 0 if entry is not None and entry["status"] == "ok" else 1, "gate")
    finally:
        lock.close()


# ---------------------------------------------------------------- compare


def mcnemar_exact(b: int, c: int) -> float:
    """The two-sided exact McNemar value; nominal, as the slots are dependent."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def compare_openings() -> str:
    """The comparison's reader of the evaluation set: hash and compare only;
    the solver reads it, and only after a settled attempt."""
    if artifacts.sha256_file(EVAL) != EVAL_SHA256:
        raise Refused("the evaluation openings are not the pinned ones")
    return EVAL


def settled_attempt(directory: str) -> tuple[dict, list, dict]:
    """One validated snapshot of a complete, valid attempt: its report, log and
    identity, bound to the actual claim and to the gate ledger's own settled
    entry. The report's ``complete`` flag is not enough: ``finalize`` also writes
    it on an invalid attempt. A refusal leaves the gate's score alone."""
    path = os.path.join(directory, "run", "gate.json")
    if not os.path.exists(path):
        raise Refused("no gate.json: no settled attempt to compare")
    with open(path, "rb") as f:
        raw = f.read()
    report = json.loads(raw)
    if (report.get("attempt_invalid_because") != [] or report.get("exit_code") != 0
            or (report.get("settlement") or {}).get("status") != "ok" or report.get("rule_id") != RULE_ID
            or report.get("games") != GAMES or report.get("openings_sha256_at_end") != EVAL_SHA256
            or report.get("protocol", {}).get("openings_sha256") != EVAL_SHA256):  # fmt: skip
        raise Refused("gate.json is not a complete valid attempt")
    ledger_entry = settlement(os.path.join(directory, "ledger.json"), "gate")
    if ledger_entry is None or ledger_entry["status"] != "ok" or ledger_entry != report["settlement"]:
        raise Refused("the gate ledger's own entry is not the report's ok settlement")
    claim_path = os.path.join(directory, "claim.json")
    if not os.path.exists(claim_path):
        raise Refused("no claim for this attempt")
    with open(claim_path, encoding="utf-8") as f:
        record = json.load(f)
    if record.get("checkpoint_sha256") != report.get("checkpoint") or record.get("attempt") != report.get("attempt"):
        raise Refused("the report is not this claim's attempt")
    with open(os.path.join(directory, "run", "gate-log.json"), "rb") as f:
        log_bytes = f.read()
    if artifacts.sha256(log_bytes) != report.get("log_sha256"):
        raise Refused("the gate log is not the one the report names")
    logs = json.loads(log_bytes)
    if any(type(row[0]) is not int or type(row[1]) is not bool or row[2] not in (1.0, 0.0, -1.0) for row in logs):
        raise Refused("the gate log holds an unknown slot or outcome")
    slots = [(row[0], row[1]) for row in logs]
    if sorted(slots) != sorted((j, first) for j in range(GAMES // 2) for first in (True, False)):
        raise Refused("the gate log is not exactly the expected slots")
    results = [row[2] for row in logs]
    wins, draws = sum(r > 0 for r in results), sum(r == 0 for r in results)
    if report.get("wdl") != [wins, draws, GAMES - wins - draws]:
        raise Refused("the gate log does not match the report's counts")
    block = report.get("attempt")
    if block is not None:
        settled = attempts.settlement(block)
        if settled is None or settled["disposition"] == "consumed-no-verdict":
            raise Refused("the journal holds no evaluated settlement for this attempt")
    identity = {"gate_json_sha256": artifacts.sha256(raw), "log_sha256": report["log_sha256"],
                "claim_sha256": artifacts.sha256_file(claim_path), "checkpoint": report["checkpoint"],
                "ledger_entry": ledger_entry}  # fmt: skip
    return report, logs, identity


def read_slots(path: str) -> dict | None:
    """The solver's exact row set, by ``(j, candidate first)``; None if it is
    not."""
    seats = {"candidate_first": True, "candidate_second": False}
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3 or not parts[0].isdecimal() or parts[1] not in seats or parts[2] not in ("win", "draw", "loss"):
                return None
            key = (int(parts[0]), seats[parts[1]])
            if key in rows:
                return None
            rows[key] = parts[2]
    return rows


def join(logs: list, solver: dict) -> dict:
    """The discordant slots, by the seat both sides name."""
    won = {(row[0], row[1]) for row in logs if row[2] > 0}
    best = {k for k, v in solver.items() if v == "win"}
    only = lambda keys: sorted([j, "candidate_first" if first else "candidate_second"] for j, first in keys)  # noqa: E731
    return {
        "solver_wins": len(best),
        "candidate_only_win_slots": only(won - best),
        "solver_only_win_slots": only(best - won),
        "mcnemar_exact_two_sided_nominal": mcnemar_exact(len(won - best), len(best - won)),
        "mcnemar_note": "nominal: the two colour slots of an opening are dependent and games can repeat",
    }


PROVISIONAL_COMPARISON = "comparison.provisional.json"


def compare_inputs(args, identity: dict) -> dict:
    observed = h2h.runtime()
    if not observed["registered"]:
        raise Refused("not the registered runtime the solver binary is pinned for")
    return {
        **identity,
        "image_digest": observed["image_digest"],
        "runtime": observed,
        "openings_sha256": EVAL_SHA256,
        "solver_slots_sha256": args.solver_slots_bin_sha256,
        "solver_sources": SOLVER_SOURCES,
    }


def compare_work(args, directory: str, clock) -> int:
    """Runs the solver and writes a provisional comparison; never touches
    ``gate.json``. Exit 1 is the ``failed`` outcome. The parent publishes."""
    _, logs, identity = settled_attempt(directory)
    inputs = compare_inputs(args, identity)
    work_dir = os.path.join(directory, "compare")
    os.makedirs(work_dir, exist_ok=True)
    slots_path = os.path.join(work_dir, "solver-slots.tsv")
    done = subprocess.run([args.solver_slots_bin, compare_openings(), slots_path, str(args.threads)],
                          timeout=max(1.0, clock.left()), check=False)  # fmt: skip
    if done.returncode != 0:
        return 1
    if eval_sha_now() != EVAL_SHA256:
        raise Refused("the evaluation file changed during the comparison")
    result = {"kind": "p2-g1-comparison", "rule_id": RULE_ID, "inputs": inputs, "threads": args.threads,
              "note": "descriptive: nothing here decides the gate, and gate.json is unchanged"}  # fmt: skip
    solver = read_slots(slots_path)
    expected = {(j, first) for j in range(GAMES // 2) for first in (True, False)}
    if solver is None or set(solver) != expected:
        result |= {"outcome": "invalid", "invalid_because": "the solver's rows are not exactly the expected slots"}
    else:
        wins = sum(v == "win" for v in solver.values())
        if wins != WINS_MIN:
            result |= {"outcome": "invalid", "solver_wins": wins, "matches_published_815": False,
                       "invalid_because": "the solver did not reproduce its published 815: investigate; the bar stays 815"}  # fmt: skip
        else:
            result |= {"outcome": "complete", "matches_published_815": True, **join(logs, solver)}
    artifacts.write_atomic(os.path.join(work_dir, PROVISIONAL_COMPARISON), artifacts.canonical_json(result))
    return 0


def publish_comparison(directory: str, rc: int, inputs: dict) -> bool:
    """``run/comparison.json``, only after exit 0 and an ok ledger entry; a
    crash before this publishes nothing, and a rerun computes it again."""
    work_dir = os.path.join(directory, "compare")
    entry = settlement(os.path.join(directory, "compare-ledger.json"), "compare")
    provisional = os.path.join(work_dir, PROVISIONAL_COMPARISON)
    if rc != 0 or entry is None or entry["status"] != "ok" or not os.path.exists(provisional):
        status = entry["status"] if entry else "missing"
        artifacts.write_atomic(os.path.join(work_dir, "outcome.json"),
                               artifacts.canonical_json({"outcome": "failed", "exit_code": rc, "ledger": status}))  # fmt: skip
        return False
    with open(provisional, "rb") as f:
        raw = f.read()
    if json.loads(raw)["inputs"] != inputs:
        raise Refused("the child compared other inputs than the parent validated")
    artifacts.write_atomic(os.path.join(directory, "run", "comparison.json"), raw)
    return True


def compare_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gate.py compare", description="The solver comparison of a settled attempt")
    p.add_argument("--root", required=True)
    p.add_argument("--development-checkpoint")
    p.add_argument("--solver-slots-bin", required=True)
    p.add_argument("--solver-slots-bin-sha256", required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--wall-cap-s", type=float, required=True)
    p.add_argument("--budget-core-hours", type=float, required=True)
    args = p.parse_args(supervisor.strip_child_flag(argv))
    directory = gate_dir(args.root, args.development_checkpoint)
    if SOLVER_SLOTS_SHA256 is None:
        raise Refused("no solver binary is pinned: the comparison stays refused until an approved build")
    if not args.threads > 0:
        raise Refused("the comparison needs at least one thread")
    if args.solver_slots_bin_sha256 != SOLVER_SLOTS_SHA256 or artifacts.sha256_file(args.solver_slots_bin) != SOLVER_SLOTS_SHA256:
        raise Refused("the solver binary is not the pinned one")
    for rel, sha in SOLVER_SOURCES.items():
        if artifacts.sha256_file(os.path.join(REPO, rel)) != sha:
            raise Refused(f"{rel} is not the pinned source")
    work = lambda clock: compare_work(args, directory, clock)  # noqa: E731
    run = lambda: supervisor.run(__file__, ["compare", *argv], work,  # noqa: E731
                                 ledger=os.path.join(directory, "compare-ledger.json"),
                                 budget_core_hours=args.budget_core_hours, cores=args.threads,
                                 wall_cap_s=args.wall_cap_s, out=os.path.join(directory, "compare"),
                                 name="compare", fresh_out=False)  # fmt: skip
    if supervisor.child_flag(argv) is not None:
        return run()
    budget, cap = args.budget_core_hours, args.wall_cap_s
    if not (math.isfinite(budget) and budget > 0 and math.isfinite(cap) and cap > supervisor.CLEANUP_GRACE_S
            and budget * 3600 >= args.threads * cap):  # fmt: skip
        raise Refused("the comparison's budget does not fund its wall cap")
    lock = loopmod.Loop(args.root).owner()
    try:
        inputs = compare_inputs(args, settled_attempt(directory)[2])
        done = os.path.join(directory, "run", "comparison.json")
        if os.path.exists(done):
            with open(done, encoding="utf-8") as f:
                if json.load(f)["inputs"] != inputs:
                    raise Refused("a comparison with other inputs exists; it is never replaced")
            entry = settlement(os.path.join(directory, "compare-ledger.json"), "compare")
            if entry is None or entry["status"] != "ok":
                raise Refused("the published comparison has no ok ledger entry")
            return 0
        provisional = os.path.join(directory, "compare", PROVISIONAL_COMPARISON)
        if os.path.exists(provisional):
            os.remove(provisional)
        rc = run()
        publish_comparison(directory, rc, inputs)
        return rc
    finally:
        lock.close()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["finalize"]:
        p = argparse.ArgumentParser(prog="gate.py finalize", description=recover.__doc__.splitlines()[0])
        p.add_argument("--root", required=True)
        p.add_argument("--development-checkpoint")
        p.add_argument("--attempts-journal", help="settles an attempt whose claim file was never written")
        done = p.parse_args(argv[1:])
        recover(done.root, done.development_checkpoint, done.attempts_journal)
        return 0
    if argv[:1] == ["compare"]:
        return compare_main(argv[1:])
    args = parse(argv)

    name = "gate" if not args.any_openings else "diagnostic-gate"

    def supervised() -> int:
        rc = supervisor.run(
            __file__, argv, lambda clock: work(args, clock), ledger=args.ledger,
            budget_core_hours=args.budget_core_hours, cores=args.cores, wall_cap_s=args.wall_cap_s,
            out=args.out, name=name, spawn_record=None if args.any_openings else os.path.join(args.gate_dir, SPAWN),
        )  # fmt: skip
        if supervisor.child_flag(argv) is None and finalize(args, rc, name).get("interrupted"):
            return rc or 1
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
