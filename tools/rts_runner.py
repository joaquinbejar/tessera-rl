#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The RTS learning gate's runner (stage 5b,
``docs/reviews/ppo-protocols/rts.md``): train, save, evaluate and decide,
one training seed at a time, with the policy bound to its record.

    rts_runner.py seed SEED --out DIR --protocol PATH --authorization SHA --frozen SHA [--rerun-of OLD]
    rts_runner.py verdict --out DIR --frozen SHA

``run_seed`` refuses before any work unless the repository's frozen
protocol sha256 (``rts_training.FROZEN_PROTOCOL_SHA256``, None until the
freeze PR) is set, the protocol file at ``--protocol`` hashes to it, and the
authorization and ``--frozen`` both equal it: typed strings alone cannot
start a run. It then takes the training seed atomically (creates
``seed-<s>.lock`` exclusively) and refuses a seed that already has a lock, a
policy or a result: the evaluation seeds are used once.

The runner writes ``seed-<s>.evaluating`` before the first evaluation game.
A crash before that marker (no evaluation game played) may be rerun once,
with the same training seed, from scratch in a fresh directory
(``--rerun-of OLD``: OLD must hold that seed's lock and neither the marker
nor a result, must not itself be a rerun, and its lock's process must be
dead on this host); OLD is taken once, atomically
(``seed-<s>.rerun-taken`` names the rerun's directory), and the rerun's
result names OLD; the crashed directory is kept and both are disclosed. A crash after the marker, or a second crash,
spends the seed: how it counts is the owner's decision in the protocol. After training it saves the
policy with its record, loads it back, and evaluates it only if the record
says the run is the protocol's: the frozen sha256, ``Hyper()`` exactly, and
the decisions trained equal to its B. The result file holds the evaluation
over exactly the protocol's 400 seeds. ``verdict`` reads the five result
files and applies the pass rule (every seed).

No run is authorized by this file; the tests never reach training or an
evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import socket
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rts_evaluation as ev  # noqa: E402
import rts_training  # noqa: E402
from rts_training import Hyper, authorize  # noqa: E402


class Refused(Exception):
    """The runner refuses to go on."""


def result_path(out: pathlib.Path, seed: int) -> pathlib.Path:
    return out / f"seed-{seed}.result.json"


def policy_path(out: pathlib.Path, seed: int) -> pathlib.Path:
    return out / f"seed-{seed}.policy.pt"


def lock_path(out: pathlib.Path, seed: int) -> pathlib.Path:
    return out / f"seed-{seed}.lock"


def marker_path(out: pathlib.Path, seed: int) -> pathlib.Path:
    """Written before the first evaluation game: from here a crash spends the seed."""
    return out / f"seed-{seed}.evaluating"


def rerun_path(out: pathlib.Path, seed: int) -> pathlib.Path:
    return out / f"seed-{seed}.rerun-of"


def taken_path(old: pathlib.Path, seed: int) -> pathlib.Path:
    """In the crashed directory: the one rerun of this seed, by its directory."""
    return old / f"seed-{seed}.rerun-taken"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def check_rerun(old: pathlib.Path, out: pathlib.Path, seed: int) -> None:
    """A rerun is allowed once, for a crash before any evaluation game."""
    if pathlib.Path(old).resolve() == pathlib.Path(out).resolve():
        raise Refused("a rerun goes to a fresh directory, not the crashed one")
    if not lock_path(old, seed).exists():
        raise Refused(f"{old} holds no lock for seed {seed}: nothing crashed there")
    if marker_path(old, seed).exists() or result_path(old, seed).exists():
        raise Refused(f"seed {seed} began its evaluation in {old}: it is spent, not rerun")
    if rerun_path(old, seed).exists():
        raise Refused(f"{old} is itself a rerun of seed {seed}: a second crash spends the seed")
    # A lock without a marker is also what a run still training looks like.
    try:
        holder = json.loads(lock_path(old, seed).read_text())
    except (ValueError, OSError) as e:
        raise Refused(f"{lock_path(old, seed)} does not name its process: cannot tell a crash from a live run") from e
    if holder.get("host") != socket.gethostname():
        raise Refused(f"seed {seed} in {old} ran on {holder.get('host')!r}, not this host: rerun it there")
    if _alive(int(holder.get("pid", -1))):
        raise Refused(f"seed {seed} in {old} is still running (pid {holder.get('pid')}): not a crash")


def check_protocol_file(path: pathlib.Path, frozen_sha256: str) -> None:
    """The protocol text the owner authorized: its sha256 must be the frozen one."""
    got = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    if got != frozen_sha256:
        raise Refused(f"the protocol file {path} hashes to {got}, not the frozen {frozen_sha256}")


def check_record(rec: dict, frozen_sha256: str) -> None:
    """The policy's record must be the protocol's run (E3)."""
    want = dataclasses.asdict(Hyper())
    if rec.get("protocol_sha256") != frozen_sha256:
        raise Refused(f"record's protocol {rec.get('protocol_sha256')!r} is not the frozen {frozen_sha256!r}")
    if rec.get("hyper") != want:
        raise Refused("record's Hyper is not the protocol's Hyper()")
    if rec.get("decisions_trained") != want["total_steps"]:
        raise Refused(f"record says {rec.get('decisions_trained')} decisions, not B = {want['total_steps']}")


def run_seed(seed: int, out: pathlib.Path, protocol: pathlib.Path, authorization: str, frozen_sha256: str,
             rerun_of: pathlib.Path | None = None) -> dict:
    """One training seed: refused before any work unless authorized against
    the repository's frozen hash and the protocol file, and the seed not yet
    taken; then train, save, load, check the record, evaluate the loaded
    policy over the protocol's seeds, write the result file."""
    authorize(authorization, frozen_sha256)
    check_protocol_file(protocol, frozen_sha256)
    if seed not in range(ev.TRAINING_SEEDS):
        raise Refused(f"training seed {seed} is not one of 0..{ev.TRAINING_SEEDS - 1}")
    out.mkdir(parents=True, exist_ok=True)
    if rerun_of is not None:
        check_rerun(rerun_of, out, seed)
    if result_path(out, seed).exists() or policy_path(out, seed).exists():
        raise Refused(f"training seed {seed} already has a policy or a result in {out}: the seeds are used once")
    if rerun_of is not None:
        # The crashed directory is taken once, atomically, before any work:
        # it names its one rerun.
        try:
            with open(taken_path(rerun_of, seed), "x", encoding="utf-8") as f:
                f.write(str(pathlib.Path(out).resolve()))
        except FileExistsError as e:
            raise Refused(f"seed {seed} in {rerun_of} was already rerun ({taken_path(rerun_of, seed)})") from e
    try:
        # Taken atomically: two starts of one seed cannot both train.
        with open(lock_path(out, seed), "x", encoding="utf-8") as f:
            f.write(json.dumps({"sha256": frozen_sha256, "pid": os.getpid(), "host": socket.gethostname()}))
    except FileExistsError as e:
        raise Refused(f"training seed {seed} is already taken ({lock_path(out, seed)}): the seeds are used once") from e
    if rerun_of is not None:
        rerun_path(out, seed).write_text(str(pathlib.Path(rerun_of).resolve()))
    h = Hyper()
    agent = rts_training.train(seed, h, authorization, frozen_sha256)
    rec = ev.record(seed, h, frozen_sha256)
    rec["decisions_trained"] = agent.decisions_trained
    ev.save(agent, policy_path(out, seed), rec)
    agent, rec = ev.load(policy_path(out, seed), h)
    check_record(rec, frozen_sha256)
    with open(marker_path(out, seed), "x", encoding="utf-8") as f:
        f.write(frozen_sha256)
    result = ev.evaluate(agent, h, authorization, frozen_sha256)
    if not result.get("protocol"):
        raise Refused("the evaluation was not over the protocol's seeds")
    result["record"] = rec
    if rerun_of is not None:
        result["rerun_of"] = str(pathlib.Path(rerun_of).resolve())
    result_path(out, seed).write_text(json.dumps(result, indent=1, sort_keys=True))
    return result


def verdict(out, frozen_sha256: str) -> dict:
    """The pass rule over the five result files, each checked against the
    same frozen protocol: its record, its training seed, the 400 games, the
    wins counted from them and its passes recomputed."""
    outs = [pathlib.Path(o) for o in (out if isinstance(out, (list, tuple)) else [out])]
    seeds = [str(s) for s in ev.EVAL_SEEDS]
    wins, sources = {}, {}
    for seed in range(ev.TRAINING_SEEDS):
        found = [result_path(o, seed) for o in outs if result_path(o, seed).exists()]
        if not found:
            raise Refused(f"no result for training seed {seed}")
        if len(found) > 1:
            raise Refused(f"training seed {seed} has results in several directories: {found}")
        path = found[0]
        r = json.loads(path.read_text())
        if not r.get("protocol") or r.get("n") != len(ev.EVAL_SEEDS):
            raise Refused(f"the result of seed {seed} is not over the protocol's seeds")
        check_record(r.get("record") or {}, frozen_sha256)
        if r["record"].get("training_seed") != seed:
            raise Refused(f"{path.name} holds the record of training seed {r['record'].get('training_seed')}")
        games = r.get("games") or {}
        if sorted(games) != sorted(seeds):
            raise Refused(f"{path.name}: its games are not exactly the protocol's 400 seeds")
        counted = sum(1 for o in games.values() if o == "win")
        if r.get("wins") != counted:
            raise Refused(f"{path.name}: wins {r.get('wins')} but {counted} games won")
        if r.get("passes") != ev.seed_passes(counted, len(ev.EVAL_SEEDS)):
            raise Refused(f"{path.name}: passes does not follow from its wins")
        source = {"dir": str(path.parent.resolve())}
        if r.get("rerun_of"):
            taken = taken_path(pathlib.Path(r["rerun_of"]), seed)
            if not taken.exists() or taken.read_text() != str(path.parent.resolve()):
                raise Refused(f"{path.name} names rerun_of {r['rerun_of']}, whose rerun is not this directory")
            source["rerun_of"] = r["rerun_of"]
        sources[seed] = source
        wins[seed] = counted
    result = ev.verdict(wins)
    result["sources"] = sources
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed")
    s.add_argument("seed", type=int)
    s.add_argument("--out", required=True, type=pathlib.Path)
    s.add_argument("--protocol", required=True, type=pathlib.Path)
    s.add_argument("--authorization", required=True)
    s.add_argument("--frozen", required=True)
    s.add_argument("--rerun-of", type=pathlib.Path, default=None)
    v = sub.add_parser("verdict")
    v.add_argument("--out", required=True, type=pathlib.Path, action="append",
                   help="a run directory; repeat it when a seed was rerun in its own directory")
    v.add_argument("--frozen", required=True)
    args = p.parse_args(argv)
    try:
        if args.cmd == "seed":
            r = run_seed(args.seed, args.out, args.protocol, args.authorization, args.frozen, args.rerun_of)
            print(json.dumps({k: r[k] for k in ("wins", "n", "lower", "passes")}))
        else:
            print(json.dumps(verdict(args.out, args.frozen), indent=1))
    except (Refused, PermissionError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
