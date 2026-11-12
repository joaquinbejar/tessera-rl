#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The RTS learning gate's runner (stage 5b,
``docs/reviews/ppo-protocols/rts.md``): train, save, evaluate and decide,
one training seed at a time, with the policy bound to its record.

    rts_runner.py seed SEED --out DIR --protocol PATH --authorization SHA --frozen SHA
    rts_runner.py verdict --out DIR --frozen SHA

``run_seed`` refuses before any work unless the repository's frozen
protocol sha256 (``rts_training.FROZEN_PROTOCOL_SHA256``, None until the
freeze PR) is set, the protocol file at ``--protocol`` hashes to it, and the
authorization and ``--frozen`` both equal it: typed strings alone cannot
start a run. It then takes the training seed atomically (creates
``seed-<s>.lock`` exclusively) and refuses a seed that already has a lock, a
policy or a result: the evaluation seeds are used once.

A crash leaves the lock (and perhaps a policy without a result). The seed
is then spent: the runner never repeats it, and whether it counts as a
failure is for the owner, through the coordinator. In particular a crash
after the first evaluation game means that seed's evaluation is used. After training it saves the
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


def run_seed(seed: int, out: pathlib.Path, protocol: pathlib.Path, authorization: str, frozen_sha256: str) -> dict:
    """One training seed: refused before any work unless authorized against
    the repository's frozen hash and the protocol file, and the seed not yet
    taken; then train, save, load, check the record, evaluate the loaded
    policy over the protocol's seeds, write the result file."""
    authorize(authorization, frozen_sha256)
    check_protocol_file(protocol, frozen_sha256)
    if seed not in range(ev.TRAINING_SEEDS):
        raise Refused(f"training seed {seed} is not one of 0..{ev.TRAINING_SEEDS - 1}")
    out.mkdir(parents=True, exist_ok=True)
    if result_path(out, seed).exists() or policy_path(out, seed).exists():
        raise Refused(f"training seed {seed} already has a policy or a result in {out}: the seeds are used once")
    try:
        # Taken atomically: two starts of one seed cannot both train.
        with open(lock_path(out, seed), "x", encoding="utf-8") as f:
            f.write(frozen_sha256)
    except FileExistsError as e:
        raise Refused(f"training seed {seed} is already taken ({lock_path(out, seed)}): the seeds are used once") from e
    h = Hyper()
    agent = rts_training.train(seed, h, authorization, frozen_sha256)
    rec = ev.record(seed, h, frozen_sha256)
    rec["decisions_trained"] = agent.decisions_trained
    ev.save(agent, policy_path(out, seed), rec)
    agent, rec = ev.load(policy_path(out, seed), h)
    check_record(rec, frozen_sha256)
    result = ev.evaluate(agent, h, authorization, frozen_sha256)
    if not result.get("protocol"):
        raise Refused("the evaluation was not over the protocol's seeds")
    result["record"] = rec
    result_path(out, seed).write_text(json.dumps(result, indent=1, sort_keys=True))
    return result


def verdict(out: pathlib.Path, frozen_sha256: str) -> dict:
    """The pass rule over the five result files, each checked against the
    same frozen protocol: its record, its training seed, the 400 games, the
    wins counted from them and its passes recomputed."""
    seeds = [str(s) for s in ev.EVAL_SEEDS]
    wins = {}
    for seed in range(ev.TRAINING_SEEDS):
        path = result_path(out, seed)
        if not path.exists():
            raise Refused(f"no result for training seed {seed}")
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
        wins[seed] = counted
    return ev.verdict(wins)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed")
    s.add_argument("seed", type=int)
    s.add_argument("--out", required=True, type=pathlib.Path)
    s.add_argument("--protocol", required=True, type=pathlib.Path)
    s.add_argument("--authorization", required=True)
    s.add_argument("--frozen", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--out", required=True, type=pathlib.Path)
    v.add_argument("--frozen", required=True)
    args = p.parse_args(argv)
    try:
        if args.cmd == "seed":
            r = run_seed(args.seed, args.out, args.protocol, args.authorization, args.frozen)
            print(json.dumps({k: r[k] for k in ("wins", "n", "lower", "passes")}))
        else:
            print(json.dumps(verdict(args.out, args.frozen), indent=1))
    except (Refused, PermissionError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
