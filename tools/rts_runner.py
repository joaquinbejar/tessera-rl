#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The RTS learning gate's runner (stage 5b,
``docs/reviews/ppo-protocols/rts.md``): train, save, evaluate and decide,
one training seed at a time, with the policy bound to its record.

    rts_runner.py seed SEED --out DIR --authorization SHA --frozen SHA
    rts_runner.py verdict --out DIR

``run_seed`` refuses before any work unless the authorization is the frozen
protocol's sha256, and refuses a training seed that already has a result
file: the evaluation seeds are used once. After training it saves the
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


def check_record(rec: dict, frozen_sha256: str) -> None:
    """The policy's record must be the protocol's run (E3)."""
    want = dataclasses.asdict(Hyper())
    if rec.get("protocol_sha256") != frozen_sha256:
        raise Refused(f"record's protocol {rec.get('protocol_sha256')!r} is not the frozen {frozen_sha256!r}")
    if rec.get("hyper") != want:
        raise Refused("record's Hyper is not the protocol's Hyper()")
    if rec.get("decisions_trained") != want["total_steps"]:
        raise Refused(f"record says {rec.get('decisions_trained')} decisions, not B = {want['total_steps']}")


def run_seed(seed: int, out: pathlib.Path, authorization: str, frozen_sha256: str) -> dict:
    """One training seed: refused before any work unless authorized and not
    yet evaluated; train, save, load, check the record, evaluate over the
    protocol's seeds, write the result file."""
    authorize(authorization, frozen_sha256)
    if seed not in range(ev.TRAINING_SEEDS):
        raise Refused(f"training seed {seed} is not one of 0..{ev.TRAINING_SEEDS - 1}")
    out.mkdir(parents=True, exist_ok=True)
    if result_path(out, seed).exists() or policy_path(out, seed).exists():
        raise Refused(f"training seed {seed} already has a policy or a result in {out}: the seeds are used once")
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


def verdict(out: pathlib.Path) -> dict:
    """The pass rule over the five result files."""
    wins = {}
    for seed in range(ev.TRAINING_SEEDS):
        path = result_path(out, seed)
        if not path.exists():
            raise Refused(f"no result for training seed {seed}")
        r = json.loads(path.read_text())
        if not r.get("protocol") or r.get("n") != len(ev.EVAL_SEEDS):
            raise Refused(f"the result of seed {seed} is not over the protocol's seeds")
        wins[seed] = r["wins"]
    return ev.verdict(wins)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed")
    s.add_argument("seed", type=int)
    s.add_argument("--out", required=True, type=pathlib.Path)
    s.add_argument("--authorization", required=True)
    s.add_argument("--frozen", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--out", required=True, type=pathlib.Path)
    args = p.parse_args(argv)
    try:
        if args.cmd == "seed":
            r = run_seed(args.seed, args.out, args.authorization, args.frozen)
            print(json.dumps({k: r[k] for k in ("wins", "n", "lower", "passes")}))
        else:
            print(json.dumps(verdict(args.out), indent=1))
    except (Refused, PermissionError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
