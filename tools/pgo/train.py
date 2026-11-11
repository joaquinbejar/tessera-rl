#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Runs the PGO training set of the manifest through the installed instrumented
wheel (docs/reviews/pgo-wheels-design.md r4, sections 3 and 4). `--holdout-build`
trains W2H, without the holdout; the default trains W2F.

    LLVM_PROFILE_FILE=<abs>/%m_%p.profraw train.py --manifest tools/pgo/manifest.toml

Refuses to start without LLVM_PROFILE_FILE, since a run without it would train
nothing and fail only later. Exits normally, as counters are written at exit.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evidence  # noqa: E402
import workload  # noqa: E402


def main(argv=None, environ=os.environ, run=workload.run_scenario) -> int:
    """`run` takes the scenario, the manifest and a steps override (None: the manifest's)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--holdout-build", action="store_true")
    parser.add_argument("--steps", type=int, default=None,
                        help="override every scenario's steps (the short format check before a long training)")
    args = parser.parse_args(argv)
    profile = environ.get("LLVM_PROFILE_FILE", "")
    if not profile or not os.path.isabs(profile) or "%p" not in profile:
        print("refused: LLVM_PROFILE_FILE must be an absolute pattern with %p", file=sys.stderr)
        return 2
    manifest = evidence.load_manifest(args.manifest)
    for scenario in evidence.training_set(evidence.scenarios(manifest), args.holdout_build):
        print(scenario.name, flush=True)
        run(scenario, manifest, args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
