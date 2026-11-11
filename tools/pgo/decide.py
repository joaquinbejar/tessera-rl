#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The shipment decision on x86-64 (docs/reviews/pgo-wheels-design.md r5,
section 6), from measured costs, the feasibility report and determinism results.

    decide.py --manifest M --kernels K --feasibility F --w0 C --w1 C
              --w1-checks J --w1-compare J
              [--w2h C --w2f C --w2f-checks J --w2f-compare J] --out R

`--w0` and the others are `costs.json` files of callgrind.py. The checks files
come from checks.py and the compare files from `determinism.py compare` against
W0; both must name the exact wheel the costs were measured on. A failed check
means the wheel is not accepted, and the ladder falls back.

Without W2 inputs only W1 is decided. The kernel list is loaded before any W2
comparison, so an unfilled list refuses it. Nothing here has seen real data.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import callgrind  # noqa: E402
import evidence  # noqa: E402


# The frozen kernel list (tools/pgo/kernels.toml at its freeze, the merge of
# the PR that filled it). A decision with W2 inputs refuses any other list.
KERNELS_SHA256 = "17568af9055c133ea8b5356dd02287e98a7e7330c863231b7c270fe0516cdd52"


class Refused(ValueError):
    """The inputs cannot support a decision."""


def _costs(report: dict) -> dict[str, evidence.Cost]:
    return {k: evidence.Cost(v) for k, v in report["costs"].items()}


REQUIRED_CHECKS = ("python_suite", "num_tests")


def section7(label: str, inputs: dict | None, wheel: str, base: str) -> list[str]:
    """The failed checks of one wheel. Inputs for another wheel are refused."""
    if inputs is None:
        raise Refused(f"{label}: no section 7 results")
    checks, compare = inputs["checks"], inputs["compare"]
    if checks.get("wheel_sha256") != wheel:
        raise Refused(f"{label}: the checks are for another wheel")
    if compare.get("wheels") != [base, wheel]:
        raise Refused(f"{label}: the comparison is not of this wheel against W0")
    missing = [c for c in REQUIRED_CHECKS if c not in checks["checks"]]
    if missing:
        raise Refused(f"{label}: checks missing: {missing}")
    failed = [k for k, v in sorted(checks["checks"].items()) if v != 0]
    if compare.get("passed") is not True:
        failed.append("determinism")
    return failed


def _check(reports: dict, feasibility: dict, manifest_sha: str, all_scenarios) -> None:
    if feasibility.get("passed") is not True:
        raise Refused("the feasibility gate has not passed")
    if feasibility["sha256"]["wheel"] != reports["W1"]["sha256"]["wheel"]:
        raise Refused("the feasibility gate was not measured on this W1")
    if feasibility["sha256"]["manifest"] != manifest_sha:
        raise Refused("the feasibility report is for another manifest")
    names = {s.name for s in all_scenarios}
    wheels = {}
    for label, report in reports.items():
        if report["sha256"]["manifest"] != manifest_sha:
            raise Refused(f"{label}: costs are for another manifest")
        if set(report["costs"]) != names:
            raise Refused(f"{label}: costs do not cover exactly the manifest")
        wheel = report["sha256"]["wheel"]
        if wheel in wheels:
            raise Refused(f"{label} and {wheels[wheel]} are the same wheel")
        wheels[wheel] = label
    # Files compared across wheels differ in the wheel hash by design; every
    # other field of the stamp (manifest, interpreter, Valgrind, CPU) must be
    # equal on both sides, and equal to the feasibility gate's.
    def rest(stamp: dict) -> dict:
        return {k: v for k, v in stamp.items() if k not in ("wheel", "kernels")}
    base_stamp = rest(feasibility["sha256"])
    for label, report in reports.items():
        if rest(report["sha256"]) != base_stamp:
            diff = sorted(k for k in set(base_stamp) | set(rest(report["sha256"]))
                          if base_stamp.get(k) != rest(report["sha256"]).get(k))
            raise Refused(f"{label}: measured under another stamp than the gate (fields {diff})")


def decide(reports: dict, feasibility: dict, seventh: dict, manifest_path: str, kernels_path: str) -> dict:
    manifest = evidence.load_manifest(manifest_path)
    all_scenarios = evidence.scenarios(manifest)
    manifest_sha = callgrind.sha256_of(manifest_path)
    _check(reports, feasibility, manifest_sha, all_scenarios)
    sets = evidence.Sets(
        tuple(feasibility["gating"]),
        tuple(feasibility["mispredict"]),
        feasibility["usable"],
        feasibility["mispredict_available"],
    )
    w0, w1 = _costs(reports["W0"]), _costs(reports["W1"])
    base = reports["W0"]["sha256"]["wheel"]
    failed = {"W1": section7("W1", seventh.get("W1"), reports["W1"]["sha256"]["wheel"], base)}
    w1_ok = evidence.accept_w1(w1, w0, sets.gating, not failed["W1"])
    out: dict = {
        "w1_accepted": w1_ok,
        "failed_checks": failed,
        "wheels": {k: v["sha256"]["wheel"] for k, v in reports.items()},
        "manifest_sha256": manifest_sha,
        "w1_ir_ratio": evidence.geomean([w1[n].ir / w0[n].ir for n in sets.gating]),
    }
    w2_ok = False
    if "W2F" in reports or "W2H" in reports:
        if not {"W2F", "W2H"} <= set(reports):
            raise Refused("W2F and W2H are needed together")
        evidence.load_kernels(kernels_path)
        out["kernels_sha256"] = callgrind.sha256_of(kernels_path)
        if out["kernels_sha256"] != KERNELS_SHA256:
            raise Refused(f"the kernel list is not the frozen one: {out['kernels_sha256']} != {KERNELS_SHA256}")
        w2f, w2h = _costs(reports["W2F"]), _costs(reports["W2H"])
        failed["W2F"] = section7("W2F", seventh.get("W2F"), reports["W2F"]["sha256"]["wheel"], base)
        w2_ok = evidence.accept_w2f(w2f, w2h, w1, all_scenarios, sets, w1_ok, not failed["W2F"])
        out["w2f_accepted"] = w2_ok
        holdout = {s.name for s in all_scenarios if s.holdout}
        trained = [n for n in sets.gating if n not in holdout]
        out["w2f_ir_ratio"] = evidence.geomean([w2f[n].ir / w1[n].ir for n in trained])
        names = [n for n in sets.mispredict if n not in holdout]
        if sets.mispredict_available and names:
            out["w2f_mispredict_ratio"] = evidence.geomean([w2f[n].mispredicts / w1[n].mispredicts for n in names])
    out["ship"] = evidence.ship(w1_ok, w2_ok)
    out["claim"] = wording(out, manifest["version"])
    return out


def wording(out: dict, version: int) -> str:
    """The only allowed claim (section 6): Callgrind on Linux x86-64, the `_native`
    object only, these scenarios. Never time, SPS or other platforms."""
    if out["ship"] == "W0":
        return "No change: the default release wheel ships."
    extra = ""
    if out["ship"] == "W1":
        pct, against = 100 * (1 - out["w1_ir_ratio"]), "the default release wheel"
    else:
        pct, against = 100 * (1 - out["w2f_ir_ratio"]), "the LTO wheel"
        if out["w2f_ir_ratio"] > evidence.PGO_IR_GAIN:  # it passed on mispredictions
            extra = f" and {100 * (1 - out['w2f_mispredict_ratio']):.1f}% fewer simulated branch mispredictions"
    return (
        f"{pct:.1f}% fewer instructions{extra} than {against} inside the _native object, "
        f"under Callgrind on Linux x86-64, in the gating scenarios of manifest v{version}. "
        "Not a time or SPS claim, and it excludes allocator and runtime work."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("manifest", "kernels", "feasibility", "w0", "w1", "w1-checks", "w1-compare", "out"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("w2h", "w2f", "w2f-checks", "w2f-compare"):
        parser.add_argument(f"--{name}")
    args = parser.parse_args(argv)

    def load(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    reports = {"W0": load(args.w0), "W1": load(args.w1)}
    if args.w2h:
        reports["W2H"] = load(args.w2h)
    if args.w2f:
        reports["W2F"] = load(args.w2f)
    seventh = {"W1": {"checks": load(args.w1_checks), "compare": load(args.w1_compare)}}
    if args.w2f_checks and args.w2f_compare:
        seventh["W2F"] = {"checks": load(args.w2f_checks), "compare": load(args.w2f_compare)}
    try:
        result = decide(reports, load(args.feasibility), seventh, args.manifest, args.kernels)
    except (Refused, ValueError, KeyError) as e:
        print(f"refused: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(result["ship"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
