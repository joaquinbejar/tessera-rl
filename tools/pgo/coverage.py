#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The profile coverage acceptance of a PGO wheel (docs/reviews/pgo-wheels-design.md,
section 3), and the check that the final wheel carries no instrumentation.

    coverage.py --kernels tools/pgo/kernels.toml --profile show.txt --build-log final.log \\
        --wheel W --out coverage.json

`--profile` is the output of `llvm-profdata show --all-functions`. A profile is
usable only if every kernel has a function with a nonzero count, and the final
build reported no control-flow-hash mismatch for a function of a `tessera_rl`
crate. Functions without a profile are counted, not gated. Exit 0 when usable.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evidence  # noqa: E402

FUNCTION = re.compile(r"^\s*(\S.*?):\s*$")
COUNT = re.compile(r"^\s*Function count:\s*(\d+)\s*$")
MISMATCH = re.compile(r"(hash mismatch|control flow change)", re.IGNORECASE)
MISSING = re.compile(r"no profile data available for function", re.IGNORECASE)


def counts(show: str) -> dict[str, int]:
    """Function name to its entry count, from `llvm-profdata show --all-functions`."""
    out: dict[str, int] = {}
    name = None
    for line in show.splitlines():
        m = COUNT.match(line)
        if m and name is not None:
            out[name] = int(m.group(1))
            name = None
            continue
        f = FUNCTION.match(line)
        if f and not line.lstrip().startswith(("Hash", "Counters", "Function count", "Total", "Maximum", "Instrumentation")):
            name = f.group(1)
    return out


def check(kernels: list[str], show: str, build_log: str, wheel: pathlib.Path) -> dict:
    found = counts(show)
    kernel_status = {}
    for k in kernels:
        # Each part as a v0 length-prefixed identifier, anchored on its left:
        # "4Grid" must not match inside "14Grid…" (another identifier).
        parts = [re.compile(rf"(?<![0-9]){len(p)}{re.escape(p)}") for p in k.split("+")]
        hits = {n: c for n, c in found.items() if all(p.search(n) for p in parts)}
        kernel_status[k] = max(hits.values()) if hits else 0
    mismatches = [l.strip() for l in build_log.splitlines() if MISMATCH.search(l) and "tessera_rl" in l]
    missing = sum(1 for l in build_log.splitlines() if MISSING.search(l))
    instrumented = []
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if name.endswith((".so", ".pyd", ".dylib")) and b"__llvm_profile" in z.read(name):
                instrumented.append(name)
    usable = all(v > 0 for v in kernel_status.values()) and not mismatches and not instrumented
    return {"usable": usable, "kernels": kernel_status, "tessera_mismatches": mismatches,
            "functions_without_profile": missing, "instrumented_objects": instrumented}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--kernels", "--profile", "--build-log", "--wheel", "--out"):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)
    try:
        kernels = evidence.load_kernels(args.kernels)
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    result = check(kernels, pathlib.Path(args.profile).read_text(encoding="utf-8"),
                   pathlib.Path(args.build_log).read_text(encoding="utf-8", errors="replace"), pathlib.Path(args.wheel))
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    if not result["usable"]:
        print(f"refused: the profile is not usable: {json.dumps(result)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
