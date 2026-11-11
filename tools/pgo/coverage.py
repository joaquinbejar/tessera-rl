#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The profile coverage acceptance of a PGO wheel (docs/reviews/pgo-wheels-design.md,
section 3), and the check that the final wheel carries no instrumentation.

    coverage.py --kernels tools/pgo/kernels.toml --profile show.txt --build-log final.log \\
        --wheel W --variant full|holdout --out coverage.json

`--profile` is the output of `llvm-profdata show --all-functions --counts` (the
`Block counts:` line of each function; without `--counts` no count is printed).
For W2F (`--variant full`) a profile is usable only if every kernel has a
function with a nonzero count; for W2H (`--variant holdout`) every kernel but
the holdout ones must be nonzero and the holdout ones must be zero. In both, the
final build must report no control-flow-hash mismatch for a function of a
`tessera_rl` crate. Functions without a profile are counted, not gated. Exit 0
when usable. The formats are those of the pinned toolchain (fixtures under
tests/python/fixtures/pgo/).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tomllib
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evidence  # noqa: E402

FUNCTION = re.compile(r"^  (\S.*):$")
BLOCKS = re.compile(r"^    Block counts: \[([0-9, ]*)\]$")
MISMATCH = re.compile(r"(hash mismatch|control flow change)", re.IGNORECASE)
MISSING = re.compile(r"no profile data available for function", re.IGNORECASE)


def counts(show: str) -> dict[str, int]:
    """Function name (the mangled part after any `module;` prefix) to its
    largest block count, from `llvm-profdata show --all-functions --counts`."""
    out: dict[str, int] = {}
    name = None
    for line in show.splitlines():
        f = FUNCTION.match(line)
        if f:
            name = f.group(1).rsplit(";", 1)[-1]
            continue
        b = BLOCKS.match(line)
        if b and name is not None:
            values = [int(v) for v in b.group(1).replace(" ", "").split(",") if v]
            out[name] = max(out.get(name, 0), max(values, default=0))
            name = None
    return out


def ident(part: str) -> str:
    """A v0 identifier: decimal length, then `_` if the name starts with `_` or a
    digit, then the name."""
    sep = "_" if part[:1] == "_" or part[:1].isdigit() else ""
    return f"{len(part)}{sep}{part}"


def _boundary(name: str, i: int) -> bool:
    """True if an identifier may start at `name[i]`: the previous character is
    not a digit, or a whole identifier ends right before `i` (as `8connect4` in
    `8connect48Connect4`). So `4Grid` inside `14GridExtended` is refused."""
    if i == 0 or not name[i - 1].isdigit():
        return True
    for m in range(i - 1, 0, -1):
        body = name[m:i]
        for length, sep in ((len(body), ""), (len(body) - 1, "_")):
            if length > 0 and (not sep or body.startswith("_")) and name[:m].endswith(str(length)):
                k = m - len(str(length))
                if k == 0 or not name[k - 1].isdigit() or _boundary(name, k):
                    return True
    return False


def contains(name: str, part: str) -> bool:
    token = ident(part)
    i = name.find(token)
    while i != -1:
        if _boundary(name, i):
            return True
        i = name.find(token, i + 1)
    return False


def kernel_count(kernel: str, found: dict[str, int]) -> int:
    parts = kernel.split("+")
    hits = [c for n, c in found.items() if all(contains(n, p) for p in parts)]
    return max(hits) if hits else 0


def check(kernels: list[str], show: str, build_log: str, wheel: pathlib.Path, holdout: list[str] | None = None) -> dict:
    """`holdout` is None for W2F; for W2H, the kernels that must be zero."""
    found = counts(show)
    kernel_status = {k: kernel_count(k, found) for k in kernels}
    zero = set(holdout or ())
    unknown = sorted(zero - set(kernels))
    if unknown:
        raise ValueError(f"holdout kernels not in the kernel list: {unknown}")
    kernels_ok = all((v == 0) if k in zero else (v > 0) for k, v in kernel_status.items())
    mismatches = [l.strip() for l in build_log.splitlines() if MISMATCH.search(l) and "tessera_rl" in l]
    missing = sum(1 for l in build_log.splitlines() if MISSING.search(l))
    instrumented = []
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if name.endswith((".so", ".pyd", ".dylib")) and b"__llvm_profile" in z.read(name):
                instrumented.append(name)
    usable = kernels_ok and not mismatches and not instrumented
    return {"usable": usable, "variant": "holdout" if holdout is not None else "full", "kernels": kernel_status,
            "must_be_zero": sorted(zero), "tessera_mismatches": mismatches,
            "functions_without_profile": missing, "instrumented_objects": instrumented}


def load_holdout(path: str) -> list[str]:
    with open(path, "rb") as f:
        value = tomllib.load(f).get("holdout_kernels")
    if not isinstance(value, list) or not value or not all(isinstance(k, str) and k for k in value):
        raise ValueError("holdout_kernels must be a nonempty list of strings")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--kernels", "--profile", "--build-log", "--wheel", "--out"):
        parser.add_argument(name, required=True)
    parser.add_argument("--variant", choices=("full", "holdout"), required=True)
    args = parser.parse_args(argv)
    try:
        kernels = evidence.load_kernels(args.kernels)
        holdout = load_holdout(args.kernels) if args.variant == "holdout" else None
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    result = check(kernels, pathlib.Path(args.profile).read_text(encoding="utf-8"),
                   pathlib.Path(args.build_log).read_text(encoding="utf-8", errors="replace"), pathlib.Path(args.wheel),
                   holdout)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    if not result["usable"]:
        print(f"refused: the profile is not usable: {json.dumps(result)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
