#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""A reporting figure for #171 (docs/reviews/171-diagnosis/NOTE-r2.md, P1a):
how much of a scenario's Ir is the native module's own, both ways.

    attribution.py FILE.callgrind [FILE.callgrind ...] [--json]

- **exclusive**: the cost inside the `_native` object, exactly what the gate
  uses (`evidence.native_cost`);
- **with memory primitives**: exclusive plus the inclusive cost of `_native`'s
  calls into libc's memory primitives (`memset`, `memcpy`, `memmove`, `bzero`
  and their `__mem*` / `_platform_mem*` variants). That is engine work done
  inside libc, such as zeroing observation rows (ADR-0058).

Report only. It must not feed the gate: about 21 G Ir of zeroing is the same in
every build, so adding it to Snake-state's about 2.7 G native Ir would show a
5% engine change as about 0.6% (NOTE-r2 section 3.1). The gate statistic and
`evidence.py` are unchanged; this tool reads `.callgrind` files and runs nothing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evidence  # noqa: E402

# The memory-primitive matching moved to evidence.py, which the ladder now uses
# too (#171); these names stay for this tool's callers and tests.
is_memory_primitive = evidence.is_memory_primitive
primitive_calls = evidence.primitive_calls


def attribution(text: str) -> dict:
    """The two native shares of one `.callgrind` file."""
    objects = evidence.parse_callgrind(text)
    total = sum(c.ir for c in objects.values())
    native = evidence.native_cost(objects).ir
    calls = primitive_calls(text)
    primitives = sum(calls.values())
    return {
        "total_ir": total,
        "native_exclusive_ir": native,
        "native_exclusive_share": native / total if total else 0.0,
        "primitives_from_native_ir": primitives,
        "native_with_primitives_share": (native + primitives) / total if total else 0.0,
        "primitives_by_function": dict(sorted(calls.items(), key=lambda kv: -kv[1])),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--json", action="store_true", help="one JSON object per file, keyed by path")
    args = parser.parse_args(argv)
    try:
        report = {path: attribution(pathlib.Path(path).read_text(errors="replace")) for path in args.files}
    except (OSError, ValueError) as e:
        print(f"attribution: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))
        return 0
    for path, r in report.items():
        print(path)
        print(f"  total Ir                       {r['total_ir']:>16,}")
        print(f"  _native exclusive (the gate's) {r['native_exclusive_ir']:>16,}  {r['native_exclusive_share']:7.2%}")
        print(f"  memory primitives from _native {r['primitives_from_native_ir']:>16,}")
        print(f"  _native with those primitives  {'':>16}  {r['native_with_primitives_share']:7.2%}  (report only)")
        for fn, ir in r["primitives_by_function"].items():
            print(f"    {fn}: {ir:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
