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
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evidence  # noqa: E402

# libc's object (glibc, or macOS's libsystem_platform) and its memory primitives.
_LIBC = re.compile(r"^(libc[.-]|libc\.so|libsystem_platform)")
_PRIMITIVE = re.compile(r"^(_platform_)?(__)?(mem(set|cpy|move)|bzero)(\b|_|@|$)")


def is_memory_primitive(obj: str, function: str) -> bool:
    """Whether `function` in object `obj` is one of libc's memory primitives."""
    return bool(_LIBC.match(os.path.basename(obj))) and bool(_PRIMITIVE.match(function))


def primitive_calls(text: str) -> dict[str, int]:
    """Inclusive Ir of every call from the `_native` object into a libc memory
    primitive, by callee function. A call's cost is the line after its
    `calls=` line; `cob=` and `cfn=` name its callee (an absent `cob=` means
    the caller's own object)."""
    positions = 1
    events: list[str] | None = None
    objects: dict[str, str] = {}
    functions: dict[str, str] = {}
    ob: str | None = None
    cob: str | None = None
    cfn: str | None = None
    pending: tuple[str, str, str] | None = None
    out: dict[str, int] = {}

    def name(table: dict[str, str], value: str) -> str:
        key, _, rest = value.strip().partition(" ")
        if key.startswith("("):
            if rest:
                table[key] = rest
            if key not in table:
                raise ValueError(f"unknown id {key}")
            return table[key]
        return value.strip()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("positions:"):
            positions = len(line.split(":", 1)[1].split())
        elif line.startswith("events:"):
            events = line.split(":", 1)[1].split()
            if "Ir" not in events:
                raise ValueError("callgrind events lack Ir")
        elif line.startswith("ob="):
            ob, cob = name(objects, line[3:]), None
        elif line.startswith("cob="):
            cob = name(objects, line[4:])
        elif line.startswith("fn="):
            name(functions, line[3:])
            cob = None
        elif line.startswith("cfn="):
            cfn = name(functions, line[4:])
        elif line.startswith("calls="):
            if ob is None or cfn is None:
                raise ValueError("a calls= line before an object or a callee")
            pending = (ob, cob or ob, cfn)
        elif line[0] in "0123456789+-*":
            if pending is None:
                continue
            if events is None:
                raise ValueError("a cost line before the events")
            caller, callee_ob, callee_fn = pending
            pending, cob = None, None
            values = line.split()[positions:]
            ir = int(values[events.index("Ir")]) if len(values) > events.index("Ir") else 0
            if os.path.basename(caller).startswith(evidence.NATIVE_OBJECT) and is_memory_primitive(callee_ob, callee_fn):
                out[callee_fn] = out.get(callee_fn, 0) + ir
    if events is None:
        raise ValueError("no events line: not a callgrind file")
    return out


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
    report = {path: attribution(pathlib.Path(path).read_text(errors="replace")) for path in args.files}
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
