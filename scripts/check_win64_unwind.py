#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""ADR-0041 on Windows: every boundary thunk and callout in the LINKED
extension module (``_native.pyd``) has unwind data, and it is the
template's (one stack allocation of 88 bytes, no frame register).

The .pyd exports only ``PyInit__native``, so the thunk addresses come from
the linker map. Build with it, for example::

    set RUSTFLAGS=-C link-arg=/MAP:target\\native.map
    maturin build --release

then::

    python scripts/check_win64_unwind.py --pyd <path>\\_native.pyd --map target\\native.map

The unwind records are read with ``llvm-readobj --unwind`` (``--readobj``;
the Rust ``llvm-tools`` component ships it). Exit 1 when a thunk has no
record, a record differs from the template's, or no thunk is found.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

# Mangled (v0) names of functions in `ffi::thunks`: the thunks, the
# callouts and the module initializer.
THUNK = re.compile(r"6thunks\d+\w+|^PyInit__native$")
TEMPLATE = {"codes": ["ALLOC_SMALL size=88"], "frame": "-"}


def parse_map(text: str) -> dict[str, int]:
    """Function name -> RVA, from an MSVC linker map."""
    base = re.search(r"Preferred load address is ([0-9a-fA-F]+)", text)
    if not base:
        raise ValueError("no preferred load address in the map")
    image_base = int(base.group(1), 16)
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*[0-9a-fA-F]{4}:[0-9a-fA-F]{8}\s+(\S+)\s+([0-9a-fA-F]{16})\s+f\b", line)
        if m and THUNK.search(m.group(1)):
            out[m.group(1)] = int(m.group(2), 16) - image_base
    return out


def parse_unwind(text: str, image_base: int = 0) -> dict[int, dict]:
    """Start RVA -> {codes, frame}, from ``llvm-readobj --unwind``."""
    out = {}
    for block in text.split("RuntimeFunction {")[1:]:
        start = re.search(r"StartAddress:[^\n]*\((0x[0-9a-fA-F]+)\)", block)
        if not start:
            continue
        addr = int(start.group(1), 16)
        if image_base and addr >= image_base:
            addr -= image_base
        codes = re.search(r"UnwindCodes \[(.*?)\]", block, re.S)
        frame = re.search(r"FrameRegister: (\S+)", block)
        out[addr] = {
            "codes": [re.sub(r"^0x[0-9a-fA-F]+:\s*", "", c.strip()) for c in (codes.group(1).strip().splitlines() if codes else []) if c.strip()],
            "frame": frame.group(1) if frame else None,
        }
    return out


def check(thunks: dict[str, int], records: dict[int, dict]) -> list[str]:
    if not thunks:
        return ["no thunk found in the map"]
    problems = []
    for name, rva in sorted(thunks.items(), key=lambda kv: kv[1]):
        r = records.get(rva)
        if r is None:
            problems.append(f"{name} (RVA 0x{rva:x}): no unwind record")
        elif r != TEMPLATE:
            problems.append(f"{name} (RVA 0x{rva:x}): unwind {r} differs from the template's {TEMPLATE}")
    return problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pyd", required=True)
    p.add_argument("--map", required=True)
    p.add_argument("--readobj", default="llvm-readobj")
    args = p.parse_args()
    map_text = open(args.map, encoding="utf-8", errors="replace").read()
    thunks = parse_map(map_text)
    image_base = int(re.search(r"Preferred load address is ([0-9a-fA-F]+)", map_text).group(1), 16)
    dump = subprocess.run([args.readobj, "--unwind", args.pyd], capture_output=True, text=True, check=True).stdout
    problems = check(thunks, parse_unwind(dump, image_base))
    for line in problems:
        print(line)
    if problems:
        return 1
    print(f"win64 unwind: {len(thunks)} thunks, each with the template's unwind record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
