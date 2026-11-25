#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Release lane kernels of this architecture hold no fused operation.

Emits the assembly of ``tessera-rl-classic``'s lane step and checks each
level's entry point: no fused mnemonic, the expected packed instructions, and
no call to a lane operation (they must inline into the ``target_feature``
function). Only the levels this architecture compiles are checked.

The integer and fixed-point lanes (#345) are checked the same way on the
``int_lanes_probe`` example of ``tessera-rl-simd``: each ``run_*_lanes``
entry point holds the 32x32->64 widening multiply of ``Fx32::saturating_mul``
and no call to a lane operation.
"""

from __future__ import annotations

import pathlib
import platform
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "target" / "simd-disasm"
LABEL = re.compile(r"^(\S+):")
CALL = re.compile(r"^\s*(?:bl|callq?)\s+(\S+)")

FUSED = {
    "aarch64": re.compile(r"^\s*f(?:n?m(?:add|sub)|ml[as])\b", re.M),
    "x86_64": re.compile(r"^\s*vf(?:n?m(?:add|sub)|m(?:addsub|subadd))\w*\b", re.M),
}
LEVELS = {
    "aarch64": {"run_neon": re.compile(r"^\s*fmul(?:\.2d\s|\s+v\d+\.2d)", re.M), "run_scalar": re.compile(r"^\s*fmul\b", re.M)},
    "x86_64": {
        "run_avx2": re.compile(r"^\s*vmulpd\b.*ymm", re.M),
        "run_sse42": re.compile(r"^\s*v?mulpd\b.*xmm", re.M),
        "run_scalar": re.compile(r"^\s*v?mul[sp]d\b", re.M),
    },
}

INT_LEVELS = {
    # The vector form only (Apple's `smull.2d v0, ...` or `smull v0.2d, ...`):
    # the scalar fallback also emits `smull x, w, w`.
    "aarch64": {"run_neon_lanes": re.compile(r"^\s*smull2?(?:\.2d\s|\s+v\d+\.2d)", re.M)},
    "x86_64": {
        "run_avx2_lanes": re.compile(r"^\s*vpmuldq\b.*ymm", re.M),
        "run_sse42_lanes": re.compile(r"^\s*v?pmuldq\b.*xmm", re.M),
    },
}


def functions(text: str) -> dict[str, str]:
    """Function bodies by symbol; local labels (``L...``) stay in the body."""
    out: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        m = LABEL.match(line)
        if m and not m.group(1).lstrip("_").startswith(("L", ".")):
            current = m.group(1)
            out[current] = []
        elif current is not None:
            out[current].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def int_problems(arch: str, text: str) -> list[str]:
    """What is wrong in the integer-lane probe's assembly; empty if nothing."""
    found = []
    bodies = functions(text)
    for level, widening in INT_LEVELS[arch].items():
        # Legacy symbol mangling drops the generic argument; the probe
        # binary instantiates each entry point once, for its one kernel.
        names = [name for name in bodies if level in name]
        if not names:
            found.append(f"{level}: no probe entry point in the assembly")
        for name in names:
            body = bodies[name]
            if not widening.search(body):
                found.append(f"{level}: widening multiply missing in {name}")
            for line in body.splitlines():
                call = CALL.match(line)
                if call and "tessera_rl" in call.group(1):
                    found.append(f"{level}: lane operation not inlined: {call.group(1)}")
    return found


def problems(arch: str, text: str) -> list[str]:
    """What is wrong in this assembly for ``arch``; empty if nothing."""
    found = []
    if m := FUSED[arch].search(text):
        found.append(f"fused operation: {m.group(0).strip()}")
    bodies = functions(text)
    for level, packed in LEVELS[arch].items():
        names = [name for name in bodies if level in name and "StepKernel" in name.replace("$LT$", "")]
        names = names or [name for name in bodies if level in name]
        if not names:
            found.append(f"{level}: no entry point in the assembly")
        for name in names:
            body = bodies[name]
            if not packed.search(body):
                found.append(f"{level}: expected instruction missing in {name}")
            for line in body.splitlines():
                call = CALL.match(line)
                if call and "tessera_rl_simd" in call.group(1):
                    found.append(f"{level}: lane operation not inlined: {call.group(1)}")
    return found


def main() -> int:
    arch = {"arm64": "aarch64", "amd64": "x86_64"}.get(platform.machine().lower(), platform.machine().lower())
    if arch not in LEVELS:
        sys.stdout.write(f"no lane levels on {arch}\n")
        return 0
    shutil.rmtree(OUT, ignore_errors=True)
    subprocess.run(
        ["cargo", "rustc", "-p", "tessera-rl-classic", "--release", "--lib", "--locked", "--target-dir", str(OUT), "--", "--emit=asm", "-Ccodegen-units=1"],
        cwd=ROOT,
        check=True,
    )
    files = sorted((OUT / "release" / "deps").glob("tessera_rl_classic-*.s"))
    if len(files) != 1:
        sys.stderr.write(f"expected one assembly file, found {files}\n")
        return 1
    found = problems(arch, files[0].read_text())
    subprocess.run(
        ["cargo", "rustc", "-p", "tessera-rl-simd", "--release", "--example", "int_lanes_probe", "--locked", "--target-dir", str(OUT), "--", "--emit=asm", "-Ccodegen-units=1"],
        cwd=ROOT,
        check=True,
    )
    probes = sorted((OUT / "release" / "examples").glob("int_lanes_probe-*.s"))
    if len(probes) != 1:
        sys.stderr.write(f"expected one probe assembly file, found {probes}\n")
        return 1
    found += int_problems(arch, probes[0].read_text())
    for p in found:
        sys.stderr.write(p + "\n")
    if not found:
        sys.stdout.write(f"{arch}: lane kernels checked\n")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
