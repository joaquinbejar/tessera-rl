#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Checks the SBOM a wheel carries (SPEC 20.1, #350; PEP 770).

    sbom_check.py --wheel W --crates crates.txt

maturin writes a CycloneDX SBOM of the extension's Rust dependencies into
the wheel's ``.dist-info/sboms/`` (``[tool.maturin.sbom]``). This checks
that exactly one CycloneDX SBOM is there, that its root component has the
wheel's version, and that it lists every crate of ``crates.txt`` (one name
per line, the extension's normal dependency tree as ``cargo tree`` prints
it for the wheel's features). Exit status 0 if so, 1 with the problems on
stderr otherwise, 2 on bad arguments.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

WHEEL_NAME = re.compile(r"(?P<dist>[^-]+)-(?P<version>[^-]+)-.+\.whl")


def problems(wheel: Path, crates: set[str]) -> list[str]:
    m = WHEEL_NAME.fullmatch(wheel.name)
    if not m:
        return [f"{wheel.name}: not a wheel file name"]
    version = m.group("version")
    with zipfile.ZipFile(wheel) as z:
        sboms = [n for n in z.namelist() if re.fullmatch(r"[^/]+\.dist-info/sboms/[^/]+\.cyclonedx\.json", n)]
        if len(sboms) != 1:
            return [f"{wheel.name}: expected one CycloneDX SBOM under .dist-info/sboms/, found {sboms}"]
        try:
            bom = json.loads(z.read(sboms[0]))
        except (ValueError, KeyError) as e:
            return [f"{sboms[0]}: unreadable ({e})"]
    found: list[str] = []
    if bom.get("bomFormat") != "CycloneDX":
        found.append(f"{sboms[0]}: bomFormat {bom.get('bomFormat')!r}, expected 'CycloneDX'")
    root = (bom.get("metadata") or {}).get("component") or {}
    if root.get("version") != version:
        found.append(f"{sboms[0]}: root component version {root.get('version')!r}, expected {version!r}")
    listed = {c.get("name") for c in bom.get("components") or []} | {root.get("name")}
    missing = sorted(crates - listed)
    if missing:
        found.append(f"{sboms[0]}: crates missing from the SBOM: {missing}")
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--crates", required=True, type=Path)
    args = parser.parse_args(argv)
    crates = {line.strip() for line in args.crates.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not crates:
        print("sbom_check: the crate list is empty", file=sys.stderr)
        return 2
    found = problems(args.wheel, crates)
    for p in found:
        print(f"sbom_check: {p}", file=sys.stderr)
    if not found:
        print(f"sbom_check: {args.wheel.name}: one CycloneDX SBOM, {len(crates)} crates listed")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
