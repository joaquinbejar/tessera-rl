#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Checks the SBOM a wheel carries (SPEC 20.1, #350; PEP 770).

    sbom_check.py --wheel W --crates crates.txt [--serial-name N --epoch E]

maturin writes a CycloneDX SBOM of the extension's Rust dependencies into
the wheel's ``.dist-info/sboms/`` (``[tool.maturin.sbom]``). This checks
that exactly one CycloneDX SBOM is there, that its root component has the
wheel's version, and that it lists every crate of ``crates.txt`` (one name
per line, the extension's normal dependency tree as ``cargo tree`` prints
it for the wheel's features), and that RECORD's line for the SBOM matches its
bytes (sha256 and size). With ``--serial-name N --epoch E``, as
``tools/sbom_reproduce.py`` was given them (#654), it also checks that
``serialNumber`` is ``urn:uuid:`` plus UUIDv5(NAMESPACE_URL, N) and that
``metadata.timestamp`` is E in UTC. Exit status 0 if so, 1 with the problems on
stderr otherwise, 2 on bad arguments.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import re
import sys
import uuid
import zipfile
from pathlib import Path

WHEEL_NAME = re.compile(r"(?P<dist>[^-]+)-(?P<version>[^-]+)-.+\.whl")


def _record_problems(record: bytes, sbom: str, data: bytes) -> list[str]:
    lines = [line for line in record.decode("utf-8").splitlines() if line.startswith(sbom + ",")]
    if len(lines) != 1:
        return [f"RECORD: {len(lines)} lines for {sbom}, expected one"]
    digest = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    if lines[0] != f"{sbom},{digest},{len(data)}":
        return [f"RECORD: {lines[0]!r} does not match the SBOM's bytes ({digest},{len(data)})"]
    return []


def problems(wheel: Path, crates: set[str], serial_name: str | None = None, epoch: int | None = None) -> list[str]:
    m = WHEEL_NAME.fullmatch(wheel.name)
    if not m:
        return [f"{wheel.name}: not a wheel file name"]
    version = m.group("version")
    with zipfile.ZipFile(wheel) as z:
        sboms = [n for n in z.namelist() if re.fullmatch(r"[^/]+\.dist-info/sboms/[^/]+\.cyclonedx\.json", n)]
        if len(sboms) != 1:
            return [f"{wheel.name}: expected one CycloneDX SBOM under .dist-info/sboms/, found {sboms}"]
        data = z.read(sboms[0])
        try:
            bom = json.loads(data)
        except (ValueError, KeyError) as e:
            return [f"{sboms[0]}: unreadable ({e})"]
        record_name = sboms[0].split("/", 1)[0] + "/RECORD"
        record = z.read(record_name) if record_name in z.namelist() else None
    found: list[str] = []
    if record is None:
        found.append(f"{wheel.name}: no {record_name}")
    else:
        found += _record_problems(record, sboms[0], data)
    if serial_name is not None:
        want = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, serial_name)}"
        if bom.get("serialNumber") != want:
            found.append(f"{sboms[0]}: serialNumber {bom.get('serialNumber')!r}, expected {want!r} (from {serial_name!r})")
    if epoch is not None:
        want = datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        got = (bom.get("metadata") or {}).get("timestamp")
        if got != want:
            found.append(f"{sboms[0]}: metadata.timestamp {got!r}, expected {want!r}")
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
    parser.add_argument("--serial-name", help="the name sbom_reproduce.py derived the serialNumber from")
    parser.add_argument("--epoch", help="the SOURCE_DATE_EPOCH sbom_reproduce.py was given")
    args = parser.parse_args(argv)
    if (args.serial_name is None) != (args.epoch is None):
        print("sbom_check: --serial-name and --epoch go together", file=sys.stderr)
        return 2
    if args.serial_name is not None and (not args.serial_name or not re.fullmatch(r"[0-9]+", args.epoch)):
        print("sbom_check: --serial-name must be non-empty and --epoch decimal seconds", file=sys.stderr)
        return 2
    crates = {line.strip() for line in args.crates.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not crates:
        print("sbom_check: the crate list is empty", file=sys.stderr)
        return 2
    epoch = None if args.epoch is None else int(args.epoch)
    found = problems(args.wheel, crates, args.serial_name, epoch)
    for p in found:
        print(f"sbom_check: {p}", file=sys.stderr)
    if not found:
        extra = "" if args.serial_name is None else ", serialNumber and timestamp as reproduced"
        print(f"sbom_check: {args.wheel.name}: one CycloneDX SBOM, {len(crates)} crates listed, RECORD matches{extra}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
