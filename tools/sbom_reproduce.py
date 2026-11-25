#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Makes a wheel's SBOM reproducible (#654), in place.

    sbom_reproduce.py --wheel W --name N --epoch E [--optional]

maturin writes the CycloneDX SBOM under ``.dist-info/sboms/`` with a random
``serialNumber`` and the build time as ``metadata.timestamp``, so two builds
of one commit differ only there. This sets:

- ``serialNumber`` to ``urn:uuid:`` plus UUIDv5(NAMESPACE_URL, N); N names the
  commit, target, profile and features, so no two wheels share one;
- ``metadata.timestamp`` to E (seconds since the epoch) as
  ``YYYY-MM-DDTHH:MM:SSZ`` in UTC.

Only those two values change in the SBOM's text, and each must match exactly
once. RECORD's line for the SBOM gets the new sha256 and size. The wheel is
then rewritten member by member, in order, with each member's name, date,
attributes, creating system and compression method kept and its bytes
recompressed with zlib at a fixed level; the uncompressed bytes of every
other member are unchanged. A second run gives the same bytes.

A wheel without exactly one SBOM is an error, unless ``--optional`` is given
and it has none, in which case the wheel is left unchanged. Exit status 0 on
success, 1 on a wheel it cannot process, 2 on bad arguments.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import sys
import tempfile
import uuid
import zipfile
import zlib
from pathlib import Path

SBOM_PATH = re.compile(r"[^/]+\.dist-info/sboms/[^/]+\.cyclonedx\.json")
SERIAL = re.compile(rb'("serialNumber"\s*:\s*")urn:uuid:[0-9A-Fa-f-]{36}(")')
TIMESTAMP = re.compile(rb'("timestamp"\s*:\s*")[^"]*(")')
LEVEL = 9


class Unprocessable(Exception):
    """The wheel cannot be processed as asked."""


def serial(name: str) -> str:
    return f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, name)}"


def timestamp(epoch: int) -> str:
    return datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")


def _replace_once(pattern: re.Pattern[bytes], text: bytes, value: str, what: str) -> bytes:
    new, count = pattern.subn(lambda m: m.group(1) + value.encode("ascii") + m.group(2), text)
    if count != 1:
        raise Unprocessable(f"the SBOM's {what} matched {count} times, expected once")
    return new


def rewrite_sbom(text: bytes, name: str, epoch: int) -> bytes:
    text = _replace_once(SERIAL, text, serial(name), "serialNumber")
    text = _replace_once(TIMESTAMP, text, timestamp(epoch), "timestamp")
    bom = json.loads(text)
    if bom.get("serialNumber") != serial(name) or (bom.get("metadata") or {}).get("timestamp") != timestamp(epoch):
        raise Unprocessable("the edited SBOM does not carry the new serialNumber and metadata.timestamp")
    return text


def rewrite_record(record: bytes, sbom: str, data: bytes) -> bytes:
    lines = record.decode("utf-8").splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.startswith(sbom + ",")]
    if len(hits) != 1:
        raise Unprocessable(f"RECORD has {len(hits)} lines for {sbom}, expected one")
    i = hits[0]
    end = lines[i][len(lines[i].rstrip("\r\n")):]
    lines[i] = f"{sbom},{record_hash(data)},{len(data)}{end}"
    return "".join(lines).encode("utf-8")


def reproduce(wheel: Path, name: str, epoch: int, optional: bool = False) -> bool:
    """Rewrites ``wheel``; returns False if it has no SBOM and ``optional`` is set."""
    with zipfile.ZipFile(wheel) as z:
        infos = z.infolist()
        sboms = [i.filename for i in infos if SBOM_PATH.fullmatch(i.filename)]
        if not sboms and optional:
            return False
        if len(sboms) != 1:
            raise Unprocessable(f"expected one CycloneDX SBOM under .dist-info/sboms/, found {sboms}")
        sbom = sboms[0]
        record = sbom.split("/", 1)[0] + "/RECORD"
        contents = {i.filename: z.read(i) for i in infos}
    if record not in contents:
        raise Unprocessable(f"no {record}")
    contents[sbom] = rewrite_sbom(contents[sbom], name, epoch)
    contents[record] = rewrite_record(contents[record], sbom, contents[sbom])
    fd, tmp = tempfile.mkstemp(dir=wheel.parent, prefix=f".{wheel.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f, zipfile.ZipFile(f, "w") as out:
            for old in infos:
                info = zipfile.ZipInfo(old.filename, date_time=old.date_time)
                info.external_attr = old.external_attr
                info.internal_attr = old.internal_attr
                info.create_system = old.create_system
                info.compress_type = old.compress_type
                out.writestr(info, contents[old.filename], compresslevel=LEVEL)
        os.replace(tmp, wheel)
    except BaseException:
        os.unlink(tmp)
        raise
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epoch", required=True)
    parser.add_argument("--optional", action="store_true", help="leave a wheel without an SBOM unchanged")
    args = parser.parse_args(argv)
    if not args.name:
        print("sbom_reproduce: --name is empty", file=sys.stderr)
        return 2
    if not re.fullmatch(r"[0-9]+", args.epoch):
        print(f"sbom_reproduce: --epoch must be decimal seconds, got {args.epoch!r}", file=sys.stderr)
        return 2
    epoch = int(args.epoch)
    try:
        done = reproduce(args.wheel, args.name, epoch, args.optional)
    except (Unprocessable, zipfile.BadZipFile, ValueError, OSError) as e:
        print(f"sbom_reproduce: {args.wheel.name}: {e}", file=sys.stderr)
        return 1
    if not done:
        print(f"sbom_reproduce: {args.wheel.name}: no SBOM; left unchanged")
        return 0
    print(f"sbom_reproduce: {args.wheel.name}: serialNumber {serial(args.name)} (from {args.name}), "
          f"timestamp {timestamp(epoch)}; zlib {zlib.ZLIB_RUNTIME_VERSION} level {LEVEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
