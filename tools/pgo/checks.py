#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Writes one wheel's section 7 check results, tied to its sha256.

    checks.py --wheel W --out checks.json python_suite=0 num_tests=1 golden_anchor=0

A status is the exit status of that check; 0 is passed. decide.py reads this.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("statuses", nargs="+", metavar="name=status")
    args = parser.parse_args(argv)
    statuses = {}
    for item in args.statuses:
        name, _, value = item.partition("=")
        if not name or not value.isdigit():
            print(f"bad status {item!r}", file=sys.stderr)
            return 2
        statuses[name] = int(value)
    with open(args.wheel, "rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"wheel_sha256": sha, "checks": statuses}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
