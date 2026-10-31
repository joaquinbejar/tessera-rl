#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Checks an installed Python environment against compat/targets.toml.

Every package of the chosen environment (default `main`) that is installed
must have exactly the pinned version. Packages that are not installed are
listed, and fail the check with `--require-all`. Exit status 1 on any
failure.

Usage: python scripts/check_targets.py [--require-all] [env]
"""

import importlib.metadata
import pathlib
import sys
import tomllib


def main():
    args = [a for a in sys.argv[1:] if a != "--require-all"]
    require_all = "--require-all" in sys.argv[1:]
    env = args[0] if args else "main"
    root = pathlib.Path(__file__).resolve().parent.parent
    targets = tomllib.loads((root / "compat" / "targets.toml").read_text())
    minimum = tuple(int(x) for x in targets["python"]["minimum"].split("."))
    ok = sys.version_info[:2] >= minimum
    print("python %d.%d (minimum %s): %s" % (*sys.version_info[:2], targets["python"]["minimum"], "ok" if ok else "TOO OLD"))
    for package, pinned in targets["env"][env].items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            print("%s: not installed (pinned %s)%s" % (package, pinned, ", FAIL" if require_all else ""))
            ok &= not require_all
            continue
        # Local version labels (for example torch's +cpu) are not part of the pin.
        match = installed.split("+")[0] == pinned
        ok &= match
        print("%s: %s %s" % (package, installed, "ok" if match else "MISMATCH, pinned %s" % pinned))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
