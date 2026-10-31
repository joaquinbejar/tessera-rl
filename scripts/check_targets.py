#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Checks an installed Python environment against compat/targets.toml.

A profile (`[profile.<name>]`) names an environment and splits its targets
into `required` and `deferred`. Every installed target must have exactly
its pinned version; a required target that is missing fails the check, a
deferred one is only listed. Exit status 1 on any failure.

Usage: python scripts/check_targets.py [profile]   (default: conformance)
"""

import importlib.metadata
import pathlib
import sys
import tomllib


def check(targets, profile_name):
    """Returns (ok, lines) for `profile_name`; raises KeyError if unknown."""
    profile = targets["profile"][profile_name]
    pins = targets["env"][profile["env"]]
    lines = []
    minimum = tuple(int(x) for x in targets["python"]["minimum"].split("."))
    ok = sys.version_info[:2] >= minimum
    lines.append("python %d.%d (minimum %s): %s" % (*sys.version_info[:2], targets["python"]["minimum"], "ok" if ok else "TOO OLD"))
    unknown = [p for p in profile["required"] + profile["deferred"] if p not in pins]
    for p in unknown:
        lines.append("%s: in profile %s but not pinned in env %s, FAIL" % (p, profile_name, profile["env"]))
        ok = False
    for package in profile["required"] + profile["deferred"]:
        if package not in pins:
            continue
        pinned = pins[package]
        required = package in profile["required"]
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            lines.append("%s: not installed (pinned %s, %s)%s" % (package, pinned, "required" if required else "deferred", ", FAIL" if required else ""))
            ok &= not required
            continue
        # Local version labels (for example torch's +cpu) are not part of the pin.
        match = installed.split("+")[0] == pinned
        ok &= match
        lines.append("%s: %s %s" % (package, installed, "ok" if match else "MISMATCH, pinned %s" % pinned))
    return ok, lines


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    targets = tomllib.loads((root / "compat" / "targets.toml").read_text())
    name = sys.argv[1] if len(sys.argv) > 1 else "conformance"
    try:
        ok, lines = check(targets, name)
    except KeyError:
        print("unknown profile %r; known: %s" % (name, sorted(targets["profile"])))
        sys.exit(2)
    print("profile %s (env %s)" % (name, targets["profile"][name]["env"]))
    for line in lines:
        print(line)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
