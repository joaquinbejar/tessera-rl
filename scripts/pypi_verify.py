# SPDX-License-Identifier: MIT OR Apache-2.0
"""Checks the three wheels of one wheels run before they go to PyPI.

    python3 -I scripts/pypi_verify.py <artifacts-dir> <tag> <commit> <out-dir>

<artifacts-dir> holds one directory per artifact of the wheels run, named as
the artifact (as `gh run download` lays them out). <tag> is the release tag
(`v0.2.0`), <commit> the full sha the wheels run built and the tag names.

The run must have exactly the three artifacts of wheels.yml:

- `wheel-x86_64-unknown-linux-gnu-wheel-<commit>-<sha256>` (W1),
- `wheel-aarch64-unknown-linux-gnu`,
- `wheel-aarch64-apple-darwin`.

For each: exactly one wheel; its sha256 equals `wheel_sha256` in the
artifact's checks.json, and for W1 also the sha256 in the artifact's name;
every check in checks.json is 0; the file name is
`tessera_rl-<version>-cp311-abi3-<platform>.whl` with the target's platform
and <version> the tag without its `v`; and the wheel's own METADATA says
`Name: tessera-rl` and `Version: <version>`.

On success the three wheels are copied to <out-dir> with a SHA256SUMS file,
and the exit status is 0. Any failure prints every problem found and exits 1;
a usage error exits 2. Nothing is copied unless every check passes.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

TAG = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
COMMIT = re.compile(r"[0-9a-f]{40}")
SHA = re.compile(r"[0-9a-f]{64}")
W1 = re.compile(r"wheel-x86_64-unknown-linux-gnu-wheel-([0-9a-f]{40})-([0-9a-f]{64}|none)")
# Artifact name pattern (or exact name) -> platform tag pattern of its wheel.
PLATFORMS = {
    "x86_64-unknown-linux-gnu": r"manylinux_2_28_x86_64",
    "aarch64-unknown-linux-gnu": r"manylinux_2_28_aarch64",
    "aarch64-apple-darwin": r"macosx_[0-9]+_[0-9]+_arm64",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _metadata(wheel: Path, version: str) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as z:
        name = f"tessera_rl-{version}.dist-info/METADATA"
        if name not in z.namelist():
            return {}
        text = z.read(name).decode("utf-8")
    head = text.split("\n\n", 1)[0]
    out: dict[str, str] = {}
    for line in head.splitlines():
        key, sep, value = line.partition(": ")
        if sep and key in ("Name", "Version") and key not in out:
            out[key] = value.strip()
    return out


def check(artifacts: Path, tag: str, commit: str) -> tuple[list[str], list[Path]]:
    problems: list[str] = []
    wheels: list[Path] = []
    version = tag[1:]
    names = sorted(p.name for p in artifacts.iterdir() if p.is_dir())
    w1 = [n for n in names if W1.fullmatch(n)]
    expected = {"wheel-aarch64-unknown-linux-gnu", "wheel-aarch64-apple-darwin"}
    others = sorted(set(names) - set(w1) - expected)
    if len(w1) != 1:
        problems.append(f"expected one W1 artifact, found {w1}")
    for name in sorted(expected - set(names)):
        problems.append(f"missing artifact {name}")
    if others:
        problems.append(f"unexpected artifacts {others}")
    if problems:
        return problems, wheels

    for name in [w1[0], "wheel-aarch64-unknown-linux-gnu", "wheel-aarch64-apple-darwin"]:
        d = artifacts / name
        target = next(t for t in PLATFORMS if t in name)
        found = sorted(d.glob("*.whl"))
        if len(found) != 1:
            problems.append(f"{name}: expected one wheel, found {[p.name for p in found]}")
            continue
        wheel = found[0]
        sha = _sha256(wheel)
        try:
            checks = json.loads((d / "checks.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            problems.append(f"{name}: no readable checks.json ({e})")
            continue
        if checks.get("wheel_sha256") != sha:
            problems.append(f"{name}: wheel sha256 {sha} != checks.json {checks.get('wheel_sha256')}")
        statuses = checks.get("checks")
        if not isinstance(statuses, dict) or not statuses:
            problems.append(f"{name}: checks.json has no checks")
        else:
            failed = {k: v for k, v in statuses.items() if v != 0}
            if failed:
                problems.append(f"{name}: failed checks {failed}")
        if name == w1[0]:
            m = W1.fullmatch(name)
            assert m is not None
            if m.group(1) != commit:
                problems.append(f"{name}: built at {m.group(1)}, not {commit}")
            if m.group(2) != sha:
                problems.append(f"{name}: wheel sha256 {sha} != the artifact name's {m.group(2)}")
        pattern = rf"tessera_rl-{re.escape(version)}-cp311-abi3-{PLATFORMS[target]}\.whl"
        if not re.fullmatch(pattern, wheel.name):
            problems.append(f"{name}: wheel {wheel.name} does not match {pattern}")
        try:
            meta = _metadata(wheel, version)
        except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as e:
            problems.append(f"{name}: unreadable wheel ({e})")
            continue
        if meta.get("Name") != "tessera-rl" or meta.get("Version") != version:
            problems.append(f"{name}: METADATA says {meta}, expected tessera-rl {version}")
        wheels.append(wheel)
    return problems, wheels


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    artifacts, tag, commit, out = Path(argv[0]), argv[1], argv[2], Path(argv[3])
    if not TAG.fullmatch(tag) or not COMMIT.fullmatch(commit) or not artifacts.is_dir():
        print(f"usage: bad tag {tag!r}, commit {commit!r} or directory {str(artifacts)!r}", file=sys.stderr)
        return 2
    if out.exists():
        print(f"usage: {out} already exists", file=sys.stderr)
        return 2
    problems, wheels = check(artifacts, tag, commit)
    for p in problems:
        print(f"pypi_verify: {p}", file=sys.stderr)
    if problems:
        return 1
    out.mkdir(parents=True)
    lines = []
    for wheel in wheels:
        shutil.copyfile(wheel, out / wheel.name)
        sha = _sha256(out / wheel.name)
        lines.append(f"{sha}  {wheel.name}\n")
        print(f"pypi_verify: {wheel.name} {sha}")
    (out / "SHA256SUMS").write_text("".join(sorted(lines, key=lambda l: l.split()[1])), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
