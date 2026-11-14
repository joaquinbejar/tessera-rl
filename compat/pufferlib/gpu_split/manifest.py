# SPDX-License-Identifier: MIT OR Apache-2.0
"""The attempt manifest of ADR-0061 item 9, fixed before the timed rounds.

It holds the commit, the full sha256 of the Tessera wheel chosen under
ADR-0061 item 3, the CUDA toolkit version and the case order of every round.
Its own sha256 is sent to the reviewer before the timed rounds start, and the
driver refuses a manifest whose sha256 is not that one.

Usage::

    python manifest.py --out manifest.json --commit SHA --wheel PATH --wheel-sha256 HEX \\
        --wheel-source pypi-release|main-wheels-run --toolkit 13.0.88 [--nvcc nvcc] [--rounds 5]

The wheel file is hashed and must equal ``--wheel-sha256``. ``nvcc --version``
must report ``--toolkit``. Either mismatch writes nothing and exits 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys

CASES = ("ocean-serial", "ocean-mp", "tessera-w1", "tessera-w2")
MODES = ("e2e", "sync", "env")
WHEEL_SOURCES = ("pypi-release", "main-wheels-run")
EXIT_MISMATCH = 2


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rotated(items, r: int) -> list:
    """``items`` rotated left by ``r`` positions."""
    k = r % len(items)
    return [*items[k:], *items[:k]]


def order(cases, modes, rounds: int) -> list[list[list[str]]]:
    """ADR-0061 item 5: in every round the cases run interleaved, mode by
    mode, in an order rotated by one position per round."""
    return [[[c, m] for m in modes for c in rotated(list(cases), r)] for r in range(rounds)]


def toolkit_version(nvcc: str) -> str | None:
    """The ``V<x.y.z>`` that ``nvcc --version`` prints, or None."""
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"\bV(\d+\.\d+\.\d+)\b", out)
    return m.group(1) if m else None


def canonical(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def build(commit, wheel, wheel_sha256, wheel_source, toolkit, nvcc, cases, modes, rounds) -> dict:
    """The manifest, or ValueError naming the first mismatch."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit must be a full 40-character sha")
    if not re.fullmatch(r"[0-9a-f]{64}", wheel_sha256):
        raise ValueError("wheel sha256 must be 64 lowercase hex characters")
    if wheel_source not in WHEEL_SOURCES:
        raise ValueError(f"wheel source must be one of {WHEEL_SOURCES}")
    actual = file_sha256(wheel)
    if actual != wheel_sha256:
        raise ValueError(f"wheel sha256 {actual} is not the declared {wheel_sha256}")
    found = toolkit_version(nvcc)
    if found != toolkit:
        raise ValueError(f"nvcc reports toolkit {found}, not the declared {toolkit}")
    return {
        "adr": "ADR-0061",
        "commit": commit,
        "wheel_sha256": wheel_sha256,
        "wheel_source": wheel_source,
        "cuda_toolkit": toolkit,
        "rounds": rounds,
        "order": order(cases, modes, rounds),
    }


def load(path: str, expected_sha256: str) -> dict:
    """The manifest at ``path``; ValueError unless its sha256 is the one sent
    to the reviewer."""
    with open(path, "rb") as f:
        data = f.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"manifest sha256 {actual} is not the declared {expected_sha256}")
    return json.loads(data)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--wheel", required=True)
    p.add_argument("--wheel-sha256", required=True)
    p.add_argument("--wheel-source", required=True, choices=WHEEL_SOURCES)
    p.add_argument("--toolkit", required=True)
    p.add_argument("--nvcc", default="nvcc")
    p.add_argument("--rounds", type=int, default=5)
    args = p.parse_args(argv)
    try:
        m = build(args.commit, args.wheel, args.wheel_sha256, args.wheel_source, args.toolkit, args.nvcc,
                  CASES, MODES, args.rounds)  # fmt: skip
    except (ValueError, OSError) as e:
        print(f"REFUSED: {e}", flush=True)
        return EXIT_MISMATCH
    data = canonical(m)
    with open(args.out, "wb") as f:
        f.write(data)
    print(hashlib.sha256(data).hexdigest(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
