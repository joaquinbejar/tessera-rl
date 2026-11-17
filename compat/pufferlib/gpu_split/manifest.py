# SPDX-License-Identifier: MIT OR Apache-2.0
"""The attempt manifest of ADR-0061 item 9, fixed before the timed rounds.

It holds the commit, the full sha256 of the Tessera wheel chosen under
ADR-0061 item 3, the CUDA toolkit version and the case order of every round.
Its own sha256 is sent to the reviewer before the timed rounds start, and the
driver refuses a manifest whose sha256 is not that one.

Usage::

    python manifest.py --out manifest.json --commit SHA --wheel PATH --wheel-sha256 HEX \\
        --wheel-source pypi-release|main-wheels-run --toolkit 13.0.88 [--nvcc nvcc] [--rounds 5] \\
        --clock-ref-host NAME --clock-ref-command CMD [--clock-cpu N] \\
        [--rerun-of-manifest PATH --rerun-of-sha256 HEX]

The clock fields (ADR-0063, docs/reviews/152-adr0063-harness-plan.md §2.1a):
the sha256 of ``clock.py`` and ``clock_ref.py`` in this checkout, the
reference host and the exact command that starts ``clock_ref.py`` there, the
sampler's guest core (``null`` for none) and the cases' cores (``null``: not
pinned). ``rerun_of`` is null for a first session; for a rerun after a void
session it names that session's manifest sha256 and this session's position
(2 or 3), and the earlier manifest must have the same commit, wheel, toolkit
and order.

The wheel file is hashed and must equal ``--wheel-sha256``. ``nvcc --version``
must report ``--toolkit``. Either mismatch writes nothing and exits 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile

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


HERE = os.path.dirname(os.path.abspath(__file__))
SAME_ON_RERUN = ("commit", "wheel_sha256", "cuda_toolkit", "order")


def rerun_of(earlier: dict, earlier_sha256: str) -> dict:
    """The ``rerun_of`` field of a rerun after the void session ``earlier``."""
    prev = earlier.get("rerun_of")
    position = 2 if prev is None else prev["position"] + 1
    if position > 3:
        raise ValueError("a third void session closes the attempt (ADR-0063 item 5(iv))")
    return {"manifest_sha256": earlier_sha256, "position": position}


def build(commit, wheel, wheel_sha256, wheel_source, toolkit, nvcc, cases, modes, rounds,
          clock_ref_host=None, clock_ref_command=None, clock_cpu=None, earlier=None) -> dict:  # fmt: skip
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
    if not clock_ref_host or not clock_ref_command:
        raise ValueError("the clock reference host and command are required (ADR-0063 item 2)")
    m = {
        "adr": "ADR-0061",
        "commit": commit,
        "wheel_sha256": wheel_sha256,
        "wheel_source": wheel_source,
        "native_sha256": wheel_native_sha256(wheel),
        "cuda_toolkit": toolkit,
        "rounds": rounds,
        "order": order(cases, modes, rounds),
        "clock_py_sha256": file_sha256(os.path.join(HERE, "clock.py")),
        "clock_ref_py_sha256": file_sha256(os.path.join(HERE, "clock_ref.py")),
        "clock_ref_host": clock_ref_host,
        "clock_ref_command": clock_ref_command,
        "clock_cpu": clock_cpu,
        "case_cpus": None,
        "rerun_of": None,
    }
    if earlier is not None:
        data, digest = earlier
        for k in SAME_ON_RERUN:
            if data.get(k) != m[k]:
                raise ValueError(f"a rerun must keep the void session's {k}")
        m["rerun_of"] = rerun_of(data, digest)
    return m


def wheel_native_sha256(wheel: str) -> str:
    """The sha256 of the wheel's ``tessera_rl/_native`` extension module,
    compared with the one ``split.py`` reports from the installed package."""
    with zipfile.ZipFile(wheel) as z:
        names = [n for n in z.namelist() if re.fullmatch(r"tessera_rl/_native\.[^/]*(so|pyd)", n)]
        if len(names) != 1:
            raise ValueError("the wheel has no single tessera_rl/_native module")
        return hashlib.sha256(z.read(names[0])).hexdigest()


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
    p.add_argument("--clock-ref-host", required=True)
    p.add_argument("--clock-ref-command", required=True)
    p.add_argument("--clock-cpu", type=int)
    p.add_argument("--rerun-of-manifest")
    p.add_argument("--rerun-of-sha256")
    args = p.parse_args(argv)
    if bool(args.rerun_of_manifest) != bool(args.rerun_of_sha256):
        p.error("--rerun-of-manifest and --rerun-of-sha256 go together")
    try:
        earlier = None
        if args.rerun_of_manifest:
            earlier = (load(args.rerun_of_manifest, args.rerun_of_sha256), args.rerun_of_sha256)
        m = build(args.commit, args.wheel, args.wheel_sha256, args.wheel_source, args.toolkit, args.nvcc,
                  CASES, MODES, args.rounds, args.clock_ref_host, args.clock_ref_command, args.clock_cpu,
                  earlier)  # fmt: skip
    except (ValueError, OSError, zipfile.BadZipFile) as e:
        print(f"REFUSED: {e}", flush=True)
        return EXIT_MISMATCH
    data = canonical(m)
    with open(args.out, "wb") as f:
        f.write(data)
    print(hashlib.sha256(data).hexdigest(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
