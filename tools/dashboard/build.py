# SPDX-License-Identifier: MIT OR Apache-2.0
"""Builds the static benchmark dashboard from ``benchmarks/results``.

``--check`` parses and judges the committed files without writing anything.
Output: ``target/dashboard``. This tool runs no benchmark and publishes
nothing.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import render
import status as st

ROOT = pathlib.Path(__file__).resolve().parents[2]


def inputs(root: pathlib.Path) -> tuple[list[st.Entry], list[str], dict]:
    configs = tomllib.loads((root / "tools" / "bench" / "configs.toml").read_text(encoding="utf-8"))
    machines = tomllib.loads((root / "compat" / "machines.toml").read_text(encoding="utf-8"))
    names = sorted(c["name"] for c in configs["config"])
    return st.load(root), names, {k: v for k, v in machines.items() if isinstance(v, dict)}


def build(root: pathlib.Path) -> pathlib.Path:
    entries, configs, machines = inputs(root)
    out = root / "target" / "dashboard"
    if out.exists():
        shutil.rmtree(out)
    render.write(out, root, entries, configs, machines)
    return out


def check(root: pathlib.Path) -> int:
    """Non-zero if any committed file is invalid."""
    entries, _, _ = inputs(root)
    bad = [e for e in entries if e.status == st.INVALID]
    for e in bad:
        sys.stderr.write(f"{e.name}: {'; '.join(e.reasons)}\n")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check(ROOT)
    sys.stdout.write(f"{build(ROOT)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
