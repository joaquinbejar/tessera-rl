#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""ADR-0041: every C entry point of tessera-rl-py goes through a boundary thunk.

Fails when pyo3-macros is a dependency, a PyO3 registration attribute or
macro appears, or a C table entry (a `method(...)` row, a PyMethodDefPointer
field, a function slot) names anything but a `thunk_*` function. Parsing is
whitespace-insensitive, so rustfmt's line breaks do not hide an entry.
"""

import pathlib
import re
import subprocess
import sys

SRC = pathlib.Path(__file__).resolve().parent.parent / "crates" / "tessera-rl-py" / "src"


def main():
    problems = []
    tree = subprocess.run(
        ["cargo", "tree", "-q", "-p", "tessera-rl-py", "-e", "normal", "--prefix", "none"],
        capture_output=True, text=True, check=True,
    ).stdout
    if any(line.startswith("pyo3-macros ") for line in tree.splitlines()):
        problems.append("pyo3-macros is a dependency of tessera-rl-py")
    for path in sorted(SRC.rglob("*.rs")):
        text = path.read_text()
        flat = re.sub(r"\s+", " ", text)
        rel = path.relative_to(SRC.parent.parent.parent)
        if re.search(r"#\[(pyclass|pymethods|pymodule|pyfunction)|wrap_pyfunction|wrap_pymodule", text):
            problems.append(f"{rel}: PyO3 registration macros are not allowed")
        for fn in re.findall(r'\bmethod\( ?c"[^"]*", ?([A-Za-z_][A-Za-z0-9_]*)', flat):
            if not fn.startswith("thunk_"):
                problems.append(f"{rel}: method table entry {fn} is not a boundary thunk")
        for fn in re.findall(r"\b(?:PyCFunction|PyCFunctionWithKeywords): ?([A-Za-z_][A-Za-z0-9_]*)", flat):
            # `f` is the parameter of the `method` constructor itself.
            if not fn.startswith("thunk_") and fn != "f":
                problems.append(f"{rel}: PyMethodDef points at {fn}, not a boundary thunk")
        for fn in re.findall(r"\bpfunc: ?([A-Za-z_][A-Za-z0-9_]*) as \*mut", flat):
            if not fn.startswith("thunk_"):
                problems.append(f"{rel}: type slot points at {fn}, not a boundary thunk")
    for p in problems:
        print(p)
    if problems:
        sys.exit(1)
    print("py entries: every registration goes through a boundary thunk")


if __name__ == "__main__":
    main()
