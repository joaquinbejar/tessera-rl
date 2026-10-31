#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""ADR-0041: every C entry point of tessera-rl-py goes through a boundary thunk.

The primary guarantee is the compiler's: C tables are built only by
`ffi::tables`, whose function rows take a `ThunkFn`, which only
`ffi::thunks` can make. This script is the secondary check. It fails when
pyo3-macros is a dependency, a PyO3 registration attribute or macro
appears, a thunk or `ThunkFn` is made outside `ffi/thunks.rs`, or a C table
row (`PyMethodDef`, `PyType_Slot`) is built outside `ffi/tables.rs`.
Parsing is whitespace-insensitive, so rustfmt's line breaks do not hide an
entry.
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
        name = path.name
        if name != "thunks.rs" and re.search(r"\bthunk!|\bThunkFn ?\(|naked_asm!|#\[unsafe\(naked\)\]", flat):
            problems.append(f"{rel}: thunks and ThunkFn values are made only in ffi/thunks.rs")
        if name != "tables.rs" and re.search(r"\b(?:ffi::)?(?:PyMethodDef|PyType_Slot) ?\{", flat):
            problems.append(f"{rel}: C table rows are built only in ffi/tables.rs")
    for p in problems:
        print(p)
    if problems:
        sys.exit(1)
    print("py entries: every registration goes through a boundary thunk")


if __name__ == "__main__":
    main()
