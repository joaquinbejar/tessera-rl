#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""ADR-0041: every C entry point of tessera-rl-py goes through a boundary thunk.

The shared boundary crate, tessera-rl-pyffi (ADR-0062 item 8), is checked
with the same rules, and must expose no public unsafe function: no item of
its public API relies on its caller for memory safety (#272 step 3), and
clippy::not_unsafe_ptr_arg_deref is never allowed or expected there. The
rustdoc-JSON lint of the same rule is scripts/check_pyffi_api.py.
In every crate scanned, a `pub safe fn` (or `static`) in an `unsafe
extern` block fails too: a public foreign function made safe to call.

The primary guarantee is the compiler's: C tables are built only by
`ffi::tables`, whose function rows take a `ThunkFn`, which only
`ffi::thunks` can make. This script is a secondary lint over the source
text, not a proof. It fails when
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

import rust_lex

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYFFI = ROOT / "crates" / "tessera-rl-pyffi" / "src"
SRCS = [ROOT / "crates" / "tessera-rl-py" / "src", PYFFI]
# `pub` (not `pub(crate)` or `pub(super)`) `unsafe fn`, `unsafe extern fn`
# (with or without an ABI string) or `unsafe trait`, with any `const` or
# `async`, in code stripped of comments and literals (scripts/rust_lex.py).
PUB_UNSAFE = re.compile(r'\bpub\s+(?:(?:const|async)\s+)*unsafe\s+(?:extern\s*(?:""\s*)?)?(?:fn|trait)\b')


# An `unsafe fn` declared in a `pub trait` (not `pub(crate)`): implementers
# and callers of a public trait reach it, whatever its own visibility says.
PUB_TRAIT = re.compile(r"\bpub\s+(?:unsafe\s+)?(?:auto\s+)?trait\s+\w+")
UNSAFE_FN = re.compile(r'\bunsafe\s+(?:extern\s*(?:""\s*)?)?fn\b')

# A `pub safe fn` (or `static`) declared in an `unsafe extern` block: a
# public, safe-to-call foreign function, which no lint can check.
UNSAFE_EXTERN = re.compile(r'\bunsafe\s+extern\s*(?:""\s*)?\{')
PUB_SAFE = re.compile(r"\bpub\s+safe\s+(?:fn|static)\b")


def bodies(code: str, header: re.Pattern):
    """The bodies (`{...}`, brace depth matched) of the items whose header
    matches `header`, in stripped code."""
    for m in header.finditer(code):
        brace = m.end() - 1 if code[m.end() - 1] == "{" else rust_lex.item_brace(code, m.end())
        if brace >= 0:
            yield code[brace:rust_lex.block_end(code, brace)]


def pub_unsafe(text: str) -> bool:
    code = rust_lex.strip(text)
    if PUB_UNSAFE.search(code):
        return True
    return any(UNSAFE_FN.search(body) for body in bodies(code, PUB_TRAIT))


def pub_safe_foreign(text: str) -> bool:
    code = rust_lex.strip(text)
    return any(PUB_SAFE.search(body) for body in bodies(code, UNSAFE_EXTERN))


def main(srcs=None, pyffi=None):
    srcs = SRCS if srcs is None else srcs
    pyffi = PYFFI if pyffi is None else pyffi
    problems = []
    tree = subprocess.run(
        ["cargo", "tree", "-q", "-p", "tessera-rl-py", "-e", "normal", "--prefix", "none"],
        capture_output=True, text=True, check=True,
    ).stdout
    if any(line.startswith("pyo3-macros ") for line in tree.splitlines()):
        problems.append("pyo3-macros is a dependency of tessera-rl-py")
    for path in sorted(p for src in srcs for p in src.rglob("*.rs")):
        text = path.read_text()
        flat = re.sub(r"\s+", " ", text)
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        if path.is_relative_to(pyffi) and pub_unsafe(text):
            problems.append(f"{rel}: a public unsafe function or trait in tessera-rl-pyffi")
        if pub_safe_foreign(text):
            problems.append(f"{rel}: a public safe foreign function (`pub safe fn` in an `unsafe extern` block)")
        if path.is_relative_to(pyffi) and "not_unsafe_ptr_arg_deref" in text:
            problems.append(f"{rel}: clippy::not_unsafe_ptr_arg_deref must stay clean in tessera-rl-pyffi, with no allow or expect")
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
    args = [pathlib.Path(a) for a in sys.argv[1:]]
    if args:
        # check_py_entries.py PY_SRC PYFFI_SRC: check copies (the lint's own
        # tests).
        main(srcs=args, pyffi=args[-1])
    else:
        main()
