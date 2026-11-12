# SPDX-License-Identifier: MIT OR Apache-2.0
"""Issue #43: every crate's copy of `tests/support/attribution.rs` is the
same code. The copies may differ only in their leading comment and module
doc (each names its crate); from the first inner attribute (`#![`) onward
they must be identical, so a fix to one copy cannot silently miss another."""

import pathlib

import pytest

ROOT = pathlib.Path(__file__).parents[2]
PATTERN = "crates/*/tests/support/attribution.rs"
ANCHOR = "#!["


def body(path: pathlib.Path) -> str:
    """The file from its first line starting with the anchor onward."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(ANCHOR):
            return "".join(lines[i:])
    pytest.fail(f"{path.relative_to(ROOT)} has no line starting with {ANCHOR!r}")


def test_attribution_copies_are_identical_after_the_module_doc():
    copies = sorted(ROOT.glob(PATTERN))
    assert len(copies) >= 2, f"expected at least two copies of {PATTERN}, found {copies}"
    reference = copies[0]
    expected = body(reference)
    drifted = [str(p.relative_to(ROOT)) for p in copies[1:] if body(p) != expected]
    assert not drifted, f"differ from {reference.relative_to(ROOT)} after the module doc: {drifted}"


def test_attribution_body_ignores_the_doc_and_catches_a_code_change(tmp_path):
    a = tmp_path / "a.rs"
    b = tmp_path / "b.rs"
    a.write_text("//! crate a\n#![x]\nfn f() {}\n", encoding="utf-8")
    b.write_text("//! crate b, longer\n//! doc\n#![x]\nfn f() {}\n", encoding="utf-8")
    assert body(a) == body(b)
    b.write_text("//! crate b\n#![x]\nfn g() {}\n", encoding="utf-8")
    assert body(a) != body(b)
