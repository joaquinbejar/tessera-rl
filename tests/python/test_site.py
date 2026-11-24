# SPDX-License-Identifier: MIT OR Apache-2.0
"""The docs site (#375): the tutorial snippets run, snippet inlining, and the
check that keeps docs/ out of the built site. The full MkDocs build is
`make site-check`, which needs the `site` dependency group."""

import importlib.util
import pathlib
import runpy
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("site_build", ROOT / "tools" / "site" / "build.py")
build = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = build
spec.loader.exec_module(build)

SNIPPETS = sorted((ROOT / "site" / "snippets").glob("*.py"))


@pytest.mark.parametrize("path", SNIPPETS, ids=[p.name for p in SNIPPETS])
def test_site_snippet_runs(path, capsys):
    runpy.run_path(str(path), run_name="__main__")


def test_site_every_snippet_marker_resolves():
    for page in (ROOT / "site" / "pages").rglob("*.md"):
        text = build.inline_snippets(page.read_text(encoding="utf-8"), ROOT / "site" / "snippets")
        assert "<!-- snippet:" not in text


def test_site_inline_snippet_missing_raises(tmp_path):
    with pytest.raises(build.SiteError):
        build.inline_snippets("<!-- snippet: nope.py -->", tmp_path)


def test_site_inline_snippet_fences_python(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    assert build.inline_snippets("A\n<!-- snippet: a.py -->\nB", tmp_path) == "A\n```python\nx = 1\n```\nB"


def _generated(out):
    for rel in build.GENERATED:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text("<html>generated</html>")


def test_site_check_flags_missing_or_placeholder_generated_pages(tmp_path):
    problems = build.check(tmp_path)
    assert all(any(rel in p and "missing" in p for p in problems) for rel in build.GENERATED)
    _generated(tmp_path)
    (tmp_path / build.GENERATED[0]).write_text(build.PLACEHOLDER)
    problems = build.check(tmp_path)
    assert any(build.GENERATED[0] in p and "placeholder" in p for p in problems)


def test_site_stage_writes_placeholders(tmp_path):
    build.stage_pages(ROOT / "site", tmp_path / "s")
    for rel in build.GENERATED:
        assert build.PLACEHOLDER in (tmp_path / "s" / "pages" / rel).read_text()


def test_site_refuses_to_overwrite_a_page(tmp_path):
    (tmp_path / "dashboard").mkdir()
    (tmp_path / "dashboard" / "page.html").write_text("a MkDocs page")
    with pytest.raises(build.SiteError):
        build.refuse_overwrite(tmp_path)
    build.refuse_overwrite(tmp_path / "empty")  # nothing there: fine


def test_site_check_flags_spec_and_adr_links(tmp_path):
    _generated(tmp_path)
    (tmp_path / "ok.html").write_text("<a href='getting-started/'>ok</a>")
    (tmp_path / "bad.html").write_text("<a href='../docs/SPEC.md'>spec</a> and /adr/0003-x")
    problems = build.check(tmp_path)
    assert any("bad.html" in p and "SPEC.md" in p for p in problems)
    assert any("bad.html" in p and "/adr/0003" in p for p in problems)
    assert not any("ok.html" in p for p in problems)


def test_site_check_flags_symlink_into_docs(tmp_path):
    _generated(tmp_path)
    (tmp_path / "leak").symlink_to(ROOT / "docs")
    assert any("links into docs/" in p for p in build.check(tmp_path))


def test_site_pages_name_no_private_document():
    for f in (ROOT / "site").rglob("*"):
        if f.is_file():
            assert not build.FORBIDDEN.search(f.read_text(encoding="utf-8")), f
