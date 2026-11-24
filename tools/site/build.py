#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Builds the docs site (SPEC 20.1, #375) into ``target/site``.

- Pages: ``site/pages`` through MkDocs (``site/mkdocs.yml``, strict). A line
  ``<!-- snippet: NAME -->`` is replaced by ``site/snippets/NAME`` as a fenced
  code block, so the published code is the code the tests run.
- Python API: pdoc over the installed ``tessera_rl`` into ``python/``.
- Benchmarks: ``tools/dashboard/build.py``'s output into ``benchmarks/``.
- The Rust API is on docs.rs; ``make site-check`` builds ``cargo doc`` only
  as a ``-D warnings`` check.

``--check`` builds into a temporary directory and then checks the result: no
file of the site comes from ``docs/`` and no page links the SPEC or the ADRs
(both stay out of the repository).

This tool publishes nothing: deploying ``target/site`` is the owner's step
(RELEASE-CHECKLIST.md).
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
SNIPPET = re.compile(r"^<!-- snippet: (?P<name>[A-Za-z0-9_.-]+) -->$", re.M)
# Links or paths into the private documents, which must not appear on the site.
FORBIDDEN = re.compile(r"(?:docs/(?:SPEC\.md|adr/|reviews/)|SPEC\.md|/adr/0\d{3})")


class SiteError(RuntimeError):
    pass


def inline_snippets(text: str, snippets: pathlib.Path) -> str:
    def repl(m: re.Match) -> str:
        path = snippets / m["name"]
        if not path.is_file():
            raise SiteError(f"snippet {m['name']} not found in {snippets}")
        lang = "python" if path.suffix == ".py" else ""
        body = path.read_text(encoding="utf-8").rstrip("\n")
        return f"```{lang}\n{body}\n```"

    return SNIPPET.sub(repl, text)


def stage_pages(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copies ``site/`` to ``dst`` with the snippets inlined."""
    shutil.copytree(src, dst)
    for page in (dst / "pages").rglob("*.md"):
        page.write_text(inline_snippets(page.read_text(encoding="utf-8"), src / "snippets"), encoding="utf-8")


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SiteError(f"{' '.join(cmd)} exited {r.returncode}")


def build(out: pathlib.Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    with tempfile.TemporaryDirectory(prefix="tessera-site-") as tmp:
        staged = pathlib.Path(tmp) / "site"
        stage_pages(ROOT / "site", staged)
        run([sys.executable, "-m", "mkdocs", "build", "--strict", "-f", str(staged / "mkdocs.yml"), "-d", str(out)])
    run([sys.executable, "-m", "pdoc", "tessera_rl", "-o", str(out / "python")])
    run([sys.executable, str(ROOT / "tools" / "dashboard" / "build.py")])
    shutil.copytree(ROOT / "target" / "dashboard", out / "benchmarks")


def check(out: pathlib.Path) -> list[str]:
    """Problems in a built site: files under docs/, and links to the SPEC
    or the ADRs."""
    problems = []
    docs = (ROOT / "docs").resolve()
    for f in sorted(out.rglob("*")):
        if f.is_symlink() and (f.resolve() == docs or docs in f.resolve().parents):
            problems.append(f"{f.relative_to(out)}: links into docs/")
        if f.is_file() and f.suffix in {".html", ".md", ".js", ".json", ".txt", ".xml"}:
            text = f.read_text(encoding="utf-8", errors="replace")
            for m in FORBIDDEN.finditer(text):
                problems.append(f"{f.relative_to(out)}: names {m.group(0)!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="build into a temporary directory and check it")
    a = p.parse_args(argv)
    try:
        if a.check:
            with tempfile.TemporaryDirectory(prefix="tessera-site-out-") as tmp:
                out = pathlib.Path(tmp) / "site"
                build(out)
                problems = check(out)
        else:
            out = ROOT / "target" / "site"
            build(out)
            problems = check(out)
    except SiteError as e:
        sys.stderr.write(f"site: {e}\n")
        return 1
    for line in problems:
        sys.stderr.write(f"site: {line}\n")
    if problems:
        return 1
    print("site: ok" if a.check else f"site: built {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
