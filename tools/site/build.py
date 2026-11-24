#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Builds the docs site (SPEC 20.1, #375) into ``target/site``.

- Pages: ``site/pages`` through MkDocs (``site/mkdocs.yml``, strict). A line
  ``<!-- snippet: NAME -->`` is replaced by ``site/snippets/NAME`` as a fenced
  code block, so the published code is the code the tests run.
- Python API: pdoc over the installed ``tessera_rl`` into ``python/``.
- Benchmarks: ``tools/dashboard/build.py``'s output into ``dashboard/`` (the
  Benchmarks page, ``benchmarks/``, links to it).
- The Rust API is on docs.rs; ``make site-check`` builds ``cargo doc`` only
  as a ``-D warnings`` check.

``--check`` builds into a temporary directory and then checks the result: no
file of the site comes from ``docs/``, no page links the SPEC or the ADRs
(both stay out of the repository), and pdoc printed no warning (#525). Every
build reports pdoc's warnings as problems.

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


# Pages the builder adds after MkDocs (pdoc, the dashboard). Placeholders
# stand in for them while MkDocs runs, so strict link validation still covers
# the links to them; the generated files then replace the placeholders.
GENERATED = ("python/tessera_rl.html", "dashboard/index.html")
PLACEHOLDER = "<!-- tessera-site placeholder: replaced after the MkDocs build -->"


class SiteError(RuntimeError):
    pass


def inline_snippets(text: str, snippets: pathlib.Path) -> str:
    def repl(m: re.Match) -> str:
        path = snippets / m["name"]
        if not path.is_file():
            raise SiteError(f"snippet {m['name']} not found in {snippets}")
        lang = {".py": "python", ".sh": "sh"}.get(path.suffix, "")
        body = path.read_text(encoding="utf-8").rstrip("\n")
        return f"```{lang}\n{body}\n```"

    return SNIPPET.sub(repl, text)


def stage_pages(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copies ``site/`` to ``dst`` with the snippets inlined."""
    shutil.copytree(src, dst)
    for page in (dst / "pages").rglob("*.md"):
        page.write_text(inline_snippets(page.read_text(encoding="utf-8"), src / "snippets"), encoding="utf-8")
    for rel in GENERATED:
        target = dst / "pages" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(PLACEHOLDER + "\n", encoding="utf-8")


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SiteError(f"{' '.join(cmd)} exited {r.returncode}")


# A pdoc warning line, as pdoc 16 prints them ("Warn: ..."), or a Python
# warning it lets through (#525).
PDOC_WARNING = re.compile(r"^(?:Warn:|.*Warning:)", re.M)


def pdoc_warnings(output: str) -> list[str]:
    """The warning lines of a pdoc run's combined output."""
    return [line.strip() for line in output.splitlines() if PDOC_WARNING.match(line)]


def run_pdoc(out: pathlib.Path) -> list[str]:
    """Runs pdoc into ``out``, echoing its output, and returns its warning
    lines: a built site has none (#525), so ``check`` reports each one."""
    r = subprocess.run(
        [sys.executable, "-m", "pdoc", "tessera_rl", "-o", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise SiteError(f"pdoc exited {r.returncode}")
    return pdoc_warnings(r.stdout + "\n" + r.stderr)


def build(out: pathlib.Path) -> list[str]:
    """Builds the site into ``out``; returns pdoc's warning lines."""
    if out.exists():
        shutil.rmtree(out)
    with tempfile.TemporaryDirectory(prefix="tessera-site-") as tmp:
        staged = pathlib.Path(tmp) / "site"
        stage_pages(ROOT / "site", staged)
        run([sys.executable, "-m", "mkdocs", "build", "--strict", "-f", str(staged / "mkdocs.yml"), "-d", str(out)])
    for rel in GENERATED:
        (out / rel).unlink(missing_ok=True)
    refuse_overwrite(out)
    warnings = run_pdoc(out / "python")
    run([sys.executable, str(ROOT / "tools" / "dashboard" / "build.py")])
    shutil.rmtree(out / "dashboard", ignore_errors=True)
    shutil.copytree(ROOT / "target" / "dashboard", out / "dashboard")
    return warnings


def refuse_overwrite(out: pathlib.Path) -> None:
    """The generated parts must not replace a MkDocs page: after the
    placeholders are removed, their directories hold nothing else."""
    for top in sorted({pathlib.PurePosixPath(rel).parts[0] for rel in GENERATED}):
        left = [p for p in (out / top).rglob("*") if p.is_file()] if (out / top).exists() else []
        if left:
            raise SiteError(f"{top}/ holds MkDocs output ({left[0].relative_to(out)}); a generated part would replace it")


def check(out: pathlib.Path) -> list[str]:
    """Problems in a built site: files under docs/, and links to the SPEC
    or the ADRs."""
    problems = []
    for rel in GENERATED:
        f = out / rel
        if not f.is_file():
            problems.append(f"{rel}: missing (its generator wrote nothing)")
        elif PLACEHOLDER in f.read_text(encoding="utf-8", errors="replace"):
            problems.append(f"{rel}: still the placeholder")
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
                warnings = build(out)
                problems = check(out)
        else:
            out = ROOT / "target" / "site"
            warnings = build(out)
            problems = check(out)
        problems += [f"pdoc: {w}" for w in warnings]
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
