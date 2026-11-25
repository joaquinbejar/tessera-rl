#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The public-API snapshot of the Python package (ADR-0094 item 2, #690).

The Python counterpart of `scripts/check_facade_api.py` (#642). Every name of
the stable Python surface is written as one line per item, with its
signature from the shipped `.pyi` stubs, to `api/tessera-rl-py.api`. `check`
rebuilds it and fails on any difference; `bless` rewrites it after a
reviewed change.

It parses the stubs and `tessera_rl/__init__.py` with `ast` and never
imports the package, so it needs no build. The signatures come from the
stubs only: `make py-typecheck`'s stubtest is what ties the stubs to the
runtime, so this check depends on it.

`ast.unparse`'s text can differ between Python minor versions, so the
script runs on the pinned 3.11 only (`uv run --python 3.11`), recorded in
the header.

Usage:
  check_py_api.py check [--root DIR]   diff against api/tessera-rl-py.api
  check_py_api.py bless [--root DIR]   rewrite api/tessera-rl-py.api
  check_py_api.py print [--root DIR]   print the snapshot
`--root` names a repository root other than this one (for tests).
"""

from __future__ import annotations

import ast
import difflib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYTHON = (3, 11)

# ADR-0094 item 2: `tessera_rl` (its `__all__`) and these modules.
MODULES = ("gym", "pz", "puffer", "replay", "render")
# Public modules outside item 2's list, not tracked until the owner's final
# item 2 names them (some are in flight). Recorded in the header.
EXPERIMENTAL_MODULES = ("eval", "jax", "masks", "search", "transforms", "viewer")

# ADR-0094 Q1, the owner's call: the host-environment API is experimental
# until decided, in Python as in Rust (`scripts/check_facade_api.py`). These
# `tessera_rl` names are left out while `INCLUDE_EXPERIMENTAL` is False; the
# owner flips both scripts with Q1. Each must be in `__all__` (a test checks
# it), so a rename cannot slip into the stable snapshot.
INCLUDE_EXPERIMENTAL = False
EXPERIMENTAL_NAMES = ("HostEnv", "HostStep")


class Refused(Exception):
    """A consistency error, reported with rc 1."""


class Stubs:
    """The package's `.pyi` files, parsed on demand."""

    def __init__(self, root: pathlib.Path):
        self.pkg = root / "python" / "tessera_rl"
        self.trees: dict[str, ast.Module] = {}

    def tree(self, module: str) -> ast.Module:
        """`module` is a dotted name under `tessera_rl` ("" for the package)."""
        if module not in self.trees:
            rel = "__init__.pyi" if module == "" else module.replace(".", "/") + ".pyi"
            f = self.pkg / rel
            if not f.exists():
                raise Refused(f"missing stub {f.relative_to(self.pkg.parent.parent)}")
            self.trees[module] = ast.parse(f.read_text(), str(f))
        return self.trees[module]

    def runtime_all(self) -> list[str]:
        """`__all__` of `tessera_rl/__init__.py`, the runtime module."""
        return all_of(ast.parse((self.pkg / "__init__.py").read_text())) or []


def all_of(tree: ast.Module) -> list[str] | None:
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    return None


def local_module(node: ast.ImportFrom, here: str) -> str | None:
    """The `tessera_rl` submodule an import names, or None outside the package."""
    if node.level:
        parts = here.split(".") if here else []
        base = parts[: len(parts) - node.level + 1] if node.level > 1 else parts[:-1] if here else []
        return ".".join([*base, *([node.module] if node.module else [])])
    if node.module == "tessera_rl":
        return ""
    if node.module and node.module.startswith("tessera_rl."):
        return node.module[len("tessera_rl.") :]
    return None


def text(node) -> str:
    return ast.unparse(node) if node is not None else ""


def decorators(node) -> list[str]:
    return [ast.unparse(d) for d in getattr(node, "decorator_list", [])]


def signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    out = ("async " if isinstance(fn, ast.AsyncFunctionDef) else "") + f"{fn.name}({ast.unparse(fn.args)})"
    if fn.returns is not None:
        out += " -> " + ast.unparse(fn.returns)
    return out


def marks(node) -> str:
    keep = [d for d in decorators(node) if d.split("(")[0].split(".")[-1] in (
        "property", "staticmethod", "classmethod", "overload", "final", "abstractmethod", "dataclass", "setter", "deprecated"
    ) or d.endswith(".setter")]
    return (" @" + " @".join(keep)) if keep else ""


class Walker:
    def __init__(self, stubs: Stubs):
        self.s = stubs
        self.lines: set[str] = set()

    def find(self, module: str, name: str, depth: int = 0):
        """The defining (module, node) of `name` as seen from `module`'s stub."""
        if depth > 10:
            raise Refused(f"import cycle resolving {name} from tessera_rl.{module}")
        tree = self.s.tree(module)
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return module, node
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
                return module, node
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return module, node
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if (a.asname or a.name) == name:
                        src = local_module(node, module)
                        if src is None:
                            return module, ("external", f"{node.module}.{a.name}")
                        return self.find(src, a.name, depth + 1)
        return None

    def public(self, module: str, names: list[str], path: str) -> None:
        for name in names:
            if name.startswith("_"):
                continue
            found = self.find(module, name)
            if found is None:
                raise Refused(f"{path}.{name} is public but has no stub definition or import")
            _, node = found
            self.item(f"{path}.{name}", node)

    def item(self, path: str, node) -> None:
        if isinstance(node, tuple):
            self.lines.add(f"alias {path} = {node[1]}")
        elif isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases] + [ast.unparse(k) for k in node.keywords]
            self.lines.add(f"class {path}({', '.join(bases)}){marks(node)}")
            for child in node.body:
                self.member(path, child)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.lines.add(f"function {path}: {signature(node)}{marks(node)}")
        elif isinstance(node, ast.AnnAssign):
            value = f" = {text(node.value)}" if node.value is not None else ""
            self.lines.add(f"const {path}: {text(node.annotation)}{value}")
        elif isinstance(node, ast.Assign):
            self.lines.add(f"const {path} = {text(node.value)}")

    def member(self, owner: str, node) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") and not (node.name.startswith("__") and node.name.endswith("__")):
                return
            self.lines.add(f"method {owner}.{signature(node)}{marks(node)}")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.startswith("_"):
                return
            value = f" = {text(node.value)}" if node.value is not None else ""
            self.lines.add(f"attr {owner}.{node.target.id}: {text(node.annotation)}{value}")
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    self.lines.add(f"attr {owner}.{t.id} = {text(node.value)}")
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            self.item(f"{owner}.{node.name}", node)


def module_names(tree: ast.Module) -> list[str]:
    """A module's public names: its `__all__`, or every top-level name without `_`."""
    declared = all_of(tree)
    if declared is not None:
        return declared
    names = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
        elif isinstance(node, ast.Assign):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.ImportFrom):
            names += [a.asname for a in node.names if a.asname and a.asname == a.name]
    return [n for n in names if not n.startswith("_")]


def snapshot(root: pathlib.Path) -> str:
    stubs = Stubs(root)
    w = Walker(stubs)
    runtime = stubs.runtime_all()
    stub_all = all_of(stubs.tree(""))
    if stub_all is not None and sorted(stub_all) != sorted(runtime):
        only_rt = sorted(set(runtime) - set(stub_all))
        only_stub = sorted(set(stub_all) - set(runtime))
        raise Refused(f"__all__ differs: in __init__.py only {only_rt}, in __init__.pyi only {only_stub}")
    # Every stub re-export of the package is in the runtime `__all__` too.
    for node in stubs.tree("").body:
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                public = a.asname or a.name
                if a.asname == a.name and not public.startswith("_") and public not in runtime:
                    raise Refused(f"__init__.pyi re-exports {public}, which __init__.py's __all__ does not list")
    missing = sorted(set(EXPERIMENTAL_NAMES) - set(runtime))
    if missing:
        raise Refused(f"EXPERIMENTAL_NAMES not in tessera_rl.__all__: {missing}")
    stable = runtime if INCLUDE_EXPERIMENTAL else [n for n in runtime if n not in EXPERIMENTAL_NAMES]
    w.public("", stable, "tessera_rl")
    for m in MODULES:
        w.public(m, module_names(stubs.tree(m)), f"tessera_rl.{m}")
    header = [
        "# The public Python API of tessera_rl (ADR-0094 item 2, #690), from the shipped stubs.",
        "# Generated by scripts/check_py_api.py; do not edit. Review a diff, then `make py-api-bless`.",
        "# Signatures come from the .pyi stubs; `make py-typecheck` (stubtest) ties them to the runtime.",
        f"# Python: {PYTHON[0]}.{PYTHON[1]}",
        "# modules: tessera_rl (__all__), " + ", ".join(MODULES),
        "# EXPERIMENTAL_MODULES (not tracked until ADR-0094 item 2 names them): " + ", ".join(EXPERIMENTAL_MODULES),
        f"# INCLUDE_EXPERIMENTAL: {INCLUDE_EXPERIMENTAL}"
        + ("" if INCLUDE_EXPERIMENTAL else " (ADR-0094 Q1 open: " + ", ".join(f"tessera_rl.{n}" for n in EXPERIMENTAL_NAMES) + " excluded)"),
    ]
    return "\n".join(header + sorted(w.lines)) + "\n"


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("check", "bless", "print"):
        print(__doc__)
        return 2
    if sys.version_info[:2] != PYTHON:
        print(
            f"py-api: Python {sys.version_info[0]}.{sys.version_info[1]}, the snapshot is made with "
            f"{PYTHON[0]}.{PYTHON[1]} (ast.unparse output differs between versions): run it with "
            f"`uv run --python {PYTHON[0]}.{PYTHON[1]} python scripts/check_py_api.py ...` (`make py-api` does)"
        )
        return 1
    root = pathlib.Path(argv[argv.index("--root") + 1]) if "--root" in argv else ROOT
    target = root / "api" / "tessera-rl-py.api"
    try:
        new = snapshot(root)
    except Refused as e:
        print(f"py-api: {e}")
        return 1
    if argv[0] == "print":
        sys.stdout.write(new)
        return 0
    if argv[0] == "bless":
        target.parent.mkdir(exist_ok=True)
        target.write_text(new)
        print(f"wrote api/tessera-rl-py.api ({new.count(chr(10))} lines)")
        return 0
    old = target.read_text() if target.exists() else ""
    if old == new:
        print("py-api: api/tessera-rl-py.api is current")
        return 0
    sys.stdout.writelines(
        difflib.unified_diff(old.splitlines(True), new.splitlines(True), "api/tessera-rl-py.api (committed)", "api/tessera-rl-py.api (built)")
    )
    print("py-api: the public Python API changed: review the diff, then `make py-api-bless`")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
