# SPDX-License-Identifier: MIT OR Apache-2.0
"""Public docstrings state their rules instead of citing ADR numbers
(#527): the ADRs are not published, so the generated API page could not
link them. Code comments may still cite them."""

import importlib
import inspect
import pathlib
import pkgutil
import re

import pytest

import tessera_rl

ADR = re.compile(r"ADR-\d{4}")


def _public_modules():
    yield tessera_rl
    for info in pkgutil.iter_modules(tessera_rl.__path__):
        if info.name.startswith("_"):
            continue
        try:
            yield importlib.import_module(f"tessera_rl.{info.name}")
        except ImportError:
            # Optional extras (JAX, PufferLib, ...) not installed here.
            continue


def _ours(obj) -> bool:
    """Defined by the package, in Python or in its native module (whose
    docstrings are Rust strings)."""
    return str(getattr(obj, "__module__", "") or "").startswith("tessera_rl")


def _docstrings():
    """(where, docstring) for every module, and every public function,
    class and method the package exposes, as the API page shows them."""
    seen = set()
    for module in _public_modules():
        yield module.__name__, module.__doc__ or ""
        for name, obj in vars(module).items():
            if name.startswith("_") or id(obj) in seen or not _ours(obj):
                continue
            seen.add(id(obj))
            yield f"{module.__name__}.{name}", inspect.getdoc(obj) or ""
            if inspect.isclass(obj):
                for attr, member in vars(obj).items():
                    if attr.startswith("_"):
                        continue
                    yield f"{module.__name__}.{name}.{attr}", inspect.getdoc(member) or ""


def test_public_docstrings_cite_no_adr_numbers():
    found = [(where, ADR.findall(doc)) for where, doc in _docstrings() if ADR.search(doc)]
    assert not found, "\n".join(f"{where}: {', '.join(refs)}" for where, refs in found)


def test_attribute_doc_comments_cite_no_adr_numbers():
    """``#:`` comments are attribute docs on the API page too."""
    root = pathlib.Path(tessera_rl.__file__).parent
    found = [
        f"{path.name}:{n}"
        for path in sorted(root.rglob("*.py"))
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if line.lstrip().startswith("#:") and ADR.search(line)
    ]
    assert not found, found


@pytest.mark.parametrize("probe", ["(ADR-0076)", "see ADR-0013"])
def test_public_docstrings_pattern_matches_citations(probe):
    assert ADR.search(probe)
