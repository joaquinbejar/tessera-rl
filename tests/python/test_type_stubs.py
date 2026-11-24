# SPDX-License-Identifier: MIT OR Apache-2.0
"""The installed package ships its type stubs (issue #348).

``make py-typecheck`` checks the stubs themselves (mypy --strict) and that
they match the runtime (stubtest); this checks that the installed package
carries them: ``py.typed`` and one ``.pyi`` per module, the native one
included.
"""

from pathlib import Path

import tessera_rl

_PACKAGE = Path(tessera_rl.__file__).parent

_MODULES = ("__init__", "_env", "_host", "_native", "gym", "jax", "masks", "puffer", "pz", "search")


def test_package_marker_installed_present():
    assert (_PACKAGE / "py.typed").is_file()


def test_stubs_every_module_has_stub():
    missing = [m for m in _MODULES if not (_PACKAGE / f"{m}.pyi").is_file()]
    assert missing == []


def test_stubs_every_python_module_listed_matches():
    modules = {p.stem for p in _PACKAGE.glob("*.py")} | {"_native"}
    assert modules == set(_MODULES)
