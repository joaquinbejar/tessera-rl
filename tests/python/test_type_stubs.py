# SPDX-License-Identifier: MIT OR Apache-2.0
"""The installed package ships its type stubs (issue #348).

``make py-typecheck`` checks the stubs themselves (mypy --strict) and that
they match the runtime (stubtest); this checks that the installed package
carries them: ``py.typed`` and one ``.pyi`` per module, the native one
included.
"""

from pathlib import Path

import pytest

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


def _native_callables():
    from tessera_rl import _native

    for name in dir(_native):
        obj = getattr(_native, name)
        if name.startswith("_") or isinstance(obj, type) and issubclass(obj, BaseException):
            continue
        if isinstance(obj, type):
            for attr in vars(obj):
                if not attr.startswith("_") or attr in ("__dlpack__", "__dlpack_device__"):
                    yield f"{name}.{attr}", getattr(obj, attr)
        elif callable(obj):
            yield name, obj


def test_native_text_signatures_every_callable_present():
    """Every public native function and method carries a text signature
    (issue #376), so stubtest checks its parameters."""
    missing = [name for name, obj in _native_callables() if getattr(obj, "__text_signature__", None) is None]
    assert missing == []


def test_native_keyword_only_positional_argument_refused():
    from tessera_rl import _native

    with pytest.raises(_native.ArgumentError, match="keyword arguments only"):
        _native.make_cartpole(1)
