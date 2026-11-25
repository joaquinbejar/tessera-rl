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

_MODULES = ("__init__", "_archive", "_env", "_envs", "_flat", "_host", "_native", "gym", "jax", "masks", "puffer", "pz", "_render", "render", "replay", "search", "transforms", "viewer")


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


def _env():
    from tessera_rl import _native

    return _native.make_cartpole(num_envs=1, workers=0, outputs="owned")


@pytest.mark.parametrize(
    "call",
    [
        lambda n, e: n.set_fp_policy("canonicalize", "extra"),
        lambda n, e: e.reset(None, None, None, "extra"),
        lambda n, e: e.state_hash(0, "extra"),
        lambda n, e: e.fork(0, [1], "extra"),
        lambda n, e: e.encode([0], "extra"),
    ],
    ids=["set_fp_policy", "reset", "state_hash", "fork", "encode"],
)
def test_native_varargs_extra_argument_refused(call):
    """A positional-only row given one argument too many raises, as its
    text signature says (#376)."""
    from tessera_rl import _native

    env = _env()
    try:
        with pytest.raises(_native.ArgumentError):
            call(_native, env)
    finally:
        env.close()


@pytest.mark.parametrize(
    "name",
    [
        "make_cartpole", "make_acrobot", "make_mountaincar", "make_pendulum", "make_tag", "make_rts", "make_lob",
        "make_connect4", "make_leduc", "make_lanterns", "make_jobshop", "make_vrp", "make_othello", "make_go9", "make_grid", "make_keydoor",
        "make_snake", "make_brickwall", "make_paddleduel", "make_search", "make_search_group", "onnx_load",
    ],
)
def test_native_keyword_only_each_positional_refused(name):
    from tessera_rl import _native

    with pytest.raises(_native.ArgumentError, match="keyword arguments only"):
        getattr(_native, name)(1)


def test_native_keyword_only_attach_onnx_positional_refused():
    from tessera_rl import _native

    search = _native.make_search(env_id="tessera/Connect4-v1")
    with pytest.raises(_native.ArgumentError, match="keyword arguments only"):
        search.attach_onnx("model.onnx")
