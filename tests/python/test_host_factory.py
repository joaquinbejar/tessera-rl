# SPDX-License-Identifier: MIT OR Apache-2.0
"""Host protocol 1, the factory part (#272 step 4, ADR-0065 item 3): the
test host wheel of tests/host-wheel (`make py-test-host`).

- The two front ends: the same config given as Python keyword arguments
  and as a Rust `ConfigSource` decodes to the same canonical bytes, or
  fails with the same error class and message, for every decoding rule.
- `make`: its named keywords, config keywords and errors.
- The exception classes are `tessera_rl`'s own, bound by name; a missing
  class fails the import.
"""

import subprocess
import sys
import textwrap

import pytest

import tessera_rl

_package = pytest.importorskip("_tessera_host", reason="needs the test host wheel (make py-test-host)")
# maturin installs the extension inside a package of the same name, whose
# `__init__` star-imports it (which skips the underscored test functions).
host = getattr(_package, "_tessera_host", _package)


def deep_table():
    d = {"x": 1}
    for _ in range(6):
        d = {"x": d}
    return {"scale": d}


# The Python side of each case of tests/host-wheel/src/lib.rs `rust_case`.
CASES = {
    "empty": {},
    "ints": {"byte": 255, "wide": -(2**63), "big": 2**64 - 1},
    "floats": {"single": 0.1, "double": -0.0},
    "f32_tie": {"single": 1.0 + 2.0**-24},
    "f32_huge": {"single": 1e300},
    "int_into_float": {"double": 2**53, "single": 2**24},
    "fixed": {"fx32": "-1.5", "fx64": "0.25"},
    "bool": {"flag": False},
    "nested": {"inner": {"scale": -2.25, "count": 7}},
    "unknown": {"colour": 1},
    "adaptive": {"level": 3},
    "byte_range": {"byte": 256},
    "float_inexact": {"double": 2**53 + 1},
    "fixed_inexact": {"fx32": "0.1"},
    "fixed_float": {"fx32": 1.5},
    "bool_int": {"flag": 1},
    "int_bool": {"byte": True},
    "nested_not_table": {"inner": 1},
    "nan": {"double": float("nan")},
    "deep_table": {"inner": deep_table()},
}


def outcome(call):
    try:
        return ("ok", call())
    except Exception as e:  # noqa: BLE001 - the class and message are compared
        return (type(e).__name__, str(e))


@pytest.mark.parametrize("name", sorted(CASES))
def test_host_factory_two_front_ends_decode_alike(name):
    python = outcome(lambda: host._config_python(**CASES[name]))
    rust = outcome(lambda: host._config_rust(name))
    assert python == rust, name


def test_host_factory_two_front_ends_cover_success_and_every_error_kind():
    kinds = {outcome(lambda n=n: host._config_rust(n))[0] for n in CASES}
    assert kinds == {"ok", "ArgumentError", "ConfigError"}, kinds
    # The decoded bytes differ between cases, so equal bytes mean something.
    ok = [outcome(lambda n=n: host._config_rust(n))[1] for n in CASES]
    ok = [b for b in ok if isinstance(b, bytes)]
    assert len(set(ok)) == len(ok)


def test_host_factory_python_only_refusals():
    for bad in ({"byte": [1]}, {"byte": None}, {"big": 2**64}, {"wide": -(2**63) - 1}):
        with pytest.raises(tessera_rl.ConfigError):
            host._config_python(**bad)
    with pytest.raises(tessera_rl.ArgumentError):
        host._config_python(1)


def test_host_factory_make_and_close():
    env = host.make(num_envs=4, seed=7, auto_reset="same_step", max_episode_steps=10, offset=-3)
    assert type(env).__name__ == "HostEnv"
    env.close()
    env.close()
    synced = host.make(num_envs=4, workers=2, reward=0.25)
    synced.close()


@pytest.mark.parametrize(
    "kwargs, cls",
    [
        ({}, tessera_rl.ArgumentError),
        ({"num_envs": 0}, tessera_rl.ArgumentError),
        ({"num_envs": 2, "auto_reset": "sometimes"}, tessera_rl.ArgumentError),
        ({"num_envs": 2, "max_episode_steps": 0}, tessera_rl.ArgumentError),
        ({"num_envs": 2, "cpu_set": [0]}, tessera_rl.ArgumentError),
        ({"num_envs": 2, "colour": 1}, tessera_rl.ArgumentError),
        ({"num_envs": 2, "episode": 0}, tessera_rl.ConfigError),
        ({"num_envs": 2, "offset": 1.5}, tessera_rl.ConfigError),
    ],
)
def test_host_factory_make_errors(kwargs, cls):
    with pytest.raises(cls):
        host.make(**kwargs)


def test_host_factory_make_refuses_positional_arguments():
    with pytest.raises(tessera_rl.ArgumentError):
        host.make(2)


def test_host_factory_errors_are_tessera_rl_classes():
    with pytest.raises(tessera_rl.TesseraError) as info:
        host.make(num_envs=2, colour=1)
    assert type(info.value) is tessera_rl.ArgumentError


MISSING_CLASS = """
import sys
sys.path.insert(0, {path!r})
try:
    import _tessera_host
    print("imported")
except ImportError as e:
    print("import-failed:", e)
"""


def test_host_factory_missing_tessera_rl_class_fails_the_import(tmp_path):
    # A stand-in tessera_rl without ForkError, first on the path.
    pkg = tmp_path / "tessera_rl"
    pkg.mkdir()
    names = [
        "TesseraError", "ConfigError", "ArgumentError", "ActionError", "PendingResetError",
        "PoisonedError", "ConcurrencyError", "ExhaustedError", "FloatingPointEnvironmentError",
    ]
    (pkg / "__init__.py").write_text("".join(f"class {n}(Exception): pass\n" for n in names))
    script = textwrap.dedent(MISSING_CLASS.format(path=str(tmp_path)))
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "import-failed: tessera_rl has no exception class ForkError" in run.stdout, run.stdout
