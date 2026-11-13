# SPDX-License-Identifier: MIT OR Apache-2.0
"""The ONNX evaluator from Python (ADR-0057).

No test loads ONNX Runtime with telemetry enabled. The gate tests never
load it: they run a child interpreter without ORT_DISABLE_TELEMETRY and a
path that is never opened. The real-library tests (marker ``onnx``) run
only through ``make py-test-onnx``, which builds the extension with the
``onnx`` feature, sets ORT_DISABLE_TELEMETRY=1 and a scratch HOME, and
TESSERA_ONNX_TEST=1; they fail if the variable or the HOME is missing, and
check afterwards that no telemetry directory appeared.
"""

import os
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import numpy as np
import pytest

import tessera_rl as t
from tessera_rl import search

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "crates" / "tessera-rl-onnx" / "tests" / "fixtures"
MISSING = "/nonexistent/tessera-onnx/libonnxruntime.so.1.31.1"
REAL = os.environ.get("TESSERA_ONNX_TEST") == "1"


def has_feature() -> bool:
    try:
        t._native.onnx_load(library=MISSING, accept_telemetry=True)
    except t.OnnxError as e:
        return "no ONNX evaluator" not in str(e)
    return True


needs_feature = pytest.mark.skipif(
    not REAL and not has_feature(), reason="extension built without the onnx feature"
)


def child(code: str, *, unset_telemetry: bool) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if unset_telemetry:
        env.pop("ORT_DISABLE_TELEMETRY", None)
    return subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_onnx_version_is_tied_to_targets_and_the_extra():
    targets = tomllib.loads((ROOT / "compat" / "targets.toml").read_text())
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pinned = targets["env"]["main"]["onnxruntime"]
    assert search.ONNXRUNTIME_VERSION == pinned
    assert pyproject["project"]["optional-dependencies"]["onnx"] == [f"onnxruntime=={pinned}"]


def test_onnx_import_does_not_import_onnxruntime():
    out = child(
        "import sys, tessera_rl, tessera_rl.search; print('onnxruntime' in sys.modules)",
        unset_telemetry=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_onnx_missing_package_is_typed(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(t.OnnxError, match=r"tessera-rl\[onnx\]"):
        search.OnnxEvaluator(FIXTURES / "constant.onnx")


@needs_feature
def test_onnx_unset_without_opt_out_is_refused_naming_the_variable():
    out = child(
        f"""
        import os, tessera_rl as t
        from tessera_rl.search import OnnxEvaluator
        before = dict(os.environ)
        try:
            OnnxEvaluator("model.onnx", library={MISSING!r})
        except t.OnnxError as e:
            print("OnnxError:", e)
        assert dict(os.environ) == before, "the environment changed"
        assert "ORT_DISABLE_TELEMETRY" not in os.environ
        """,
        unset_telemetry=True,
    )
    assert out.returncode == 0, out.stderr
    assert "OnnxError:" in out.stdout
    assert "ORT_DISABLE_TELEMETRY=1" in out.stdout


@needs_feature
def test_onnx_opt_out_passes_the_gate_to_the_missing_library():
    out = child(
        f"""
        import tessera_rl as t
        from tessera_rl.search import OnnxEvaluator
        try:
            OnnxEvaluator("model.onnx", library={MISSING!r}, accept_telemetry=True)
        except t.OnnxError as e:
            print("OnnxError:", e)
        """,
        unset_telemetry=True,
    )
    assert out.returncode == 0, out.stderr
    assert "no ONNX Runtime library" in out.stdout, out.stdout
    assert "ORT_DISABLE_TELEMETRY=1" not in out.stdout


# Real-library tests (make py-test-onnx).

real = pytest.mark.skipif(not REAL, reason="opens the real ONNX Runtime: run through `make py-test-onnx`")


@pytest.fixture
def home():
    """Rule T: refuses unless telemetry is off and HOME is the scratch dir;
    afterwards no telemetry directory may exist."""
    assert os.environ.get("ORT_DISABLE_TELEMETRY") == "1", (
        "refusing to load ONNX Runtime with telemetry enabled: set ORT_DISABLE_TELEMETRY=1"
    )
    home = Path(os.environ["HOME"])
    assert home.parts[-2:] == ("target", "onnx-test-home"), f"HOME must be the scratch dir, got {home}"
    yield home
    for d in (
        home / ".cache/Microsoft/DeveloperTools/.onnxruntime",
        home / "Library/Application Support/Microsoft/DeveloperTools/.onnxruntime",
    ):
        assert not d.exists(), f"telemetry directory appeared: {d}"


def fresh_search():
    return search.Search("tessera/Connect4-v1", capacity=4096, simulations=200, seed=7)


def constant(obs, legal, hashes):
    n = len(hashes)
    return np.full((n, 7), 1.0 / 7.0), np.full(n, 0.25)


@real
def test_onnx_constant_model_equals_the_python_evaluator(home):
    onnx = search.OnnxEvaluator(FIXTURES / "constant.onnx")
    a, b = fresh_search(), fresh_search()
    for _ in range(3):
        a.run(constant)
        b.run(onnx)
        assert a.root_visits().tolist() == b.root_visits().tolist()
        assert a.select_action() == b.select_action()


@real
def test_onnx_provenance_is_a_read_only_record(home):
    onnx = search.OnnxEvaluator(FIXTURES / "constant.onnx")
    assert onnx.provenance is None
    fresh_search().run(onnx)
    p = onnx.provenance
    assert p["library_version"] == search.ONNXRUNTIME_VERSION
    assert p["provider"] == "CPUExecutionProvider"
    assert (p["intra_threads"], p["inter_threads"]) == (1, 1)
    assert p["optimization_level"] == "basic"
    assert p["telemetry_enabled"] is False
    assert re.fullmatch(r"[0-9a-f]{64}", p["model_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", p["library_sha256"])
    with pytest.raises(TypeError):
        p["provider"] = "other"
    with pytest.raises(AttributeError):
        onnx.provenance = {}


@real
def test_onnx_model_refusals_are_onnx_errors(home):
    onnx = search.OnnxEvaluator(FIXTURES / "wrong_input.onnx")
    s = fresh_search()
    with pytest.raises(t.OnnxError, match="no input named"):
        s.run(onnx)
    assert s.root_visits().sum() == 0


@real
def test_onnx_works_next_to_the_onnxruntime_package(home):
    import onnxruntime  # noqa: F401  (its own runtime, in the same process)

    onnx = search.OnnxEvaluator(FIXTURES / "constant.onnx")
    s = fresh_search()
    s.run(onnx)
    assert s.root_visits().sum() > 0
