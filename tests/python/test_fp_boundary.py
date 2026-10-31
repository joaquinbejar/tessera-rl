# SPDX-License-Identifier: MIT OR Apache-2.0
"""ADR-0041 conformance: the native floating-point boundary under hostile
caller controls. Each scenario runs in a fresh process (fp_child.py)."""

import json
import pathlib
import subprocess
import sys

import pytest

CHILD = pathlib.Path(__file__).with_name("fp_child.py")


def child(scenario, *flags):
    proc = subprocess.run([sys.executable, *flags, str(CHILD), scenario], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_fp_boundary_results_identical_under_hostile_rounding():
    r = child("identical")
    assert r["identical"], "every public binding returns the same bytes"
    assert r["round_restored"], "the caller's rounding mode survives every call"
    assert r["counted"] > 0
    assert r["warnings"] == 1, "one RuntimeWarning per process"


def test_fp_boundary_keeps_sticky_flags_raised_before_the_call():
    r = child("flags")
    assert r["control_kept"], "the flag is observable in this process"
    assert r["flag_kept"], "native calls keep a flag the caller raised"


def test_fp_boundary_reject_policy_raises_before_any_work():
    r = child("reject")
    # A method and a module function return NULL, getbuffer -1, each with
    # the exception set.
    assert r["errors"] == {
        "method": "FloatingPointEnvironmentError",
        "getbuffer": "FloatingPointEnvironmentError",
        "module_function": "FloatingPointEnvironmentError",
    }
    assert r["unchanged"], "a rejected step mutates nothing"
    assert r["round_restored"]
    assert r["policy"] == "canonicalize"


def test_fp_boundary_failing_warning_propagates_before_mutation():
    r = child("werror")
    assert r["raised"] == "RuntimeWarning"
    assert r["unchanged"], "the warning failed before the step ran"


def test_fp_boundary_unmasked_exceptions_never_trap():
    r = child("unmasked")
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert r["identical"]
    assert r["enabled_after"] != 0, "the caller's unmasked exceptions are restored"


@pytest.mark.torch
def test_fp_boundary_torch_flush_denormal_is_invisible():
    pytest.importorskip("torch")
    r = child("torch_ftz")
    assert r["supported"], "torch can enable flush-to-zero on this CPU"
    assert r["identical"]


def test_fp_boundary_policy_names_are_checked():
    import tessera_rl as t

    with pytest.raises(t.ArgumentError):
        t.set_fp_policy("lenient")
    assert t.get_fp_policy() == "canonicalize"


def test_fp_boundary_user_conversions_never_run_inside_native():
    # codex R-PR26 P1: exact types only, so no __index__ runs inside Rust.
    r = child("probe")
    assert r["raised"] == "TypeError"
    assert r["user_code_ran"] == []
    assert r["python_layer_ok"], "the Python layer converts user objects first"


def test_fp_boundary_gc_finalizer_changing_controls_does_not_leak_into_rust():
    r = child("gc_finalizer")
    assert r["finalizers_ran"] > 0, "gc ran the finalizers during native allocations"
    assert r["identical"], "Rust never resumed under the finalizer's controls"
    if sys.version_info < (3, 12):
        # Python 3.11 collects inside the allocation, so inside the callout.
        # From 3.12 the collection waits for the eval breaker and the
        # finalizers run in Python, never inside the native module.
        assert r["counted"] > 0, "the callout counted the change"


def test_fp_boundary_reports_its_variant():
    import platform

    import tessera_rl._native as n

    variant = n.fp_boundary()
    if platform.system() == "Windows":
        assert variant.startswith("none")
    elif platform.machine() in ("arm64", "aarch64"):
        assert variant == "AAPCS64"
    else:
        assert variant == "System V x86-64"


def test_fp_boundary_callout_lint_rejects_a_direct_call(tmp_path):
    import shutil

    root = pathlib.Path(__file__).resolve().parents[2]
    src = tmp_path / "src"
    shutil.copytree(root / "crates" / "tessera-rl-py" / "src", src)
    lint = root / "scripts" / "check_py_callouts.py"
    ok = subprocess.run([sys.executable, str(lint), str(src)], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout
    api = src / "api.rs"
    api.write_text(api.read_text() + "\nfn sneaky() { unsafe { pyo3::ffi::PyList_New(0); } }\n")
    bad = subprocess.run([sys.executable, str(lint), str(src)], capture_output=True, text=True)
    assert bad.returncode != 0 and "PyList_New" in bad.stdout
    capi = src / "ffi" / "capi.rs"
    shutil.copy(root / "crates" / "tessera-rl-py" / "src" / "api.rs", api)
    capi.write_text(capi.read_text() + "\nfn sneaky() { unsafe { ffi::PyDict_New(); } }\n")
    bad = subprocess.run([sys.executable, str(lint), str(src)], capture_output=True, text=True)
    assert bad.returncode != 0 and "PyDict_New" in bad.stdout


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="from 3.12 gc never runs inside a native allocation")
def test_fp_boundary_injected_container_values_drop_under_caller_controls():
    # codex R-PR26 R2 P1: dict_set replacing a value a gc callback stored.
    r = child("gc_inject")
    assert r["injected"] > 0 and r["finalizers"] > 0, r
    assert r["all_caller_controls"], r


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="from 3.12 gc never runs inside a native allocation")
def test_fp_boundary_unwind_drops_injected_containers_under_caller_controls():
    import tessera_rl._native as n

    if not hasattr(n, "_fail_after"):
        pytest.skip("needs the test-hooks build (make py-test-hooks)")
    r = child("gc_unwind")
    assert r["injected"] > 0 and r["failed"] > 0 and r["finalizers"] > 0, r
    assert r["all_caller_controls"], r
