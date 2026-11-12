# SPDX-License-Identifier: MIT OR Apache-2.0
"""ADR-0041 conformance: the native floating-point boundary under hostile
caller controls. Each scenario runs in a fresh process (fp_child.py)."""

import json
import os
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


def test_fp_boundary_search_callback_runs_under_the_caller_controls():
    r = child("search_callback")
    assert r["identical"], "visits and tree equal those of a default-mode caller"
    assert r["clean_modes_default"]
    assert r["callback_saw_caller_mode"], "the evaluator runs under the caller's rounding"
    assert r["round_after"], "the caller's rounding is restored after run"
    assert r["counted"] > 0
    # An evaluator that alters the controls and leaves them changed.
    assert r["altered_identical"], "canonical arithmetic after the callback"
    assert r["altered_counted"] > 0
    assert r["restored_after_success"], "the caller's mode after a successful run"
    assert r["raised"] and r["unchanged_after_exception"]
    assert r["restored_after_exception"], "the caller's mode after a failed run"


EXPECT_TRAPS = "TESSERA_EXPECT_FP_TRAPS"


def check_unmasked(r, expect_traps=None):
    """The verdict on an "unmasked" child result.

    Everything except the enables is asserted on every CPU: identical
    results, the caller's controls (rounding, flush-to-zero, default NaN on
    AArch64) restored, the caller's sticky flag kept. Only the enables
    assertions are skipped, and only when the enables read back as zero
    before any library call (issue #225). With TESSERA_EXPECT_FP_TRAPS=1 (a
    CI job on hardware that implements traps) that case fails instead."""
    if expect_traps is None:
        expect_traps = os.environ.get(EXPECT_TRAPS) == "1"
    probe = f"cpu: {r.get('cpu')}; enables read back: {r.get('enabled_set')}; controls: {r.get('controls_with_enables')}"
    print(probe)
    if "skipped" in r:
        assert not expect_traps, f"{EXPECT_TRAPS}=1 but the scenario could not run: {r['skipped']}"
        pytest.skip(f"{r['skipped']} ({probe})")
    assert r["identical"]
    assert r["controls_after"] == r["controls_before"], "the caller's controls are restored"
    if sys.platform != "win32":
        assert r["flag_set"], "the flag is observable in this process"
    if r["flag_set"]:
        assert r["flag_kept"], "native calls keep a flag the caller raised"
    if r["enables_reason"] is not None:
        assert r["enabled_set"] == 0, "a skip is decided only by a zero read-back"
        assert not expect_traps, f"{EXPECT_TRAPS}=1 but {r['enables_reason']} ({probe})"
        pytest.skip(f"enables assertions only: {r['enables_reason']} ({probe})")
    assert r["enabled_set"] != 0, "the enables took effect before the call"
    assert r["enabled_after"] != 0, "the caller's unmasked exceptions are restored"
    if "raw_em_exact" in r:
        assert r["raw_em_exact"], "exactly divide-by-zero and invalid were unmasked"


def test_fp_boundary_unmasked_exceptions_never_trap():
    check_unmasked(child("unmasked"))


def test_unmasked_skip_reason_zero_read_back_skips():
    import fp_child

    reason = fp_child.unmasked_skip_reason(0)
    assert reason == (
        "this CPU does not implement trapped floating-point exceptions (FPCR enables read back as zero)"
    )


def test_unmasked_skip_reason_enables_set_runs():
    import fp_child

    assert fp_child.unmasked_skip_reason(0x03) is None


def _result(**kw):
    r = {
        "cpu": "test",
        "enabled_set": 0x03,
        "enabled_after": 0x03,
        "enables_reason": None,
        "identical": True,
        "controls_before": 7,
        "controls_after": 7,
        "flag_set": True,
        "flag_kept": True,
    }
    r.update(kw)
    return r


IGNORED = "this CPU does not implement trapped floating-point exceptions (FPCR enables read back as zero)"


def test_check_unmasked_enables_ignored_skips():
    r = _result(enabled_set=0, enabled_after=0, enables_reason=IGNORED)
    with pytest.raises(pytest.skip.Exception, match="read back as zero"):
        check_unmasked(r, expect_traps=False)


def test_check_unmasked_enables_ignored_with_traps_expected_fails():
    r = _result(enabled_set=0, enabled_after=0, enables_reason=IGNORED)
    with pytest.raises(AssertionError, match=EXPECT_TRAPS):
        check_unmasked(r, expect_traps=True)


def test_check_unmasked_enables_ignored_still_checks_controls():
    # The skip covers the enables only: a lost rounding mode still fails.
    r = _result(enabled_set=0, enabled_after=0, enables_reason=IGNORED, controls_after=0)
    with pytest.raises(AssertionError, match="controls are restored"):
        check_unmasked(r, expect_traps=False)


def test_check_unmasked_enables_ignored_still_checks_flags():
    r = _result(enabled_set=0, enabled_after=0, enables_reason=IGNORED, flag_kept=False)
    with pytest.raises(AssertionError, match="keep a flag"):
        check_unmasked(r, expect_traps=False)


def test_check_unmasked_library_cleared_enables_fails():
    # The enables took effect and the library left them cleared: a failure,
    # never a skip.
    with pytest.raises(AssertionError, match="unmasked exceptions are restored"):
        check_unmasked(_result(enabled_after=0), expect_traps=False)


def test_check_unmasked_restored_enables_passes():
    check_unmasked(_result(), expect_traps=False)


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
        assert variant == "Win64"
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
    capi.write_text(capi.read_text() + "\nfn sneaky() -> CResult<Mine> { Mine::new(unsafe { thunks::c_dict_new() }) }\n")
    bad = subprocess.run([sys.executable, str(lint), str(src)], capture_output=True, text=True)
    assert bad.returncode != 0 and "own it with Mine::container" in bad.stdout


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


def test_fp_boundary_windows_flush_to_zero_is_invisible():
    r = child("windows_ftz")
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert r["identical"] and r["restored"]
