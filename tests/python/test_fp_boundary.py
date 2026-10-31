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
