# SPDX-License-Identifier: MIT OR Apache-2.0
"""ADR-0044 conformance: the x86-64 boundary preserves the caller's x87
control word (preserve only: not canonicalized, counted or rejected), read
raw, independently of MXCSR. Each scenario runs in a fresh process
(x87_child.py)."""

import json
import pathlib
import signal
import subprocess
import sys

import pytest

CHILD = pathlib.Path(__file__).with_name("x87_child.py")


WIN64 = sys.platform == "win32"
# The Win64 NTSTATUS code of the precondition fault: STATUS_FLOAT_DIVIDE_BY_ZERO,
# observed for both precondition scenarios in the native session on main
# 26e5a5f (docs/reviews/win64-x87-validation-26e5a5f.md, #163).
WIN64_FAULTS = {0xC000008E}
TIMEOUT = 60


def run(scenario):
    try:
        return subprocess.run([sys.executable, str(CHILD), scenario], capture_output=True, text=True,
                              timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        pytest.fail(f"x87 child {scenario!r} did not finish in {TIMEOUT} s")


def last_json(proc):
    lines = proc.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines and lines[-1].startswith("{") else None


def child(scenario):
    proc = run(scenario)
    r = last_json(proc)
    assert proc.returncode == 0, (proc.returncode, r, proc.stderr)
    assert r is not None, ("the child printed no report", proc.stdout, proc.stderr)
    if "skipped" in r:
        pytest.skip(r["skipped"])
    assert r["self_test"]["ok"], r["self_test"]
    return r


def test_x87_entry_preserves_an_x87_only_caller_uncounted():
    r = child("entry")
    assert r["identical"], "results do not depend on the x87 word"
    assert r["after"] == r["want"], "the caller's x87 word and MXCSR after every entry"
    assert r["counted"] == 0, "an x87-only change is not a non-canonical entry"
    assert r["mxcsr_only_after"] == r["mxcsr_only_want"], "an MXCSR-only change leaves the x87 word alone"


def test_x87_hostile_rounding_restores_both_raw_registers():
    r = child("fesetround")
    assert r["want_is_toward_zero"], "fesetround wrote both registers"
    assert r["after"] == r["want"], "both raw registers restored after every binding"
    assert r["identical"]


def test_x87_reject_policy_ignores_the_x87_word_and_restores_it():
    r = child("reject")
    assert r["x87_only"] == "ran", "strict reject never rejects an x87-only change"
    assert r["x87_only_after"] == r["want"]
    assert r["both"] == "rejected", "MXCSR still decides the policy"
    assert r["both_after"] == r["want_both"], "both registers restored on the error path"
    assert r["cleanup_after"] == r["want"], "cleanup entries restore the x87 word"


@pytest.mark.parametrize("scenario", ["callout", "callout_raises", "callout_interrupt"])
def test_x87_callout_change_never_outlives_the_callout(scenario):
    r = child(scenario)
    assert r["hook_entry"] == r["want"], "the host runs under the caller's x87 word and MXCSR"
    assert r["hook_x87"] == r["want_hook_x87"], "the host's own word is in force inside the callout"
    assert r["after"] == r["want"], "the host's change does not reach the caller"
    expected = {"callout": "ran", "callout_raises": "ValueError", "callout_interrupt": "KeyboardInterrupt"}
    assert r["outcome"] == expected[scenario]


def test_x87_nested_entry_inside_a_callout_restores_its_immediate_caller():
    r = child("callout_nested")
    assert r["after_nested"] == r["want_nested"], "the inner entry restores the hook's own words"
    assert r["after"] == r["want"], "the outer entry restores the original caller's"


def test_x87_precondition_supported_state_exits_cleanly():
    r = child("pending_supported")
    assert r["status_ze"], "the masked flag is left raised: the status word is not touched"
    assert r["after"] == r["want"]


@pytest.mark.parametrize("scenario", ["pending_masked_newly_unmasked", "pending_current_unmasked"])
def test_x87_precondition_violation_faults_with_sigfpe(scenario):
    # ADR-0044: a pending unmasked x87 exception at a callback return is an
    # unsupported precondition; the restore faults (#MF) and the process
    # dies: SIGFPE on System V, an NTSTATUS float code on Win64. It is never
    # converted into a Python exception. Three outcomes are told apart: the
    # fault, a ctypes OSError (Win64: a stub took the fault), and a normal
    # return; only the first passes.
    child("entry")  # skips where the raw probes cannot run
    proc = run(scenario)
    r = last_json(proc)
    assert not (r and r.get("outcome") == "ctypes_oserror"), f"a stub took the fault, not engine code: {r}"
    assert r is None, f"the child returned normally: {r}"
    if WIN64:
        code = proc.returncode & 0xFFFFFFFF
        assert code in WIN64_FAULTS, (hex(code), proc.stdout, proc.stderr)
    else:
        assert proc.returncode == -signal.SIGFPE, (proc.returncode, proc.stdout, proc.stderr)
