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


def run(scenario):
    return subprocess.run([sys.executable, str(CHILD), scenario], capture_output=True, text=True)


def child(scenario):
    proc = run(scenario)
    assert proc.returncode == 0, proc.stderr
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    if "skipped" in r:
        pytest.skip(r["skipped"])
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
    # dies with SIGFPE. It is never converted into a Python exception.
    child("entry")  # skips where the raw probes cannot run
    proc = run(scenario)
    assert proc.returncode == -signal.SIGFPE, (proc.returncode, proc.stdout, proc.stderr)
