# SPDX-License-Identifier: MIT OR Apache-2.0
"""The target checker fails on a missing required target and tolerates a
missing deferred one (a controlled probe on a synthetic targets file)."""

import importlib.util
import pathlib

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "check_targets.py"
spec = importlib.util.spec_from_file_location("check_targets", SCRIPT)
check_targets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_targets)

TARGETS = {
    "python": {"minimum": "3.11"},
    "env": {"probe": {"tessera-surely-missing-a": "1.0", "tessera-surely-missing-b": "2.0"}},
    "profile": {
        "everything-missing": {"env": "probe", "required": ["tessera-surely-missing-a"], "deferred": ["tessera-surely-missing-b"]},
        "only-deferred": {"env": "probe", "required": [], "deferred": ["tessera-surely-missing-a", "tessera-surely-missing-b"]},
        "unknown-name": {"env": "probe", "required": ["not-pinned"], "deferred": []},
    },
}


def test_missing_required_target_fails():
    ok, lines = check_targets.check(TARGETS, "everything-missing")
    assert not ok
    assert any("required), FAIL" in line for line in lines)


def test_missing_deferred_targets_pass():
    ok, _ = check_targets.check(TARGETS, "only-deferred")
    assert ok


def test_profile_naming_an_unpinned_target_fails():
    ok, _ = check_targets.check(TARGETS, "unknown-name")
    assert not ok
