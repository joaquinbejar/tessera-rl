# SPDX-License-Identifier: MIT OR Apache-2.0
"""The changelog_draft tests and the git binary (#644), each checked in a
child pytest: under a `PATH` without git, as the musllinux wheel images
have, every git test skips with the stated reason and the parser test still
passes; with a git on `PATH` outside `/usr/bin:/bin`, as manylinux_2_28 has,
the fixtures run that git."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TARGET = HERE / "test_changelog_draft.py"


def test_changelog_draft_without_git_on_path_skips_the_git_tests(tmp_path: Path):
    empty = tmp_path / "bin"
    empty.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PATH"] = str(empty)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", str(TARGET)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    summary = re.search(r"(\d+) passed, (\d+) skipped", out)
    assert summary is not None, out
    passed, skipped = int(summary.group(1)), int(summary.group(2))
    # Every test but the parser test needs a repository.
    total = len(re.findall(r"^def test_", TARGET.read_text(), flags=re.M))
    assert (passed, skipped) == (1, total - 1), out
    assert "git is not on PATH: these tests build fixture git repositories (#644)" in out


def test_changelog_draft_git_off_the_system_path_is_the_git_used(tmp_path: Path):
    """A git on PATH but not in /usr/bin:/bin (manylinux_2_28, #644) runs the
    fixtures: the child's only PATH entry is a logging wrapper around the real
    git, and the fixtures' `git init` must go through it. A fixture that ran
    git from a fixed system PATH would miss that git, or bypass the wrapper."""
    real = shutil.which("git")
    if real is None:
        pytest.skip("this host has no git to wrap")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "git.log"
    wrapper = bin_dir / "git"
    wrapper.write_text(f'#!/bin/sh\necho "$1" >> "{log}"\nexec "{real}" "$@"\n')
    wrapper.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PATH"] = str(bin_dir)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(TARGET)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert " failed" not in out and " error" not in out, out
    calls = log.read_text().split() if log.exists() else []
    assert "init" in calls, (calls, out)
