# SPDX-License-Identifier: MIT OR Apache-2.0
"""The changelog_draft tests skip where no `git` is on `PATH` (#644): run
them in a child pytest under a `PATH` without git, as the musllinux wheel
images have, and check that every git test skips with the stated reason and
the parser test still passes."""

import os
import re
import subprocess
import sys
from pathlib import Path

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
