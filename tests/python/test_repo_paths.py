# SPDX-License-Identifier: MIT OR Apache-2.0
"""Every tracked path stays short enough for a Windows checkout (#789).

Windows refuses paths past MAX_PATH (260 characters) unless long paths are
enabled, and the runner's checkout directory comes before the repository
path. 200 characters leaves room for that prefix.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MAX_TRACKED_PATH = 200


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_repo_paths_every_tracked_path_is_short_enough_for_windows() -> None:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, text=True, check=True
    ).stdout
    paths = [p for p in out.split("\0") if p]
    assert paths
    long = sorted((len(p), p) for p in paths if len(p) > MAX_TRACKED_PATH)
    assert not long, long
