# SPDX-License-Identifier: MIT OR Apache-2.0
"""Runs the wheels workflow's preflight step itself against a stand-in curl."""

import json
import os
import re
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "wheels.yml"
ZOMBIE = 42064531284
ME = 7


def _step() -> str:
    match = re.search(
        r"name: Refuse while other runs.*?\n        run: \|\n(.*?)\n\n  linux:",
        WORKFLOW.read_text(),
        re.DOTALL,
    )
    assert match is not None
    return textwrap.dedent(match.group(1))


def _preflight(tmp_path: Path, runs: list[tuple[int, str]]) -> subprocess.CompletedProcess[str]:
    for status in ("in_progress", "queued"):
        listed = [{"id": i, "name": f"run-{i}", "status": s} for i, s in runs if s == status]
        (tmp_path / f"{status}.json").write_text(json.dumps({"workflow_runs": listed}))
    curl = tmp_path / "curl"
    curl.write_text(
        '#!/bin/bash\nfor a in "$@"; do case "$a" in\n'
        f'*status=in_progress*) cat {tmp_path}/in_progress.json;;\n'
        f'*status=queued*) cat {tmp_path}/queued.json;;\n'
        "esac; done\n"
    )
    curl.chmod(curl.stat().st_mode | stat.S_IXUSR)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "TOKEN": "t",
        "WINDOW": "w",
        "GITHUB_REPOSITORY": "o/r",
        "GITHUB_RUN_ID": str(ME),
    }
    return subprocess.run(["bash", "-e", "-c", _step()], env=env, capture_output=True, text=True, check=False)


def test_preflight_only_own_and_zombie_runs_passes(tmp_path: Path) -> None:
    assert _preflight(tmp_path, [(ME, "in_progress"), (ZOMBIE, "queued")]).returncode == 0


@pytest.mark.parametrize(
    "other",
    [
        (1, "queued"),
        (1, "in_progress"),
        (ZOMBIE + 1, "queued"),
        (ZOMBIE - 1, "queued"),
    ],
)
def test_preflight_any_other_active_run_refuses_beside_the_zombie(tmp_path: Path, other: tuple[int, str]) -> None:
    assert _preflight(tmp_path, [(ZOMBIE, "queued"), other]).returncode != 0


def test_preflight_zombie_id_refuses_once_it_is_in_progress(tmp_path: Path) -> None:
    assert _preflight(tmp_path, [(ZOMBIE, "in_progress")]).returncode != 0
