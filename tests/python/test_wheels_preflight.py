# SPDX-License-Identifier: MIT OR Apache-2.0
"""Runs the wheels workflow's preflight listing step itself against a stand-in curl (#787)."""

import json
import os
import re
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

# The preflight is a bash step of a Linux workflow and its fake is a bash script.
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX only: a bash workflow step with a bash fake")

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "wheels.yml"
ZOMBIE = 42064531284
ME = 7


def _step() -> str:
    match = re.search(
        r"name: List other runs.*?\n        run: \|\n(.*?)\n\n  linux:",
        WORKFLOW.read_text(),
        re.DOTALL,
    )
    assert match is not None
    return textwrap.dedent(match.group(1))


def _preflight(tmp_path: Path, runs: list[tuple[int, str]], fail: bool = False) -> subprocess.CompletedProcess[str]:
    for status in ("in_progress", "queued"):
        listed = [{"id": i, "name": f"run-{i}", "status": s} for i, s in runs if s == status]
        (tmp_path / f"{status}.json").write_text(json.dumps({"workflow_runs": listed}))
    curl = tmp_path / "curl"
    body = "exit 22" if fail else (
        'for a in "$@"; do case "$a" in\n'
        f"*status=in_progress*) cat {tmp_path}/in_progress.json;;\n"
        f"*status=queued*) cat {tmp_path}/queued.json;;\n"
        "esac; done"
    )
    curl.write_text(f"#!/bin/bash\n{body}\n")
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


def test_preflight_only_own_run_passes_and_lists_nothing(tmp_path: Path) -> None:
    result = _preflight(tmp_path, [(ME, "in_progress")])
    assert result.returncode == 0
    assert "in_progress []" in result.stdout
    assert "queued []" in result.stdout


@pytest.mark.parametrize("other", [(1, "queued"), (1, "in_progress"), (ZOMBIE, "queued"), (ZOMBIE, "in_progress")])
def test_preflight_other_active_run_is_listed_and_never_refuses(tmp_path: Path, other: tuple[int, str]) -> None:
    result = _preflight(tmp_path, [(ME, "in_progress"), other])
    assert result.returncode == 0
    assert f"{other[1]} [({other[0]}, 'run-{other[0]}')]" in result.stdout


def test_preflight_api_error_is_logged_and_never_refuses(tmp_path: Path) -> None:
    result = _preflight(tmp_path, [], fail=True)
    assert result.returncode == 0
    assert "could not list (not a refusal)" in result.stdout
