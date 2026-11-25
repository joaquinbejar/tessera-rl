# SPDX-License-Identifier: MIT OR Apache-2.0
"""Every compat uv project that builds tessera-rl from this checkout (a path
source) keeps a lock that `uv lock --check` accepts.

Such a lock records tessera-rl's own metadata, its dependency groups
included, so a change to the root pyproject.toml makes it stale. Then
`uv sync --locked` fails before anything runs, as every reference-libraries
night did after the root gained its `site` group. Checked offline: an
up-to-date lock needs no index; a stale one fails either way."""

import os
import pathlib
import re
import shutil
import subprocess

import pytest

# A repository property, not a wheel property: make py-test runs these;
# scripts/wheel_job.sh deselects the marker (an empty uv cache in a release
# leg is no evidence about the locks).
pytestmark = pytest.mark.repo

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATH_SOURCE = re.compile(r'^tessera-rl\s*=\s*\{[^}]*\bpath\s*=', re.MULTILINE)


def path_source_projects() -> list[pathlib.Path]:
    """The uv projects under compat/ whose pyproject.toml takes tessera-rl
    from a path and that have a lock."""
    return sorted(
        p.parent
        for p in (ROOT / "compat").rglob("pyproject.toml")
        if PATH_SOURCE.search(p.read_text()) and (p.parent / "uv.lock").exists()
    )


def test_compat_locks_the_path_source_projects_are_found():
    names = {p.relative_to(ROOT).as_posix() for p in path_source_projects()}
    # The reference-libraries nightly's three projects at least.
    assert {"compat/training/sb3", "compat/training/tianshou", "compat/training/rllib"} <= names, names


@pytest.mark.parametrize("project", path_source_projects(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_compat_lock_is_up_to_date_with_the_root_pyproject(project, tmp_path):
    # A missing uv fails rather than skips: a skipped guard reads as green.
    uv = shutil.which("uv")
    assert uv is not None, "uv is not on PATH; the lock check needs it"
    r = subprocess.run(
        [uv, "lock", "--check", "--offline", "--project", str(project)],
        capture_output=True,
        text=True,
        env={**os.environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache")},
    )
    assert r.returncode == 0, (
        f"{project.relative_to(ROOT)}: uv.lock is stale; run `uv lock --project {project.relative_to(ROOT)}` "
        f"with the pinned uv and commit it.\n{r.stdout}{r.stderr}"
    )


@pytest.mark.parametrize("script", ["wheel_job.sh", "pgo_candidate_checks.sh"])
def test_compat_locks_wheel_runs_deselect_repo_tests(script):
    """The scripts that run tests/python against a built wheel deselect the
    `repo` tests."""
    text = (ROOT / "scripts" / script).read_text()
    suite = next(line for line in text.splitlines() if '--ignore="$root/tests/python/test_bench.py"' in line)
    assert '-m "not repo"' in suite, suite
