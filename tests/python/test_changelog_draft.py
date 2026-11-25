# SPDX-License-Identifier: MIT OR Apache-2.0
"""scripts/changelog_draft.py (SPEC 20.1, #350): drafts and checks against
fixture git histories in a temporary repository.

The tests that need git skip, with `GIT_MISSING`, where no `git` is on
`PATH`, as in the musllinux wheel images (#644); the parser tests still run.
`GIT`, resolved once on the process `PATH`, is both what decides the skip and
what the fixtures run, so a git outside `/usr/bin:/bin` (manylinux_2_28) is
used rather than missed. `test_changelog_draft_skip.py` checks both."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "changelog_draft.py"
spec = importlib.util.spec_from_file_location("changelog_draft", SCRIPT)
cd = importlib.util.module_from_spec(spec)
# Registered first: @dataclass looks its module up in sys.modules.
sys.modules[spec.name] = cd
spec.loader.exec_module(cd)

GIT_MISSING = "git is not on PATH: these tests build fixture git repositories (#644)"

# The git binary on the process PATH, or None: the skip and every fixture
# call use this one resolution.
GIT = shutil.which("git")


def require_git() -> None:
    """Skips the calling test where `git` is not on `PATH`."""
    if GIT is None:
        pytest.skip(GIT_MISSING)


def git(repo: Path, *args: str) -> str:
    assert GIT is not None, "git() called without require_git()"
    # A minimal environment: a fixed identity, no user or system config, and
    # only the resolved git's own directory on PATH.
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid", "HOME": str(repo), "GIT_CONFIG_NOSYSTEM": "1",
           "PATH": os.path.dirname(GIT)}
    return subprocess.run([GIT, *args], cwd=repo, env=env, capture_output=True, text=True, check=True).stdout.strip()


def commit(repo: Path, message: str) -> str:
    (repo / "f").write_text(message)
    git(repo, "add", "f")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    require_git()
    git(tmp_path, "init", "-q", "-b", "main")
    commit(tmp_path, "chore: start")
    git(tmp_path, "tag", "v0.0.1")
    return tmp_path


def run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo), *args], capture_output=True, text=True)


def test_changelog_draft_types_map_to_sections(repo):
    commit(repo, "feat(core): a new trait (#10)")
    commit(repo, "fix(py): a crash (#11)")
    commit(repo, "perf(engine): faster step (#12)")
    commit(repo, "deprecate(core): EnvAssets (#13)")
    commit(repo, "test(core): more tests (#14)")
    out = run(repo, "v0.0.1").stdout
    assert "### Added\n\n- a new trait (#10)" in out
    assert "### Fixed\n\n- a crash (#11)" in out
    assert "### Changed\n\n- faster step (#12)" in out
    assert "### Deprecated\n\n- EnvAssets (#13)" in out
    assert "more tests" not in out
    assert "test: more tests (#14)" in run(repo, "--all", "v0.0.1").stdout


def test_changelog_draft_breaking_bang_and_trailer(repo):
    commit(repo, "feat(api)!: rename step (#20)")
    commit(repo, "fix(api): drop a field (#21)\n\nBREAKING CHANGE: the field is gone")
    out = run(repo, "v0.0.1").stdout
    assert out.index("### Breaking") < out.index("### Added")
    assert "- rename step (#20)" in out.split("### Added")[0]
    assert "- drop a field (#21)" in out.split("### Added")[0]


def test_changelog_draft_squash_subject_is_not_conventional(repo):
    sha = commit(repo, "squash (#487)")
    out = run(repo, "v0.0.1").stdout
    assert f"### Not conventional\n\n- squash (#487) ({sha[:8]})" in out


def test_changelog_draft_revert_drops_both_in_range(repo):
    sha = commit(repo, "feat(x): a thing (#30)")
    commit(repo, f'Revert "feat(x): a thing (#30)"\n\nThis reverts commit {sha}.')
    out = run(repo, "v0.0.1").stdout
    assert "a thing" not in out


def test_changelog_draft_revert_of_earlier_commit_is_reverted(repo):
    sha = commit(repo, "feat(x): a thing (#30)")
    git(repo, "tag", "v0.0.2")
    commit(repo, f'Revert "feat(x): a thing (#30)"\n\nThis reverts commit {sha}.')
    out = run(repo, "v0.0.2").stdout
    assert '### Reverted\n\n- feat(x): a thing (#30)' in out


def test_changelog_draft_empty_range(repo):
    r = run(repo, "v0.0.1")
    assert r.returncode == 0 and r.stdout == "## [Unreleased]\n"


def test_changelog_draft_bad_ref_exits_2(repo):
    r = run(repo, "no-such-tag")
    assert r.returncode == 2 and "changelog_draft:" in r.stderr


def test_changelog_draft_refs_from_subject_and_body():
    c = cd.parse("a" * 40, "feat(x): y (#5)", "Closes #6\nPart of #7\nfixes #5\nsee #8")
    assert c.refs == ["5", "6", "7"]


def test_changelog_draft_check_reports_missing_and_exits_1(repo, tmp_path):
    commit(repo, "feat(a): listed (#40)")
    commit(repo, "fix(b): only in the body\n\nCloses #41")
    commit(repo, "feat(c): not listed (#42)")
    commit(repo, "squash (#43)")
    commit(repo, "test(d): ignored (#44)")
    log = tmp_path / "CHANGELOG.md"
    log.write_text("# Changelog\n\n## [Unreleased]\n\n### Added\n\n- listed (#40)\n- body ref (#41)\n\n## [0.0.1] - 2026-01-01\n\n- old (#42)\n")
    r = run(repo, "--check", str(log), "v0.0.1")
    assert r.returncode == 1
    assert "not listed (#42)" in r.stdout and "[#42]" in r.stdout
    assert "squash (#43)" in r.stdout
    assert "listed (#40)" not in r.stdout.replace("not listed", "")
    assert "only in the body" not in r.stdout
    assert "ignored" not in r.stdout
    assert "2 of the checked commits" in r.stdout


def test_changelog_draft_check_passes_when_all_listed(repo, tmp_path):
    commit(repo, "feat(a): x (#50)")
    log = tmp_path / "CHANGELOG.md"
    log.write_text("## [Unreleased]\n\n- x (#50)\n")
    r = run(repo, "--check", str(log), "v0.0.1")
    assert r.returncode == 0 and "0 of the checked commits" in r.stdout


def test_changelog_draft_check_commit_without_ref_is_missing(repo, tmp_path):
    commit(repo, "fix(a): no reference at all")
    log = tmp_path / "CHANGELOG.md"
    log.write_text("## [Unreleased]\n\n- something (#1)\n")
    r = run(repo, "--check", str(log), "v0.0.1")
    assert r.returncode == 1 and "[no reference]" in r.stdout


def test_changelog_draft_real_history_parses():
    require_git()
    assert GIT is not None
    tags = subprocess.run([GIT, "tag", "--list", "v*"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    if not tags:
        pytest.skip("no release tag in this checkout (shallow CI clone)")
    r = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(ROOT), sorted(tags)[-1]], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("## [Unreleased]")
