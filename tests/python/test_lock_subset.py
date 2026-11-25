# SPDX-License-Identifier: MIT OR Apache-2.0
"""scripts/lock_subset.sh, the lock-subset check of cargo tessera's
end-to-end test (#351, #633): a scaffold lock with a package the
repository's lock lacks fails; one collation for sort and comm, so no
"not in sorted order" warning under a UTF-8 locale."""

import os
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "lock_subset.sh"

# Names whose order differs between bytewise and UTF-8 locale collation
# ('-' and '_' are ignored at the first level of en_US.UTF-8).
PACKAGES = [("foo-bar", "1.0.0"), ("foo_bar", "1.0.0"), ("foobar", "2.0.0"), ("Zeta", "0.1.0"), ("alpha", "0.3.0")]


def lock(path: pathlib.Path, packages) -> pathlib.Path:
    path.write_text("".join(f'[[package]]\nname = "{n}"\nversion = "{v}"\n\n' for n, v in packages))
    return path


def run(repo, scaffold, crate="scaffold", locale="en_US.UTF-8"):
    env = {**os.environ, "LANG": locale, "LC_ALL": locale}
    return subprocess.run(["bash", str(SCRIPT), str(repo), str(scaffold), crate], capture_output=True, text=True, env=env)


@pytest.mark.parametrize("locale", ["en_US.UTF-8", "C.UTF-8", "C"])
def test_lock_subset_a_subset_passes_without_a_sort_warning(tmp_path, locale):
    repo = lock(tmp_path / "repo.lock", PACKAGES)
    scaffold = lock(tmp_path / "scaffold.lock", PACKAGES[:3] + [("scaffold", "0.1.0")])
    r = run(repo, scaffold, locale=locale)
    assert r.returncode == 0, r.stderr
    assert "sorted order" not in r.stderr and r.stderr == "", r.stderr


@pytest.mark.parametrize("extra", [("bar-baz", "1.0.0"), ("foo_bar", "9.9.9"), ("zzz", "0.0.1")])
def test_lock_subset_an_extra_package_or_version_fails(tmp_path, extra):
    repo = lock(tmp_path / "repo.lock", PACKAGES)
    scaffold = lock(tmp_path / "scaffold.lock", PACKAGES + [extra])
    r = run(repo, scaffold)
    assert r.returncode == 1, (r.stdout, r.stderr)
    assert f'"{extra[0]}" "{extra[1]}"' in r.stderr
    assert "sorted order" not in r.stderr


def test_lock_subset_excepts_the_scaffold_crate_only(tmp_path):
    repo = lock(tmp_path / "repo.lock", PACKAGES)
    scaffold = lock(tmp_path / "scaffold.lock", PACKAGES + [("scaffold", "0.1.0"), ("scaffold-extra", "0.1.0")])
    r = run(repo, scaffold)
    assert r.returncode == 1
    assert '"scaffold-extra"' in r.stderr and '"scaffold" "0.1.0"' not in r.stderr


def test_lock_subset_on_the_repository_lock_is_clean():
    r = run(ROOT / "Cargo.lock", ROOT / "Cargo.lock")
    assert r.returncode == 0 and r.stderr == "", r.stderr


def test_lock_subset_usage_error():
    r = subprocess.run(["bash", str(SCRIPT), "only-one"], capture_output=True, text=True)
    assert r.returncode == 2 and "usage" in r.stderr
