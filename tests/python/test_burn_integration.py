# SPDX-License-Identifier: MIT OR Apache-2.0
"""The Burn integration stays outside the root workspace (#367)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BURN = ROOT / "integrations" / "burn"
WORKSPACE = (BURN / "Cargo.toml").read_text()
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
NIGHTLY = (ROOT / ".github" / "workflows" / "learning-test.yml").read_text()


def test_burn_is_excluded_from_the_root_workspace_and_lock():
    root = (ROOT / "Cargo.toml").read_text()
    assert re.search(r'^exclude = \[.*"integrations/burn".*\]$', root, re.M)
    lock = (ROOT / "Cargo.lock").read_text()
    assert 'name = "burn"' not in lock
    assert (BURN / "Cargo.lock").is_file()


def test_burn_pin_is_exact_cpu_only_and_without_defaults():
    m = re.search(r"^burn = \{(.*)\}$", WORKSPACE, re.M)
    assert m, WORKSPACE
    spec = m.group(1)
    assert 'version = "=0.22.1"' in spec
    assert "default-features = false" in spec
    assert 'features = ["std", "flex", "autodiff", "optim"]' in spec
    main = (BURN / "ppo-cartpole" / "src" / "main.rs").read_text()
    assert 'pub const BURN_VERSION: &str = "0.22.1";' in main
    assert "backend flex" in main


def test_burn_crates_are_unpublished():
    assert re.search(r"^publish = false$", WORKSPACE, re.M)
    for crate in ("tessera-rl-burn", "ppo-cartpole"):
        assert "publish.workspace = true" in (BURN / crate / "Cargo.toml").read_text()


def test_burn_ci_job_is_separate_and_runs_the_sanity_gate():
    job = CI.split("\n  burn-check:\n", 1)[1].split("\n\n  ", 1)[0]
    assert "manifest-path: integrations/burn/Cargo.toml" in job
    assert "arguments: --config integrations/burn/deny.toml" in job
    assert job.count("--manifest-path integrations/burn/Cargo.toml") == 4
    assert "        run: make burn-gate" in job
    others = CI.replace(job, "")
    assert "integrations/burn" not in others.replace("burn-check", "")
    assert "burn" not in NIGHTLY


def test_burn_deny_policy_is_the_root_one_plus_one_exception():
    local = (BURN / "deny.toml").read_text()
    root = (ROOT / "deny.toml").read_text()
    exception = re.findall(r"^exceptions = .*$", local, re.M)
    assert exception == ['exceptions = [{ crate = "colored", allow = ["MPL-2.0"] }]']
    assert "MPL-2.0" not in root
    body = lambda t: [l for l in t.splitlines() if l.strip() and not l.startswith("#")]
    assert [l for l in body(local) if not l.startswith("exceptions")] == body(root)


PATHS = ROOT / "scripts" / "paths_changed.sh"


def test_burn_check_heavy_steps_all_wait_for_the_paths_filter():
    job = CI.split("\n  burn-check:\n", 1)[1].split("\n\n  ", 1)[0]
    steps = job.split("    steps:\n", 1)[1].split("\n      - ")
    guarded = [s for s in steps if "if: steps.paths.outputs.changed == 'true'" in s]
    # fmt, clippy, test, deny, smoke, gate.
    assert len(guarded) == 6
    for s in steps:
        if "integrations/burn" in s or "burn-gate" in s:
            assert s in guarded or s.startswith("id: paths"), s
    paths = next(s for s in steps if s.startswith("id: paths"))
    for prefix in ("integrations/burn/", "crates/tessera-rl-core/", "crates/tessera-rl-engine/", "crates/tessera-rl-classic/", ".github/workflows/ci.yml"):
        assert prefix in paths
    assert "fetch-depth: 0" in job and "shell: bash" in paths


def _repo(tmp_path):
    import subprocess

    def git(*a):
        return subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.txt").write_text("1")
    git("add", "-A")
    git("commit", "-qm", "one")
    base = git("rev-parse", "HEAD")
    return git, base


def _run(tmp_path, base, head, *prefixes):
    import subprocess

    return subprocess.run(["bash", str(PATHS), base, head, *prefixes], cwd=tmp_path, capture_output=True, text=True)


def test_paths_changed_true_only_for_a_listed_prefix(tmp_path):
    git, base = _repo(tmp_path)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "y.txt").write_text("2")
    git("add", "-A")
    git("commit", "-qm", "two")
    head = git("rev-parse", "HEAD")
    assert _run(tmp_path, base, head, "a/").stdout == "changed=false\n"
    assert _run(tmp_path, base, head, "a/", "b/").stdout == "changed=true\n"


def test_paths_changed_unknown_or_zero_base_counts_as_changed(tmp_path):
    git, base = _repo(tmp_path)
    for b in ("", "0" * 40, "f" * 40):
        p = _run(tmp_path, b, base, "zzz/")
        assert p.returncode == 0 and p.stdout == "changed=true\n", (b, p.stdout, p.stderr)


def test_paths_changed_needs_prefixes(tmp_path):
    git, base = _repo(tmp_path)
    assert _run(tmp_path, base, base).returncode == 2
