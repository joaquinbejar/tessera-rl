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


def test_burn_ci_job_is_separate_and_the_nightly_gate_reports_only():
    job = CI.split("\n  burn-check:\n", 1)[1].split("\n\n  ", 1)[0]
    assert "manifest-path: integrations/burn/Cargo.toml" in job
    assert job.count("--manifest-path integrations/burn/Cargo.toml") == 4
    others = CI.replace(job, "")
    assert "integrations/burn" not in others.replace("burn-check", "")
    nightly = NIGHTLY.split("\n  burn-sanity:\n", 1)[1]
    assert "    continue-on-error: true\n" in nightly
    assert "make burn-gate" in nightly
