# SPDX-License-Identifier: MIT OR Apache-2.0
"""scripts/wheel_job.sh refuses the caller's build environment before it creates, fetches or
builds anything (issue #212)."""

import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "wheel_job.sh"

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX only: a bash script with sh fakes")

# The refused set, literally: it mirrors scripts/pgo_candidate_checks.sh, scripts/pgo_w2_build.sh
# and the L1 run script, which name the same set; plus every CARGO_PROFILE_* / CARGO_TARGET_* name.
REFUSED = ["RUSTFLAGS", "RUSTDOCFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_ENCODED_RUSTDOCFLAGS", "CARGO_BUILD_RUSTFLAGS",
           "CARGO_BUILD_RUSTDOCFLAGS", "CARGO_BUILD_TARGET", "CARGO_BUILD_TARGET_DIR", "RUSTC", "RUSTC_WRAPPER",
           "RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_RUSTC", "CARGO_BUILD_RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER"]
PATTERN = "grep -Eq '^CARGO_(PROFILE|TARGET)_[A-Za-z0-9_]*='"
PATTERNED = ["CARGO_TARGET_DIR", "CARGO_PROFILE_RELEASE_LTO", "CARGO_PROFILE_RELEASE_OPT_LEVEL",
             "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS", "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER"]
# The caller's environment as CI's ci.yml sets it, and worse.
HOSTILE = {"RUSTFLAGS": "-D warnings", "CARGO_TERM_COLOR": "always", "CARGO_ENCODED_RUSTFLAGS": "-Cdebuginfo=2",
           "CARGO_TARGET_DIR": "/nonexistent/target", "CARGO_PROFILE_RELEASE_LTO": "fat", "RUSTC_WRAPPER": "/nonexistent"}
SCRUB = tuple(REFUSED)


def run(tmp_path: Path, env: dict, profile: str = "release"):
    """The real script with a logging curl first on PATH; returns the process, the log and WHEEL_RUN."""
    fake, log, wheel_run = tmp_path / "fake", tmp_path / "log", tmp_path / "wheel-run"
    fake.mkdir()
    curl = fake / "curl"
    curl.write_text(f"#!/bin/sh\necho \"curl $*\" >> {log}\nexit 1\n")
    curl.chmod(curl.stat().st_mode | stat.S_IXUSR)
    env = {**env, "PATH": f"{fake}:{env['PATH']}", "WHEEL_RUN": str(wheel_run)}
    p = subprocess.run(["bash", str(SCRIPT), "x86_64-unknown-linux-gnu", "manylinux_2_28", "0" * 64, "0" * 64, profile],
                       env=env, capture_output=True, text=True, check=False, cwd=tmp_path)
    return p, log.read_text() if log.exists() else "", wheel_run


def clean() -> dict:
    return {k: v for k, v in os.environ.items() if k not in SCRUB and not k.startswith(("CARGO_PROFILE_", "CARGO_TARGET_"))}


def test_wheel_job_names_the_refused_set():
    text = SCRIPT.read_text()
    start = text.index('BUILD_VARS="') + len('BUILD_VARS="')
    assert text[start:text.index('"', start)].split() == REFUSED
    assert PATTERN in text
    # Before the script exports its own CARGO_TARGET_DIR and creates anything.
    assert text.index(PATTERN) < text.index("export RUSTUP_HOME=") < text.index("mkdir -p")


@pytest.mark.parametrize("name", REFUSED + PATTERNED)
@pytest.mark.parametrize("value", ["x", ""], ids=["set", "empty"])
def test_wheel_job_refuses_an_ambient_build_variable_before_anything(tmp_path, name, value):
    p, log, wheel_run = run(tmp_path, {**clean(), name: value})
    msg = (f"wheel_job: {name} is set in the environment; refused" if name in REFUSED
           else f"wheel_job: {name} set in the environment; refused")
    assert p.returncode == 2 and msg in p.stderr, p.stderr
    assert log == "" and not wheel_run.exists()


def test_wheel_job_refuses_the_hostile_environment_unscrubbed(tmp_path, monkeypatch):
    # The caller's environment as is, not through clean().
    for k, v in HOSTILE.items():
        monkeypatch.setenv(k, v)
    p, log, wheel_run = run(tmp_path, dict(os.environ))
    assert p.returncode == 2 and "set in the environment; refused" in p.stderr, p.stderr
    assert log == "" and not wheel_run.exists()


def test_wheel_job_clean_environment_passes_the_check(tmp_path, monkeypatch):
    # CARGO_TERM_COLOR is cosmetic and not refused; the script goes on to its first fetch.
    monkeypatch.setenv("CARGO_TERM_COLOR", "always")
    p, log, wheel_run = run(tmp_path, clean())
    assert "refused" not in p.stderr
    assert log.startswith("curl ") and "rustup-init" in log and wheel_run.is_dir()


@pytest.mark.parametrize("profile", ["", "bench", "dev", "Release", "wheel "], ids=["missing", "bench", "dev", "case", "space"])
def test_wheel_job_refuses_a_profile_other_than_release_or_wheel(tmp_path, profile):
    p, log, wheel_run = run(tmp_path, clean(), profile)
    assert p.returncode == 2
    assert "profile must be release or wheel" in p.stderr
    assert log == "" and not wheel_run.exists()


@pytest.mark.parametrize("profile", ["release", "wheel"])
def test_wheel_job_accepts_release_and_wheel(tmp_path, profile):
    p, log, wheel_run = run(tmp_path, clean(), profile)
    assert "profile must be" not in p.stderr and "refused" not in p.stderr
    assert log.startswith("curl ") and wheel_run.is_dir()
