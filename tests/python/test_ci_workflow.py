# SPDX-License-Identifier: MIT OR Apache-2.0
"""Which CI jobs may run on either Linux x86-64 host, and which stay on xaskar."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
CI = (WORKFLOWS / "ci.yml").read_text()
WHEELS = (WORKFLOWS / "wheels.yml").read_text()

FORK_GUARD = (
    "if: github.event_name == 'push' || "
    "github.event.pull_request.head.repo.full_name == github.repository"
)
HOST_PRINT = "- run: hostname && grep -m1 'model name' /proc/cpuinfo"
MOVED = [
    "fmt",
    "fuzz-check",
    "build-id",
    "loom",
    "msrv",
    "docs",
    "readme",
    "pufferlib",
    "python",
]
ON_XASKAR = ["gungraun", "deny"]


def _jobs(text):
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


JOBS = _jobs(CI)


def _runs_on(job):
    return re.search(r"^    runs-on: (.*)$", JOBS[job], re.M).group(1)


@pytest.mark.parametrize("job", MOVED)
def test_ci_moved_job_runs_on_either_linux_host_and_names_it(job):
    assert _runs_on(job) == "[self-hosted, linux-x64]"
    steps = JOBS[job].split("    steps:\n", 1)[1]
    first = re.search(r"^      - .*$", steps, re.M).group(0)
    assert first == "      " + HOST_PRINT


@pytest.mark.parametrize("job", ON_XASKAR)
def test_ci_pinned_job_stays_on_xaskar(job):
    assert _runs_on(job) == "[self-hosted, xaskar]"


def test_ci_only_the_moved_jobs_use_linux_x64():
    assert sorted(j for j in JOBS if "linux-x64" in _runs_on(j)) == sorted(MOVED)
    assert "linux-x64" not in WHEELS


@pytest.mark.parametrize(
    "job,entries",
    [
        ("clippy-test", ["os: [xaskar, m4-linux-arm64, m4]"]),
        ("math-diff", ["os: [xaskar, m4-linux-arm64]"]),
        ("musl", ["- os: xaskar\n            target: x86_64-unknown-linux-musl"]),
    ],
)
def test_ci_matrix_job_keeps_its_xaskar_entry(job, entries):
    assert _runs_on(job) == '[self-hosted, "${{ matrix.os }}"]'
    for entry in entries:
        assert entry in JOBS[job]


def test_ci_merge_condition_job_names_are_unchanged():
    assert "name: clippy and test (${{ matrix.os }})" in JOBS["clippy-test"]
    assert "name: math differential (${{ matrix.os }})" in JOBS["math-diff"]


def test_ci_wheels_linux_x86_64_stays_on_xaskar():
    assert "- runner: xaskar" in WHEELS


@pytest.mark.parametrize("job", sorted(JOBS))
def test_ci_self_hosted_job_keeps_the_fork_guard(job):
    if "self-hosted" in _runs_on(job):
        assert FORK_GUARD in JOBS[job]


def test_ci_python_job_expects_fp_traps():
    # #225: on x86-64 the unmasked FP boundary test must not skip.
    assert re.search(r'^    env:\n(?:      .*\n)*?      TESSERA_EXPECT_FP_TRAPS: "1"$', JOBS["python"], re.M)


def test_ci_python_job_requires_dash():
    # The bless workflow's run blocks are checked under dash there, never skipped.
    assert re.search(r'^    env:\n(?:      .*\n)*?      TESSERA_REQUIRE_DASH: "1"$', JOBS["python"], re.M)


def test_ci_only_the_python_job_requires_dash():
    assert [j for j, body in JOBS.items() if "TESSERA_REQUIRE_DASH" in body] == ["python"]
    assert "TESSERA_REQUIRE_DASH" not in WHEELS


def test_ci_only_the_python_job_expects_fp_traps():
    assert [j for j, body in JOBS.items() if "TESSERA_EXPECT_FP_TRAPS" in body] == ["python"]


MAKEFILE = (ROOT / "Makefile").read_text()
EXAMPLE = ROOT / "examples" / "host_orderbook"


def test_ci_python_job_runs_the_host_example_after_the_test_host_wheel():
    steps = JOBS["python"]
    assert "run: make py-test-orderbook" in steps
    assert steps.index("run: make py-test-host") < steps.index("run: make py-test-orderbook")


def test_ci_host_example_target_checks_installs_and_tests_the_example():
    target = MAKEFILE.split("\npy-test-orderbook:\n", 1)[1].split("\n\n", 1)[0]
    assert "cargo fmt --manifest-path $(ORDERBOOK) --check" in target
    assert "cargo clippy --manifest-path $(ORDERBOOK) --all-targets --locked -- -D warnings" in target
    assert "--reinstall ./examples/host_orderbook" in target
    assert "--group torch pytest -q tests/python/test_host_orderbook.py" in target
    assert target.rstrip().endswith("uv sync --locked --python 3.11 || status=1; exit $$status")
    assert "ORDERBOOK = examples/host_orderbook/Cargo.toml" in MAKEFILE


def test_ci_host_example_is_its_own_unpublished_workspace_with_a_lockfile():
    manifest = (EXAMPLE / "Cargo.toml").read_text()
    assert "\n[workspace]\n" in manifest and "publish = false" in manifest
    assert (EXAMPLE / "Cargo.lock").is_file()
    assert '"examples/host_orderbook"' in (ROOT / "Cargo.toml").read_text().split("default-members", 1)[0]
    pyproject = (EXAMPLE / "pyproject.toml").read_text()
    assert '"example/OrderBook-v1" = "tessera_example_orderbook._tessera_host:make"' in pyproject
