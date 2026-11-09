# SPDX-License-Identifier: MIT OR Apache-2.0
"""The fixes from wheels.yml's first run (42618212792) stay in place."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github" / "workflows" / "wheels.yml").read_text()
JOB = (ROOT / "scripts" / "wheel_job.sh").read_text()


def test_wheels_container_reaps_orphans_and_runs_as_the_runner_user():
    limits = re.search(r"limits: (.*)", WORKFLOW).group(1).split()
    assert {"--init", "--user", "1000:1000", "--cpuset-cpus=12-15"} <= set(limits)


def test_wheels_defers_aarch64_linux_and_says_so():
    assert "runner: m4-linux-arm64" not in WORKFLOW
    assert "aarch64 Linux is deferred" in WORKFLOW


def test_wheels_reports_are_uploaded_when_the_job_fails():
    uploads = re.findall(r"upload-artifact@\S+[^\n]*\n\s+if: always\(\)\n", WORKFLOW)
    assert len(uploads) == WORKFLOW.count("upload-artifact@") == 2


def test_wheels_leaves_nothing_needing_a_chown():
    assert "chown" not in WORKFLOW


def test_wheel_job_excludes_only_the_bench_tests():
    assert re.findall(r"--ignore\S*", JOB) == ['--ignore="$root/tests/python/test_bench.py"']
    assert "--deselect" not in JOB
