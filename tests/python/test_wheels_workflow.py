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


def _job(name: str) -> str:
    match = re.search(rf"\n  {name}:\n(.*?)(?=\n  [a-z][\w-]*:\n|\Z)", WORKFLOW, re.DOTALL)
    assert match is not None
    return match.group(1)


def test_wheels_aarch64_linux_avoids_the_dockerless_self_hosted_runner():
    assert "m4-linux-arm64" not in WORKFLOW.split("jobs:", 1)[1]


def test_wheels_aarch64_linux_runs_on_the_hosted_arm_runner_in_the_pinned_image():
    job = _job("linux-aarch64")
    assert re.search(r"^    runs-on: ubuntu-24.04-arm$", job, re.M)
    image = re.search(r"image: (\S+)", job).group(1)
    assert re.fullmatch(r"quay.io/pypa/manylinux_2_28_aarch64@sha256:[0-9a-f]{64}", image)


def test_wheels_aarch64_linux_mirrors_the_x86_64_job_script_and_reports():
    job = _job("linux-aarch64")
    run = re.search(r"run: (scripts/wheel_job.sh .*)", job).group(1).split()
    assert run[1:3] == ["aarch64-unknown-linux-gnu", "manylinux_2_28"]
    assert all(re.fullmatch(r"[0-9a-f]{64}", sha) for sha in run[3:5]) and len(run) == 5
    assert re.search(r"options: (.*)", job).group(1).split() == ["--init"]
    assert "name: wheel-aarch64-unknown-linux-gnu" in job
    assert re.search(r"venv/bin/python\" -I -c 'import tessera_rl", job)
    assert re.search(r"^    permissions:\n      contents: read\n(?!      )", job, re.M)
    for report in ("dist/*.whl", "golden-report.json", "checks.json", "determinism.json"):
        assert report in job


def test_wheels_hosted_job_is_the_only_one_and_runs_only_on_dispatch_here():
    assert WORKFLOW.count("runs-on: ubuntu-") == 1
    triggers = re.search(r"^on:\n(.*?)\n\S", WORKFLOW, re.M | re.DOTALL).group(1)
    assert re.findall(r"^  (\w+):", triggers, re.M) == ["workflow_dispatch"]
    guard = re.search(r"^    if: (.*)$", _job("linux-aarch64"), re.M).group(1)
    assert "github.event_name == 'workflow_dispatch'" in guard
    assert "github.repository == 'joaquinbejar/tessera-rl'" in guard
    assert re.search(r"^    needs: preflight$", _job("linux-aarch64"), re.M)


def test_wheels_reports_are_uploaded_when_the_job_fails():
    uploads = re.findall(r"upload-artifact@\S+[^\n]*\n\s+if: always\(\)\n", WORKFLOW)
    assert len(uploads) == WORKFLOW.count("upload-artifact@") == 3


def test_wheels_leaves_nothing_needing_a_chown():
    assert "chown" not in WORKFLOW


def test_wheel_job_excludes_only_the_bench_tests():
    assert re.findall(r"--ignore\S*", JOB) == ['--ignore="$root/tests/python/test_bench.py"']
    assert "--deselect" not in JOB
