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
    assert all(re.fullmatch(r"[0-9a-f]{64}", sha) for sha in run[3:5]) and len(run) == 6
    assert run[5] == "release"
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


def test_wheels_x86_64_expects_fp_traps_and_the_hosted_arm_job_does_not():
    # #225: x86-64 implements trapped FP exceptions, so the unmasked FP
    # boundary test must check the enables there; the hosted arm64 core
    # may not, so its job leaves the variable unset.
    linux = _job("linux")
    entry = re.search(r"- runner: xaskar\n(.*?)\n    container:", linux, re.DOTALL).group(1)
    assert "target: x86_64-unknown-linux-gnu" in entry
    assert re.search(r'^            expect_fp_traps: "1"$', entry, re.M)
    assert "      TESSERA_EXPECT_FP_TRAPS: ${{ matrix.expect_fp_traps }}\n" in linux
    assert "TESSERA_EXPECT_FP_TRAPS" not in _job("linux-aarch64")
    assert WORKFLOW.count("TESSERA_EXPECT_FP_TRAPS") == 1


CARGO = (ROOT / "Cargo.toml").read_text()
W1_BUILD = (ROOT / "scripts" / "pgo_w1_build.sh").read_text()


def _entries():
    linux = _job("linux")
    return re.findall(r"^          - runner: (.*?)(?=^          - runner: |^    container:)", linux, re.M | re.DOTALL)


def test_wheels_linux_x86_64_builds_with_the_wheel_profile():
    entries = _entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.startswith("xaskar") and "target: x86_64-unknown-linux-gnu" in entry
    assert re.search(r"^            profile: wheel$", entry, re.M)
    run = re.search(r"run: (scripts/wheel_job.sh .*)", _job("linux")).group(1)
    assert run.endswith('"${{ matrix.uv_sha }}" "${{ matrix.profile }}"')
    assert re.search(r"^    name: W1 \(\$\{\{ matrix.target \}\}\)$", _job("linux"), re.M)


def test_wheels_other_targets_keep_the_default_release_profile():
    for job in ("linux-aarch64", "macos"):
        run = re.search(r"run: (scripts/wheel_job.sh .*)", _job(job)).group(1).split()
        assert run[-1] == "release", job
    assert WORKFLOW.count("profile: wheel") == 1
    assert "--profile wheel" not in WORKFLOW


def test_wheel_job_builds_with_the_named_profile_only():
    assert 'args=(--profile "$profile" --locked --target "$target" --out dist "${features[@]}")' in JOB
    # The only --release build is W0, for the comparison, and only under the wheel profile.
    assert JOB.count("--release --locked --target") == 1
    assert 'w0_args=(--release --locked --target "$target" --out "$WHEEL_RUN/w0-dist" "${features[@]}")' in JOB
    assert re.search(r'^if \[ "\$profile" = wheel \]; then\n.*\n    nice -n 10 uvx [^\n]* build "\$\{w0_args\[@\]\}"\nfi$', JOB, re.M)
    assert re.search(r'^    release \| wheel\) ;;$', JOB, re.M)


def test_wheel_profile_is_fat_lto_one_unit_and_matches_the_measured_w1_build():
    section = CARGO.split("[profile.wheel]\n", 1)[1].split("\n\n", 1)[0]
    settings = [ln for ln in section.splitlines() if ln and not ln.startswith("#")]
    assert settings == ['inherits = "release"', 'lto = "fat"', "codegen-units = 1", 'panic = "unwind"']
    # The published W1 is built as the PGO ladder measured it.
    assert "build --locked --profile wheel" in W1_BUILD and "--compatibility manylinux_2_28" in W1_BUILD


def test_wheels_x86_64_artifact_is_named_with_profile_commit_and_sha256():
    linux = _job("linux")
    assert re.search(r"^      - uses: actions/checkout@[0-9a-f]{40}.*\n        id: checkout$", linux, re.M)
    built = re.search(r"^      - id: built\n(.*?)(?=^      - )", linux, re.M | re.DOTALL).group(1)
    assert "COMMIT: ${{ steps.checkout.outputs.commit }}" in built and "PROFILE: ${{ matrix.profile }}" in built
    assert 'echo "name=wheel-$TARGET-$PROFILE-$COMMIT-$sha" >> "$GITHUB_OUTPUT"' in built
    assert "if: always()" in built
    assert "          name: ${{ steps.built.outputs.name }}\n" in linux
    for report in ("determinism-w0.json", "compare.json"):
        assert report in linux.split("upload-artifact@", 1)[1]
        assert report in re.search(r"run: (rm -rf .*)", linux).group(1)


def test_wheel_job_suite_names_its_skips():
    # -rs: every skipped test is listed with its reason in the job log.
    suite = re.search(r'^nice -n 10 "\$venv/bin/python" -I -m pytest (.*)$', JOB, re.M).group(1)
    assert suite.split()[:3] == ["-q", "-rs", "-p"]


def test_wheel_job_enables_onnx_only_on_linux_glibc_and_macos():
    assert re.search(r'^    \*-unknown-linux-gnu \| \*-apple-darwin\) features=\(--features onnx\) ;;$', JOB, re.M)
    # No Windows wheel job exists, so no wheel is built for Windows with it.
    assert not re.search(r"runs-on: .*windows", WORKFLOW)
