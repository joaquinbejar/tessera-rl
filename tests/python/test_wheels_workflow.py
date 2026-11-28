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
    assert run[5] == "wheel"
    assert re.search(r"options: (.*)", job).group(1).split() == ["--init"]
    assert "name: wheel-aarch64-unknown-linux-gnu" in job
    assert re.search(r"venv/bin/python\" -I -c 'import tessera_rl", job)
    assert re.search(r"^    permissions:\n      contents: read\n(?!      )", job, re.M)
    for report in ("dist/*.whl", "golden-report.json", "checks.json", "determinism.json", "determinism-w0.json", "compare.json"):
        assert report in job


def test_wheels_hosted_jobs_are_the_arm_ones_and_run_only_on_dispatch_here():
    # The aarch64 manylinux and musllinux legs (#149, #350).
    assert WORKFLOW.count("runs-on: ubuntu-") == 2
    triggers = re.search(r"^on:\n(.*?)\n\S", WORKFLOW, re.M | re.DOTALL).group(1)
    assert re.findall(r"^  (\w+):", triggers, re.M) == ["workflow_dispatch"]
    for job in ("linux-aarch64", "musl-aarch64"):
        assert "    runs-on: ubuntu-24.04-arm\n" in _job(job)
        guard = re.search(r"^    if: (.*)$", _job(job), re.M).group(1)
        assert "github.event_name == 'workflow_dispatch'" in guard
        assert "github.repository == 'joaquinbejar/tessera-rl'" in guard
        assert re.search(r"^    needs: preflight$", _job(job), re.M)


def test_wheels_reports_are_uploaded_when_the_job_fails():
    uploads = re.findall(r"upload-artifact@\S+[^\n]*\n\s+if: always\(\)\n", WORKFLOW)
    assert len(uploads) == WORKFLOW.count("upload-artifact@") == 6


def test_wheels_leaves_nothing_needing_a_chown():
    assert "chown" not in WORKFLOW


def test_wheel_job_excludes_only_the_bench_and_compat_lock_tests():
    # test_compat_locks.py checks the checkout's locks with uv, which the
    # wheel images do not have; make py-test runs it.
    assert re.findall(r"--ignore\S*", JOB) == [
        '--ignore="$root/tests/python/test_bench.py"',
        '--ignore="$root/tests/python/test_compat_locks.py"',
    ]
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
    assert "TESSERA_EXPECT_FP_TRAPS: ${{" not in _job("linux-aarch64")
    assert "TESSERA_EXPECT_FP_TRAPS" not in _job("linux-aarch64")
    # musllinux (#350): neither leg expects the traps (musl has no
    # feenableexcept).
    assert "TESSERA_EXPECT_FP_TRAPS:" not in _job("musl-x86_64")
    assert "TESSERA_EXPECT_FP_TRAPS:" not in _job("musl-aarch64")
    # Windows x86-64 (#350): traps through _controlfp_s, so expected.
    assert '      TESSERA_EXPECT_FP_TRAPS: "1"\n' in _job("windows")
    assert WORKFLOW.count("TESSERA_EXPECT_FP_TRAPS:") == 2


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
    for job in ("macos", "musl-x86_64", "windows"):
        run = re.search(r"run: (scripts/wheel_job.sh .*)", _job(job)).group(1).split()
        assert run[-1] == "release", job
    assert WORKFLOW.count("profile: wheel") == 1
    assert "--profile wheel" not in WORKFLOW


def test_wheel_job_builds_with_the_named_profile_only():
    assert 'args=(--profile "$profile" --locked --target "$target" --out dist "${features[@]}")' in JOB
    # The only --release build of the first-party wheel is W0, for the
    # comparison, and only under the wheel profile. The other --release build
    # is the host example's (ob_args, #313), which is not tessera-rl.
    assert JOB.count("--release --locked --target") == 2
    assert JOB.count('ob_args=(--release --locked --target "$target" --out "$WHEEL_RUN/orderbook-dist"') == 1
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
    suite = re.search(r'^nice -n 10 "\$venv/\$bindir/python\$exe" -I -m pytest (.*)$', JOB, re.M).group(1)
    assert suite.split()[:3] == ["-q", "-rs", "-p"]


def test_wheel_job_enables_onnx_only_on_linux_and_macos():
    # musl too (#350, ADR-0057 section 6: a dynamically linked musl build).
    assert re.search(
        r'^    \*-unknown-linux-gnu \| \*-unknown-linux-musl \| \*-apple-darwin\) features=\(--features onnx\) ;;$',
        JOB,
        re.M,
    )
    # The Windows wheel (#350, option A) is built without it: ADR-0057 has no
    # Windows loading rows, and the onnx smoke checks the no-evaluator error.
    assert re.search(r"^    x86_64-pc-windows-msvc\) features=\(\) ;;$", JOB, re.M)
    assert "has no ONNX evaluator" in JOB


# #313: the host example wheel's checks on aarch64 Linux and macOS arm64.


def _env_flag(job: str):
    m = re.search(r'^      TESSERA_WHEEL_ORDERBOOK: "?(\w+)"?$', _job(job), re.M)
    return m and m.group(1)


def test_wheels_orderbook_leg_runs_on_aarch64_linux_and_macos_only():
    assert _env_flag("linux-aarch64") == "1"
    assert _env_flag("macos") == "1"
    # x86-64 runs it in CI's python job (make py-test-orderbook).
    assert _env_flag("linux") is None
    assert WORKFLOW.count("TESSERA_WHEEL_ORDERBOOK") == 2


def test_wheels_orderbook_leg_stays_dispatch_only_with_an_approved_window():
    # No tag or schedule trigger: release builds stay a manual dispatch
    # with the coordinator's window (the 0.3.0 plan).
    triggers = re.search(r"^on:\n(.*?)\n\S", WORKFLOW, re.M | re.DOTALL).group(1)
    assert re.findall(r"^  (\w+):", triggers, re.M) == ["workflow_dispatch"]
    assert re.search(r"^      approved_window:\n(?:        .*\n)*?        required: true$", triggers, re.M)
    assert re.search(r"^    needs: \[preflight, linux\]$", _job("macos"), re.M)


def test_wheel_job_orderbook_leg_builds_installs_and_tests_the_example():
    # Off unless asked, and only 0 or 1.
    assert 'orderbook_leg="${TESSERA_WHEEL_ORDERBOOK:-0}"' in JOB
    assert re.search(r"^    0 \| 1\) ;;$", JOB, re.M)
    # Built for the same target and tag, release profile, from its own lock.
    build = re.search(r"ob_args=\((.*?)\)\n", JOB, re.DOTALL).group(1).split()
    assert {"--release", "--locked", "--target", '"$target"', "-m", "examples/host_orderbook/Cargo.toml"} <= set(build)
    assert 'ob_args+=(--compatibility "$compatibility")' in JOB
    # Installed with no dependencies of its own, next to the first-party wheel.
    assert re.search(r'uv pip install --python "\$venv/\$bindir/python\$exe" --no-deps "\$WHEEL_RUN"/orderbook-dist/\*\.whl', JOB)
    # It must import, and its test file runs on its own: a skip cannot pass.
    assert "import tessera_example_orderbook._tessera_host" in JOB
    assert re.search(r'-m pytest -q -rs -p no:cacheprovider \\\n\s+"\$root/tests/python/test_host_orderbook.py"', JOB)
    # A failing import or test sets the status (not `|| true`), which is
    # recorded and fails the job.
    assert re.search(r'--rootdir "\$root"; \} \|\| orderbook=\$\?$', JOB, re.M)
    assert 'statuses+=(orderbook="$orderbook")' in JOB
    assert JOB.rstrip().endswith('[ "$orderbook" -eq 0 ]')
    # The example wheel is not uploaded.
    assert "orderbook-dist" not in WORKFLOW


def test_wheels_orderbook_golden_is_one_pinned_pair_asserted_by_the_example():
    # The book is integer-only, so one GOLDEN_11 holds on every platform
    # (#313); this pins its shape and the assertion, and the legs' runs
    # check the value.
    src = (ROOT / "tests" / "python" / "test_host_orderbook.py").read_text()
    assert re.search(r"^GOLDEN_11 = \(\n    \"[0-9a-f]{64}\",\n    \"[0-9a-f]{64}\",\n\)$", src, re.M)
    assert "assert first == GOLDEN_11" in src


def test_wheels_windows_job_is_hosted_guarded_and_pinned():
    # #350, option A: GitHub-hosted, the fork and dispatch guard of the other
    # hosted legs, SHA-pinned actions, and the tools by their sha256.
    job = _job("windows")
    assert re.search(r"^    runs-on: windows-\d{4}$", job, re.M), job
    assert "if: github.event_name == 'workflow_dispatch' && github.repository == 'joaquinbejar/tessera-rl'" in job
    for uses in re.findall(r"uses: (\S+)", job):
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", uses), uses
    run = re.search(r"scripts/wheel_job\.sh x86_64-pc-windows-msvc \"\" ([0-9a-f]{64}) ([0-9a-f]{64}) release", job)
    assert run, job
    assert "name: wheel-x86_64-pc-windows-msvc" in job


WHEEL_JOBS = ["linux", "linux-aarch64", "macos", "musl-x86_64", "musl-aarch64", "windows"]


def test_wheels_preflight_outputs_the_commit_and_its_time():
    # #654: the legs' images have no git, so preflight derives both.
    job = _job("preflight")
    assert "    outputs:\n      commit: ${{ steps.rev.outputs.commit }}\n      epoch: ${{ steps.rev.outputs.epoch }}\n" in job
    rev = re.search(r"^      - id: rev\n(.*?)(?=^      - |\Z)", job, re.M | re.DOTALL).group(1)
    assert 'epoch="$(git log -1 --format=%ct "$GITHUB_SHA")"' in rev
    assert 'echo "commit=$GITHUB_SHA" >> "$GITHUB_OUTPUT"' in rev and 'echo "epoch=$epoch" >> "$GITHUB_OUTPUT"' in rev
    assert job.index("actions/checkout@") < job.index("- id: rev")


def test_wheels_every_leg_gets_the_sbom_inputs_from_preflight():
    assert sorted(WHEEL_JOBS) == sorted(j for j in re.findall(r"^  ([\w-]+):\n", WORKFLOW.split("\njobs:\n", 1)[1], re.M)
                                        if j != "preflight")
    for name in WHEEL_JOBS:
        job = _job(name)
        assert "scripts/wheel_job.sh" in job, name
        env = re.search(r"^    env:\n((?:      .*\n)+)", job, re.M).group(1)
        assert "      WHEEL_COMMIT: ${{ needs.preflight.outputs.commit }}\n" in env, name
        assert "      SOURCE_DATE_EPOCH: ${{ needs.preflight.outputs.epoch }}\n" in env, name
        needs = re.search(r"^    needs: (.*)$", job, re.M).group(1)
        assert needs in ("preflight", "[preflight, linux]"), (name, needs)


def test_wheels_w1_legs_are_x86_64_linux_and_both_aarch64_linux():
    # #770: the aarch64 glibc and musl legs ship W1; macOS stays W0 until its leg runs.
    w1 = {"linux"}
    for job in ("linux-aarch64", "musl-aarch64"):
        run = re.search(r"(scripts/wheel_job.sh .*)", _job(job)).group(1).split()
        assert run[-1] == "wheel", job
        assert re.search(r"^    name: W1 \(aarch64-unknown-linux-(gnu|musl)\)$", _job(job), re.M), job
        assert "determinism-w0.json" in _job(job) and "compare.json" in _job(job), job
        assert re.search(r"^    timeout-minutes: 150$", _job(job), re.M), job
        w1.add(job)
    for job in WHEEL_JOBS:
        if job not in w1:
            assert re.search(r"^    name: W0 \(", _job(job), re.M), job


def test_wheels_musl_aarch64_runs_wheel_job_in_the_pinned_image_without_a_job_container():
    # #790: JavaScript actions cannot run in an Alpine container on arm64, so
    # checkout and upload run on the host and only wheel_job.sh is in the image.
    job = _job("musl-aarch64")
    assert "    container:" not in job
    image = re.search(r"^      IMAGE: (\S+)$", job, re.M).group(1)
    assert re.fullmatch(r"quay.io/pypa/musllinux_1_2_aarch64@sha256:[0-9a-f]{64}", image)
    run = re.search(r"^      - run: (docker run .*)$", job, re.M).group(1)
    assert run.startswith('docker run --rm --init -e WHEEL_RUN=/work/.wheel-run -e WHEEL_COMMIT -e SOURCE_DATE_EPOCH ')
    assert '-v "$GITHUB_WORKSPACE:/work" -w /work "$IMAGE" bash scripts/wheel_job.sh aarch64-unknown-linux-musl musllinux_1_2 ' in run
    assert job.index("actions/checkout@") < job.index("- run: docker run") < job.index("actions/upload-artifact@")
