# SPDX-License-Identifier: MIT OR Apache-2.0
"""The gungraun-bless workflow (ADR-0059 section 5): it blesses in the gate's
own environment, uploads one artifact, and can never write to the repository."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
BLESS = (WORKFLOWS / "gungraun-bless.yml").read_text()
CI = (WORKFLOWS / "ci.yml").read_text()


def _job(text, name):
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))[name]


GATE = _job(CI, "gungraun")
JOB = _job(BLESS, "bless")


def _one(pattern, text):
    found = re.findall(pattern, text, re.M)
    assert len(found) == 1, (pattern, found)
    return found[0]


def _steps(job):
    return job.split("    steps:\n", 1)[1]


def test_gungraun_bless_is_dispatch_only():
    on = BLESS.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
    assert re.findall(r"^  ([a-z_]+):", on, re.M) == ["workflow_dispatch"]


def test_gungraun_bless_runs_on_the_gate_runner():
    runs_on = r"^    runs-on: (.*)$"
    assert _one(runs_on, JOB) == "[self-hosted, xaskar]"
    assert _one(runs_on, JOB) == _one(runs_on, GATE)


def test_gungraun_bless_image_digest_equals_the_gate_job():
    image = r"^      image: (\S+)$"
    gate = _one(image, GATE)
    assert re.fullmatch(r"rust:[0-9.]+@sha256:[0-9a-f]{64}", gate)
    assert _one(image, JOB) == gate
    env = r"^      TESSERA_GUNGRAUN_IMAGE: (\S+)$"
    assert _one(env, JOB) == _one(env, GATE) == gate


def test_gungraun_bless_container_options_and_tools_equal_the_gate_job():
    for pattern in [
        r"^      options: (.*)$",
        r"^      - run: (apt-get .*valgrind.*)$",
        r"^      - run: (cargo install --locked gungraun-runner .*)$",
    ]:
        assert _one(pattern, JOB) == _one(pattern, GATE)


def test_gungraun_bless_workflow_env_equals_ci():
    head = r"^env:\n((?:  .*\n)+)"
    assert _one(head, BLESS) == _one(head, CI)


def test_gungraun_bless_blesses_then_checks():
    runs = re.findall(r"^      - run: (scripts/gungraun\.sh \w+)$", _steps(JOB), re.M)
    assert runs == ["scripts/gungraun.sh bless", "scripts/gungraun.sh check"]


def test_gungraun_bless_permissions_are_read_only():
    assert _one(r"^permissions:\n((?:  .*\n)+)", BLESS) == "  contents: read\n"
    assert "write" not in BLESS.split("\njobs:\n", 1)[0].split("\npermissions:\n", 1)[1]
    assert "permissions:" not in JOB
    assert "contents: write" not in BLESS


def test_gungraun_bless_never_pushes():
    assert "git push" not in BLESS
    assert not re.search(r"\bgit (commit|push|tag)\b", BLESS)
    assert "persist-credentials: false" in JOB
    assert "secrets." not in BLESS
    assert "GITHUB_TOKEN" not in BLESS


def test_gungraun_bless_uploads_one_artifact_of_baselines_and_manifest():
    uploads = re.findall(r"uses: actions/upload-artifact@([0-9a-f]{40}) ", JOB)
    assert len(uploads) == 1
    path = _one(r"^          path: \|\n((?:            .*\n)+)", JOB)
    assert [p.strip() for p in path.splitlines()] == [
        "benches/gungraun/**/*.base@ref",
        "benches/gungraun/environment.toml",
    ]
    assert "if-no-files-found: error" in JOB


def test_gungraun_bless_logs_hashes_and_digest():
    assert "| xargs sha256sum" in JOB
    assert "steps.upload.outputs.artifact-digest" in JOB


def test_gungraun_bless_pins_actions_by_sha():
    for ref in re.findall(r"uses: (\S+)", BLESS):
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", ref), ref


def test_gungraun_bless_gives_the_workspace_back():
    assert _steps(JOB).rstrip().endswith(
        'run: chown -R "$(stat -c %u:%g "$GITHUB_WORKSPACE/..")" "$GITHUB_WORKSPACE"'
    )
    assert _steps(GATE).rstrip().endswith(
        'run: chown -R "$(stat -c %u:%g "$GITHUB_WORKSPACE/..")" "$GITHUB_WORKSPACE"'
    )


def _checkouts(job):
    return re.findall(r"^      - uses: actions/checkout@[0-9a-f]{40}.*\n((?:        .*\n)*)", _steps(job), re.M)


def test_gungraun_bless_checks_out_exactly_the_branch_input():
    checkouts = _checkouts(JOB)
    assert len(checkouts) == 1
    assert len(re.findall(r"actions/checkout@", BLESS)) == 1
    refs = re.findall(r"^          ref: (.*)$", checkouts[0], re.M)
    assert refs == ["${{ inputs.branch }}"]


def _run_scripts(text):
    """The text of every `run:` value, block scalars included."""
    lines, out, i = text.splitlines(), [], 0
    while i < len(lines):
        m = re.match(r"^(\s*)(?:- )?run: ?(.*)$", lines[i])
        if not m:
            i += 1
            continue
        indent, value = len(m.group(1)), m.group(2)
        block = [value]
        if value.strip() in ("|", ">", "|-", ">-"):
            i += 1
            while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent):
                block.append(lines[i])
                i += 1
        else:
            i += 1
        out.append("\n".join(block))
    return out


def test_gungraun_bless_no_run_interpolates_an_expression():
    scripts = _run_scripts(BLESS)
    assert len(scripts) >= 6
    for script in scripts:
        assert "${{" not in script, script


def test_gungraun_bless_records_the_blessed_commit_in_the_artifact_name():
    steps = _steps(JOB)
    head = re.search(r"^      - id: head\n((?:        .*\n)+)", steps, re.M)
    assert head, "no step with id head"
    body = head.group(1)
    assert "COMMIT: ${{ steps.checkout.outputs.commit }}" in body
    assert '[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] ||' in body
    assert 'echo "blessed commit: $COMMIT"' in body
    assert 'echo "sha=$COMMIT" >> "$GITHUB_OUTPUT"' in body
    assert re.search(r"^      - uses: actions/checkout@[0-9a-f]{40}.*\n        id: checkout\n", steps, re.M)
    assert steps.index("- id: head") < steps.index("scripts/gungraun.sh bless")
    name = _one(r"^          name: (gungraun-baseline-.*)$", JOB)
    assert name == "gungraun-baseline-${{ github.run_id }}-${{ steps.head.outputs.sha }}"


def test_gungraun_bless_job_has_a_timeout():
    minutes = int(_one(r"^    timeout-minutes: (\d+)$", JOB))
    assert 0 < minutes <= 180


def test_gungraun_bless_no_run_block_calls_git():
    # The job container runs as root over a workspace the runner user owns;
    # git refuses such a repository ("dubious ownership"), which failed the
    # first run on main. No step may depend on git working there.
    for script in _run_scripts(BLESS):
        assert not re.search(r"(^|[\s;&|(`$])git(\s|$)", script), script

def _indent(line):
    return len(line) - len(line.lstrip())


def test_gungraun_bless_expressions_only_in_step_with_and_env_values():
    # Allow-list: an expression may appear only as the value of a key under
    # a step's `with:` or `env:` (step keys at 8 spaces, their entries at 10).
    # Anywhere else (a run script, a shell, a name, an if, the job) it fails.
    lines = BLESS.splitlines()
    found = 0
    for i, line in enumerate(lines):
        if "${{" not in line:
            continue
        found += 1
        assert _indent(line) == 10 and re.match(r"^ {10}[A-Za-z0-9_-]+: ", line), (i + 1, line)
        parent = next(lines[j] for j in range(i - 1, -1, -1) if lines[j].strip() and _indent(lines[j]) < 10)
        assert parent in ("        with:", "        env:"), (i + 1, line, parent)
    assert found == 5


def test_gungraun_bless_no_step_or_job_sets_a_shell():
    assert not re.search(r"^\s*(?:- )?shell:", BLESS, re.M)
    assert "defaults:" not in BLESS
