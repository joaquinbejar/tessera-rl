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
