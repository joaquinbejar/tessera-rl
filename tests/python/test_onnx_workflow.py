# SPDX-License-Identifier: MIT OR Apache-2.0
"""The CI steps that load the real ONNX Runtime (ADR-0057): pinned wheels,
no installed tools, telemetry off, and a no-network ephemeral container on
Linux hosts with Docker."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
FETCH = (ROOT / "scripts" / "fetch_onnxruntime.py").read_text()
RUNNER = (ROOT / "scripts" / "onnx_test.sh").read_text()

# PyPI's sha256 for each cp313 wheel of onnxruntime 1.31.1 (also recorded by
# spike S1).
PINNED = {
    "onnxruntime-1.31.1-cp313-cp313-manylinux_2_28_x86_64.whl": (
        "74a6a52994616c4d18b33db1e13c226333a70a6ba07480a5cab9c2a50dff1430"
    ),
    "onnxruntime-1.31.1-cp313-cp313-manylinux_2_28_aarch64.whl": (
        "dcaedc50333876f3829ab966b242147f0e548229e29c36fe44622ecf748c25ee"
    ),
    "onnxruntime-1.31.1-cp313-cp313-macosx_14_0_arm64.whl": (
        "e4ad067d36e2220218e085d7223283b2ee1bc73d887e811fd14d21cf901a11ee"
    ),
}


def _job(name):
    body = CI.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))[name]


CLIPPY_TEST = _job("clippy-test")


def _onnx_steps():
    steps = CLIPPY_TEST.split("    steps:\n", 1)[1]
    return [s for s in re.split(r"^      - ", steps, flags=re.M) if "onnx" in s]


def test_onnx_wheels_are_pinned_by_sha256():
    for wheel, sha in PINNED.items():
        assert re.search(rf'{re.escape(wheel)}",\n\s+"{sha}"', FETCH), wheel
    assert "hashlib.sha256(data).hexdigest()" in FETCH
    assert "if got != sha256:" in FETCH


def test_onnx_fetch_version_matches_compat_targets():
    targets = tomllib.loads((ROOT / "compat" / "targets.toml").read_text())
    pinned = targets["env"]["main"]["onnxruntime"]
    assert f'VERSION = "{pinned}"' in FETCH
    assert all(f"onnxruntime-{pinned}-" in wheel for wheel in PINNED)


def test_onnx_fetch_installs_and_imports_nothing():
    imports = set(re.findall(r"^import (\w+)|^from (\w+)", FETCH, re.M))
    names = {a or b for a, b in imports}
    assert names <= {"hashlib", "platform", "sys", "urllib", "zipfile", "pathlib"}


def test_onnx_container_has_no_network_and_is_ephemeral():
    run = RUNNER.split("docker run", 1)[1]
    for flag in ("--rm", "--pull never", "--network none", "--read-only"):
        assert flag in run, flag
    assert '--user "$(id -u):$(id -g)"' in run
    assert "-e ORT_DISABLE_TELEMETRY=1" in run
    assert '"$PWD":/w:ro' in run
    assert 'docker image inspect "$image"' in RUNNER


def test_onnx_clippy_test_installs_no_tool():
    for tool in ("pipx", "uv ", "uv\n", "pip install", "setup-python"):
        assert tool not in CLIPPY_TEST, tool


def test_onnx_telemetry_disabled_at_job_level():
    env = CLIPPY_TEST.split("    steps:\n", 1)[0]
    assert re.search(r'^    env:\n(?:      .*\n)*?      ORT_DISABLE_TELEMETRY: "1"$', env, re.M)


def test_onnx_steps_have_no_expressions_in_run():
    steps = _onnx_steps()
    assert len(steps) == 2
    for step in steps:
        run = step.split("env:", 1)[0]
        assert "${{" not in run, step


def test_onnx_container_image_is_gungraun_digest():
    image = re.search(r"ONNX_TEST_IMAGE: (\S+)", CLIPPY_TEST).group(1)
    assert "@sha256:" in image
    assert f"image: {image}" in _job("gungraun")
