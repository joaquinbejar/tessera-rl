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

# The wheels whose library spike S1 examined (ADR-0057), and that library:
# (wheel, wheel sha256 = PyPI's digest, library, library sha256).
PINNED = [
    (
        "onnxruntime-1.31.1-cp313-cp313-manylinux_2_28_x86_64.whl",
        "74a6a52994616c4d18b33db1e13c226333a70a6ba07480a5cab9c2a50dff1430",
        "libonnxruntime.so.1.31.1",
        "fc44112fbe06d7d689c57e11d3862031f64711007e4ee020f8eed5185fd2c1d3",
    ),
    (
        "onnxruntime-1.31.1-cp313-cp313-manylinux_2_28_aarch64.whl",
        "dcaedc50333876f3829ab966b242147f0e548229e29c36fe44622ecf748c25ee",
        "libonnxruntime.so.1.31.1",
        "8aa3b76b28c778a2b1dd660d34161adb1d508c9ccd04ff9dc80e26eca5df8467",
    ),
    (
        "onnxruntime-1.31.1-cp312-cp312-macosx_14_0_arm64.whl",
        "96dd2564b10ea7153ad9c207e7cd8e5917cb3ba521e0e0a62ac03486fdc8d56c",
        "libonnxruntime.1.31.1.dylib",
        "d586cbe89effc085a09d3341b3cfbd5c322517f9d397aeb20814af21433f2b36",
    ),
]
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")


def _job(name):
    body = CI.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))[name]


CLIPPY_TEST = _job("clippy-test")


def _onnx_steps():
    steps = CLIPPY_TEST.split("    steps:\n", 1)[1]
    # A step's own first line (its run command) names onnx; comments that
    # precede a step belong to the previous chunk and are not counted.
    return [s for s in re.split(r"^      - ", steps, flags=re.M) if "onnx" in s.split("\n", 1)[0]]


def test_onnx_wheels_and_libraries_are_pinned_by_sha256():
    for wheel, sha, library, library_sha in PINNED:
        assert re.search(
            rf'{re.escape(wheel)}",\n\s+"{sha}",\n\s+"{re.escape(library)}",\n\s+"{library_sha}"',
            FETCH,
        ), wheel
    assert "if got != sha256:" in FETCH
    assert "if got != library_sha256:" in FETCH
    # The library is checked before it is written.
    assert FETCH.index("if got != library_sha256:") < FETCH.index("path.write_bytes(payload)")


def test_onnx_pins_live_only_in_the_fetch_script():
    pinned = {h for _, sha, _, lib in PINNED for h in (sha, lib)}
    assert set(HEX64.findall(FETCH)) == pinned
    for text in (CI, RUNNER):
        assert not pinned & set(HEX64.findall(text))
    assert "python3 scripts/fetch_onnxruntime.py" in CLIPPY_TEST


def test_onnx_fetch_version_matches_compat_targets():
    targets = tomllib.loads((ROOT / "compat" / "targets.toml").read_text())
    pinned = targets["env"]["main"]["onnxruntime"]
    assert f'VERSION = "{pinned}"' in FETCH
    assert all(f"onnxruntime-{pinned}-" in wheel for wheel, *_ in PINNED)


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


def test_onnx_native_mode_per_runner():
    legs = dict(re.findall(r"- os: (\S+)\n\s+onnx_native: (\S+)", CLIPPY_TEST))
    assert legs == {"xaskar": "forbid", "m4-linux-arm64": "if-no-docker", "m4": "always"}
    assert "ONNX_TEST_NATIVE: ${{ matrix.onnx_native }}" in CLIPPY_TEST


def test_onnx_runner_defaults_to_forbid_and_caps_the_container():
    assert "mode=${ONNX_TEST_NATIVE:-forbid}" in RUNNER
    run = RUNNER.split("docker run", 1)[1]
    for flag in ("--cpus 2", "--memory 4g", "--memory-swap 4g", "size=2g"):
        assert flag in run, flag
