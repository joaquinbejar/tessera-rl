#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Runs the Python ONNX evaluator tests (tests/python/test_onnx_evaluator.py,
# ADR-0057) against a library fetched by scripts/fetch_onnxruntime.py, in an
# ephemeral container with no network (the owner's choice for the Linux
# x86-64 runner). No test loads ONNX Runtime with telemetry enabled.
#
# Usage: scripts/py_onnx_test.sh <path-to-libonnxruntime> <image>
#
# The extension must already be built into the project's virtual
# environment with the `onnx` feature (`MATURIN_PEP517_ARGS="--features
# onnx" uv sync ...`), without the [onnx] extra: the library is pinned only
# by fetch_onnxruntime.py, so the tests that need the onnxruntime package
# skip. The container sees the workspace, the environment and its Python
# installation read-only, at their host paths, runs as the invoking uid, and
# is capped at 2 CPUs and 4 GiB. The image must already be present: it is
# never pulled, and a missing Docker or image is exit 1.
set -euo pipefail

lib=${1:?usage: py_onnx_test.sh <libonnxruntime> <image>}
image=${2:?usage: py_onnx_test.sh <libonnxruntime> <image>}
lib="$(cd "$(dirname "$lib")" && pwd)/$(basename "$lib")"

refuse() {
  echo "py-onnx-test: refused: $1" >&2
  exit 1
}
command -v docker >/dev/null 2>&1 || refuse "no docker binary"
docker image inspect "$image" >/dev/null 2>&1 || refuse "image $image not present (it is never pulled)"

venv=${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}
cfg="$venv/pyvenv.cfg"
[ -f "$cfg" ] || refuse "no virtual environment at $venv"
# The environment's Python installation: pyvenv.cfg's `home` is its bin dir.
home_bin=$(sed -n 's/^home = //p' "$cfg")
[ -n "$home_bin" ] || refuse "no 'home' in $cfg"
python_root=$(dirname "$home_bin")

echo "py-onnx-test: container $image, library $lib"
docker run --rm --pull never --network none --read-only \
  --cpus 2 --memory 4g --memory-swap 4g \
  --tmpfs /tmp:rw,exec,size=1g \
  --user "$(id -u):$(id -g)" \
  -e ORT_DISABLE_TELEMETRY=1 \
  -e TESSERA_ONNX_TEST=1 \
  -e HOME=/tmp/target/onnx-test-home \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e TESSERA_ONNXRUNTIME_LIB="/ort/$(basename "$lib")" \
  -v "$PWD":"$PWD":ro \
  -v "$venv":"$venv":ro \
  -v "$python_root":"$python_root":ro \
  -v "$(dirname "$lib")":/ort:ro \
  -w "$PWD" "$image" \
  sh -ec 'mkdir -p "$HOME" && "'"$venv"'/bin/python" -m pytest -q -rA -p no:cacheprovider tests/python/test_onnx_evaluator.py'
