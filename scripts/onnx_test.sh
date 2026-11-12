#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Runs the real-library test of tessera-rl-onnx (ADR-0057) against a library
# fetched by scripts/fetch_onnxruntime.py. No test loads ONNX Runtime with
# telemetry enabled.
#
# Usage: scripts/onnx_test.sh <path-to-libonnxruntime> [image]
#
# - Linux with Docker and <image> already present (checked, never pulled):
#   the test is built and run inside an ephemeral container with no network,
#   as the invoking uid, with the workspace read-only. The crates are fetched
#   on the host first; the container builds offline from that registry.
# - Otherwise (macOS, or Linux without Docker or without the image): natively,
#   through `make onnx-test` (telemetry off, scratch HOME).
# The path taken is printed first: "onnx-test: container <image>" or
# "onnx-test: native (<reason>)".
set -euo pipefail

lib=${1:?usage: onnx_test.sh <libonnxruntime> [image]}
# Absolute: cargo runs the test from the crate's directory.
lib="$(cd "$(dirname "$lib")" && pwd)/$(basename "$lib")"
image=${2:-}
export ORT_DISABLE_TELEMETRY=1

reason=""
if [ "$(uname -s)" != Linux ]; then
  reason="not Linux"
elif [ -z "$image" ]; then
  reason="no image given"
elif ! command -v docker >/dev/null 2>&1; then
  reason="no docker"
elif ! docker image inspect "$image" >/dev/null 2>&1; then
  reason="image $image not present"
fi

if [ -n "$reason" ]; then
  echo "onnx-test: native ($reason)"
  exec make onnx-test ONNXRUNTIME_LIB="$lib"
fi

echo "onnx-test: container $image"
cargo fetch --locked
lib_dir=$(dirname "$lib")
lib_name=$(basename "$lib")
toolchain="1.99.1-$(uname -m)-unknown-linux-gnu"
docker run --rm --pull never --network none --read-only \
  --tmpfs /tmp:rw,exec,size=4g \
  --user "$(id -u):$(id -g)" \
  -e ORT_DISABLE_TELEMETRY=1 \
  -e HOME=/tmp/target/onnx-test-home \
  -e CARGO_HOME=/tmp/cargo \
  -e CARGO_TARGET_DIR=/tmp/target \
  -e CARGO_NET_OFFLINE=true \
  -e RUSTUP_TOOLCHAIN="$toolchain" \
  -e TESSERA_ONNXRUNTIME_LIB="/ort/$lib_name" \
  -v "$PWD":/w:ro \
  -v "${CARGO_HOME:-$HOME/.cargo}/registry":/tmp/cargo/registry:ro \
  -v "$lib_dir":/ort:ro \
  -w /w "$image" \
  sh -ec 'mkdir -p "$HOME" && cargo test -p tessera-rl-onnx --features onnx --locked -- --include-ignored'
