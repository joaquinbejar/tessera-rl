#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Runs the real-library test of tessera-rl-onnx (ADR-0057) against a library
# fetched by scripts/fetch_onnxruntime.py. No test loads ONNX Runtime with
# telemetry enabled.
#
# Usage: scripts/onnx_test.sh <path-to-libonnxruntime> [image]
#
# ONNX_TEST_NATIVE says where the library may be loaded; the caller (the CI
# matrix leg) chooses it, following the owner's decision of 2026-11-12:
# - forbid (the default when unset): only in the container. Linux with
#   Docker and <image> already present (checked, never pulled); otherwise
#   exit 1 naming what is missing.
# - if-no-docker: the container when Docker is present; natively only when
#   there is no docker binary at all. Docker present but <image> absent is
#   exit 1 (no pull, no native fallback).
# - always: natively (the macOS runner).
# The container: ephemeral, no network, read-only root and workspace, the
# invoking uid, capped at 2 CPUs and 4 GiB (tmpfs included). It builds the
# test offline from the host's registry (crates fetched on the host first).
# Natively: `make onnx-test` (telemetry off, scratch HOME).
# The mode and the path taken are printed first.
set -euo pipefail

lib=${1:?usage: onnx_test.sh <libonnxruntime> [image]}
# Absolute: cargo runs the test from the crate's directory.
lib="$(cd "$(dirname "$lib")" && pwd)/$(basename "$lib")"
image=${2:-}
mode=${ONNX_TEST_NATIVE:-forbid}
export ORT_DISABLE_TELEMETRY=1

native() {
  echo "onnx-test: mode $mode, native ($1)"
  exec make onnx-test ONNXRUNTIME_LIB="$lib"
}

refuse() {
  echo "onnx-test: mode $mode, refused: $1" >&2
  exit 1
}

case "$mode" in
  always)
    native "mode always"
    ;;
  forbid | if-no-docker) ;;
  *)
    refuse "unknown ONNX_TEST_NATIVE '$mode' (forbid, if-no-docker or always)"
    ;;
esac

if [ "$(uname -s)" != Linux ]; then
  refuse "the container path needs Linux, this is $(uname -s)"
elif ! command -v docker >/dev/null 2>&1; then
  if [ "$mode" = if-no-docker ]; then
    native "no docker binary"
  fi
  refuse "no docker binary"
elif [ -z "$image" ]; then
  refuse "no image given"
elif ! docker image inspect "$image" >/dev/null 2>&1; then
  refuse "image $image not present (it is never pulled)"
fi

echo "onnx-test: mode $mode, container $image"
cargo fetch --locked
lib_dir=$(dirname "$lib")
lib_name=$(basename "$lib")
toolchain="1.99.1-$(uname -m)-unknown-linux-gnu"
docker run --rm --pull never --network none --read-only \
  --cpus 2 --memory 4g --memory-swap 4g \
  --tmpfs /tmp:rw,exec,size=2g \
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
