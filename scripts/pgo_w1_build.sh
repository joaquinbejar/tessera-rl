#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# The W1 build of the PGO feasibility gate (docs/reviews/pgo-feasibility-run-plan.md):
# the `wheel` profile (fat LTO), manylinux_2_28 x86-64, and the venv the
# measurement container uses, since that one has no network.
#
#   scripts/pgo_w1_build.sh <rustup sha256> <uv sha256> <run dir>
#
# Writes only under <run dir>: tools in w1/, the wheel and its sha256 in dist/,
# the venv in venv/. Runs inside the W0 manylinux image as the runner's user.
# Never run on a host yet.
set -euo pipefail

rustup_sha="$1"
uv_sha="$2"
run="$3"
target=x86_64-unknown-linux-gnu

RUSTUP_VERSION=1.29.3
UV_VERSION=0.13.6
MATURIN_VERSION=1.16.0
NUMPY_VERSION=2.4.6

w1="$run/w1"
export RUSTUP_HOME="$w1/rustup" CARGO_HOME="$w1/cargo" CARGO_TARGET_DIR="$w1/target"
export UV_CACHE_DIR="$w1/uv-cache" UV_PYTHON_INSTALL_DIR="$w1/python"
# A bare `docker run --user` leaves HOME at /, which is not writable.
export HOME="$w1/home"
mkdir -p "$HOME" "$w1/bin" "$CARGO_HOME/bin" "$run/dist"
export PATH="$w1/bin:$CARGO_HOME/bin:$PATH"

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1
}

fetch() { # url, destination, expected sha256
    curl --proto '=https' --tlsv1.2 -fsSL --retry 3 -o "$2" "$1"
    local got
    got="$(sha256 "$2")"
    [ "$got" = "$3" ] || { echo "pgo_w1_build: sha256 mismatch for $1: $got" >&2; exit 1; }
}

fetch "https://static.rust-lang.org/rustup/archive/$RUSTUP_VERSION/$target/rustup-init" \
    "$w1/rustup-init" "$rustup_sha"
chmod +x "$w1/rustup-init"
"$w1/rustup-init" -y --no-modify-path --profile minimal --default-toolchain none
rustup toolchain install # the version of rust-toolchain

fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$target.tar.gz" \
    "$w1/uv.tar.gz" "$uv_sha"
tar -xzf "$w1/uv.tar.gz" -C "$w1/bin" --strip-components=1

uvx --python 3.11 "maturin==$MATURIN_VERSION" build --locked --profile wheel \
    --target "$target" --compatibility manylinux_2_28 --out "$run/dist"

shopt -s nullglob
wheels=("$run"/dist/*.whl)
shopt -u nullglob
[ "${#wheels[@]}" -eq 1 ] || { echo "pgo_w1_build: expected one wheel, found ${#wheels[@]}" >&2; exit 1; }
sha256 "${wheels[0]}" > "$run/dist/w1.sha256"

uv venv --python 3.11 "$run/venv"
uv pip install --python "$run/venv/bin/python" "${wheels[0]}" "numpy==$NUMPY_VERSION"
"$run/venv/bin/python" -I -c "import tessera_rl._native"
echo "pgo_w1_build: W1 $(cat "$run/dist/w1.sha256")"
