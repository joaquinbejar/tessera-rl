#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Installs rustup from its pinned rustup-init, verified by sha256 before it
# runs, then the toolchain of rust-toolchain.toml. Shared by
# scripts/wheel_job.sh (every wheel leg) and the Windows native-wait CI job
# (#358), so no unverified installer runs on any leg.
#
#   scripts/install_rustup.sh <target> <rustup-init sha256> <download directory>
#
# RUSTUP_HOME and CARGO_HOME are the caller's (set and exported before the
# call), and "$CARGO_HOME/bin" must be on PATH for the toolchain step.
set -euo pipefail

target="$1"
rustup_sha="$2"
dir="$3"
: "${RUSTUP_HOME:?RUSTUP_HOME must be set by the caller}"
: "${CARGO_HOME:?CARGO_HOME must be set by the caller}"

RUSTUP_VERSION=1.29.3

exe=""
case "$target" in *-windows-*) exe=.exe ;; esac

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1
}

init="$dir/rustup-init$exe"
url="https://static.rust-lang.org/rustup/archive/$RUSTUP_VERSION/$target/rustup-init$exe"
mkdir -p "$dir"
curl --proto '=https' --tlsv1.2 -fsSL --retry 3 -o "$init" "$url"
got="$(sha256 "$init")"
if [ "$got" != "$rustup_sha" ]; then
    echo "install_rustup: sha256 mismatch for $url: $got" >&2
    rm -f "$init"
    exit 1
fi
chmod +x "$init"
"$init" -y --no-modify-path --profile minimal --default-toolchain none
rustup toolchain install # the version of rust-toolchain
