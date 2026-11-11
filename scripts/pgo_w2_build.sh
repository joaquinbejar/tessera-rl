#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# A PGO wheel of the ladder (docs/reviews/pgo-wheels-design.md, section 3):
# instrumented build, training with the manifest, merge, final build, the
# profile coverage acceptance and the no-instrumentation check.
#
#   scripts/pgo_w2_build.sh <rustup sha256> <uv sha256> <run dir> <full|holdout>
#
# `full` makes W2F (trained on the whole manifest), `holdout` makes W2H
# (without the holdout rows). Run from the source tree of W1's commit, inside
# the W0 manylinux image, as the runner's user. Writes only under
# <run dir>/w2-<variant>/ and <run dir>/dist-<variant>/. Never run on a host yet.
set -euo pipefail

rustup_sha="$1"
uv_sha="$2"
run="$3"
variant="$4"
target=x86_64-unknown-linux-gnu

RUSTUP_VERSION=1.29.3
UV_VERSION=0.13.6
MATURIN_VERSION=1.16.0
NUMPY_VERSION=2.4.6

case "$variant" in
    full) train_flag=() ;;
    holdout) train_flag=(--holdout-build) ;;
    *) echo "pgo_w2_build: variant must be full or holdout, not $variant" >&2; exit 2 ;;
esac
case "$run" in /*) ;; *) echo "pgo_w2_build: the run dir must be absolute (profile paths are)" >&2; exit 2 ;; esac

w="$run/w2-$variant"
dist="$run/dist-$variant"
prof="$w/profraw"
export RUSTUP_HOME="$w/rustup" CARGO_HOME="$w/cargo"
export UV_CACHE_DIR="$w/uv-cache" UV_PYTHON_INSTALL_DIR="$w/python"
export HOME="$w/home"
mkdir -p "$HOME" "$w/bin" "$CARGO_HOME/bin" "$prof" "$w/gen" "$dist"
export PATH="$w/bin:$CARGO_HOME/bin:$PATH"

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1
}

fetch() { # url, destination, expected sha256
    curl --proto '=https' --tlsv1.2 -fsSL --retry 3 -o "$2" "$1"
    local got
    got="$(sha256 "$2")"
    [ "$got" = "$3" ] || { echo "pgo_w2_build: sha256 mismatch for $1: $got" >&2; exit 1; }
}

one_wheel() { # dir
    shopt -s nullglob
    local found=("$1"/*.whl)
    shopt -u nullglob
    [ "${#found[@]}" -eq 1 ] || { echo "pgo_w2_build: expected one wheel in $1, found ${#found[@]}" >&2; exit 1; }
    echo "${found[0]}"
}

fetch "https://static.rust-lang.org/rustup/archive/$RUSTUP_VERSION/$target/rustup-init" \
    "$w/rustup-init" "$rustup_sha"
chmod +x "$w/rustup-init"
"$w/rustup-init" -y --no-modify-path --profile minimal --default-toolchain none
rustup toolchain install # the version of rust-toolchain
rustup component add llvm-tools

fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$target.tar.gz" \
    "$w/uv.tar.gz" "$uv_sha"
tar -xzf "$w/uv.tar.gz" -C "$w/bin" --strip-components=1

# llvm-profdata from the pinned rustc's llvm-tools, same LLVM major version.
profdata="$(rustc --print sysroot)/lib/rustlib/$target/bin/llvm-profdata"
[ -x "$profdata" ] || { echo "pgo_w2_build: no llvm-profdata at $profdata" >&2; exit 1; }
rustc_llvm="$(rustc -vV | sed -n 's/^LLVM version: \([0-9]*\).*/\1/p')"
tool_llvm="$("$profdata" --version | sed -n 's/.*LLVM version \([0-9]*\).*/\1/p' | head -1)"
[ -n "$rustc_llvm" ] && [ "$rustc_llvm" = "$tool_llvm" ] \
    || { echo "pgo_w2_build: LLVM $tool_llvm of llvm-profdata does not match rustc's $rustc_llvm" >&2; exit 1; }

build() { # rustflags, target dir, out dir, log
    RUSTFLAGS="$1" CARGO_TARGET_DIR="$2" uvx --python 3.11 "maturin==$MATURIN_VERSION" build --locked \
        --profile wheel --target "$target" --compatibility manylinux_2_28 --out "$3" 2>&1 | tee "$4"
}

# 1. Instrumented build, its own target dir; explicit --target keeps the flag off build scripts.
build "-Cprofile-generate=$prof" "$w/target-gen" "$w/gen" "$w/gen.log"
gen="$(one_wheel "$w/gen")"

# 2. Training in a venv on a uv-managed Python, with the instrumented wheel.
uv venv --python 3.11 "$w/venv" --python-preference only-managed
py="$w/venv/bin/python"
uv pip install --python "$py" "$gen" "numpy==$NUMPY_VERSION"
"$py" -I -c "import sys; sys.path.insert(0, 'tools/pgo'); import evidence; evidence.load_kernels('tools/pgo/kernels.toml')" \
    || { echo "pgo_w2_build: tools/pgo/kernels.toml is not filled" >&2; exit 1; }
LLVM_PROFILE_FILE="$prof/%m_%p.profraw" "$py" -I tools/pgo/train.py --manifest tools/pgo/manifest.toml ${train_flag[@]+"${train_flag[@]}"}

# 3. Merge; no profraw or an empty merge fails.
shopt -s nullglob
raws=("$prof"/*.profraw)
shopt -u nullglob
[ "${#raws[@]}" -gt 0 ] || { echo "pgo_w2_build: training wrote no profraw" >&2; exit 1; }
"$profdata" merge -o "$w/merged.profdata" "${raws[@]}"
[ -s "$w/merged.profdata" ] || { echo "pgo_w2_build: the merged profile is empty" >&2; exit 1; }
"$profdata" show --all-functions "$w/merged.profdata" > "$w/profile.txt"

# 4 and 5. Final build with the profile; maturin repairs for manylinux_2_28.
build "-Cprofile-use=$w/merged.profdata -Cllvm-args=-pgo-warn-missing-function" "$w/target-use" "$dist" "$w/final.log"
whl="$(one_wheel "$dist")"

# 6 and the coverage acceptance.
"$py" -I tools/pgo/coverage.py --kernels tools/pgo/kernels.toml --profile "$w/profile.txt" \
    --build-log "$w/final.log" --wheel "$whl" --out "$dist/coverage.json"

printf '%s  %s\n%s  %s\n' "$(sha256 "$whl")" "$(basename "$whl")" "$(sha256 "$w/merged.profdata")" merged.profdata \
    > "$dist/w2-$variant.sha256"
echo "pgo_w2_build: W2-$variant $(head -1 "$dist/w2-$variant.sha256"), profile $(sed -n 2p "$dist/w2-$variant.sha256")"
