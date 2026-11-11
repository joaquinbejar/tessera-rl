#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# One W0 wheel job of wheels.yml: verified tools, the build, the golden anchor
# and the Python suite against the installed wheel.
#
#   scripts/wheel_job.sh <target> <compatibility or ""> <rustup sha256> <uv sha256>
#
# Everything it writes goes under $WHEEL_RUN, which the workflow removes, except
# dist/ and the report that it copies to the workspace for upload. Runs at nice 10.
set -euo pipefail

target="$1"
compatibility="$2"
rustup_sha="$3"
uv_sha="$4"
: "${WHEEL_RUN:?the per-run directory}"

# The build environment is the script's own: an ambient flag, target, profile setting,
# compiler or wrapper would change what is built without leaving a trace, so it is refused,
# not inherited. The refused set (scripts/pgo_candidate_checks.sh, scripts/pgo_w2_build.sh and
# the L1 run script name the same): these names, and every name that starts with CARGO_PROFILE_
# or CARGO_TARGET_ (CARGO_TARGET_DIR included: the caller's is refused here, before the script
# sets its own).
BUILD_VARS="RUSTFLAGS RUSTDOCFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_ENCODED_RUSTDOCFLAGS CARGO_BUILD_RUSTFLAGS
CARGO_BUILD_RUSTDOCFLAGS CARGO_BUILD_TARGET CARGO_BUILD_TARGET_DIR RUSTC RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER
CARGO_BUILD_RUSTC CARGO_BUILD_RUSTC_WRAPPER CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER"
for v in $BUILD_VARS; do
    if [ -n "${!v+x}" ]; then echo "wheel_job: $v is set in the environment; refused" >&2; exit 2; fi
done
if env | grep -Eq '^CARGO_(PROFILE|TARGET)_[A-Za-z0-9_]*='; then
    echo "wheel_job: $(env | grep -Eo '^CARGO_(PROFILE|TARGET)_[A-Za-z0-9_]*' | tr '\n' ' ')set in the environment; refused" >&2; exit 2
fi

RUSTUP_VERSION=1.29.3
UV_VERSION=0.13.6
MATURIN_VERSION=1.16.0

export RUSTUP_HOME="$WHEEL_RUN/rustup" CARGO_HOME="$WHEEL_RUN/cargo" CARGO_TARGET_DIR="$WHEEL_RUN/target"
export UV_CACHE_DIR="$WHEEL_RUN/uv-cache" UV_PYTHON_INSTALL_DIR="$WHEEL_RUN/python"
export MACOSX_DEPLOYMENT_TARGET=13.0
mkdir -p "$WHEEL_RUN/bin" "$CARGO_HOME/bin"
export PATH="$WHEEL_RUN/bin:$CARGO_HOME/bin:$PATH"

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1
}

fetch() { # url, destination, expected sha256
    curl --proto '=https' --tlsv1.2 -fsSL --retry 3 -o "$2" "$1"
    local got
    got="$(sha256 "$2")"
    [ "$got" = "$3" ] || { echo "wheel_job: sha256 mismatch for $1: $got" >&2; exit 1; }
}

fetch "https://static.rust-lang.org/rustup/archive/$RUSTUP_VERSION/$target/rustup-init" \
    "$WHEEL_RUN/rustup-init" "$rustup_sha"
chmod +x "$WHEEL_RUN/rustup-init"
"$WHEEL_RUN/rustup-init" -y --no-modify-path --profile minimal --default-toolchain none
rustup toolchain install # the version of rust-toolchain

fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$target.tar.gz" \
    "$WHEEL_RUN/uv.tar.gz" "$uv_sha"
tar -xzf "$WHEEL_RUN/uv.tar.gz" -C "$WHEEL_RUN/bin" --strip-components=1

args=(--release --locked --target "$target" --out dist)
[ -z "$compatibility" ] || args+=(--compatibility "$compatibility")
nice -n 10 uvx --python 3.11 "maturin==$MATURIN_VERSION" build "${args[@]}"

TESSERA_GOLDEN_ACTIONS="$WHEEL_RUN/golden-actions" nice -n 10 \
    cargo test --release --locked -p tessera-rl-classic -p tessera-rl-games --test golden

# A directory of its own and -I, so that the installed wheel is what is imported.
root="$PWD"
wheel="$(ls "$root"/dist/*.whl)"
venv="$WHEEL_RUN/venv"
uv venv --python 3.11 "$venv"
uv pip install --python "$venv/bin/python" dist/*.whl numpy==2.4.6 xxhash==4.0.2 \
    pytest==9.2.0 gymnasium==1.4.0 pettingzoo==1.27.0
mkdir -p "$WHEEL_RUN/run"
cd "$WHEEL_RUN/run"
# Every check runs, and its exit status is recorded against the wheel's sha256.
anchor=0 suite=0 num=0
nice -n 10 "$venv/bin/python" -I "$root/tools/wheel_golden.py" \
    --actions "$WHEEL_RUN/golden-actions" --out "$root/golden-report.json" || anchor=$?
# test_bench.py is the only exclusion: it builds the bench runner with cargo and
# reads it from the repository's target/, which this job redirects. It has no
# wheel surface and is tested in CI's python job. test_pilot, test_loop and
# test_gate skip here because the venv has no torch: adding torch to it needs a
# decision about them.
nice -n 10 "$venv/bin/python" -I -m pytest -q -p no:cacheprovider "$root/tests/python" --rootdir "$root" \
    --ignore="$root/tests/python/test_bench.py" || suite=$?
nice -n 10 "$venv/bin/python" -I "$root/tools/pgo/determinism.py" record --wheel "$wheel" --out "$root/determinism.json"

cd "$root"
scripts/wheel_num_tests.sh || num=$?
"$venv/bin/python" -I tools/pgo/checks.py --wheel "$wheel" --out checks.json \
    golden_anchor="$anchor" python_suite="$suite" num_tests="$num"
[ "$anchor" -eq 0 ] && [ "$suite" -eq 0 ] && [ "$num" -eq 0 ]
