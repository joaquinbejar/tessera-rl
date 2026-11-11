#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Section 7 of docs/reviews/pgo-wheels-design.md for a candidate wheel that is
# already built (W1, W2F): the golden anchor, the Python suite and determinism
# record against the installed wheel, the num crate's tests under the wheel's
# cargo profile, then the comparison with W0's determinism record and checks.py.
# wheel_job.sh does the same for the wheel it builds itself (W0); this script
# takes the exact candidate bytes instead, so the checked wheel is the measured one.
#
#   scripts/pgo_candidate_checks.sh <wheel> <wheel sha256> <cargo profile> <rustup sha256> <uv sha256> <W0 determinism.json>
#
# Run from the candidate's source tree (W1's commit), inside the W0 manylinux
# image. Writes under $WHEEL_RUN: checks.json, determinism.json, compare.json,
# golden-report.json. Exit 0 only if every check passed.
set -euo pipefail

wheel="$1"
wheel_sha="$2"
cargo_profile="$3"
rustup_sha="$4"
uv_sha="$5"
w0_det="$6"
: "${WHEEL_RUN:?the per-run directory}"
# The build environment is the script's own: an ambient flag or target directory would change
# what the checks build and test without leaving a trace, so it is refused, not inherited.
# The refused set (the L1 run script names the same): these names, and every name that starts
# with CARGO_PROFILE_ or CARGO_TARGET_.
BUILD_VARS="RUSTFLAGS RUSTDOCFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_ENCODED_RUSTDOCFLAGS CARGO_BUILD_RUSTFLAGS
CARGO_BUILD_RUSTDOCFLAGS CARGO_BUILD_TARGET CARGO_BUILD_TARGET_DIR RUSTC RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER
CARGO_BUILD_RUSTC CARGO_BUILD_RUSTC_WRAPPER CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER"
for v in $BUILD_VARS; do
    if [ -n "${!v+x}" ]; then echo "pgo_candidate_checks: $v is set in the environment; refused" >&2; exit 2; fi
done
if env | grep -Eq '^CARGO_(PROFILE|TARGET)_[A-Za-z0-9_]*='; then
    echo "pgo_candidate_checks: $(env | grep -Eo '^CARGO_(PROFILE|TARGET)_[A-Za-z0-9_]*' | tr '\n' ' ')set in the environment; refused" >&2; exit 2
fi
target=x86_64-unknown-linux-gnu

RUSTUP_VERSION=1.29.3
UV_VERSION=0.13.6

for f in "$wheel" "$w0_det"; do
    case "$f" in /*) ;; *) echo "pgo_candidate_checks: $f must be absolute" >&2; exit 2 ;; esac
    [ -f "$f" ] || { echo "pgo_candidate_checks: no file $f" >&2; exit 2; }
done

got="$(if command -v sha256sum >/dev/null; then sha256sum "$wheel"; else shasum -a 256 "$wheel"; fi | cut -d' ' -f1)"
[ "$got" = "$wheel_sha" ] || { echo "pgo_candidate_checks: $wheel is $got, not the candidate $wheel_sha" >&2; exit 2; }

export RUSTUP_HOME="$WHEEL_RUN/rustup" CARGO_HOME="$WHEEL_RUN/cargo" CARGO_TARGET_DIR="$WHEEL_RUN/target"
export UV_CACHE_DIR="$WHEEL_RUN/uv-cache" UV_PYTHON_INSTALL_DIR="$WHEEL_RUN/python"
export HOME="$WHEEL_RUN/home"
mkdir -p "$HOME" "$WHEEL_RUN/bin" "$CARGO_HOME/bin" "$WHEEL_RUN/run"
export PATH="$WHEEL_RUN/bin:$CARGO_HOME/bin:$PATH"

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1
}

fetch() { # url, destination, expected sha256
    curl --proto '=https' --tlsv1.2 -fsSL --retry 3 -o "$2" "$1"
    local got
    got="$(sha256 "$2")"
    [ "$got" = "$3" ] || { echo "pgo_candidate_checks: sha256 mismatch for $1: $got" >&2; exit 1; }
}

fetch "https://static.rust-lang.org/rustup/archive/$RUSTUP_VERSION/$target/rustup-init" \
    "$WHEEL_RUN/rustup-init" "$rustup_sha"
chmod +x "$WHEEL_RUN/rustup-init"
"$WHEEL_RUN/rustup-init" -y --no-modify-path --profile minimal --default-toolchain none
rustup toolchain install # the version of rust-toolchain

fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$target.tar.gz" \
    "$WHEEL_RUN/uv.tar.gz" "$uv_sha"
tar -xzf "$WHEEL_RUN/uv.tar.gz" -C "$WHEEL_RUN/bin" --strip-components=1

# The golden actions, as wheel_job.sh writes them for W0.
TESSERA_GOLDEN_ACTIONS="$WHEEL_RUN/golden-actions" nice -n 10 \
    cargo test --release --locked -p tessera-rl-classic -p tessera-rl-games --test golden

root="$PWD"
venv="$WHEEL_RUN/venv"
uv venv --python 3.11 "$venv"
uv pip install --python "$venv/bin/python" "$wheel" numpy==2.4.6 xxhash==4.0.2 \
    pytest==9.2.0 gymnasium==1.4.0 pettingzoo==1.27.0
cd "$WHEEL_RUN/run"
anchor=0 suite=0 num=0
nice -n 10 "$venv/bin/python" -I "$root/tools/wheel_golden.py" \
    --actions "$WHEEL_RUN/golden-actions" --out "$WHEEL_RUN/golden-report.json" || anchor=$?
# The same exclusion as wheel_job.sh: test_bench.py builds with cargo into the repository's target/.
nice -n 10 "$venv/bin/python" -I -m pytest -q -p no:cacheprovider "$root/tests/python" --rootdir "$root" \
    --ignore="$root/tests/python/test_bench.py" || suite=$?
nice -n 10 "$venv/bin/python" -I "$root/tools/pgo/determinism.py" record --wheel "$wheel" --out "$WHEEL_RUN/determinism.json"
cd "$root"
WHEEL_CARGO_PROFILE="$cargo_profile" scripts/wheel_num_tests.sh || num=$?
"$venv/bin/python" -I tools/pgo/checks.py --wheel "$wheel" --out "$WHEEL_RUN/checks.json" \
    golden_anchor="$anchor" python_suite="$suite" num_tests="$num"
compare=0
"$venv/bin/python" -I tools/pgo/determinism.py compare "$w0_det" "$WHEEL_RUN/determinism.json" \
    --out "$WHEEL_RUN/compare.json" || compare=$?
echo "pgo_candidate_checks: anchor=$anchor suite=$suite num=$num compare=$compare"
[ "$anchor" -eq 0 ] && [ "$suite" -eq 0 ] && [ "$num" -eq 0 ] && [ "$compare" -eq 0 ]
