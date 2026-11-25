#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# The study's offline grader (#639, design r2 section 5):
#
#   study/grading/grade.sh <crate-dir> <sheet>
#
# <sheet> is a, b or c (the T1 rule sheet), or `none` for a crate graded
# without a sheet (the untouched scaffold). It prints one PASS or FAIL line
# per check and nothing else, so two runs on the same inputs print the same
# bytes; each check's full output goes to <crate>/target/study-grade/.
#
# Checks:
#   check        `cargo tessera check`
#   clippy       `cargo clippy --all-targets -- -D warnings` (the crate's
#                lints and clippy.toml, ADR-0008)
#   conformance  the fixed action script against the reference trajectory
#                (study/grading/conform/conform.rs), sheets a, b and c only
#   safety       `cargo clippy --lib` with unsafe code forbidden and
#                unwrap, expect, panic and unchecked indexing denied in src/
#
# Everything runs --locked --offline on the repository's pinned toolchain.
# A crate without a Cargo.lock gets one derived from the repository's lock
# (copied, then pruned by `cargo metadata --offline`), as the cargo-tessera
# end-to-end test does. STUDY_BLESS=1 records the sheet's reference
# trajectory instead of comparing it (references only).

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: grade.sh <crate-dir> <a|b|c|none>" >&2
    exit 2
fi
crate="$(cd "$1" && pwd)"
sheet="$2"
case "$sheet" in a | b | c | none) ;; *)
    echo "grade.sh: sheet must be a, b, c or none" >&2
    exit 2
    ;;
esac

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
RUSTUP_TOOLCHAIN="$(sed -n 's/^channel = "\(.*\)"$/\1/p' "$root/rust-toolchain.toml")"
export RUSTUP_TOOLCHAIN
logs="$crate/target/study-grade"
mkdir -p "$logs"

# The tool, built from this checkout.
cargo build --locked --offline --quiet --manifest-path "$root/Cargo.toml" -p cargo-tessera \
    > "$logs/tool.log" 2>&1
tool_dir="${CARGO_TARGET_DIR:-$root/target}/debug"
export PATH="$tool_dir:$PATH"
echo "grade: $(cargo tessera --version) at $(git -C "$root" rev-parse HEAD)"
echo "grade: sheet $sheet"

if [ ! -e "$crate/Cargo.lock" ]; then
    cp "$root/Cargo.lock" "$crate/Cargo.lock"
    cargo metadata --offline --format-version 1 --manifest-path "$crate/Cargo.toml" \
        > /dev/null 2> "$logs/lock.log"
    "$root/scripts/lock_subset.sh" "$root/Cargo.lock" "$crate/Cargo.lock" \
        "$(sed -n 's/^name = "\(.*\)"$/\1/p' "$crate/Cargo.toml" | head -n 1)" >> "$logs/lock.log" 2>&1
fi

failed=0
result() {
    if [ "$2" -eq 0 ]; then
        echo "PASS $1"
    else
        echo "FAIL $1"
        failed=1
    fi
}

cd "$crate"
rc=0
cargo tessera check --locked --offline > "$logs/check.log" 2>&1 || rc=$?
result check "$rc"

rc=0
cargo clippy --all-targets --locked --offline -- -D warnings > "$logs/clippy.log" 2>&1 || rc=$?
result clippy "$rc"

if [ "$sheet" != none ]; then
    lib="$(awk '/^\[lib\]/{l=1;next} /^\[/{l=0} l && /^name = /{gsub(/"/,"",$3); print $3}' Cargo.toml)"
    if [ -z "$lib" ]; then
        lib="$(sed -n 's/^name = "\(.*\)"$/\1/p' Cargo.toml | head -n 1 | tr '-' '_')"
    fi
    conform="tests/study_conform.rs"
    trap 'rm -f "$crate/$conform"' EXIT
    sed "s/__LIB__/$lib/g" "$here/conform/conform.rs" > "$conform"
    rc=0
    STUDY_SHEET="$sheet" STUDY_GOLDEN="$here/golden/$sheet.txt" \
        cargo test --locked --offline --test study_conform > "$logs/conformance.log" 2>&1 || rc=$?
    rm -f "$conform"
    result conformance "$rc"
fi

rc=0
cargo clippy --lib --locked --offline -- -F unsafe_code -D clippy::unwrap_used \
    -D clippy::expect_used -D clippy::panic -D clippy::indexing_slicing \
    > "$logs/safety.log" 2>&1 || rc=$?
result safety "$rc"

exit "$failed"
