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
# bytes; each check's full output goes to the kit checkout's
# target/study-grade/<crate-name>/.
#
# The participant's crate is never written to: grade.sh rsyncs it (without
# target/) into a temporary directory, rewrites its relative `path =`
# dependencies to absolute ones, and grades that copy, which an EXIT trap
# removes. If grading crashes, even by SIGKILL, the crate is byte-identical
# to how it was; only the temporary copy is left behind.
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
# A crate without a Cargo.lock gets one, in the copy, derived from the repository's lock
# (copied, then pruned by `cargo metadata --offline`), as the cargo-tessera
# end-to-end test does. STUDY_BLESS=1 records the sheet's reference
# trajectory instead of comparing it (references only).

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: grade.sh <crate-dir> <a|b|c|none>" >&2
    exit 2
fi
src="$(cd "$1" && pwd)"
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
name="$(sed -n 's/^name = "\(.*\)"$/\1/p' "$src/Cargo.toml" | head -n 1)"
logs="${CARGO_TARGET_DIR:-$root/target}/study-grade/${name:-crate}"
rm -rf "${logs:?}"
mkdir -p "$logs"

# The copy that is graded (the participant's crate stays untouched).
work="$(mktemp -d)"
trap 'rm -rf "${work:?}"' EXIT
crate="$work/$(basename "$src")"
rsync -a --exclude /target "$src/" "$crate/"
python3 -I - "$crate/Cargo.toml" "$src" <<'PY'
import os, re, sys
toml, base = sys.argv[1], sys.argv[2]
text = open(toml).read()
def absolute(m):
    q, value = m.group(1), m.group(2)
    return m.group(0) if os.path.isabs(value) else f"path = {q}{os.path.normpath(os.path.join(base, value))}{q}"
open(toml, "w").write(re.sub(r"""path = (['"])([^'"]*)\1""", absolute, text))
PY

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
    mkdir -p tests
    sed "s/__LIB__/$lib/g" "$here/conform/conform.rs" > "$conform"
    rc=0
    STUDY_SHEET="$sheet" STUDY_GOLDEN="$here/golden/$sheet.txt" \
        cargo test --locked --offline --test study_conform > "$logs/conformance.log" 2>&1 || rc=$?
    result conformance "$rc"
fi

rc=0
cargo clippy --lib --locked --offline -- -F unsafe_code -D clippy::unwrap_used \
    -D clippy::expect_used -D clippy::panic -D clippy::indexing_slicing \
    > "$logs/safety.log" 2>&1 || rc=$?
result safety "$rc"

exit "$failed"
