#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# The kit's acceptance run (#639, design r2 section 8), offline, on Linux:
#
#   study/grading/acceptance.sh
#
# 1. Each T2 variant fails to build with its pre-declared diagnostic, and
#    the fixed crate builds.
# 2. The T3 crate fails `cargo tessera check` with the pre-declared
#    comparison kind, `Inline vs Inline (same process)`, and the fixed
#    version passes.
# 3. Each reference passes grade.sh against its own sheet, and the
#    untouched scaffold (the fixed T2 crate) against no sheet; grade.sh
#    prints the same bytes on a second run.
# 4. Each seeded wrong solution fails exactly the check it names.
#
# Work happens on copies under a temporary directory; the kit is not
# modified. It prints one line per assertion and exits non-zero on the
# first failure.

set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
study="$(cd "$here/.." && pwd)"
root="$(cd "$study/.." && pwd)"
RUSTUP_TOOLCHAIN="$(sed -n 's/^channel = "\(.*\)"$/\1/p' "$root/rust-toolchain.toml")"
export RUSTUP_TOOLCHAIN
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

ok() { echo "ok: $*"; }
die() {
    echo "FAILED: $*" >&2
    exit 1
}

cargo build --locked --offline --quiet --manifest-path "$root/Cargo.toml" -p cargo-tessera
export PATH="${CARGO_TARGET_DIR:-$root/target}/debug:$PATH"

# A copy of a kit crate that keeps its relative path dependencies.
copy() {
    local src="$1" name="$2" dest
    dest="$work/$(dirname "${src#"$root"/}")/$name"
    mkdir -p "$(dirname "$dest")"
    rm -rf "$dest"
    cp -R "$src" "$dest"
    rm -rf "$dest/target"
    # The copy sits at the same depth, so ../ paths to crates/ must point back.
    sed -i.bak "s|path = '\(\.\./\)*crates/|path = '$root/crates/|" "$dest/Cargo.toml"
    rm -f "$dest/Cargo.toml.bak"
    echo "$dest"
}

# 1. T2.
t2="$study/tasks/t2-derive"
declare -A want=(
    [bool-field]='`bool` is not allowed in state; use `Bool8` (ADR-0029)'
    [padding]='has trailing padding after field `reached`'
    [plain-enum]='`Phase` cannot be used as a state field'
)
for v in bool-field padding plain-enum; do
    d="$(copy "$t2" "t2-$v")"
    cp "$t2/variants/$v.rs" "$d/src/lib.rs"
    if (cd "$d" && cargo build --locked --offline > build.out 2>&1); then
        die "T2 $v builds"
    fi
    grep -qF "${want[$v]}" "$d/build.out" || { cat "$d/build.out"; die "T2 $v: diagnostic missing: ${want[$v]}"; }
    ok "T2 $v fails to build with: ${want[$v]}"
done
d="$(copy "$t2" "t2-fixed")"
cp "$here/solutions/t2-derive-lib.rs" "$d/src/lib.rs"
(cd "$d" && cargo build --locked --offline > build.out 2>&1) || { cat "$d/build.out"; die "T2 fixed fails to build"; }
ok "T2 fixed builds"

# 2. T3.
t3="$study/tasks/t3-divergence"
d="$(copy "$t3" "t3-seeded")"
for run in 1 2; do
    if (cd "$d" && cargo tessera check --locked --offline > "check$run.out" 2>&1); then
        die "T3 seeded passes the check (run $run)"
    fi
    grep -qF 'Inline vs Inline (same process)' "$d/check$run.out" \
        || { cat "$d/check$run.out"; die "T3 run $run: kind not reported"; }
    ok "T3 seeded fails, run $run: $(grep -m1 -F 'Inline vs Inline' "$d/check$run.out" | cut -c1-40)"
done
d="$(copy "$t3" "t3-fixed")"
cp "$here/solutions/t3-divergence-lib.rs" "$d/src/lib.rs"
(cd "$d" && cargo tessera check --locked --offline > check.out 2>&1) || { cat "$d/check.out"; die "T3 fixed fails the check"; }
ok "T3 fixed passes the check"

# 3. References and the untouched scaffold, graded twice.
grade_twice() {
    local src="$1" sheet="$2" name="$3" d
    d="$(copy "$src" "$name")"
    # grade.sh resolves the kit from its own location: run the kit's copy.
    "$here/grade.sh" "$d" "$sheet" > "$d/grade1.out" || { cat "$d/grade1.out"; die "$name fails grade.sh"; }
    "$here/grade.sh" "$d" "$sheet" > "$d/grade2.out" || die "$name fails grade.sh on the second run"
    cmp -s "$d/grade1.out" "$d/grade2.out" || die "$name: grade.sh output differs between runs"
    sed 's/^/  /' "$d/grade1.out"
    ok "$name passes grade.sh ($sheet), byte-identical over two runs"
}
grade_twice "$here/reference/a-gridwalk" a ref-a
grade_twice "$here/reference/b-nim" b ref-b
grade_twice "$here/reference/c-pickup" c ref-c
grade_twice "$t2" none scaffold

# 4. Seeded wrong solutions: each must fail the check it names. The last
#    one drops the crate's unwrap lint, so only the safety check catches it.
wrong() {
    local src="$1" sheet="$2" name="$3" check="$4" patch="$5" d out
    d="$(copy "$src" "$name")"
    python3 - "$d/src/lib.rs" "$d/Cargo.toml" <<<"$patch"
    out="$("$here/grade.sh" "$d" "$sheet" || true)"
    echo "$out" | sed 's/^/  /'
    echo "$out" | grep -qx "FAIL $check" || die "$name: expected FAIL $check"
    ok "$name fails $check"
}
wrong "$here/reference/a-gridwalk" a wrong-a-goal conformance "
import sys; p=sys.argv[1]; s=open(p).read()
old='const GOAL: (u8, u8) = (4, 4);'; assert old in s
open(p,'w').write(s.replace(old,'const GOAL: (u8, u8) = (4, 3);'))"
wrong "$here/reference/b-nim" b wrong-b-misere conformance "
import sys; p=sys.argv[1]; s=open(p).read()
old='out.set_reward(me, 1.0);\n            out.set_reward(other, -1.0);'; assert old in s
open(p,'w').write(s.replace(old,'out.set_reward(me, -1.0);\n            out.set_reward(other, 1.0);'))"
wrong "$here/reference/c-pickup" c wrong-c-mask conformance "
import sys; p=sys.argv[1]; s=open(p).read()
old='let legal = cell.is_some_and(|(r, c)| moved(r, c, a).is_some());'; assert old in s
open(p,'w').write(s.replace(old,'let legal = cell.is_some();'))"
wrong "$here/reference/a-gridwalk" a wrong-a-unwrap safety "
import sys; p, t = sys.argv[1], sys.argv[2]; s=open(p).read()
old='let (row, col) = FREE.get(draw % FREE.len()).copied().unwrap_or((0, 0));'; assert old in s
open(p,'w').write(s.replace(old,'let (row, col) = FREE.get(draw % FREE.len()).copied().unwrap();'))
c=open(t).read(); assert 'unwrap_used = \"deny\"\n' in c
open(t,'w').write(c.replace('unwrap_used = \"deny\"\n',''))"

echo "acceptance: all passed"
