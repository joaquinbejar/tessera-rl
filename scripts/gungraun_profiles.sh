#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# MEASUREMENT ONLY (branch measure/gungraun-profiles, do not merge).
#
#   scripts/gungraun_profiles.sh <check log>
#
# Called by `scripts/gungraun.sh check` after its own comparison. Rebuilds
# the instruction benches under two more profiles and prints one table of
# Ir per case:
#   (a) the check's own run (bench profile as committed: 16 codegen units,
#       no LTO), read from <check log>, with its committed baseline;
#   (b) CARGO_PROFILE_BENCH_CODEGEN_UNITS=1;
#   (c) CARGO_PROFILE_BENCH_CODEGEN_UNITS=1 CARGO_PROFILE_BENCH_LTO=fat.
# (b) and (c) run with their own --home in a temporary directory and no
# --baseline or --save-baseline, so no committed baseline file is read or
# written. Their failure is reported in the table and never changes the
# exit status of check. Always exits 0.
set -uo pipefail

log_a="${1:?usage: $0 <check log>}"
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# The bench list of scripts/gungraun.sh at this commit (no async_instructions yet).
benches=(-p tessera-rl-classic --bench classic_instructions -p tessera-rl-engine --bench snapshot_instructions --bench turns_instructions -p tessera-rl-games --bench games_instructions -p tessera-rl-search --bench search_instructions)

# "<case> <Ir> <baseline>" per measured case, from a gungraun log.
parse() {
    sed 's/\x1b\[[0-9;]*m//g' "$1" | awk '
        /^[a-z_]+_instructions::[^ ]+ [^ :]+:/ { split($0, p, ":\\("); name = p[1]; next }
        name != "" && /^ +Instructions: +[0-9]+\|/ {
            split($2, v, "|"); printf "%s\t%s\t%s\n", name, v[1], v[2]; name = ""
        }'
}

run_mode() {
    local mode="$1"; shift
    echo "gungraun-profiles: build ($mode): $*"
    (
        cd "$root" || exit 1
        export "$@"
        export CARGO_TARGET_DIR="$root/target/gungraun-profile-$mode"
        # Show that the profile override reached rustc.
        CARGO_TERM_COLOR=never cargo bench --locked --no-run -v "${benches[@]}" 2>&1 >/dev/null \
            | grep -E 'Running .*crate-name (tessera_rl_(games|search)|search_instructions) ' \
            | grep -oE 'crate-name [a-z_]+|-C (codegen-units|lto)=[^ ]+' | paste -sd' ' \
            | sed 's/^/gungraun-profiles: rustc flags ('"$mode"'): /'
        cargo bench --locked "${benches[@]}" -- --home="$work/home-$mode"
    ) > "$work/log-$mode" 2>&1
    local status=$?
    echo "gungraun-profiles: build ($mode) exit status $status"
    grep '^gungraun-profiles: rustc flags' "$work/log-$mode" || true
    [ "$status" -eq 0 ] || tail -20 "$work/log-$mode" | sed "s/^/gungraun-profiles: ($mode) /"
}

run_mode b CARGO_PROFILE_BENCH_CODEGEN_UNITS=1
run_mode c CARGO_PROFILE_BENCH_CODEGEN_UNITS=1 CARGO_PROFILE_BENCH_LTO=fat

parse "$log_a" > "$work/a"
parse "$work/log-b" > "$work/b"
parse "$work/log-c" > "$work/c"

echo "gungraun-profiles: Ir per case; % vs the committed baseline (base) and vs (a)"
awk -F'\t' '
    function num(x) { return x ~ /^[0-9]+$/ }
    function pct(x, y) { return (num(x) && num(y) && y > 0) ? sprintf("%+.2f%%", (x - y) * 100 / y) : "n/a" }
    function add(k) { if (!(k in seen)) { seen[k] = 1; order[++n] = k } }
    FILENAME ~ /\/a$/ { add($1); a[$1] = $2; base[$1] = $3; next }
    FILENAME ~ /\/b$/ { add($1); b[$1] = $2; next }
    FILENAME ~ /\/c$/ { add($1); c[$1] = $2; next }
    END {
        printf "| %-62s | %10s | %10s | %8s | %10s | %8s | %8s | %10s | %8s | %8s |\n", "case", "base", "(a) Ir", "a/base", "(b) Ir", "b/base", "b/a", "(c) Ir", "c/base", "c/a"
        for (i = 1; i <= n; i++) {
            k = order[i]; aa = (k in a) ? a[k] : "-"; bs = (k in base) ? base[k] : "-"; bb = (k in b) ? b[k] : "-"; cc = (k in c) ? c[k] : "-"
            printf "| %-62s | %10s | %10s | %8s | %10s | %8s | %8s | %10s | %8s | %8s |\n", k, bs, aa, pct(aa, bs), bb, pct(bb, bs), pct(bb, aa), cc, pct(cc, bs), pct(cc, aa)
        }
    }' "$work/a" "$work/b" "$work/c" | sed 's/^/gungraun-profiles: /'
exit 0
