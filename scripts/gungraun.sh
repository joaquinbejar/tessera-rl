#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Instruction-count gate (ADR-0025 level 1) with gungraun on Valgrind's
# Callgrind (cache simulation on, fixed geometry).
#
#   scripts/gungraun.sh check   compare with the committed baseline; fails
#                               when Ir moves by more than 2% either way
#                               (ADR-0059: a fall beyond the band asks for
#                               a reviewed re-bless)
#   scripts/gungraun.sh bless   record a new baseline (see benches/gungraun/README.md)
#
# Both fail closed. The environment the counts depend on (image, rustc,
# glibc, Valgrind, gungraun-runner, the glibc dispatch pin) is recorded in
# benches/gungraun/environment.toml by bless and must match it for check.
# check also requires a committed baseline for every expected benchmark,
# requires that every benchmark was compared, and requires that no baseline
# file is created or changed. TESSERA_GUNGRAUN_IMAGE names the image the run
# uses (the CI job sets it).
#
# The benches build with Cargo's `bench-gate` profile (one codegen unit, no
# LTO; ADR-0059), and both modes fail closed unless the rustc command lines
# show it.
#
# Linux only: Valgrind does not run on macOS, where this exits 0 with a
# message.
set -euo pipefail

mode="${1:-check}"
root="$(cd "$(dirname "$0")/.." && pwd)"
home="$root/benches/gungraun"
manifest="$home/environment.toml"

if [ "$(uname -s)" != "Linux" ]; then
    echo "gungraun: skipped on $(uname -s): Valgrind needs Linux (run it on ref-x86 or in CI)"
    exit 0
fi

fail() { echo "gungraun: $*" >&2; exit 1; }

# The benchmarks of the gate, as gungraun lays them out under $home.
expected=(
    tessera-rl-classic/classic_instructions/classic/cartpole_physics.steps_1000
    tessera-rl-classic/classic_instructions/classic/cartpole_pool_step.n1024
    tessera-rl-classic/classic_instructions/classic/cartpole_pool_steady.n1024
    tessera-rl-classic/classic_instructions/classic/tag_pool_step.n256
    tessera-rl-engine/snapshot_instructions/engine/step_inline.n1024
    tessera-rl-engine/snapshot_instructions/engine/snapshot_all_copy.n1024
    tessera-rl-engine/snapshot_instructions/engine/restore_with_reobserve.n1024
    tessera-rl-engine/snapshot_instructions/engine/fork_with_reobserve.n1024
    tessera-rl-engine/snapshot_instructions/engine/fork_branched_with_reobserve.n1024
    tessera-rl-engine/turns_instructions/turns/masked_step.n1024
    tessera-rl-engine/turns_instructions/turns/sequential_step.n1024
    tessera-rl-engine/async_instructions/async_mode/drain.round_256
    tessera-rl-games/games_instructions/games/connect4_pool_step.n1024
    tessera-rl-games/games_instructions/games/bot_choose.positions_32
    tessera-rl-search/search_instructions/search/expand_batch.n7
    tessera-rl-search/search_instructions/search/puct_run.sims_256
)
# The profile the gate builds with, its band, and the crates whose rustc
# lines must show the profile: the bench binaries and the measured packages.
profile=bench-gate
band=2.0
profile_crates=(classic_instructions snapshot_instructions turns_instructions async_instructions games_instructions search_instructions tessera_rl_classic tessera_rl_engine tessera_rl_games tessera_rl_search)
baseline_of() { local d="$1"; local b; b="$(basename "$d")"; echo "$home/$d/callgrind.$b.out.base@ref"; }

command -v valgrind >/dev/null || fail "valgrind is not installed"
command -v gungraun-runner >/dev/null || fail "gungraun-runner is not installed"

# The settings of the gate's profile in Cargo.toml, as `key=value;...`.
profile_settings() {
    sed -n "/^\[profile\.$profile\]\$/,/^\[/{/^\[/d;/^#/d;/^\$/d;p;}" "$root/Cargo.toml" | tr -d ' "' | paste -sd';' -
}
[ -n "$(profile_settings)" ] || fail "Cargo.toml has no [profile.$profile]"

# The observed environment.
observed() {
    cat <<TOML
image = "${TESSERA_GUNGRAUN_IMAGE:-}"
rustc = "$(rustc --version)"
glibc = "$(ldd --version | head -1)"
valgrind = "$(valgrind --version)"
valgrind_package = "$(dpkg-query -W -f '${Version}' valgrind 2>/dev/null || echo unknown)"
gungraun_runner = "$(gungraun-runner --version)"
target = "$(rustc -vV | sed -n 's/^host: //p')"
glibc_tunables_sha256 = "$(sha256sum "$home/GLIBC_TUNABLES" | cut -d' ' -f1)"
bench_profile = "$profile: $(profile_settings)"
TOML
}
[ -n "${TESSERA_GUNGRAUN_IMAGE:-}" ] || fail "TESSERA_GUNGRAUN_IMAGE is not set (the image@digest of this run)"

echo "gungraun: cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
observed | sed 's/^/gungraun: /'

# Builds the benches with the gate's profile, from clean for the measured
# packages so that every named crate prints its rustc line, and checks it.
build_with_profile() {
    local build_log
    build_log="$(mktemp)"
    cargo clean --profile "$profile" -p tessera-rl-classic -p tessera-rl-engine -p tessera-rl-games -p tessera-rl-search
    CARGO_TERM_COLOR=never cargo bench --locked --profile "$profile" --no-run -v "${benches[@]}" > "$build_log" 2>&1 \
        || { tail -30 "$build_log" >&2; fail "the benches did not build with the $profile profile"; }
    "$root/scripts/gungraun_profile_check.sh" "$build_log" "${profile_crates[@]}"
}

benches=(-p tessera-rl-classic --bench classic_instructions -p tessera-rl-engine --bench snapshot_instructions --bench turns_instructions --bench async_instructions -p tessera-rl-games --bench games_instructions -p tessera-rl-search --bench search_instructions)
case "$mode" in
    check)
        [ -f "$manifest" ] || fail "no environment manifest: bless first"
        if ! diff <(grep -v '^#' "$manifest") <(observed) >/dev/null; then
            diff <(grep -v '^#' "$manifest") <(observed) >&2 || true
            fail "the environment differs from the baseline's: a reviewed re-bless is required, not a regression"
        fi
        for d in "${expected[@]}"; do
            [ -f "$(baseline_of "$d")" ] || fail "missing committed baseline for $d"
        done
        sums_before="$(cd "$home" && find . -name '*.base@ref' | sort | xargs sha256sum)"
        build_with_profile
        log="$(mktemp)"
        status=0
        cargo bench --locked --profile "$profile" "${benches[@]}" -- --home="$home" --baseline=ref --callgrind-limits="ir=${band}%" 2>&1 | tee "$log" || status=$?
        sums_after="$(cd "$home" && find . -name '*.base@ref' | sort | xargs sha256sum)"
        [ "$sums_before" = "$sums_after" ] || fail "check created or changed baseline files"
        # A regression (gungraun exits 3) or a failed run is reported as is.
        [ "$status" -eq 0 ] || exit "$status"
        # Strip ANSI colours first: CI sets CARGO_TERM_COLOR=always, and
        # gungraun then colours the counts.
        compared="$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -cE '^ +Instructions: +[0-9]+\|[0-9]+ ' || true)"
        [ "$compared" -eq "${#expected[@]}" ] || fail "compared $compared benchmarks against the baseline, expected ${#expected[@]}"
        # The lower side of the band (gungraun's limit is upward only).
        "$root/scripts/gungraun_band.sh" "$log" "$band" "${#expected[@]}"
        ;;
    bless)
        build_with_profile
        cargo bench --locked --profile "$profile" "${benches[@]}" -- --home="$home" --save-baseline=ref
        for d in "${expected[@]}"; do
            [ -f "$(baseline_of "$d")" ] || fail "bless did not produce $d"
        done
        { echo "# The environment of the committed baseline (written by scripts/gungraun.sh bless)."; observed; } > "$manifest"
        echo "gungraun: baseline and $manifest written; commit them in their own reviewed commit"
        ;;
    *)
        echo "usage: $0 check|bless" >&2
        exit 2
        ;;
esac
