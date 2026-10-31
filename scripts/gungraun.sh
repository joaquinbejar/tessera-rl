#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Instruction-count gate (ADR-0025 level 1) with gungraun on Valgrind's
# Callgrind, cache simulation on.
#
#   scripts/gungraun.sh check   compare with the committed baseline; fails
#                               when Ir regresses by more than 2%
#   scripts/gungraun.sh bless   record a new baseline (see benches/gungraun/README.md)
#
# Linux only: Valgrind does not run on macOS, where this exits 0 with a
# message. Needs valgrind and gungraun-runner 0.20.0 (the version of the
# gungraun library in Cargo.lock) on PATH.
set -euo pipefail

mode="${1:-check}"
root="$(cd "$(dirname "$0")/.." && pwd)"
home="$root/benches/gungraun"

if [ "$(uname -s)" != "Linux" ]; then
    echo "gungraun: skipped on $(uname -s): Valgrind needs Linux (run it on ref-x86 or in CI)"
    exit 0
fi
command -v valgrind >/dev/null || { echo "gungraun: valgrind is not installed" >&2; exit 1; }
runner="$(gungraun-runner --version 2>/dev/null || true)"
case "$runner" in
    *" 0.20.0"*) ;;
    *) echo "gungraun: needs gungraun-runner 0.20.0 on PATH, found '${runner:-none}'" >&2; exit 1 ;;
esac

benches=(-p tessera-rl-classic --bench classic_instructions -p tessera-rl-engine --bench snapshot_instructions)
common=(--home="$home")
case "$mode" in
    check)
        cargo bench --locked "${benches[@]}" -- "${common[@]}" --baseline=ref --callgrind-limits='ir=2.0%'
        ;;
    bless)
        cargo bench --locked "${benches[@]}" -- "${common[@]}" --save-baseline=ref
        ;;
    *)
        echo "usage: $0 check|bless" >&2
        exit 2
        ;;
esac
