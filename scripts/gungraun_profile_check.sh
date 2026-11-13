#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# The gate's benches were built with the gate's profile (ADR-0059).
#
#   scripts/gungraun_profile_check.sh <cargo -v log> <crate name>...
#
# Reads the rustc command lines of a `cargo bench --profile bench-gate -v`
# build (colours stripped, though callers run it with CARGO_TERM_COLOR=never)
# and fails closed unless every named crate was compiled at least once in it,
# every one of its rustc lines carries `-C codegen-units=1`, and none carries
# an LTO setting other than off. A crate that cargo found Fresh prints no
# rustc line, so the caller cleans the named packages first.
set -euo pipefail

log="${1:?usage: $0 <cargo -v log> <crate name>...}"
shift
[ "$#" -gt 0 ] || { echo "usage: $0 <cargo -v log> <crate name>..." >&2; exit 2; }

fail() { echo "gungraun: $*" >&2; exit 1; }
[ -r "$log" ] || fail "no build log at $log"

lines="$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -E '^ +Running `.*rustc .*--crate-name [a-z0-9_]+ ' || true)"
for crate in "$@"; do
    mine="$(printf '%s\n' "$lines" | grep -E -- "--crate-name $crate " || true)"
    [ -n "$mine" ] || fail "no rustc line for $crate in the build log: cannot show it was built with the gate's profile"
    if printf '%s\n' "$mine" | grep -vqE -- '-C codegen-units=1( |`|$)'; then
        fail "$crate was compiled without -C codegen-units=1: not the gate's profile"
    fi
    if printf '%s\n' "$mine" | grep -qE -- '-C lto=(fat|thin|true|yes|on)'; then
        fail "$crate was compiled with LTO: not the gate's profile"
    fi
done
echo "gungraun: profile check passed: $* built with -C codegen-units=1, no LTO"
