#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# The instruction-count band, both sides (ADR-0059 section 4).
#
#   scripts/gungraun_band.sh <gungraun log> <band %> <expected cases>
#
# gungraun's own limit (--callgrind-limits) already fails a rise beyond the
# band before this runs; this checks the rise again from the log, and fails
# a fall beyond the band, which gungraun lets pass and which would leave the
# baseline above reality. It reads every "Instructions: new|base" line
# of a check run's log (ANSI colours stripped: CI sets CARGO_TERM_COLOR),
# fails closed unless exactly <expected cases> were compared, and fails if
# any case is above or below its baseline by more than <band %>, naming
# each one (and, for a fall, the re-bless the author must do).
set -euo pipefail

log="${1:?usage: $0 <gungraun log> <band %> <expected cases>}"
band="${2:?usage: $0 <gungraun log> <band %> <expected cases>}"
expected="${3:?usage: $0 <gungraun log> <band %> <expected cases>}"

fail() { echo "gungraun: $*" >&2; exit 1; }
[ -r "$log" ] || fail "no log at $log"

# "<case>\t<new>\t<base>" per compared case.
cases="$(sed 's/\x1b\[[0-9;]*m//g' "$log" | awk '
    /^[a-z_]+_instructions::[^ ]+ [^ :]+:/ { split($0, p, ":\\("); name = p[1]; next }
    /^ +Instructions: +[0-9]+\|[0-9]+ / {
        split($2, v, "|"); printf "%s\t%s\t%s\n", (name == "" ? "?" : name), v[1], v[2]; name = ""
    }')"
compared="$(printf '%s' "$cases" | grep -c . || true)"
[ "$compared" -eq "$expected" ] || fail "the band check read $compared compared cases, expected $expected"

below="$(printf '%s\n' "$cases" | awk -F'\t' -v band="$band" '
    $3 > 0 && ($2 - $3) * 100 < -band * $3 {
        printf "  %s: %d against the baseline %d (%+.2f%%)\n", $1, $2, $3, ($2 - $3) * 100 / $3
    }')"
above="$(printf '%s\n' "$cases" | awk -F'\t' -v band="$band" '
    $3 > 0 && ($2 - $3) * 100 > band * $3 {
        printf "  %s: %d against the baseline %d (%+.2f%%)\n", $1, $2, $3, ($2 - $3) * 100 / $3
    }')"
if [ -n "$above" ]; then
    { echo "gungraun: above the baseline by more than ${band}% (a regression):"; echo "$above"; } >&2
    exit 1
fi
if [ -n "$below" ]; then
    {
        echo "gungraun: below the baseline by more than ${band}% (ADR-0059: an improvement beyond the band is a reviewed re-bless, not a silent pass):"
        echo "$below"
        echo "gungraun: dispatch the 'gungraun bless' workflow on this pull request's branch, commit the artifact's files unchanged in this pull request, and put the workflow run id and the artifact's sha256 in the pull request description."
    } >&2
    exit 1
fi
echo "gungraun: band check passed: $compared cases within ${band}% of their baselines"
