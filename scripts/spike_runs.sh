#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Independent repeated runs of the layout spike (ADR-0025: at least 10
# repetitions). Each run measures every layout in a freshly shuffled order
# (one `cargo bench` process per layout, a Criterion baseline per run and
# layout), so drift and ordering effects spread over the layouts.
#
# Usage: scripts/spike_runs.sh [runs] [criterion dir]
# Then:  python3 scripts/spike_stats.py --runs [criterion dir]
set -euo pipefail
runs="${1:-10}"
dir="${2:-target/criterion}"
layouts=(L0-parallel L1-combined L2-soa L3-aosoa)
export SPIKE_WARMUP_S="${SPIKE_WARMUP_S:-2}" SPIKE_MEASURE_S="${SPIKE_MEASURE_S:-2}" SPIKE_SAMPLES="${SPIKE_SAMPLES:-10}"
cargo bench --locked -p tessera-rl-spike-layout --bench layout --no-run >/dev/null 2>&1
for r in $(seq 1 "$runs"); do
  order=$(printf '%s\n' "${layouts[@]}" | python3 -c "import random,sys; l=sys.stdin.read().split(); random.Random($r).shuffle(l); print(' '.join(l))")
  echo "run $r: $order" >&2
  for layout in $order; do
    cargo bench --locked -p tessera-rl-spike-layout --bench layout -- \
      --save-baseline "run$r" --noplot "/$layout/" >/dev/null 2>&1
  done
done
echo "done: $runs runs in $dir" >&2
