#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Median and 95% bootstrap confidence intervals from Criterion samples.

ADR-0025 asks for the median with a 95% bootstrap CI; Criterion reports a
CI on the mean, so this script recomputes from the raw samples
(`target/criterion/<group>/<layout>/<param>/new/sample.json`, per sample
`times[i] / iters[i]` nanoseconds per iteration).

For each group and parameter it prints a Markdown table: the median, its
CI, and the ratio to the L0 layout with a CI of the ratio (paired by
bootstrap resampling of both sides). Standard library only; the resampling
seed is fixed so the report is reproducible.

Usage: python3 scripts/spike_stats.py [target/criterion] > report.md
"""

import json
import os
import random
import statistics
import sys

RESAMPLES = 10_000
SEED = 20261031
BASELINE = "L0-parallel"


def samples(path):
    with open(path) as f:
        data = json.load(f)
    return [t / n for t, n in zip(data["times"], data["iters"])]


def bootstrap(xs, rng):
    n = len(xs)
    meds = sorted(
        statistics.median(rng.choices(xs, k=n)) for _ in range(RESAMPLES)
    )
    return meds[int(0.025 * RESAMPLES)], meds[int(0.975 * RESAMPLES) - 1]


def ratio_ci(xs, base, rng):
    rs = sorted(
        statistics.median(rng.choices(xs, k=len(xs)))
        / statistics.median(rng.choices(base, k=len(base)))
        for _ in range(RESAMPLES)
    )
    return rs[int(0.025 * RESAMPLES)], rs[int(0.975 * RESAMPLES) - 1]


def fmt(ns):
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if ns >= scale:
            return "%.3f %s" % (ns / scale, unit)
    return "%.1f ns" % ns


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "target/criterion"
    table = {}
    for dirpath, _, files in os.walk(root):
        if "sample.json" not in files or os.path.basename(dirpath) != "new":
            continue
        parts = os.path.relpath(dirpath, root).split(os.sep)[:-1]
        if len(parts) != 3:
            continue
        group, layout, param = parts
        table.setdefault((group, param), {})[layout] = samples(
            os.path.join(dirpath, "sample.json")
        )
    rng = random.Random(SEED)
    for (group, param) in sorted(table):
        rows = table[(group, param)]
        print("\n### %s, %s\n" % (group, param))
        print("| Layout | Median | 95% CI | Ratio to L0 | 95% CI of ratio | n |")
        print("| --- | --- | --- | --- | --- | --- |")
        base = rows.get(BASELINE)
        for layout in sorted(rows):
            xs = rows[layout]
            lo, hi = bootstrap(xs, rng)
            med = statistics.median(xs)
            if base is not None:
                r = med / statistics.median(base)
                rlo, rhi = ratio_ci(xs, base, rng)
                ratio = "%.3f" % r
                rci = "[%.3f, %.3f]" % (rlo, rhi)
            else:
                ratio, rci = "", ""
            print(
                "| %s | %s | [%s, %s] | %s | %s | %d |"
                % (layout, fmt(med), fmt(lo), fmt(hi), ratio, rci, len(xs))
            )


if __name__ == "__main__":
    main()
