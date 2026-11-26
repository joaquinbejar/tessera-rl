<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Benchmark results

The inputs of the benchmark dashboard (`make dashboard`). One JSON file per
run, schema `tessera-dashboard-result/1`, wrapping an unchanged
`tools/bench/bench.py` report.

File names follow `<UTC date>-<machine>-<short commit>.json`; this is a
convention, and the builder does not check it.

Admission: a file enters through a reviewed PR, and is never edited once
merged. A correction adds a new file with `"supersedes": "<file name>"`.
Only measured results belong here; test fixtures do not.

## Flagship results (#640)

The builder's flagship table (`flagships.html`) lists every flagship
number (configurations with `flagship = "<id>"` in
`tools/bench/configs.toml`) from the valid and indicative files here, with
its configuration, metric, median, 95% CI, workers, schedule, whether the
throughput includes illegal-action no-ops (`illegal_action_noops`),
machine, the machine's current `compat/machines.toml` status, commit and
`valid_for_claims`. Scaling rows are "efficiency vs k×Inline": k-worker
throughput over k times the Inline throughput, not over one Sync worker. Until a
reference machine exists (#151), flagship files are indicative: they keep
`valid_for_claims: false` and their `claim_check` reasons, and use the full
method. The phase 4 gate is met only by a reference-machine run with
`--runs 10`. Flagship files are one per flagship invocation, named
`<UTC date>-<machine>-<short commit>-<flagship>.json`, because the six
runs share date, machine and commit.

Flagship runs follow the measurement load rule of `tools/bench/README.md`:
before each process run, load5 must be below the bound B (2 for the
indicative runs, waited for within a 2 h budget per invocation); a run is
void when load5 after it is at least B + k_run (k_run is the run's worker
count). Each file carries the invocation's start and end loads and every
run's loads with both bounds in its top-level `loads`
(`tools/dashboard/envelope.py` writes it). A report marked `load_void` is
refused. If the reference candidate cannot meet the bound, the fallback
host is the `ref-x86-alt` record (status `candidate`) of
`compat/machines.toml`. The first flagship files (#640, PR #750) are six indicative runs on `ref-x86-alt` at commit `cd5efdda`, one per flagship; they are not claims, and their numbers are not compared across flagships.
