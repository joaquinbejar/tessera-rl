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
its configuration, metric, median, 95% CI, machine, the machine's current
`compat/machines.toml` status, commit and `valid_for_claims`. Until a
reference machine exists (#151), flagship files are indicative: they keep
`valid_for_claims: false` and their `claim_check` reasons, and use the full
method. The phase 4 gate is met only by a reference-machine run with
`--runs 10`. No flagship file has been admitted yet.
