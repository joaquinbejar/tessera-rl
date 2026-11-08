<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Benchmark results

The inputs of the benchmark dashboard (`make dashboard`). One JSON file per
run, schema `tessera-dashboard-result/1`, wrapping an unchanged
`tools/bench/bench.py` report.

Admission: a file enters through a reviewed PR, and is never edited once
merged. A correction adds a new file with `"supersedes": "<file name>"`.
Only measured results belong here; test fixtures do not.
