<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Benchmark harness

The harness implements the method of ADR-0025 and SPEC 19. Every number it
reports carries a named configuration, the machine, the software versions
and the method, and can be reproduced with one command.

```sh
make bench MACHINE=ref-x86 RUNS=10   # claim method, JSON report in target/bench/
make bench-quick                  # smoke run (not valid for claims)
make bench-kernels                # Criterion benches of the kernels
```

## Pieces

| Piece | Role |
| --- | --- |
| `configs.toml` | The method (warm-up, repetitions, bootstrap) and the named configurations; derived metrics. |
| `bench.py` | Runs the configurations and writes the report: statistics, machine, hardware, frequency policy, load, versions. |
| `tools/bench-runner` | Rust runner: raw samples for throughput, latency, snapshot costs and memory, as JSON. |
| `crates/*/benches` | Criterion benches of the kernels (CartPole physics, classic pool steps, Sync-mode wait backends). |

## Method

- At least 2 s of warm-up and 10 measured repetitions per configuration.
  The report gives the median and the 95% bootstrap confidence interval of
  the median (10,000 seeded resamples), with the raw samples.
- Process-level repetitions: `--runs N` runs every configuration in N
  separate processes, in a seeded shuffled order recorded in the report.
  The claim statistic is the median and CI over the process medians. One
  run (the default) gives in-process statistics, which are only indicative.
- The same work everywhere: every configuration, Rust or Python, steps the
  same seeded action plan (a 64-bit LCG, 64 batches) under the same
  options: seed 1, `next_step` auto-reset, and `max_episode_steps` (500 by
  default, CartPole-v1's). The report records each side's options and the
  plan's FNV-1a fingerprint. A derived ratio marked `require_same_work` is
  refused when they differ.
- Every measured operation is warmed up first, snapshot costs included.
  The report records the effective method of each metric.
- Pinning: each machine's `bench_cpu_set` in `compat/machines.toml` lists
  distinct physical cores, avoids CPU 0 and leaves SMT siblings idle.
  Single-core and Python runs use its first CPU (`taskset`). A k-worker run
  pins the process to the first k CPUs and passes them to the pool as the
  workers' `cpu_set`. The report records the requested set and the process
  affinity the OS reports. Without `taskset` (macOS) runs are unpinned, and
  the report says so.
- Every Python result names its output mode. `owned` numbers are never
  presented as borrowed-path throughput.
- Derived ratios (scaling efficiency, Python overhead, speedup) carry a 95%
  CI that bootstraps each operand's median independently.
- Claims fail closed. `valid_for_claims` holds only when
  `claim_check.reasons` is empty, which requires all of the following:
  - the full method;
  - at least `runs_for_claims` process runs;
  - release builds (runner and native module);
  - pinning applied, with the observed affinity matching it;
  - an observed CPU matching the record's `cpu_id`;
  - enough samples;
  - a machine whose `compat/machines.toml` status is `reference`;
  - a qualification record (`--qualification`, TOML with `machine` and
    `exclusive_window = true`) for what cannot be checked automatically. Today no machine is
  characterized (`ref-x86` is a candidate on a shared host), so every
  report is indicative. Quick runs are never valid.

## Metrics (SPEC 19.1)

| Metric | Configurations |
| --- | --- |
| Raw env throughput | `rust/*` with `metric = "throughput"`: auto-reset and observation writing included |
| Scaling efficiency | `scaling_efficiency/*`: k-worker throughput over k times one worker |
| Step latency | `rust/*/latency`: p50 and p99 per batch, Sync mode |
| Python overhead | `python_overhead/*`: binding throughput over raw Rust, borrowed and owned separately |
| Snapshot cost | `rust/*/snapshot`, per instance. Copy only: `snapshot_copy` and `snapshot_all_copy`; the latter is also reported in ns/KiB against ADR-0010's budget of at most 1 µs/KB. Including re-observation (ADR-0039): `restore_with_reobserve`, `import_with_validation_and_reobserve`, `fork_with_reobserve`, `fork_branched_with_reobserve` |
| Memory | `rust/*/memory`: continuation bytes per instance, mask bytes per step, output storage bytes |
| Head to head | `py/gymnasium-sync/*` against `py/gym-adapter/*`: same environment count, NEXT_STEP auto-reset, same action stream |

End-to-end SPS (reference PPO) and the authoring study come later.

## Note: DLPack data alignment

The output storage aligns every region to 8 bytes, which covers every
element type. The DLPack header says the tensor `data` pointer is "always
aligned to 256 bytes as in CUDA", with `byte_offset` locating the first
element. Today's exports set `data` to the region and `byte_offset = 0`, so
`data` is only 8-byte aligned. NumPy and PyTorch on the CPU accept that,
and the tests cover them. Two options:

- Align every region to 256 bytes in the storage, which costs at most 255
  bytes per region.
- Export `data` rounded down to 256 bytes with a matching `byte_offset`.

This is still to be decided, with a benchmark showing whether alignment
changes copy or kernel throughput for the consumers in `compat/targets.toml`.
