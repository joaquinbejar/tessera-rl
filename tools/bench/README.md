<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Benchmark harness

The harness implements the method of ADR-0025 and SPEC 19. Every number it
reports carries a named configuration, the machine, the software versions
and the method, and can be reproduced with one command.

```sh
make bench MACHINE=ref-mac        # full method, JSON report in target/bench/
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
- Fixed seeds and pregenerated seeded action streams. Workers are pinned by
  the engine's default policy (worker `w` on core `w`). On the hybrid
  `ref-x86` candidate, single-core runs go under `taskset -c 2`, as
  `compat/machines.toml` says.
- Every Python result names its output mode. `owned` numbers are never
  presented as borrowed-path throughput.
- `valid_for_claims` is true only with the full method on a machine whose
  `compat/machines.toml` status is `reference`. Today no machine is
  characterized (`ref-x86` is a candidate on a shared host), so every
  report is indicative. Quick runs are never valid.

## Metrics (SPEC 19.1)

| Metric | Configurations |
| --- | --- |
| Raw env throughput | `rust/*` with `metric = "throughput"`: auto-reset and observation writing included |
| Scaling efficiency | `scaling_efficiency/*`: k-worker throughput over k times one worker |
| Step latency | `rust/*/latency`: p50 and p99 per batch, Sync mode |
| Python overhead | `python_overhead/*`: binding throughput over raw Rust, borrowed and owned separately |
| Snapshot cost | `rust/*/snapshot`: per-instance snapshot, restore, fork, `fork_branched` |
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
