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
| Raw env throughput | `rust/*` with `metric = "throughput"`: auto-reset and observation writing included. Every catalog environment has an entry (`rust/<env>/inline/n1024`, SPEC 16.1). Masked games with a flat mask (Lanterns among them) set `legal_actions`: each planned action moves to the next legal choice inside the timed step, so they step real moves; RTS and Tag step under the no-op illegal-action policy. Every Rust row records `action_policy` (`planned` or `legal_actions`) |
| Scaling efficiency | `scaling_efficiency/*`: k-worker throughput over k times one worker (`cartpole`: a one-worker Sync run); the flagships use "efficiency vs k×Inline" (below) |
| Step latency | `rust/*/latency`: p50 and p99 per batch, Sync mode |
| Python overhead | `python_overhead/*`: binding throughput over raw Rust, borrowed and owned separately |
| Snapshot cost | `rust/*/snapshot`, per instance. Copy only: `snapshot_copy` and `snapshot_all_copy`; the latter is also reported in ns/KiB against ADR-0010's budget of at most 1 µs/KB. Including re-observation (ADR-0039): `restore_with_reobserve`, `import_with_validation_and_reobserve`, `fork_with_reobserve`, `fork_branched_with_reobserve` |
| Memory | `rust/*/memory`: continuation bytes per instance, mask bytes per step, output storage bytes |
| Head to head | `py/gymnasium-sync/*` against `py/gym-adapter/*`: same environment count, NEXT_STEP auto-reset, same action stream |
| Against EnvPool | `speedup/native-vs-envpool/*`: `py/native/<env>/n1024/borrowed` over `py/envpool/<env>/n1024` (CartPole, Acrobot, MountainCar, Pendulum), one thread each, same plan (continuous plans for Pendulum) and options. The output paths differ (Tessera's borrowed views; EnvPool's fresh arrays plus an info dict each step), so every speedup carries a `note` saying so, and a published number must state it. The `py/envpool/*` configurations declare `requires = "envpool"`; without that package they are skipped and listed under `skipped` in the report |

End-to-end SPS (reference PPO) and the authoring study come later.

## Flagship benchmarks (#640)

The SPEC 22 phase 4 gate, "flagship benchmarks published", covers six
flagships: `tessera/RTS-v1`, `tessera/LimitOrderBook-v1`,
`tessera/Lanterns-v1`, `tessera/Survival-v1`, `tessera/JobShop-v1` and
`tessera/Vrp-v1`. Their configurations carry `flagship = "<id>"`, and each
flagship is measured on all four SPEC 19.1 metrics at one N, so its rows
are comparable:

| Metric | Configurations |
| --- | --- |
| Raw throughput | `rust/<env>/inline/n<N>`: Inline, one pinned core |
| Scaling | `rust/<env>/sync4/n<N>` (4 workers) and `rust/<env>/syncfull/n<N>` (`workers = "full"`: one worker per CPU of the machine's `bench_cpu_set`, the effective count recorded as `workers`); `scaling_efficiency/<env>/sync4/n<N>` and `.../syncfull/n<N>`: "efficiency vs k×Inline", k-worker throughput over k times the Inline entry |
| Snapshot cost | `rust/<env>/n<N>/snapshot`: snapshot, restore, import, fork, fork_branched |
| Memory | `rust/<env>/n<N>/memory`: continuation bytes per instance, mask bytes per step |

N is 1024, except Survival (16 agents on 32 x 32) at 256. Every
multi-worker configuration names its `schedule` (ADR-0076, ADR-0087):
`dynamic` for RTS and Survival, whose step cost varies per instance, and
`static` for the others; the runner takes it as `--schedule`, and the
report records it in the options and as `schedule`.

Scaling efficiency is labelled "efficiency vs k×Inline" in the report
(the derived entry's `note`) and in the flagship table: its baseline is k
times the Inline throughput, not a one-worker Sync run. Inline has no
worker threads, so the number includes the Sync scheduling overhead; it
is the honest figure for choosing a mode, not parallel efficiency against
one worker.

Survival keeps N = 256, which at the engine's default chunk of 64 is only
4 chunks, so a CPU set larger than 4 would leave workers idle by
construction. Its multi-worker configurations therefore set
`chunk_size = 16` (16 chunks), at least as many chunks as the largest
`bench_cpu_set` in `compat/machines.toml`; the runner takes it as
`--chunk-size`, and every row records the effective `chunk_size`. Its
Inline entry keeps the default chunk (it has no workers to feed), and a
test checks that every flagship multi-worker configuration has at least
as many chunks as the largest `bench_cpu_set`. An unpinned `syncfull` run
uses every CPU the process may use, which can exceed that.

Every flagship configuration states `illegal_action_noops`, shown as a
column of the flagship table: whether its throughput includes
illegal-action no-ops. The masked flagships step as their Inline entries
do:

| Flagship | Action policy | Includes illegal-action no-ops |
| --- | --- | --- |
| `tessera/RTS-v1` | planned, no-op policy (conditioned heads) | yes |
| `tessera/LimitOrderBook-v1` | planned, no-op policy | yes |
| `tessera/Lanterns-v1` | `legal_actions` | no |
| `tessera/Survival-v1` | planned, no-op policy | yes |
| `tessera/JobShop-v1` | planned, no-op policy | yes |
| `tessera/Vrp-v1` | planned, no-op policy | yes |

A comparison with an alternative must step the same policy or state the
difference.

The gate is Tessera's own numbers. Each flagship's closest alternative is
either named with a same-machine comparison (same harness discipline,
pinned version, the step semantics stated) or recorded as "none found"
with the search done; a comparison is not required to close the gate, and
no alternative is downloaded without announcing it first.

Until a reference machine exists (#151), flagship numbers are indicative:
`valid_for_claims` is false and the `claim_check` reasons are kept. The
method does not change for them (at least 2 s of warm-up, 10 repetitions,
a 95% CI). The gate is met only by a reference-machine run with
`--runs 10`, admitted through a reviewed PR. The dashboard's flagship
table (`flagships.html`) shows every valid or indicative flagship row with
its machine's current `compat/machines.toml` status.

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

## Nightly wall-clock gate (SPEC 19.4)

`tools/bench/nightly.py` and `.github/workflows/nightly-bench.yml` (#374).

- **What it compares:** each night compares today's head with the **accepted base**, a commit kept in the previous night's artifact.
- **How it measures:** paired, interleaved process runs of the two runner binaries, on the single-core `rust/*/inline` cases in `CASES`. The change per pair is `base / head - 1` throughput (positive is slower); its 95% interval is a seeded bootstrap of the median over the pairs.
- **When a case is flagged:** a case is flagged when the interval lies entirely above +3%.
- **How the base moves:**
  - It advances to the head only on a qualifying night with no flag.
  - A flagged night holds it, so a lasting regression is compared again instead of being absorbed.
  - A flag clears when a later head is no longer flagged, or when the owner acknowledges the head in `compat/bench/nightly-ack.toml` (a reviewed commit).
- **When it goes red:** the same case is flagged on two consecutive *qualifying* nights.
- **Qualifying nights:** a night does not qualify when the load exceeds the limit, other jobs ran on the host during the window, or an A/A half-width exceeds 1.5%. Such a night neither flags, nor moves the base, nor counts for the streak. With no new commit, the night runs an A/A.
- **Binding:** the gate binds only on a machine whose `compat/machines.toml` status is `reference`; otherwise it reports what it would have done and stays green.
- **Slow drift:** a regression below the threshold each night can accumulate, because the base advances after every clean night. The state keeps `weekly_base`, the base accepted at least seven qualifying nights earlier, for a periodic comparison against a fixed base (reported, never flagged).

