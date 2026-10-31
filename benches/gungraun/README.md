<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Instruction-count baseline (gungraun)

This directory holds the committed baseline of the ADR-0025 level 1 gate
(gungraun 0.20.0 on Valgrind's Callgrind, cache simulation on): the
`*.base@ref` files under one directory per crate, bench and benchmark.
Everything else gungraun writes here is ignored.

`make gungraun` (`scripts/gungraun.sh check`) compares a run with this
baseline and fails when the instruction count (`Ir`) of any benchmark
regresses by more than 2%. Cache misses and hit rates are reported, not
gated. On macOS the target exits 0 with a message: Valgrind needs Linux.

## What is measured

| Benchmark | Work |
| --- | --- |
| `classic_instructions::cartpole_physics` | 1,000 CartPole physics steps |
| `classic_instructions::cartpole_pool_step` | One Inline pool step, 1,024 CartPole instances, auto-reset and observation writing |
| `classic_instructions::tag_pool_step` | One Inline pool step, 256 Tag instances (masks on) |
| `snapshot_instructions::step_inline` | One Inline pool step, 1,024 instances of the engine's counter test environment |
| `snapshot_instructions::snapshot_all_copy` | `snapshot_all` into a preallocated buffer (copy only) |
| `snapshot_instructions::restore_with_reobserve` | `restore` of every instance (with re-observation, ADR-0039) |
| `snapshot_instructions::fork_with_reobserve` | `fork` of instance 0 into the other 1,023 |
| `snapshot_instructions::fork_branched_with_reobserve` | `fork_branched` of instance 0 into the other 1,023 |

Setup runs outside the measured region, and every benchmark returns its
inputs so that dropping them, and the allocator, is not counted.

## Environment of the baseline

Instruction counts depend on the binary, so the baseline fixes the build
and the tools:

- Image `rust:1.99.1` (Docker, x86-64), with Valgrind 3.24.0 from its
  Debian packages and `gungraun-runner` 0.20.0 (`cargo install --locked`).
- Target `x86_64-unknown-linux-gnu`, Cargo `bench` profile (release
  optimizations), the workspace's `Cargo.lock`.
- ASLR disabled by gungraun. Under Docker this needs
  `--security-opt seccomp=unconfined`, because the default profile blocks
  `personality`.
- No SIMD dispatch exists yet. When `tessera-rl-simd` lands, the gate must
  force the dispatch to fixed targets (scalar and AVX2, ADR-0025), because
  Valgrind does not support AVX-512 and runners differ.

The CI job `gungraun` runs in the same image.

## Re-blessing

A new baseline is needed when an intended change moves the counts by more
than the limit, or when the environment above changes (toolchain, image,
Valgrind, gungraun). To re-bless:

1. Run `scripts/gungraun.sh bless` in the environment above, then
   `scripts/gungraun.sh check`, which must pass.
2. Commit only the `*.base@ref` files, in their own commit, with the old
   and new `Ir` of every changed benchmark and the reason in the message.
3. The PR needs the coordinator's approval and both reviewers' (Codex and
   TESSERA-REVIEWER). A re-bless never rides along with an unrelated change.
