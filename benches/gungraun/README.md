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
Run it in the baseline's image:

```sh
docker run --rm --security-opt seccomp=unconfined \
  -e TESSERA_GUNGRAUN_IMAGE=rust:1.99.1@sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137 \
  -v "$PWD":/src -w /src rust:1.99.1@sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137 \
  sh -c 'apt-get update && apt-get install -y valgrind=1:3.24.0-3 &&
         cargo install --locked gungraun-runner --version 0.20.0 &&
         scripts/gungraun.sh check'
```

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
| `turns_instructions::masked_step` | One Inline pool step, 1,024 instances of the counter environment with masks on (mask writing) |
| `turns_instructions::sequential_step` | One Inline pool step, 1,024 instances of the Sequential race fixture (target 100) with masks on: masks, the acting update and the turn check |
| `games_instructions::connect4_pool_step` | One Inline mid-game pool step, 1,024 Connect Four instances 10 plies in (masks, the acting update and the turn check). Auto-reset is enabled but not exercised: no game ends in the measured step |
| `games_instructions::bot_choose` | The pinned P2-G1 bot's choice (depth 4) on 32 fixed mid-game positions |
| `search_instructions::expand_batch` | One batched expansion of the 7 children of the empty Connect Four board (restore, one scratch step, snapshot, commit) |
| `search_instructions::puct_run` | One PUCT call of 256 simulations from the empty board, batch 8, uniform evaluator: selection, virtual loss, batched expansion, evaluation and backup |

The cost of the serial turn check (`check_turns`) inside `sequential_step`
was measured by an A/B run in the baseline's image: the same bench with
the check disabled in an uncommitted copy ran 14,339 fewer instructions,
1.04% of the step (about 14 per instance). That is why the check stays a
serial pass over the headers; the benchmark pins its cost.

Setup runs outside the measured region, and every benchmark returns its
inputs so that dropping them, and the allocator, is not counted.

## Environment of the baseline

Instruction counts depend on the binary and on the libc code it runs, so
the baseline fixes both. `environment.toml`, written by `bless`, records
the environment, and `check` refuses to run in a different one, with a
diagnostic: a changed environment needs a reviewed re-bless and is never
reported as a regression. The recorded environment is:

- Image `rust:1.99.1@sha256:09027f87...` (`TESSERA_GUNGRAUN_IMAGE` names
  it; the CI job uses the same digest), Valgrind 3.24.0 (Debian package
  `1:3.24.0-3`), `gungraun-runner` 0.20.0 (`cargo install --locked`),
  rustc 1.99.1, glibc 2.41, target `x86_64-unknown-linux-gnu`, Cargo
  `bench` profile.
- **glibc dispatch pin.** glibc chooses its `memcpy` and `memset` variants
  by CPU (ifunc), and the copy benchmarks run them. The benches set
  `GLIBC_TUNABLES` from the committed file `GLIBC_TUNABLES`, which masks
  every optional x86 feature and preferred-string flag glibc dispatches on
  and fixes the cache sizes and copy thresholds. glibc then runs its x86-64
  base (`sse2_unaligned`) variants on any x86-64 CPU. The validation on
  xaskar, in the PR, shows:
  - without the pin the counts differ;
  - masking even more features changes no count;
  - with the pin the counts are stable.
  `environment.toml` records the file's hash.
- A fixed cache geometry (`--I1=32768,8,64 --D1=32768,8,64
  --LL=8388608,16,64`), so cache metrics compare across hosts. The gate is
  on `Ir` only (ADR-0025, as amended). Cache misses and hit rates are
  reported, not gated.
- ASLR disabled by gungraun. Under Docker this needs
  `--security-opt seccomp=unconfined`, because the default profile blocks
  `personality`.
- Our own code has no SIMD dispatch yet. When `tessera-rl-simd` lands, the
  gate must also force its dispatch to fixed targets (scalar and AVX2,
  ADR-0025).

## Fail closed

`check` fails, without blessing anything, when:

- the environment differs from `environment.toml`;
- a benchmark of the expected list (`scripts/gungraun.sh`) has no
  committed baseline;
- not every benchmark was compared against its baseline;
- any baseline file is created or changed by the run;
- `Ir` regresses by more than 2% (exit 3).

## Re-blessing

A new baseline is needed when an intended change moves the counts by more
than the limit, or when the environment above changes (toolchain, image,
Valgrind, gungraun). To re-bless:

1. Run `scripts/gungraun.sh bless` in the environment above (it rewrites
   `environment.toml`), then `scripts/gungraun.sh check`, which must pass.
   A new or renamed benchmark also updates the expected list in the script.
2. Commit only the `*.base@ref` files and `environment.toml`, in their own
   commit. The message gives the old and new `Ir` of every changed
   benchmark and the reason.
3. The PR needs the coordinator's approval and both reviewers' (Codex and
   TESSERA-REVIEWER). A re-bless never rides along with an unrelated change.
