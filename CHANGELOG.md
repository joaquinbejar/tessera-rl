<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Changelog

All notable changes to this project are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/). Before 1.0
the API is not stable, and any 0.x release may break it.

## [Unreleased]

### Added

- **PettingZoo adapters** (`tessera_rl.pz`, with the `pettingzoo` extra):
  `parallel_env` and `aec_env`. Each environment declares the interfaces
  it supports and any other request is refused at construction: Tag is
  Parallel natively and AEC through cycle buffering, Connect Four is AEC
  only. They pass the official PettingZoo API tests, and tests cover
  terminal reward timing, agent death, a conditional second-player
  decision and equality with the native pool. `turn_masked_env` is
  declared by no environment yet. Not yet tested: agent respawn, which
  ADR-0013 requires, because no shipped environment spawns agents; the
  test lands with the first one that does (#71).
- **Search** (`tessera-rl-search`): player-perspective search on
  imperfect-information environments. `SearchPool::new_determinized` and
  `set_root_determinized` root the tree at a state sampled by
  `Determinize` from the searching agent's information state, with a
  caller's sampler seed and sample index. The root's hidden state and RNG
  lineage never reach the search: the sampler and the rollout draw with
  their own identities (the `SAMPLER` and `ROLLOUT` branch labels), the
  header keeps only the step clock, `alive`, `generation` and adaptive
  values, and the record passes the engine's import checks. A
  determinized tree is never re-rooted. `SearchPool::new_privileged`
  searches the full record and is labelled by `is_privileged`;
  `SearchPool::new` still refuses imperfect information.
- **Engine** (`tessera-rl-engine`): `EnvPool::config`, and `EnvPool::contain`,
  which runs author code that reads only the configuration under the
  pool's containment (the canonical floating-point scope, a panic caught
  and reported as `None`).

## [0.1.0] - 2026-11-04

The first functional release. Six of these crate names were reserved on
crates.io at 0.0.1; `tessera-rl-games` and `tessera-rl-search` are new.

### Added

- **Core** (`tessera-rl-core`): the `Environment` trait and its family
  (`ActionMask`, `Validate`, `Heuristic`, `ImperfectInformation`,
  `Determinize`, `Render`, `EnvAssets`), spaces, `StepOut`, agent slots,
  the canonical encoding and `StateHash`, and the plain-data state
  contract (`Bool8`, `Reserved<N>`, `Pool<T, CAP>`, `StateEnum`).
- **Numerics** (`tessera-rl-num`): the counter-based Philox RNG with
  per-instance identities and branch salts, the `Fx32` and `Fx64` fixed-point
  types, frozen deterministic math, and the floating-point scope guard.
- **Macros** (`tessera-rl-macros`): the `EnvState`, `EnvConfig`,
  `Observation` and `StateEnum` derives, with compile-time rejection of
  padding, missing `#[repr(C)]` and non-plain fields.
- **Engine** (`tessera-rl-engine`): `EnvPool` with the `Inline` and `Sync`
  modes (pinned worker threads, futex waits on Linux), the `NextStep`,
  `SameStep` and `Disabled` auto-reset modes, action masks, `Sequential`
  and `Mixed` turns, poisoning, the validation boundary, and snapshot,
  restore, fork and branched fork of instance records and pool context.
- **Search** (`tessera-rl-search`): the bounded `SearchPool` and a batched
  PUCT MCTS helper with virtual loss and root noise.
- **Environments**: `tessera/CartPole-v1` and `tessera/Tag-v1`
  (`tessera-rl-classic`), and `tessera/Connect4-v1` (`tessera-rl-games`),
  with golden trajectory and continuation tests.
- **Facade** (`tessera-rl`): re-exports of the crates, a `prelude`, and the
  `debug-checks` feature forwarded to the engine
  (`tessera_rl::engine::DEBUG_CHECKS` reports it).
- **Python** (`tessera-rl-py`, built with maturin, abi3, Python 3.11+):
  `tessera_rl.make`, owned and borrowed outputs, a Gymnasium 1.x `Env` and
  `VectorEnv` (`tessera_rl.gym`), and `Search` and `SearchGroup` over a
  Python evaluator.
- **Example**: an AlphaZero-style Connect Four training pipeline
  (`examples/alphazero_connect4`) with a core-hour ledger, a supervised
  phase harness and a remote phase transport.

### Not included in 0.1.0

- **The phase 2 exit gate has not been run.** Whether the AlphaZero-style
  example beats the scripted Connect Four bot in at least 90% of 1,000
  games is not established.
- **The example's training run is incomplete.** It was stopped after 8
  generations (51 of its 200 core-hour budget) by the owner's decision and
  is deferred. The best held-out look so far won 310 of 400 games; the
  looks are a selection aid, not a gate estimate.
- **`SearchGroup` scope.** Its throughput was measured once and it was
  adopted (results in
  `examples/alphazero_connect4/results/group-bench-20261103`), but the
  example's training loop still uses single-tree search. The measurement is
  shared-host throughput on one machine, not a published benchmark.
- **No throughput figure** is published with this release.
- **0.1.0 is unverified on Windows.** CI covers Linux x86-64, Linux
  aarch64 and macOS (aarch64); Windows is not in CI. The last Windows runs
  were on 2026-11-01: a native hand check of the Python Win64
  floating-point boundary at `cc5c97d` (#29), and GitHub's Windows CI of
  the Rust workspace, last green on main at `105be6c`. Both the Rust code
  (search and engine) and the Python native boundary have changed since.
  In particular the Win64 thunk in this release has never run on Windows.
- **Planned, not implemented:** PettingZoo and PufferLib adapters,
  `Async` and `Pipelined` modes, persisted snapshots, Othello, Go 9x9 and
  Leduc hold'em (phase 2); SIMD lanes, JAX and GPU paths, Snake, Pong and
  Breakout (phase 3); the RTS, market simulator, card games, survival world,
  league tooling and viewer (phase 4). Their crates do not exist yet.
- The Python package is not published to PyPI.

### Known deviations from the specification

Code, comments and documents cite a project specification ("SPEC") and
architecture decision records ("ADR-00nn") that are not published in this
repository, so a reader cannot open them. Where the specification and the
code disagree, the specification is the target, and these are open
defects:

- `debug-checks` does not provide the NaN detection and allocation
  tracking the specification requires (#62). This release is knowingly
  non-conformant with that requirement.
- `tessera_rl::register!` is not implemented (#63).
- There is no typed `Discrete<N>` action type, and the specification's
  authoring example does not compile: it also omits `type Header`,
  `observation_space` and `action_space`, which the trait requires (#64).
  The README quick start is a compiled doctest of the real API.

### Known issues

- #43: `alloc_sync` counted 4 allocations in the `Sync` step window once,
  on a GitHub-hosted ubuntu-latest runner with default features. It has not
  been reproduced: 1,200 earlier runs, then 5,000 runs in a local Linux
  container and 10,000 on an x86-64 Linux host (two variants, 4 pinned
  CPUs). The GitHub-hosted environment could not be retested because of
  account billing. The test now records each counted allocation (its size,
  the thread and its name, whether the thread was in a step or reset kernel
  or in neither, and that thread's kernel call count), so a recurrence will
  be diagnosable. The assertion
  stays at zero allocations.

### Notes for dependent crates

- Depend on the facade `tessera-rl` only; the derives resolve their paths
  through it.
- This repository's `clippy.toml` and `rust-toolchain.toml` do not apply
  to a dependent crate. To keep simulation kernels deterministic across
  platforms, configure the same `disallowed_methods` lints (no `mul_add`,
  no `powi`, no `std` transcendental methods; use `tessera_rl::num::math`)
  in your own crate.

[0.1.0]: https://github.com/joaquinbejar/tessera-rl/releases/tag/v0.1.0
