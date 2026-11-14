<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Tessera

Tessera (`tessera-rl`) is a Rust engine for writing and running
reinforcement learning environments. It is built around three
properties that fast environment libraries rarely offer together:

- **Cheap snapshot, restore and fork** of an environment's state, so
  search (MCTS, AlphaZero-style self-play) and model-based planning can
  run on the real environment.
- **Bit-exact determinism**: a counter-based RNG keyed by instance
  identity, fixed-point types and frozen math, with golden trajectory
  hashes checked in CI.
- **A native multi-agent model**: agent slots, action masks, and
  simultaneous, sequential and mixed turns in one `Environment` trait.

Environments are plain-data Rust structs with a few derives. The engine
runs many instances in one pool (inline, or on worker threads pinned
where the OS allows)
with zero allocations per step. Python bindings expose them through a
Gymnasium-compatible interface.

Tessera is not a training library. It ships no production trainer; the
reference training scripts only validate the environments.

## Status

Version 0.2.0 is an early release. The API is not yet stable and may
change in any 0.x release.

| Area | State in 0.2.0 |
| --- | --- |
| Core, engine, macros, numerics (phase 1a) | Implemented: the `Environment` trait and spaces, the canonical encoding and state hashes, the `Sync` and `Inline` pools, auto-reset modes, poisoning, snapshot, restore, fork and branched fork, `debug-checks` for NaN and allocations, the `register!` environment registry |
| Python and Gymnasium (phase 1b) | Implemented: `PyO3` bindings (abi3, Python 3.11+), a Gymnasium 1.x `Env` and `VectorEnv`, owned and borrowed outputs, single-agent environments, `MultiDiscrete` and continuous (`Box`) actions |
| Masks, turns, modes, search (phase 2) | Implemented: action masks and conditional masks, `Sequential` and `Mixed` turns, the `Pipelined` and `Async` modes, persisted snapshots, `SearchPool` with batched PUCT and the Gumbel root, determinized and privileged search roots, an ONNX evaluator (CPU, Linux and macOS), Connect Four, Othello, Go 9x9 and Leduc hold'em |
| Adapters (phase 2 and 3) | Implemented: PettingZoo AEC and Parallel, PufferLib (Sync and Pipelined), JAX handles through ordered `io_callback` (CPU) |
| Classic games (phase 3) | Implemented: MountainCar, Acrobot, Pendulum, Grid, GridKeyDoor, Snake, PaddleDuel and BrickWall |
| SIMD lanes (phase 3) | Partial: `tessera-rl-simd` has a standalone lane-wise CartPole kernel; no pool uses it |
| RTS (phase 4) | Implemented: `tessera-rl-rts`, with fog of war, multi-head actions and scripted opponents |
| GPU paths, market simulator, card games, league tooling, viewer | Planned |

Known limits of 0.2.0:

- The phase 2 exit gate (the AlphaZero-style Connect Four example against the
  scripted bot, P2-G1) has **not been run** (#154).
- No throughput figure is published with this release.
- **Windows is not in CI**, and no Windows wheel is built. CI covers Linux
  x86-64, Linux aarch64 and macOS (aarch64). The Python Win64
  floating-point boundary has been checked natively by hand (#166, #183,
  #185), not on every change. The ONNX evaluator is not available on
  Windows.
- Wheels exist for Linux x86-64, Linux aarch64 and macOS arm64 only.

## Environments

| Id | Crate | Agents | Action |
| --- | --- | --- | --- |
| `tessera/CartPole-v1` | `tessera-rl-classic` | 1 | `Discrete(2)` |
| `tessera/MountainCar-v1` | `tessera-rl-classic` | 1 | `Discrete(3)`, Gymnasium `MountainCar-v0` dynamics |
| `tessera/Acrobot-v1` | `tessera-rl-classic` | 1 | `Discrete(3)`, Gymnasium `Acrobot-v1` dynamics |
| `tessera/Pendulum-v1` | `tessera-rl-classic` | 1 | `Box(f32, [1])`, Gymnasium `Pendulum-v1` dynamics |
| `tessera/Tag-v1` | `tessera-rl-classic` | 2 to 8 | `Discrete(5)`, masked, simultaneous |
| `tessera/Grid-v1` | `tessera-rl-classic` | 1 | `Discrete(4)` |
| `tessera/GridKeyDoor-v1` | `tessera-rl-classic` | 1 to 4 | `Discrete(5)`, masked, simultaneous, generated worlds |
| `tessera/Snake-v1` | `tessera-rl-classic` | 1 to 2 | `Discrete(4)`, masked, simultaneous, pixel or state observation |
| `tessera/PaddleDuel-v1` | `tessera-rl-classic` | 1 to 2 | `Discrete(3)`, simultaneous, frameskip, pixel or state observation |
| `tessera/BrickWall-v1` | `tessera-rl-classic` | 1 | `Discrete(3)`, frameskip, pixel or state observation |
| `tessera/Connect4-v1` | `tessera-rl-games` | 2 | `Discrete(7)`, masked, sequential |
| `tessera/Leduc-v1` | `tessera-rl-games` | 2 | `Discrete(3)`, masked, sequential, imperfect information |
| `tessera/Othello-v1` | `tessera-rl-games` | 2 | `Discrete(65)`, masked, sequential, forced pass |
| `tessera/Go9-v1` | `tessera-rl-games` | 2 | `Discrete(82)`, masked, sequential, positional superko |
| `tessera/RTS-v1` | `tessera-rl-rts` | 2, or 1 against a scripted opponent | `MultiDiscrete`, multi-head, conditional masks, fog of war |

Leduc hold'em rewards and returns are normalized to `[-1, 1]`:
multiplying them by 13 gives chips.

## Installation

From crates.io:

```toml
[dependencies]
tessera-rl = "0.2.0"
```

From the git tag:

```toml
[dependencies]
tessera-rl = { git = "https://github.com/joaquinbejar/tessera-rl", tag = "v0.2.0" }
```

Depend on the facade `tessera-rl` only: the derive macros resolve their
paths through it. The minimum supported Rust version is 1.95 (edition
2024).

The Python package is not published to `PyPI` in this release. Build it
from a checkout with [maturin](https://www.maturin.rs/), whose `develop`
command installs into the active virtual environment, so create and
activate one first (the `source` line is for Unix shells):

```
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release
```

## Quick start

An environment is one trait implementation on a plain-data state. This
example compiles and runs as a test of this crate.

```rust
use core::num::NonZeroU32;
use tessera_rl::prelude::*;

#[derive(EnvConfig, Debug, Clone)]
pub struct WalkConfig {
    #[config(default = 8, range = 1..=100)]
    length: u32,
}

// Plain data: `#[repr(C)]`, no padding, no pointers.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct WalkState {
    pos: i32,
    steps: u32,
}

#[derive(Observation)]
pub struct WalkObs {
    #[space(box(low = -100, high = 100, shape = [2]))]
    state: ObsLeaf<i32>,
}

pub struct Walk;

impl Environment for Walk {
    type State = WalkState;
    type Config = WalkConfig;
    type Obs = WalkObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &WalkConfig) -> Result<Space, SpaceError> {
        WalkObs::space(cfg)
    }

    fn action_space(_: &WalkConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(_: &WalkConfig, _: &mut StepRng, state: &mut WalkState) {
        *state = WalkState::default();
    }

    fn step(
        cfg: &WalkConfig,
        state: &mut WalkState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let right = actions.discrete(AgentSlot::FIRST, 0) == Some(1);
        state.pos += if right { 1 } else { -1 };
        state.steps += 1;
        out.set_reward(AgentSlot::FIRST, if right { 1.0 } else { 0.0 });
        if state.steps >= cfg.length {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &WalkConfig,
        state: &WalkState,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, WalkObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(WalkObs::STATE) {
            leaf.set(0, state.pos);
            leaf.set(1, i32::try_from(state.steps).unwrap_or(i32::MAX));
        }
    }
}

// Four instances, one seed; every instance has its own random stream.
let options = PoolOptions {
    auto_reset: AutoReset::SameStep,
    ..PoolOptions::new(NonZeroU32::new(4).unwrap(), Seed::new(7))
};
let mut pool = EnvPool::<Walk, NoMasks>::new(WalkConfig::default(), options)?;
pool.reset_all(Reseed::Keep)?;

// Snapshot instance 0, step every instance, then restore it.
let saved = pool.snapshot(0)?;
let actions = [1, 0, 1, 1];
let out = pool.step(ActionInput { discrete: &actions, continuous: &[] })?;
assert_eq!(out.rewards(), &[1.0, 0.0, 1.0, 1.0]);
pool.restore(0, &saved)?;
assert_eq!(pool.state_hash(0)?, saved.hash());

// Fork: instances 1 to 3 become exact copies of instance 0.
pool.fork(0, &[1, 2, 3])?;
assert_eq!(pool.state_hash(3)?, pool.state_hash(0)?);
```

With the `Sync` mode, `with_workers(SyncOptions::new(n))` runs the same
pool on `n` worker threads (pinned where the OS allows), with
bit-identical results.

From Python:

```python
import tessera_rl

env = tessera_rl.make("tessera/CartPole-v1", num_envs=8, seed=0)
```

and, with the `gymnasium` extra, `tessera_rl.gym` provides `TesseraEnv` and
`TesseraVectorEnv` for Gymnasium 1.x; Snake and paddle duel are exposed there
with one agent (`agents=1`), and with two through PettingZoo.

With the `pettingzoo` extra, `tessera_rl.pz.parallel_env` and
`tessera_rl.pz.aec_env` expose the multi-agent environments, each only
through the PettingZoo interfaces it declares.

## Crates

| Crate | Responsibility |
| --- | --- |
| `tessera-rl` | This facade: re-exports the workspace crates and the `prelude` |
| `tessera-rl-core` | The `Environment` trait family, spaces, `StepOut`, errors, canonical encoding |
| `tessera-rl-num` | Philox RNG, `Fx32` and `Fx64` fixed point, frozen math, the floating-point scope guard |
| `tessera-rl-macros` | The `EnvState`, `EnvConfig`, `Observation` and `StateEnum` derives |
| `tessera-rl-engine` | Instance records, output storage, the scheduler, `EnvPool`, snapshot, restore and fork |
| `tessera-rl-search` | `SearchPool` and the batched PUCT MCTS helper |
| `tessera-rl-classic` | CartPole and Tag |
| `tessera-rl-games` | Connect Four |

The Python bindings (`tessera-rl-py`) are built with maturin and are not
a crates.io crate.

## Feature flags

| Feature | Effect |
| --- | --- |
| `debug-checks` | Strict legality checks at the validation boundary (illegal masked actions fail the call), and an invalid agent event, a NaN in state or rewards (outside `#[state(allow_nan)]` fields) or, with `engine::debug::TrackingAllocator` installed, an allocation inside a chunk step poisons the pool (ADR-0055); `engine::DEBUG_CHECKS` reports whether it is on |

## Rules for environment authors

- State is plain data: `#[repr(C)]`, no padding (use `Reserved<N>`),
  no pointers or heap fields; use `Bool8` and `#[derive(StateEnum)]`
  instead of `bool` and plain enums. Variable-size data uses a fixed
  capacity `Pool<T, CAP>`.
- Randomness only through `StepRng` and `ObsRng`.
- `reset`, `step` and `observe` are infallible and must not allocate,
  lock or make system calls; validation happens at the boundary.
- For cross-platform determinism, use the fixed-point types and
  `tessera_rl::num::math` instead of the `std` transcendental functions.
  This repository enforces that with clippy `disallowed_methods`; a
  repository-local `clippy.toml` does not reach a dependent crate, so
  set the same lints in your own crate for its simulation kernels.

## Development

The workspace's local gate is `make pre-push`: formatting, clippy with
`-D warnings`, the Rust and Python tests, documentation and this README
check. Other targets include `make bench`, `make gym-equivalence` and
`make py-test`. This README is generated from the facade's crate
documentation with `cargo-readme` (`make readme`).

## Contribution and Contact

We welcome contributions to this project! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure that the project still builds and all tests pass (`make pre-push`).
4. Commit your changes and push your branch to your forked repository.
5. Submit a pull request to the main repository.

CI does not run for pull requests from forks: it runs on the maintainer's
own machines. A maintainer reviews the contribution and pushes it to a
branch of this repository, where CI runs. See [CONTRIBUTING.md](CONTRIBUTING.md).

If you have any questions, issues, or would like to provide feedback, please feel free to contact the project
maintainer:

### **Contact Information**
- **Author**: Joaquín Béjar García
- **Email**: jb@taunais.com
- **Telegram**: [@joaquin_bejar](https://t.me/joaquin_bejar)
- **Repository**: <https://github.com/joaquinbejar/tessera-rl>
- **Documentation**: <https://docs.rs/tessera-rl>

We appreciate your interest and look forward to your contributions!

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

**License**: MIT OR Apache-2.0

<!-- related-projects:start -->
## Related projects

No other repositories by the same author depend on this project, and it depends on none of them. See all repositories at https://github.com/joaquinbejar?tab=repositories.

<!-- related-projects:end -->
