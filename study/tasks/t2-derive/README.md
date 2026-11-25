# study-t2-derive: `study-t2-derive/StudyT2Derive-v1`

A [Tessera](https://github.com/joaquinbejar/tessera-rl) environment on the
`Environment` trait, scaffolded by `cargo tessera new --kind env`
0.3.0. It starts as a counter game (1 agent(s),
simultaneous turns); replace it with your environment in `src/lib.rs`.

**Rust only.** A separately compiled `Environment` cannot be loaded into
the prebuilt `tessera-rl` wheel (ADR-0032), so this crate has no Python
package and `cargo tessera check --python` refuses it. It runs through
`tessera_rl_engine::EnvPool`: every engine mode, snapshot, restore and
fork, and `tessera-rl-search`. For an environment usable from Python,
scaffold a host environment instead: `cargo tessera new <name>` (the
default `--kind host`, ADR-0062).

## Layout

| File | Role |
| --- | --- |
| `Cargo.toml` | Its own package and workspace, with the Tessera workspace's lints. |
| `clippy.toml` | The banned float methods of ADR-0008 and the test allowances. |
| `src/lib.rs` | The state, config, observation and `Environment`, `ActionMask`, `Heuristic`, `Validate` and `Persist` implementations. |
| `tests/check.rs` | The checks and the golden hashes (`tests/golden.json`). |
| `benches/step.rs` | The criterion bench. |

The transition kernel in `src/lib.rs` uses `saturating_*` arithmetic only
where the game's rules bound the value, and a comment at each call states
the bound (the counter never exceeds the target, the step count never
exceeds the episode length). Keep that rule when you replace the game: a
saturating call needs a documented bound, never a silent clamp, and
checked arithmetic belongs outside the kernel.

## Dependencies

The Tessera crates come from a local checkout, by path (`--tessera-path`).
Before publishing, depend on a released version instead.

## Checks

```sh
cargo tessera check --bless    # record tests/golden.json (commit it)
cargo tessera check            # determinism, continuation, snapshots, spaces, validation, goldens
cargo clippy --all-targets -- -D warnings
cargo tessera bench --quick    # steps per second on this machine
```

- Determinism: two Inline pools, and pools with 4 workers at chunk size 1
  and at the default, give equal hashes per call.
- Continuation: every instance, exported mid-run and imported into a fresh
  pool with another seed, continues exactly as the uninterrupted run.
- The goldens hash every record and output of every call of a seeded run.
  When a change to the outputs is intended, bump the version in the id and
  `ENV_VERSION`, then re-bless.

## Use

```rust
use core::num::NonZeroU32;
use tessera_rl_engine::{EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

let options = PoolOptions::new(NonZeroU32::new(8).unwrap(), Seed::new(1));
let mut pool = EnvPool::<study_t2_derive::Counter, Masked>::with_masks(Default::default(), options)?;
pool.reset_all(Reseed::Keep)?;
```

## Moving it into the Tessera repository

The template does not edit the repository: a first-party environment also
needs `register!`, a Python maker, the registry and adapter entries, and
goldens in the repository's harness. Copy `src/lib.rs` into a first-party
crate and follow an existing environment (for example
`crates/tessera-rl-classic/src/grid`).

## Choosing an id

The namespace `tessera/` is reserved for first-party environments.
