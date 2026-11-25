# Writing an `Environment`-trait environment

`cargo tessera new --kind env` scaffolds a Rust crate on the `Environment`
trait (SPEC 10). Use this path when you want the engine's own modes,
snapshot, restore and fork, and search. It is Rust only: a separately
compiled `Environment` cannot be loaded into the prebuilt `tessera-rl` wheel
(ADR-0032). For Python access, follow
[writing an environment](writing-an-environment.md), which scaffolds a host
environment.

```sh
cargo tessera new my-env --kind env --agents 2 --turns sequential
cd my-env
cargo tessera check --bless     # record tests/golden.json, then commit it
cargo tessera check             # determinism, continuation, snapshots, spaces
cargo clippy --all-targets -- -D warnings
```

Inside a Tessera checkout, add `--tessera-path <checkout>` to depend on its
crates by path instead of the released version.

## What the scaffold contains

- `src/lib.rs`: a counter game. Replace it with your rules.
- `tests/check.rs`: the checks `cargo tessera check` runs. It names your
  types (`Counter`, `CounterConfig`, `CounterState`) and the observation
  size, so rename and resize them along with your environment.
- `benches/step.rs`: the bench `cargo tessera bench` runs.
- `Cargo.toml` and `clippy.toml`: the Tessera workspace's lints, including
  ADR-0008's banned float methods, so `cargo clippy -- -D warnings` holds the
  crate to the engine's rules.

## The four derives

- `#[derive(EnvConfig)]` on the configuration: every field has a
  `#[config(default = ..., range = ...)]`, and the engine checks the range at
  construction.
- `#[derive(EnvState)]` on the per-instance state. The state is plain data
  (ADR-0029): `#[repr(C)]`, fixed-width fields, no pointers or heap fields,
  and no padding. Three rules catch most mistakes:
  - use `Bool8`, not `bool`;
  - make every padding byte explicit with a `Reserved<N>` field (the error
    names the field after which the padding falls and the `N` to add);
  - give enums `#[derive(StateEnum)]` and an integer `#[repr]`.
- `#[derive(StateEnum)]` on each enum stored in the state, with a
  `#[default]` variant.
- `#[derive(Observation)]` on the observation, one `ObsLeaf` per leaf with a
  `#[space(...)]` attribute giving its bounds and shape.

When a derive refuses a type, the compiler error quotes the rule and its ADR.
Read the first error first: later errors usually follow from it.

## The kernel rules

`reset`, `step` and `observe` are infallible and allocate nothing; the engine
validates actions before the step. Randomness comes only from the `StepRng`
and `ObsRng` the engine passes, which are keyed per instance (ADR-0007). The
state record and those two generators are the only state a kernel may read:
no `static`, no `thread_local!`, no `HashMap` iteration, no wall-clock time.

## When the determinism check fails

`cargo tessera check` compares several runs call by call. A failure names the
comparison that diverged, the first diverging call (numbered as in
`tests/golden.json`, where call 0 is the first reset) and the first instance
whose record differs there, with a hint at the usual cause:

| Comparison | Usual cause |
| --- | --- |
| `Inline vs Inline (same process)` | state that outlives a pool: a `static`, a `thread_local!` or `HashMap` iteration order |
| `Inline vs 4 workers at chunk size N` | state shared across instances, or keyed by storage slot or thread |
| `import at call N vs the uninterrupted run` | state the step reads that the record does not hold |
| `restore in place vs the uninterrupted run` | the same, seen through a snapshot |

To find the source, start from the named call and instance: look for
anything `reset`, `step` or `observe` reads besides the state record, the
configuration, the actions and the two generators. The engine's
`Recorder` and `Replay` (in `tessera_rl_engine`) can replay one instance
call by call when the code path is not obvious.

`check --bless` only rewrites `tests/golden.json`. When a trajectory change
is intended, bump the id's version and `ENV_VERSION` before blessing.
