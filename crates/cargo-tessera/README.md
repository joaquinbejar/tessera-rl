<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# cargo-tessera

`cargo tessera` scaffolds, checks and benches a Tessera **host environment** crate (SPEC 10.2, ADR-0032, ADR-0062). Part of [Tessera](https://github.com/joaquinbejar/tessera-rl). Licensed under MIT OR Apache-2.0.

```sh
cargo install cargo-tessera
cargo tessera new my-env --id acme/MyEnv-v1 --agents 2 --turns sequential --snapshots
cd my-env
cargo tessera check --bless     # record tests/golden.json, then commit it
cargo tessera check --python    # Rust checks, goldens and the Python adapter tests
cargo tessera bench --quick
```

- `new` writes a crate with its own workspace and wheel: a counter game implementing `HostEnvironment` (and `HostSnapshot` with `--snapshots`), `tests/check.rs`, `tests/test_adapters.py`, `benches/step.rs`, `pyproject.toml` and the Python package. It refuses an existing directory. The `tessera/` id namespace is reserved.
- The scaffold depends on the Tessera crates at exactly this tool's version (`=X.Y.Z`): the host API is experimental until 1.0 and may change in a minor release. `--tessera-path <checkout>` uses path dependencies instead.
- `check` runs `cargo test --test check`: `check_determinism` (Inline and with workers), `check_continuation`, `check_spaces`, `check_invariants`, the snapshot checks with `--snapshots`, and the golden hashes. `--python` installs the wheel with `maturin develop` (or `uv pip install` with `--uv`), both taken from the PATH, and runs pytest. `--locked` and `--offline` reach every cargo call.
- `bench` runs the criterion bench and prints steps per second at 1, 1024 and 8192 instances, Inline and with 4 workers, and snapshot costs.

- `new --kind env` (#628) writes a Rust-only crate on the `Environment` trait instead: the same counter game with plain-data state, `ActionMask`, `Heuristic`, `Validate` and `Persist`, a `tests/check.rs` (determinism with Inline and 4 workers at chunk sizes 1 and the default, continuation by import into a fresh pool, a snapshot round trip, spaces, validation and goldens), `benches/step.rs`, and the Tessera workspace's lints with its `clippy.toml` (ADR-0008). It has no Python package: a separately compiled `Environment` cannot be loaded into the prebuilt wheel (ADR-0032), so `check --python` refuses it and `--snapshots` is host only. It runs through `EnvPool`, every engine mode, snapshot, restore and fork, and search.

Neither kind edits the Tessera repository: a first-party environment also needs `register!`, the Python maker, the registry and adapter entries, which a template cannot add safely.
