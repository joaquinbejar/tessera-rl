# Writing an environment

`cargo tessera` scaffolds a new environment crate with its own workspace and
Python wheel, then checks and benchmarks it. The scaffold is a *host
environment* (a counter game implementing `HostEnvironment`), which you
replace with your own rules.

<!-- snippet: writing_an_environment.sh -->

What each command does:

- `new` writes the crate: the environment, `tests/check.rs`,
  `tests/test_adapters.py`, `benches/step.rs`, `pyproject.toml` and the
  Python package. `--snapshots` adds snapshot support; `--agents` and
  `--turns` set the agent slots and the turn structure.
- `check --bless` records `tests/golden.json`, the golden trajectory hashes.
  Commit it, so a later change that alters a trajectory fails.
- `check --python` runs the Rust checks (determinism Inline and with
  workers, continuation, spaces, invariants, snapshots, goldens), installs the
  wheel and runs the Python adapter tests.
- `bench --quick` prints steps per second at 1, 1024 and 8192 instances,
  Inline and with 4 workers.

The crate's own README (`crates/cargo-tessera/README.md`) lists every option.
