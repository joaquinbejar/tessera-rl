# A Tessera host environment crate (ADR-0062), scaffolded by
# `cargo tessera new`. Its own package, workspace and wheel.

[package]
name = "{{crate_name}}"
description = "A Tessera host environment: {{env_id}}"
version = "0.1.0"
edition = "2024"
rust-version = "1.95"
publish = false

[lib]
name = "{{py_package}}"
# cdylib: the Python extension module; rlib: the Rust tests and bench.
crate-type = ["cdylib", "rlib"]

[features]
# The Python extension module. maturin enables it (pyproject.toml); the
# Rust tests and the bench build without it, so they never link Python.
python = ["dep:tessera-rl-pyffi", "dep:pyo3"]

{{#path_deps}}
# The Tessera crates of a local checkout, by path.
{{/path_deps}}
{{^path_deps}}
# The Tessera crates at exactly the version of the cargo-tessera that
# scaffolded this crate: the host Rust API is excluded from the 1.x
# stability promise (ADR-0094 item 3) and may change in a minor release.
# Raise all of them together.
{{/path_deps}}
[dependencies]
tessera-rl-core = {{dep_core}}
tessera-rl-macros = {{dep_macros}}
tessera-rl-pyffi = {{dep_pyffi}}
pyo3 = { version = "0.30", default-features = false, features = ["abi3-py311", "extension-module"], optional = true }

[dev-dependencies]
# check_determinism, check_continuation, check_spaces, check_invariants and
# check_snapshot_contract (feature `testing`).
tessera-rl-engine = {{dep_engine}}
tessera-rl-num = {{dep_num}}
criterion = { version = "0.8", default-features = false, features = ["cargo_bench_support"] }

[[test]]
name = "check"

[[bench]]
name = "step"
harness = false

# Not part of any other workspace.
[workspace]

[lints.rust]
unsafe_code = "forbid"

[lints.clippy]
all = { level = "warn", priority = -1 }
pedantic = { level = "warn", priority = -1 }
