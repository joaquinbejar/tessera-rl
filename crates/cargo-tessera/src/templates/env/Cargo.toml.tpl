# A Tessera environment crate on the `Environment` trait (SPEC 10),
# scaffolded by `cargo tessera new --kind env`. Its own package and
# workspace, Rust only: no Python package (ADR-0032).

[package]
name = "{{crate_name}}"
description = "A Tessera environment: {{env_id}}"
version = "0.1.0"
edition = "2024"
rust-version = "1.95"
publish = false

# Read by `cargo tessera check`.
[package.metadata.tessera]
kind = "env"

[lib]
name = "{{py_package}}"

{{#path_deps}}
# The Tessera crates of a local checkout, by path.
{{/path_deps}}
{{^path_deps}}
# The Tessera crates at exactly the version of the cargo-tessera that
# scaffolded this crate: the API is not stable before 1.0 and may change in
# a minor release. Raise all of them together.
{{/path_deps}}
[dependencies]
tessera-rl-core = {{dep_core}}
tessera-rl-macros = {{dep_macros}}
tessera-rl-num = {{dep_num}}

[dev-dependencies]
# EnvPool: the checks and the bench.
tessera-rl-engine = {{dep_engine_env}}
criterion = { version = "0.8", default-features = false, features = ["cargo_bench_support"] }

[[test]]
name = "check"

[[bench]]
name = "step"
harness = false

# Not part of any other workspace.
[workspace]

# The Tessera workspace's lints, so `cargo clippy -- -D warnings` holds this
# crate to the same rules (with clippy.toml, ADR-0008).
[lints.rust]
missing_docs = "deny"
unsafe_code = "forbid"

[lints.clippy]
pedantic = { level = "warn", priority = -1 }
allow_attributes = "deny"
allow_attributes_without_reason = "deny"
dbg_macro = "deny"
expect_used = "deny"
indexing_slicing = "deny"
panic = "deny"
print_stderr = "deny"
print_stdout = "deny"
todo = "deny"
unimplemented = "deny"
unreachable = "deny"
unwrap_used = "deny"
