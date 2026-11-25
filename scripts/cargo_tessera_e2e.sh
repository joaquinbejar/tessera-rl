#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# The end-to-end test of cargo tessera (#351, #628), `make cargo-tessera-e2e`:
# it builds the tool, then
# - scaffolds a host environment crate with path dependencies on this
#   checkout and runs `check --python --bless`, `check --python` and
#   `bench --quick` in it;
# - scaffolds a `--kind env` crate the same way and runs `check --bless`,
#   `check`, `cargo clippy --all-targets -- -D warnings` (the scaffold
#   carries the workspace's lints and clippy.toml) and `bench --quick`;
# and prints the wall time of each phase.
#
# Nothing is resolved from the network (#351 review r1, point 1):
# - the scaffold's Cargo.lock starts as a copy of the repository's lock;
#   `cargo metadata --offline` prunes it to the scaffold's graph, keeping
#   every locked version, and the script fails if the pruned lock holds any
#   package or version the repository's lock does not;
# - every later cargo call is `--locked --offline` (the wheel build too,
#   through MATURIN_PEP517_ARGS), and uv runs `--locked` / `--offline`.
# The one network step is `cargo fetch --locked` of the repository's own
# lock, which downloads exactly the locked crates (criterion's included)
# when the runner's cache lacks them and resolves nothing.
#
# The Python side reuses the repository's uv environment (.venv, group
# `test`), installs the scaffold's wheel into it with `--uv`, and restores
# the default environment on exit, as `make py-test-orderbook` does.

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
# The scaffold lives outside the repository, where rust-toolchain.toml does
# not apply: pin the repository's toolchain for every cargo call here.
RUSTUP_TOOLCHAIN="$(sed -n 's/^channel = "\(.*\)"$/\1/p' "$root/rust-toolchain.toml")"
if [ -z "$RUSTUP_TOOLCHAIN" ]; then
    echo "cargo-tessera-e2e: no channel in rust-toolchain.toml" >&2
    exit 1
fi
export RUSTUP_TOOLCHAIN
target="${CARGO_TARGET_DIR:-$root/target}"
work="$(mktemp -d)"
crate="e2e-env"
env_crate="e2e-envtrait"
start=$(date +%s)
last=$start

cleanup() {
    status=$?
    (cd "$root" && uv sync --locked --python 3.11) || status=1
    rm -rf "$work"
    exit "$status"
}
trap cleanup EXIT

phase() {
    local now
    now=$(date +%s)
    echo "cargo-tessera-e2e: phase $1: $((now - last)) s"
    last=$now
}

cargo fetch --locked --manifest-path "$root/Cargo.toml"
phase fetch

cargo build --locked --offline --manifest-path "$root/Cargo.toml" -p cargo-tessera
export PATH="$target/debug:$PATH"
phase build-tool

cd "$work"
cargo tessera new "$crate" --tessera-path "$root" --snapshots
phase new

cp "$root/Cargo.lock" "$crate/Cargo.lock"
cargo metadata --offline --format-version 1 --manifest-path "$crate/Cargo.toml" >/dev/null
"$root/scripts/lock_subset.sh" "$root/Cargo.lock" "$crate/Cargo.lock" "$crate"
phase lock

(cd "$root" && uv sync --locked --python 3.11 --group test)
export VIRTUAL_ENV="$root/.venv"
phase uv-sync

cd "$work/$crate"
cargo tessera check --python --uv --bless --locked --offline
test -s tests/golden.json
phase check-bless
cargo tessera check --python --uv --locked --offline
phase check
cargo tessera bench --quick --locked --offline
phase bench-quick

cd "$work"
cargo tessera new "$env_crate" --kind env --tessera-path "$root" --agents 2 --turns sequential
test ! -e "$env_crate/pyproject.toml"
phase env-new
# The same lock-subset check as the host pass above, on the env crate.
cp "$root/Cargo.lock" "$env_crate/Cargo.lock"
cargo metadata --offline --format-version 1 --manifest-path "$env_crate/Cargo.toml" >/dev/null
"$root/scripts/lock_subset.sh" "$root/Cargo.lock" "$env_crate/Cargo.lock" "$env_crate"
phase env-lock
cd "$work/$env_crate"
if cargo tessera check --python --locked --offline 2>/dev/null; then
    echo "cargo tessera check --python must refuse a --kind env crate" >&2
    exit 1
fi
cargo tessera check --bless --locked --offline
test -s tests/golden.json
phase env-check-bless
cargo tessera check --locked --offline
phase env-check
cargo clippy --all-targets --locked --offline -- -D warnings
phase env-clippy
cargo tessera bench --quick --locked --offline
phase env-bench-quick
# A seeded nondeterminism (a static reset counter, #645): the check must
# fail and name the comparison kind and the first diverging call.
cd "$work"
rm -rf "$env_crate-seeded"
cp -R "$env_crate" "$env_crate-seeded"
cd "$work/$env_crate-seeded"
grep -q 'let counter = (rng.next_u32() % CHOICES).min(cfg.target);' src/lib.rs
sed -i.orig 's|let counter = (rng.next_u32() % CHOICES).min(cfg.target);|let counter = ((rng.next_u32() ^ RESETS.fetch_add(1, core::sync::atomic::Ordering::Relaxed)) % CHOICES).min(cfg.target);|' src/lib.rs
rm -f src/lib.rs.orig
cat >> src/lib.rs <<'RS'

/// Counts resets (the seeded defect).
static RESETS: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
RS
if cargo tessera check --locked --offline > seeded.out 2>&1; then
    echo "the check must fail on a seeded static reset counter" >&2
    exit 1
fi
grep -q 'Inline vs Inline (same process)' seeded.out
grep -Eq 'first divergence at call [0-9]+, (instance [0-9]+|the outputs only)' seeded.out
phase env-seeded-report

echo "cargo-tessera-e2e: total: $(($(date +%s) - start)) s"
