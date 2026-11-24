// SPDX-License-Identifier: MIT OR Apache-2.0

//! Registers the counter's host factory from a separate rlib (ADR-0065
//! item 3). A binary references [`linked`] so the linker keeps this crate's
//! `inventory` submission (ADR-0054).

tessera_rl_engine::register_host_factory!(tessera_rl_test_host_env::Counter, snapshots);

/// Referenced by the binaries that use this crate's factory.
#[must_use]
pub const fn linked() -> &'static str {
    tessera_rl_test_host_env::COUNTER_ID
}
