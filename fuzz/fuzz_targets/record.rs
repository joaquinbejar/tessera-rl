// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persisted snapshots: see `fuzz/harness/record.rs` (shared with the
//! stable replay of `fuzz/regressions/record/` in tessera-rl-classic).

#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../harness/record.rs"]
mod harness;

fuzz_target!(|data: &[u8]| harness::run(data));
