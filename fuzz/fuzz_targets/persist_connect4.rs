// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persisted files of Connect4: see `fuzz/harness/persist.rs`.

#![no_main]

use core::num::NonZeroU32;

use libfuzzer_sys::fuzz_target;
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_engine::{EnvPool, Reseed, Masked, PoolOptions};
use tessera_rl_num::identity::Seed;

#[path = "../harness/persist.rs"]
mod harness;

fuzz_target!(|data: &[u8]| {
    let options = PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(3));
    let mut p = EnvPool::<Connect4, Masked>::with_masks(Connect4Config {}, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    harness::run(p, data, None);
});
