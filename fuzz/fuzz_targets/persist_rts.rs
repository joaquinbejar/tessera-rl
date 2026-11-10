// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persisted files of the RTS: see `fuzz/harness/persist.rs`.

#![no_main]

use core::num::NonZeroU32;

use libfuzzer_sys::fuzz_target;
use tessera_rl_engine::{EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;
use tessera_rl_rts::{Rts, RtsConfig};

#[path = "../harness/persist.rs"]
mod harness;

fuzz_target!(|data: &[u8]| {
    let options = PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(3));
    let cfg = RtsConfig {
        size: 8,
        max_units: 4,
        ..RtsConfig::default()
    };
    let mut p = EnvPool::<Rts, Masked>::with_masks(cfg, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    harness::run(p, data, None);
});
