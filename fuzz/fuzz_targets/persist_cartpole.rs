// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persisted files of CartPole: see `fuzz/harness/persist.rs`.

#![no_main]

use core::num::NonZeroU32;

use libfuzzer_sys::fuzz_target;
use tessera_rl_classic::{CartPole, CartPoleConfig};
use tessera_rl_engine::{EnvPool, Reseed, NoMasks, PoolOptions};
use tessera_rl_num::identity::Seed;

#[path = "../harness/persist.rs"]
mod harness;

fuzz_target!(|data: &[u8]| {
    let options = PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(3));
    let mut p = EnvPool::<CartPole, NoMasks>::new(CartPoleConfig::default(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    harness::run(p, data, None);
});
