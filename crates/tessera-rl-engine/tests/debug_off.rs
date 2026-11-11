// SPDX-License-Identifier: MIT OR Apache-2.0

//! ADR-0055, default build: without `debug-checks` nothing is detected. A
//! NaN in state and an allocation in `step` leave the pool working and
//! unpoisoned, and the debug errors are never constructed.

#![cfg(all(not(loom), not(feature = "debug-checks")))]

#[path = "debug_env/mod.rs"]
mod debug_env;

use core::num::{NonZeroU32, NonZeroUsize};

use debug_env::{ProbeConfig, ProbeEnv};
use tessera_rl_engine::{ActionInput, EnvPool, PoolOptions, Reseed, SyncOptions};
use tessera_rl_num::identity::Seed;

#[test]
fn test_debug_off_nan_state_and_allocation_are_not_detected() {
    const { assert!(!tessera_rl_engine::DEBUG_CHECKS) };
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(2).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(6).unwrap(), Seed::new(9))
    };
    let cfg = ProbeConfig {
        nan_state_at: 1,
        nan_on_reset: true,
        alloc_always: true,
        ..ProbeConfig::default()
    };
    let mut p = EnvPool::<ProbeEnv>::new(cfg, options)
        .unwrap()
        .with_workers(SyncOptions::new(NonZeroUsize::new(2).unwrap()))
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    for _ in 0..4 {
        p.step(ActionInput {
            discrete: &[0; 6],
            continuous: &[],
        })
        .unwrap();
    }
    assert!(!p.is_poisoned());
}
