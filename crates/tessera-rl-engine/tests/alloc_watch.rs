// SPDX-License-Identifier: MIT OR Apache-2.0

//! A served live-watch boundary never allocates (ADR-0083 item 7): with a
//! request pending at every step, Sync mode serves one record per step into
//! the buffer allocated at attach, and the handle's `request` and `take`
//! copy into a caller buffer. The counting allocator of
//! `common/alloc_diag.rs` counts on every thread while enabled, so this
//! binary holds a single test.

#![cfg(not(loom))]

mod common;

#[path = "common/alloc_diag.rs"]
mod alloc_diag;

use core::num::{NonZeroU32, NonZeroUsize};

use alloc_diag::Tagged;
use common::CountConfig;
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;

#[test]
fn test_watch_served_boundary_does_not_allocate_on_any_thread() {
    let options = PoolOptions {
        auto_reset: AutoReset::SameStep,
        chunk_size: NonZeroU32::new(8).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(1))
    };
    let cfg = CountConfig {
        length: 4,
        ..CountConfig::default()
    };
    let mut pool = EnvPool::<Tagged, Masked>::with_masks(cfg, options)
        .unwrap()
        .with_workers(SyncOptions::new(NonZeroUsize::new(4).unwrap()))
        .unwrap();
    let mut handle = pool.attach_watch().unwrap();
    let mut out = vec![0_u8; handle.record_len()];
    let actions = vec![1i32; 128];
    // Warm up: the first epoch may initialize thread-local runtime state.
    pool.reset_all(Reseed::Keep).unwrap();
    pool.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap();
    alloc_diag::start();
    let mut served = 0_u32;
    for k in 0..20_u32 {
        let _ = handle.request(k % 64);
        let _ = std::hint::black_box(pool.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        }));
        if let Ok(Some(_)) = handle.take(&mut out) {
            served += 1;
        }
    }
    let n = alloc_diag::stop();
    assert_eq!(served, 20, "every step served the pending request");
    assert_eq!(
        n,
        0,
        "allocations in the measured window:\n{}",
        alloc_diag::report()
    );
}
