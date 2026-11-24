// SPDX-License-Identifier: MIT OR Apache-2.0

//! Transforms allocate nothing per step (ADR-0078 item 1.2, SPEC 5.5): a
//! Sync pool with a frame stack and a normalized leaf steps, auto-resets
//! and resets without allocating, on any thread. One test per binary: the
//! counting allocator counts every thread while enabled.

mod common;

#[path = "common/alloc_diag.rs"]
mod alloc_diag;

use core::num::{NonZeroU32, NonZeroUsize};

use alloc_diag::Tagged;
use common::CountConfig;
use tessera_rl_core::space::Space;
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Reseed, SyncOptions, Transforms,
};
use tessera_rl_num::identity::Seed;

#[test]
fn test_transforms_step_and_reset_do_not_allocate_on_any_thread() {
    let options = PoolOptions {
        auto_reset: AutoReset::SameStep,
        chunk_size: NonZeroU32::new(8).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(1))
    };
    let cfg = CountConfig {
        length: 4,
        ..CountConfig::default()
    };
    let pool = EnvPool::<Tagged, Masked>::with_masks(cfg, options).unwrap();
    let leaf = match pool.observation_space().unwrap() {
        Space::Dict(entries) => entries.first().map(|e| e.0.clone()).unwrap(),
        _ => String::new(),
    };
    let mut pool = pool
        .with_transforms(
            Transforms::new()
                .normalize(leaf.clone(), 5.0)
                .frame_stack(leaf, 4),
        )
        .unwrap()
        .with_workers(SyncOptions::new(NonZeroUsize::new(4).unwrap()))
        .unwrap();
    let actions = vec![1i32; 128];
    pool.reset_all(Reseed::Keep).unwrap();
    pool.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap();
    alloc_diag::start();
    for k in 0..20 {
        let _ = std::hint::black_box(pool.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        }));
        if k % 5 == 4 {
            let _ = std::hint::black_box(pool.epoch_barrier());
        }
    }
    let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
    let n = alloc_diag::stop();
    assert_eq!(
        n,
        0,
        "allocations in the measured window:\n{}",
        alloc_diag::report()
    );
}
