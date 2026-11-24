// SPDX-License-Identifier: MIT OR Apache-2.0

//! A Sync pool with a heuristic slot (ADR-0079 item 3.4) never allocates
//! after construction, on any thread: the heuristic pass, its validation
//! and the copy to `heuristic_actions` included.
//!
//! The counting allocator of `common/alloc_diag.rs` counts on every thread
//! while enabled, so this binary holds a single test.

mod common;

#[path = "common/alloc_diag.rs"]
mod alloc_diag;

use core::num::{NonZeroU32, NonZeroUsize};

use alloc_diag::Tagged;
use common::{CountConfig, Counts};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::StepRng;

/// A heuristic that draws from its RNG: action 1 or 2, legal for agent 1.
impl Heuristic for Tagged {
    fn act(
        _: &CountConfig,
        _: &Counts,
        _: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let draw = rng.below_u32(NonZeroU32::new(2).unwrap_or(NonZeroU32::MIN));
        out.set_discrete(0, 1 + i32::try_from(draw).unwrap_or(0));
    }
}

#[test]
fn test_sync_step_with_a_heuristic_slot_does_not_allocate_on_any_thread() {
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
        .with_heuristic_slots(&[1])
        .unwrap()
        .with_workers(SyncOptions::new(NonZeroUsize::new(4).unwrap()))
        .unwrap();
    // Slot 1's rows are garbage: never read.
    let actions: Vec<i32> = (0..128).map(|k| if k % 2 == 0 { 1 } else { -9 }).collect();
    // Warm up: the first epoch may initialize thread-local runtime state.
    pool.reset_all(Reseed::Keep).unwrap();
    pool.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap();
    alloc_diag::start();
    for _ in 0..20 {
        let _ = std::hint::black_box(pool.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        }));
    }
    let _ = std::hint::black_box(pool.heuristic_actions());
    let n = alloc_diag::stop();
    assert_eq!(
        n,
        0,
        "allocations in the measured window:\n{}",
        alloc_diag::report()
    );
    assert!(!pool.is_poisoned());
}
