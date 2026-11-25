// SPDX-License-Identifier: MIT OR Apache-2.0

//! A Sync pool with the privileged state buffer (ADR-0079 item 7) never
//! allocates after construction, on any thread: the privileged pass after
//! each step, and reading the rows, included.
//!
//! The counting allocator of `common/alloc_diag.rs` counts on every thread
//! while enabled, so this binary holds a single test.

mod common;

#[path = "common/alloc_diag.rs"]
mod alloc_diag;

use core::num::{NonZeroU32, NonZeroUsize};

use alloc_diag::Tagged;
use common::{CountConfig, Counts};
use tessera_rl_core::io::ObsWriter;
use tessera_rl_core::space::Space;
use tessera_rl_core::{ObsLeaf, Observation, PrivilegedState, SpaceError};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_macros::Observation;
use tessera_rl_num::identity::Seed;

/// Both counters and the step count.
#[derive(Observation)]
struct TaggedState {
    #[space(box(low = -1_000_000, high = 1_000_000, shape = [3]))]
    all: ObsLeaf<i32>,
}

impl PrivilegedState for Tagged {
    fn state_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        TaggedState::space(cfg)
    }

    fn write_state(_: &CountConfig, state: &Counts, out: &mut ObsWriter<'_, '_, ()>) {
        if let Some(mut leaf) = out.leaf_mut(TaggedState::ALL) {
            leaf.set(0, state.value[0]);
            leaf.set(1, state.value[1]);
            leaf.set(2, i32::try_from(state.steps).unwrap_or(i32::MAX));
        }
    }
}

#[test]
fn test_sync_step_with_privileged_state_does_not_allocate_on_any_thread() {
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
        .with_privileged_state()
        .unwrap()
        .with_workers(SyncOptions::new(NonZeroUsize::new(4).unwrap()))
        .unwrap();
    let actions = vec![1; 128];
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
        let _ = std::hint::black_box(pool.outputs().ok().and_then(|o| o.privileged_state(0)));
    }
    let n = alloc_diag::stop();
    assert_eq!(
        n,
        0,
        "allocations in the measured window:\n{}",
        alloc_diag::report()
    );
    assert!(!pool.is_poisoned());
    let rows = pool.outputs().unwrap().privileged_state(0).unwrap();
    assert!(rows.iter().any(|&b| b != 0), "the rows were written");
}
