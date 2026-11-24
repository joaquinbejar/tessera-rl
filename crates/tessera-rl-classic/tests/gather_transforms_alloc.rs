// SPDX-License-Identifier: MIT OR Apache-2.0

//! Transforms on `Set` leaves allocate nothing per step (ADR-0092, ADR-0078
//! item 1.2, SPEC 5.5): Gather-v1 with a masked `Normalize` and a set-wide
//! `FrameStack` on a Sync pool steps, auto-resets (both modes), passes epoch
//! barriers and resets without allocating, on any thread. The counting
//! allocator of `support/counting.rs` attributes allocator calls by thread.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_classic::{Gather, GatherConfig};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, NoMasks, PoolOptions, Reseed, SyncOptions, Transforms,
};
use tessera_rl_num::identity::Seed;

#[path = "support/counting.rs"]
mod counting;
use counting::window;

/// Allocation calls of the counted threads while `f` runs.
fn allocations(f: impl FnOnce()) -> u64 {
    window(f).1.calls
}

fn test_counter_detects_heap_use() {
    assert!(allocations(|| drop(std::hint::black_box(vec![1u8; 16]))) > 0);
}

fn test_gather_set_transforms_step_reset_and_barrier_do_not_allocate() {
    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
        let options = PoolOptions {
            auto_reset,
            chunk_size: NonZeroU32::new(8).unwrap(),
            ..PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(1))
        };
        // Short episodes, so auto-reset runs in the window.
        let cfg = GatherConfig {
            max_steps: 5,
            ..GatherConfig::default()
        };
        let mut pool = EnvPool::<Gather, NoMasks>::new(cfg, options)
            .unwrap()
            .with_transforms(
                Transforms::new()
                    .normalize("items", 5.0)
                    .frame_stack("items", 4),
            )
            .unwrap()
            .with_workers(SyncOptions::new(NonZeroUsize::new(4).unwrap()))
            .unwrap();
        let actions: Vec<i32> = (0..64).map(|i| i % 4).collect();
        let input = || ActionInput {
            discrete: &actions,
            continuous: &[],
        };
        // Warm-up: the first reset and step.
        pool.reset_all(Reseed::Keep).unwrap();
        pool.step(input()).unwrap();
        let mut ended = false;
        let n = allocations(|| {
            for k in 0..30 {
                let out = pool.step(input()).unwrap();
                ended |= out.episode_truncated().iter().any(|f| f.get())
                    || out.episode_terminated().iter().any(|f| f.get());
                if k % 10 == 9 {
                    let _ = std::hint::black_box(pool.epoch_barrier());
                }
            }
            let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
        });
        assert_eq!(n, 0, "{auto_reset:?}");
        assert!(ended, "no episode ended, so auto-reset was not exercised");
    }
}

/// The scenarios this binary runs.
const SCENARIOS: usize = 2;

#[test]
fn test_gather_transforms_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_counter_detects_heap_use",
                test_counter_detects_heap_use,
            ),
            (
                "test_gather_set_transforms_step_reset_and_barrier_do_not_allocate",
                test_gather_set_transforms_step_reset_and_barrier_do_not_allocate,
            ),
        ],
    );
}
