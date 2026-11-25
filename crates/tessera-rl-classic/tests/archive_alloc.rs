// SPDX-License-Identifier: MIT OR Apache-2.0

//! A step that archive-resets never allocates (ADR-0075 item 6, SPEC 5.5):
//! the archive and its weight table exist before the step, and the reset
//! copies an entry. Both auto-reset modes, Inline and with workers, and
//! `reset_from`. The counting allocator of `support/counting.rs`
//! attributes calls by thread; this binary holds one measuring test.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_classic::{CartPole, CartPoleConfig};
use tessera_rl_engine::{
    ActionInput, Archive, AutoReset, EnvPool, NoMasks, PoolOptions, Reseed, ResetSource,
    SyncOptions,
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

fn archived_pool(auto_reset: AutoReset, workers: usize) -> EnvPool<CartPole, NoMasks> {
    let options = PoolOptions {
        auto_reset,
        ..PoolOptions::new(NonZeroU32::new(8).unwrap(), Seed::new(3))
    };
    // Entries: snapshots of a pool after a few steps.
    let mut src =
        EnvPool::<CartPole, NoMasks>::new(CartPoleConfig::default(), options.clone()).unwrap();
    src.reset_all(Reseed::Keep).unwrap();
    let a = [1i32; 8];
    for _ in 0..3 {
        src.step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
    }
    let records: Vec<_> = (0..4).map(|i| src.snapshot(i).unwrap()).collect();
    let archive = Archive::new(&CartPoleConfig::default(), &records).unwrap();
    let mut pool = EnvPool::<CartPole, NoMasks>::new(CartPoleConfig::default(), options).unwrap();
    if workers > 0 {
        pool = pool
            .with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
            .unwrap();
    }
    pool.set_reset_source(ResetSource::Archive {
        archive,
        weights: Some(vec![1, 2, 0, 3]),
    })
    .unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    pool
}

fn test_archive_auto_resets_do_not_allocate() {
    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
        for workers in [0, 2] {
            let mut pool = archived_pool(auto_reset, workers);
            // Pushing one way ends every episode within a few dozen steps.
            let actions = [1i32; 8];
            let mut ended = false;
            let n = allocations(|| {
                for _ in 0..300 {
                    let out = pool
                        .step(ActionInput {
                            discrete: &actions,
                            continuous: &[],
                        })
                        .unwrap();
                    ended |= out.episode_terminated().iter().any(|f| f.get());
                }
                let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
            });
            assert_eq!(n, 0, "{auto_reset:?} workers={workers}");
            assert!(ended, "no episode ended, so no archive reset ran");
        }
    }
}

fn test_reset_from_does_not_allocate() {
    let mut pool = archived_pool(AutoReset::Disabled, 0);
    let ids = [0u32, 3, 5];
    let entries = [3u32, 0, 1];
    let n = allocations(|| {
        for _ in 0..100 {
            let _ = std::hint::black_box(pool.reset_from(&ids, &entries));
        }
    });
    assert_eq!(n, 0);
}

/// The scenarios this binary runs.
const SCENARIOS: usize = 3;

#[test]
fn test_archive_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_counter_detects_heap_use",
                test_counter_detects_heap_use,
            ),
            (
                "test_archive_auto_resets_do_not_allocate",
                test_archive_auto_resets_do_not_allocate,
            ),
            (
                "test_reset_from_does_not_allocate",
                test_reset_from_does_not_allocate,
            ),
        ],
    );
}
