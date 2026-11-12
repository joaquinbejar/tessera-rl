// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Inline hot path never allocates after construction (SPEC 5.5,
//! ADR-0004).
//!
//! The counting allocator of `support/counting.rs` attributes allocator
//! calls by thread (issue #43). This binary holds a single measuring test,
//! which runs each scenario (the former tests, by name) with its own pool
//! and windows, and reports every failing scenario together.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

mod common;

#[path = "support/counting.rs"]
mod counting;

use core::num::NonZeroU32;

use common::{CountConfig, Counter};
use counting::window;
use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

/// Allocation calls of the counted threads while `f` runs.
fn allocations(f: impl FnOnce()) -> u64 {
    window(f).1.calls
}

fn test_counter_detects_heap_use() {
    assert!(allocations(|| drop(std::hint::black_box(vec![1u8; 16]))) > 0);
}

fn test_inline_reset_and_step_do_not_allocate() {
    let cfg = CountConfig {
        length: 1000,
        ..CountConfig::default()
    };
    let options = PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(1));
    let mut pool = EnvPool::<Counter>::new(cfg, options).unwrap();
    let actions = vec![1i32; 128];
    let ids = [3u32, 9, 27];
    let seeds = [Seed::new(5), Seed::new(6), Seed::new(7)];
    let n = allocations(|| {
        let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
        for _ in 0..20 {
            let input = ActionInput {
                discrete: &actions,
                continuous: &[],
            };
            let _ = std::hint::black_box(pool.step(input));
        }
        let _ = std::hint::black_box(pool.reset(&ids, Reseed::List(&seeds)));
        let _ = std::hint::black_box(pool.reset(&ids, Reseed::Base(Seed::new(2))));
    });
    assert_eq!(n, 0);
    assert!(pool.outputs().unwrap().rewards().iter().any(|&r| r > 0.0));
}

fn test_masked_same_step_pool_does_not_allocate() {
    let cfg = CountConfig {
        length: 3,
        ..CountConfig::default()
    };
    let options = PoolOptions {
        auto_reset: AutoReset::SameStep,
        ..PoolOptions::new(NonZeroU32::new(16).unwrap(), Seed::new(1))
    };
    let mut pool = EnvPool::<Counter, Masked>::with_masks(cfg, options).unwrap();
    let actions = vec![2i32; 32];
    let n = allocations(|| {
        let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
        for _ in 0..10 {
            let input = ActionInput {
                discrete: &actions,
                continuous: &[],
            };
            let _ = std::hint::black_box(pool.step(input));
        }
    });
    assert_eq!(n, 0);
}

fn test_snapshot_restore_and_fork_do_not_allocate() {
    let options = PoolOptions::new(NonZeroU32::new(8).unwrap(), Seed::new(3));
    let mut pool = EnvPool::<Counter, Masked>::with_masks(CountConfig::default(), options).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let mut records = [pool.snapshot(0).unwrap(); 8];
    let n = allocations(|| {
        let record = pool.snapshot(3).unwrap();
        let _ = std::hint::black_box(pool.restore(5, &record));
        let _ = std::hint::black_box(pool.fork(1, &[2, 4]));
        let _ = std::hint::black_box(pool.fork_branched(1, &[6, 7], Some(&[10, 11])));
        let _ = std::hint::black_box(pool.snapshot_all(&mut records));
        let _ = std::hint::black_box(pool.restore_all(&records));
        let _ = std::hint::black_box(pool.state_hash(0));
    });
    assert_eq!(n, 0);
}

/// The scenarios this binary runs.
const SCENARIOS: usize = 4;

#[test]
fn test_alloc_inline_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_counter_detects_heap_use",
                test_counter_detects_heap_use,
            ),
            (
                "test_inline_reset_and_step_do_not_allocate",
                test_inline_reset_and_step_do_not_allocate,
            ),
            (
                "test_masked_same_step_pool_does_not_allocate",
                test_masked_same_step_pool_does_not_allocate,
            ),
            (
                "test_snapshot_restore_and_fork_do_not_allocate",
                test_snapshot_restore_and_fork_do_not_allocate,
            ),
        ],
    );
}
