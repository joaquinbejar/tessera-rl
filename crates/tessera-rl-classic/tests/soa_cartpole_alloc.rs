// SPDX-License-Identifier: MIT OR Apache-2.0

//! A lane-wise pool's hot path never allocates after construction (SPEC 5.5,
//! ADR-0090 decision 7), auto-reset included: `SoaCartPole`'s groups of 8
//! run `step_lanes` and `observe_lanes` on the columns, and a tail shorter
//! than 8 runs the one-instance gather, scalar step and scatter. The counting
//! allocator of `support/counting.rs` attributes allocator calls by thread.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};
use core::sync::atomic::Ordering;

#[path = "../benches/support/soa_cartpole.rs"]
mod soa_cartpole;
use soa_cartpole::{LANE_STEPS, SoaCartPole};
use tessera_rl_classic::cartpole::CartPoleConfig;
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, NoMasks, PoolOptions, Reseed, SyncOptions,
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

/// 67 instances in chunks of 8: eight full groups take the lane kernels, and
/// the last chunk's 3 instances take the per-instance fallback.
fn run(workers: Option<usize>, auto_reset: AutoReset) {
    const N: u32 = 67;
    let options = PoolOptions {
        auto_reset,
        chunk_size: NonZeroU32::new(8).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(3))
    };
    let pool = EnvPool::<SoaCartPole, NoMasks>::new(CartPoleConfig {}, options).unwrap();
    let mut pool = match workers {
        None => pool,
        Some(w) => pool
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap(),
    };
    // Always push right: every pole falls within a few dozen steps.
    let actions = [1i32; N as usize];
    let before = LANE_STEPS.load(Ordering::Relaxed);
    let mut ended = false;
    let n = allocations(|| {
        let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
        for _ in 0..200 {
            let input = ActionInput {
                discrete: &actions,
                continuous: &[],
            };
            let out = pool.step(input).unwrap();
            ended |= out.episode_terminated().iter().any(|f| f.get());
        }
    });
    assert_eq!(n, 0, "workers {workers:?}, {auto_reset:?}");
    assert!(ended, "no episode ended, so auto-reset was not exercised");
    assert!(
        LANE_STEPS.load(Ordering::Relaxed) > before,
        "the lane kernels never ran"
    );
}

fn test_soa_cartpole_inline_steps_do_not_allocate() {
    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
        run(None, auto_reset);
    }
}

fn test_soa_cartpole_sync_steps_do_not_allocate() {
    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
        run(Some(2), auto_reset);
    }
}

/// The scenarios this binary runs.
const SCENARIOS: usize = 3;

#[test]
fn test_soa_cartpole_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_counter_detects_heap_use",
                test_counter_detects_heap_use,
            ),
            (
                "test_soa_cartpole_inline_steps_do_not_allocate",
                test_soa_cartpole_inline_steps_do_not_allocate,
            ),
            (
                "test_soa_cartpole_sync_steps_do_not_allocate",
                test_soa_cartpole_sync_steps_do_not_allocate,
            ),
        ],
    );
}
