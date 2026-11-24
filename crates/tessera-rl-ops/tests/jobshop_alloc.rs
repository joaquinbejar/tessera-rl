// SPDX-License-Identifier: MIT OR Apache-2.0

//! The JobShop-v1 hot path never allocates after construction (SPEC 5.5,
//! ADR-0004): reset, step, masks, observations and auto-reset through
//! whole episodes at the largest shape, and the dispatching rules. The
//! counting allocator of the games crate attributes calls by thread.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;
use tessera_rl_ops::jobshop::baseline::{self, Rule};
use tessera_rl_ops::{JobShop, JobShopConfig};

#[path = "../../tessera-rl-games/tests/support/counting.rs"]
mod counting;
use counting::window;

/// Allocation calls of the counted threads while `f` runs.
fn allocations(f: impl FnOnce()) -> u64 {
    window(f).1.calls
}

fn test_counter_detects_heap_use() {
    assert!(allocations(|| drop(std::hint::black_box(vec![1u8; 16]))) > 0);
}

fn test_jobshop_reset_and_steps_through_auto_reset_do_not_allocate() {
    let cfg = JobShopConfig {
        jobs: 64,
        machines: 32,
        max_duration: 255,
    };
    let width = 64;
    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
        let options = PoolOptions {
            auto_reset,
            ..PoolOptions::new(NonZeroU32::new(4).unwrap(), Seed::new(1))
        };
        let mut pool = EnvPool::<JobShop, Masked>::with_masks(cfg.clone(), options).unwrap();
        let mut actions = vec![0i32; 4];
        let mut ended = false;
        let n = allocations(|| {
            let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
            // One episode is 2,048 steps: run past it so auto-reset runs.
            for _ in 0..2_100 {
                let masks = pool.outputs().unwrap().masks();
                for (row, action) in masks.chunks(width).zip(actions.iter_mut()) {
                    let last = row.iter().rposition(|f| f.get()).unwrap_or(0);
                    *action = i32::try_from(last).unwrap();
                }
                let input = ActionInput {
                    discrete: &actions,
                    continuous: &[],
                };
                let out = pool.step(input).unwrap();
                ended |= out.episode_terminated().iter().any(|f| f.get());
            }
        });
        assert_eq!(n, 0, "{auto_reset:?}");
        assert!(ended, "no episode ended, so auto-reset was not exercised");
    }
}

fn test_jobshop_rules_do_not_allocate() {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(2));
    let cfg = JobShopConfig {
        jobs: 64,
        machines: 32,
        max_duration: 255,
    };
    let mut pool = EnvPool::<JobShop, Masked>::with_masks(cfg, options).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let state = *pool.state(0).unwrap();
    let n = allocations(|| {
        for rule in [Rule::Spt, Rule::Mwkr, Rule::Fifo] {
            std::hint::black_box(baseline::act(rule, &state));
        }
    });
    assert_eq!(n, 0);
}

/// The scenarios this binary runs.
const SCENARIOS: usize = 3;

#[test]
fn test_jobshop_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_counter_detects_heap_use",
                test_counter_detects_heap_use,
            ),
            (
                "test_jobshop_reset_and_steps_through_auto_reset_do_not_allocate",
                test_jobshop_reset_and_steps_through_auto_reset_do_not_allocate,
            ),
            (
                "test_jobshop_rules_do_not_allocate",
                test_jobshop_rules_do_not_allocate,
            ),
        ],
    );
}
