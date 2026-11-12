// SPDX-License-Identifier: MIT OR Apache-2.0

//! The pendulum hot path never allocates after construction (SPEC 5.5,
//! ADR-0004), terminations, truncations and auto-reset included. The
//! counting allocator of `support/counting.rs` attributes allocator calls by
//! thread (issue #43). This binary holds a single measuring test, which runs
//! each scenario (the former tests, by name) with its own objects and
//! windows, and reports every failing scenario together.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::{Pendulum, PendulumConfig};
use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, NoMasks, PoolOptions, Reseed};
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

fn test_pendulum_reset_and_steps_through_auto_reset_do_not_allocate() {
    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
        let options = PoolOptions {
            auto_reset,
            max_episode_steps: NonZeroU32::new(150),
            ..PoolOptions::new(NonZeroU32::new(8).unwrap(), Seed::new(1))
        };
        let mut pool =
            EnvPool::<Pendulum, NoMasks>::new(PendulumConfig::default(), options).unwrap();
        let actions = [1.5f32; 8];
        let mut ended = false;
        let n = allocations(|| {
            let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
            for _ in 0..400 {
                let input = ActionInput {
                    discrete: &[],
                    continuous: &actions,
                };
                let out = pool.step(input).unwrap();
                ended |= out.episode_truncated().iter().any(|f| f.get());
            }
        });
        assert_eq!(n, 0, "{auto_reset:?}");
        assert!(ended, "no episode ended, so auto-reset was not exercised");
    }
}

/// The scenarios this binary runs.
const SCENARIOS: usize = 2;

#[test]
fn test_pendulum_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_counter_detects_heap_use",
                test_counter_detects_heap_use,
            ),
            (
                "test_pendulum_reset_and_steps_through_auto_reset_do_not_allocate",
                test_pendulum_reset_and_steps_through_auto_reset_do_not_allocate,
            ),
        ],
    );
}
