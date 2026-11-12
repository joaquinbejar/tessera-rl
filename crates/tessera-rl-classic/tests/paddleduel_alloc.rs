// SPDX-License-Identifier: MIT OR Apache-2.0

//! The paddle duel hot path never allocates after construction (SPEC 5.5,
//! ADR-0004), serves, points and auto-reset included, in both observations.
//! The counting allocator of `support/counting.rs` attributes allocator
//! calls by thread (issue #43). This binary holds a single measuring test,
//! which runs each scenario (the former tests, by name) with its own objects
//! and windows, and reports every failing scenario together.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::paddleduel::{OBS_PIXELS, OBS_STATE};
use tessera_rl_classic::{PaddleDuel, PaddleDuelConfig};
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

fn test_paddleduel_reset_and_steps_through_auto_reset_do_not_allocate() {
    for agents in [1, 2] {
        for obs in [OBS_PIXELS, OBS_STATE] {
            for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
                let options = PoolOptions {
                    auto_reset,
                    ..PoolOptions::new(NonZeroU32::new(8).unwrap(), Seed::new(1))
                };
                let cfg = PaddleDuelConfig {
                    agents,
                    frameskip: 8,
                    max_score: 1,
                    bounce: 1,
                    obs,
                };
                let mut pool = EnvPool::<PaddleDuel, NoMasks>::new(cfg, options).unwrap();
                // Slot 0 always up: it misses balls aimed low, so one point
                // ends each episode within the loop.
                let actions = [0i32; 16];
                let mut ended = false;
                let n = allocations(|| {
                    let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
                    for _ in 0..80 {
                        let input = ActionInput {
                            discrete: &actions,
                            continuous: &[],
                        };
                        let out = pool.step(input).unwrap();
                        ended |= out.episode_terminated().iter().any(|f| f.get());
                    }
                });
                assert_eq!(n, 0, "{agents} {obs} {auto_reset:?}");
                assert!(ended, "no episode ended, so auto-reset was not exercised");
            }
        }
    }
}

/// The scenarios this binary runs.
const SCENARIOS: usize = 2;

#[test]
fn test_paddleduel_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_counter_detects_heap_use",
                test_counter_detects_heap_use,
            ),
            (
                "test_paddleduel_reset_and_steps_through_auto_reset_do_not_allocate",
                test_paddleduel_reset_and_steps_through_auto_reset_do_not_allocate,
            ),
        ],
    );
}
