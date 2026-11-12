// SPDX-License-Identifier: MIT OR Apache-2.0

//! The brick wall hot path never allocates after construction (SPEC 5.5,
//! ADR-0004), serves, misses and auto-reset included, in both observations.
//! The counting allocator of `support/counting.rs` attributes allocator
//! calls by thread (issue #43). This binary holds a single measuring test,
//! which runs each scenario (the former tests, by name) with its own objects
//! and windows, and reports every failing scenario together.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::brickwall::{OBS_PIXELS, OBS_STATE};
use tessera_rl_classic::{BrickWall, BrickWallConfig};
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

fn test_brickwall_reset_and_steps_through_auto_reset_do_not_allocate() {
    for obs in [OBS_PIXELS, OBS_STATE] {
        for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
            let options = PoolOptions {
                auto_reset,
                ..PoolOptions::new(NonZeroU32::new(8).unwrap(), Seed::new(1))
            };
            let cfg = BrickWallConfig {
                frameskip: 8,
                lives: 1,
                obs,
            };
            let mut pool = EnvPool::<BrickWall, NoMasks>::new(cfg, options).unwrap();
            // Always left: the paddle leaves the serve's path, so the one ball
            // is missed and every episode ends within the loop.
            let actions = [0i32; 8];
            let mut ended = false;
            let n = allocations(|| {
                let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
                for _ in 0..60 {
                    let input = ActionInput {
                        discrete: &actions,
                        continuous: &[],
                    };
                    let out = pool.step(input).unwrap();
                    ended |= out.episode_terminated().iter().any(|f| f.get());
                }
            });
            assert_eq!(n, 0, "{obs} {auto_reset:?}");
            assert!(ended, "no episode ended, so auto-reset was not exercised");
        }
    }
}

/// The scenarios this binary runs.
const SCENARIOS: usize = 2;

#[test]
fn test_brickwall_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_counter_detects_heap_use",
                test_counter_detects_heap_use,
            ),
            (
                "test_brickwall_reset_and_steps_through_auto_reset_do_not_allocate",
                test_brickwall_reset_and_steps_through_auto_reset_do_not_allocate,
            ),
        ],
    );
}
