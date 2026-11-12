// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Go 9x9 hot path never allocates after construction (SPEC 5.5,
//! ADR-0004): reset, step, masks, observations and auto-reset through whole
//! games. The counting allocator of `support/counting.rs` attributes
//! allocator calls by thread (issue #43). This binary holds a single
//! measuring test, which runs each scenario (the former tests, by name) with
//! its own objects and windows, and reports every failing scenario together.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_games::{Go9, Go9Config};
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

fn test_go9_reset_and_steps_through_auto_reset_do_not_allocate() {
    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
        let options = PoolOptions {
            auto_reset,
            ..PoolOptions::new(NonZeroU32::new(8).unwrap(), Seed::new(1))
        };
        let mut pool = EnvPool::<Go9, Masked>::with_masks(Go9Config::default(), options).unwrap();
        let mut actions = vec![0i32; 16];
        let mut ended = false;
        let n = allocations(|| {
            let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
            for _ in 0..400 {
                // The first legal action of each row, read from the masks: games run
                // to the board's limits, with captures and superko scans.
                let masks = pool.outputs().unwrap().masks();
                for (row, action) in masks.chunks(82).zip(actions.iter_mut()) {
                    let first = row.iter().position(|f| f.get()).unwrap_or(81);
                    *action = i32::try_from(first).unwrap();
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
        assert!(ended, "no game ended, so auto-reset was not exercised");
    }
}

/// The scenarios this binary runs.
const SCENARIOS: usize = 2;

#[test]
fn test_go9_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_counter_detects_heap_use",
                test_counter_detects_heap_use,
            ),
            (
                "test_go9_reset_and_steps_through_auto_reset_do_not_allocate",
                test_go9_reset_and_steps_through_auto_reset_do_not_allocate,
            ),
        ],
    );
}
