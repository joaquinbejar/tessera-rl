// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Lanterns hot path never allocates after construction (SPEC 5.5,
//! ADR-0004): reset, step, masks, observations and auto-reset through whole
//! games, and the exact determinization and the baseline, which use
//! fixed-size buffers only. The counting allocator of `support/counting.rs`
//! attributes allocator calls by thread (issue #43).

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_games::lanterns::{ACTIONS, MAX_AGENTS, baseline, deal, info_of};
use tessera_rl_games::{Lanterns, LanternsConfig};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

#[path = "support/counting.rs"]
mod counting;
use counting::window;

/// Allocation calls of the counted threads while `f` runs.
fn allocations(f: impl FnOnce()) -> u64 {
    window(f).1.calls
}

fn cfg() -> LanternsConfig {
    LanternsConfig {
        players: 4,
        zero_on_fuse: false,
    }
}

fn test_counter_detects_heap_use() {
    assert!(allocations(|| drop(std::hint::black_box(vec![1u8; 16]))) > 0);
}

fn test_lanterns_reset_and_steps_through_auto_reset_do_not_allocate() {
    let width = ACTIONS as usize;
    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
        let options = PoolOptions {
            auto_reset,
            ..PoolOptions::new(NonZeroU32::new(8).unwrap(), Seed::new(1))
        };
        let mut pool = EnvPool::<Lanterns, Masked>::with_masks(cfg(), options).unwrap();
        let mut actions = vec![0i32; 8 * MAX_AGENTS];
        let mut ended = false;
        let n = allocations(|| {
            let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
            for _ in 0..400 {
                // The first legal action of each agent row, from the masks;
                // the last legal one favours discards and hints over plays,
                // so games reach the end of the deck too.
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
        assert!(ended, "no game ended, so auto-reset was not exercised");
    }
}

fn test_lanterns_determinize_and_baseline_do_not_allocate() {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(2));
    let mut pool = EnvPool::<Lanterns, Masked>::with_masks(cfg(), options).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let state = *pool.snapshot(0).unwrap().state();
    let info = info_of(&state, 1);
    let id = RngIdentity::new(Seed::new(3), LogicalId::DEFAULT);
    let mut rng = StepRng::new(&RngAuthority::engine_only(), &id, StreamId::DYNAMICS);
    let n = allocations(|| {
        for _ in 0..64 {
            let s = deal::sample(&info, &mut rng);
            std::hint::black_box(baseline::act_info(&info_of(&s, 1)));
        }
    });
    assert_eq!(n, 0);
}

/// The scenarios this binary runs.
const SCENARIOS: usize = 3;

#[test]
fn test_lanterns_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_counter_detects_heap_use",
                test_counter_detects_heap_use,
            ),
            (
                "test_lanterns_reset_and_steps_through_auto_reset_do_not_allocate",
                test_lanterns_reset_and_steps_through_auto_reset_do_not_allocate,
            ),
            (
                "test_lanterns_determinize_and_baseline_do_not_allocate",
                test_lanterns_determinize_and_baseline_do_not_allocate,
            ),
        ],
    );
}
