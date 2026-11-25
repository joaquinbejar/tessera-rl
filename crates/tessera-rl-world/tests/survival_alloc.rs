// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Survival-v1 hot path never allocates after construction (SPEC 5.5,
//! ADR-0004): reset (map generation and the flood fill included), steps
//! through deaths, respawns and auto-reset, at 16 and at 128 agents. The
//! counting allocator of the games crate attributes calls by thread.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, NoMasks, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;
use tessera_rl_world::survival::MAX_AGENTS;
use tessera_rl_world::{Survival, SurvivalConfig};

#[path = "../../tessera-rl-games/tests/support/counting.rs"]
mod counting;
use counting::window;

fn allocations(f: impl FnOnce()) -> u64 {
    window(f).1.calls
}

fn test_counter_detects_heap_use() {
    assert!(allocations(|| drop(std::hint::black_box(vec![1u8; 16]))) > 0);
}

fn test_survival_reset_and_steps_through_deaths_and_auto_reset_do_not_allocate() {
    for (agents, size) in [(16, 32), (128, 96)] {
        for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
            let cfg = SurvivalConfig {
                agents,
                size,
                steps: 40,
                respawn: 2,
                ..SurvivalConfig::default()
            };
            let options = PoolOptions {
                auto_reset,
                ..PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(9))
            };
            let mut pool = EnvPool::<Survival, NoMasks>::new(cfg, options).unwrap();
            let actions: Vec<i32> = (0..2 * MAX_AGENTS)
                .flat_map(|i| {
                    [
                        i32::try_from(i % 5).unwrap(),
                        i32::try_from(3 + i % 4).unwrap(),
                    ]
                })
                .collect();
            let mut ended = false;
            let n = allocations(|| {
                let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
                for _ in 0..90 {
                    let out = pool
                        .step(ActionInput {
                            discrete: &actions,
                            continuous: &[],
                        })
                        .unwrap();
                    ended |= out.episode_truncated().iter().any(|f| f.get());
                }
            });
            assert_eq!(n, 0, "{agents} agents, {auto_reset:?}");
            assert!(ended, "no episode ended, so auto-reset was not exercised");
        }
    }
}

const SCENARIOS: usize = 2;

#[test]
fn test_survival_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_counter_detects_heap_use",
                test_counter_detects_heap_use,
            ),
            (
                "test_survival_reset_and_steps_through_deaths_and_auto_reset_do_not_allocate",
                test_survival_reset_and_steps_through_deaths_and_auto_reset_do_not_allocate,
            ),
        ],
    );
}
