// SPDX-License-Identifier: MIT OR Apache-2.0

//! A scripted bot's act never allocates: it runs inside the step when a
//! scripted opponent plays (SPEC zero-allocation hot path). The counting
//! allocator of `support/counting.rs` attributes allocator calls by thread
//! (issue #43). This binary holds a single measuring test, which runs each
//! scenario (the former tests, by name) with its own objects and windows,
//! and reports every failing scenario together.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test scenarios: a failure is the assertion"
)]
#![expect(clippy::many_single_char_names, reason = "test fixtures")]

use core::num::NonZeroU32;

use tessera_rl_core::AgentSlot;
use tessera_rl_core::io::ActionWriter;
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_rts::bots::Bot;
use tessera_rl_rts::env::HEADS_PER_UNIT;
use tessera_rl_rts::{Rts, RtsConfig};

#[path = "support/counting.rs"]
mod counting;
use counting::window;

/// Allocation calls of the counted threads while `f` runs.
fn allocations(f: impl FnOnce()) -> u64 {
    window(f).1.calls
}

fn test_bots_alloc_counter_detects_heap_use() {
    assert!(allocations(|| drop(std::hint::black_box(vec![1u8; 16]))) > 0);
}

fn test_bots_act_never_allocates() {
    let c = RtsConfig {
        size: 16,
        decision_ticks: 4,
        max_units: 16,
        step_limit: 600,
        seat_swap: false,
        opponent: 0,
    };
    let options = PoolOptions::new(NonZeroU32::new(1).unwrap(), Seed::new(14));
    let mut p: EnvPool<Rts, Masked> = EnvPool::with_masks(c.clone(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let identity = RngIdentity::new(Seed::new(1), LogicalId::new(0));
    let mut r = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
    let n = usize::from(c.max_units) * HEADS_PER_UNIT;
    let (mut heads, mut a) = (vec![0; n], vec![0; 2 * n]);
    for _ in 0..200 {
        let s = *p.state(0).unwrap();
        for bot in [Bot::Rush, Bot::Prospector, Bot::RandomLegal] {
            for agent in [0, 1] {
                let slot = AgentSlot::new(agent, 2).unwrap();
                let mut w = ActionWriter::new(&mut heads, &mut []);
                assert_eq!(
                    allocations(|| bot.act(&c, &s, slot, &mut r, &mut w)),
                    0,
                    "{bot:?}"
                );
            }
        }
        for (agent, bot) in [(0u16, Bot::Rush), (1, Bot::Prospector)] {
            let slot = AgentSlot::new(agent, 2).unwrap();
            let lo = usize::from(agent) * n;
            let mut w = ActionWriter::new(&mut a[lo..lo + n], &mut []);
            bot.act(&c, &s, slot, &mut r, &mut w);
        }
        p.step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
    }
}

/// The scenarios this binary runs.
const SCENARIOS: usize = 2;

#[test]
fn test_bots_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_bots_alloc_counter_detects_heap_use",
                test_bots_alloc_counter_detects_heap_use,
            ),
            (
                "test_bots_act_never_allocates",
                test_bots_act_never_allocates,
            ),
        ],
    );
}
