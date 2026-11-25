// SPDX-License-Identifier: MIT OR Apache-2.0

//! RTS heuristic slots (#586, ADR-0079): a heuristic slot of a policy map
//! plays `act_info` on its information state, the fog-respecting Rush,
//! with garbage in its action rows ignored, at every worker count and
//! schedule.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    reason = "test fixtures"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, ImperfectInformation, InfoHeuristic, PolicyGroup};
use tessera_rl_engine::{
    ActionInput, EngineError, EnvPool, Masked, PoolOptions, Reseed, Schedule, SyncOptions,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_rts::bots::Bot;
use tessera_rl_rts::env::HEADS_PER_UNIT;
use tessera_rl_rts::{Rts, RtsConfig, RtsState};

type Pool = EnvPool<Rts, Masked>;

fn cfg() -> RtsConfig {
    RtsConfig {
        size: 16,
        decision_ticks: 4,
        max_units: 16,
        step_limit: 600,
        seat_swap: false,
        opponent: 0,
    }
}

/// Heads of one agent's action row.
fn width(c: &RtsConfig) -> usize {
    usize::from(c.max_units) * HEADS_PER_UNIT
}

fn rng() -> StepRng {
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS)
}

fn slot(agent: usize) -> AgentSlot {
    AgentSlot::new(u16::try_from(agent).unwrap(), 2).unwrap()
}

/// `act_info` of `agent` on its information state in `s`. Rush draws no
/// randomness, so the stream does not matter.
fn act_info(c: &RtsConfig, s: &RtsState, agent: usize) -> Vec<i32> {
    let info = Rts::info_state(c, s, slot(agent));
    let mut heads = vec![0; width(c)];
    Rts::act_info(
        c,
        &info,
        slot(agent),
        &mut rng(),
        &mut ActionWriter::new(&mut heads, &mut []),
    );
    heads
}

/// The learner's legal orders: Prospector on its side.
fn learner(c: &RtsConfig, s: &RtsState) -> Vec<i32> {
    let mut heads = vec![0; width(c)];
    Bot::Prospector.act(
        c,
        s,
        slot(0),
        &mut rng(),
        &mut ActionWriter::new(&mut heads, &mut []),
    );
    heads
}

fn options(n: u32, seed: u64, schedule: Schedule) -> PoolOptions {
    PoolOptions {
        chunk_size: NonZeroU32::new(2).unwrap(),
        max_episode_steps: NonZeroU32::new(120),
        schedule,
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed))
    }
}

#[test]
fn test_info_heuristic_slots_require_info_heuristic_or_privileged() {
    let c = cfg();
    let build = || Pool::with_masks(c.clone(), options(4, 1, Schedule::Static));
    // Imperfect information: the omniscient path is refused.
    assert!(matches!(
        build().unwrap().with_heuristic_slots(&[1]),
        Err(EngineError::PolicyMap { .. })
    ));
    let info = build().unwrap().with_info_heuristic_slots(&[1]).unwrap();
    assert_eq!(info.policy_groups()[1], PolicyGroup::Heuristic);
    assert_eq!(info.policy_groups()[0], PolicyGroup::Learner);
}

/// Runs RTS with slot 1 a heuristic slot and garbage in its rows; checks
/// each step's heuristic actions against `act_info` on the states before
/// the step and returns the final state hashes.
fn run(workers: Option<usize>, schedule: Schedule) -> Vec<u64> {
    let c = cfg();
    let n = 6;
    let w = width(&c);
    let mut p = Pool::with_masks(c.clone(), options(n, 9, schedule))
        .unwrap()
        .with_info_heuristic_slots(&[1])
        .unwrap();
    if let Some(k) = workers {
        p = p
            .with_workers(SyncOptions::new(NonZeroUsize::new(k).unwrap()))
            .unwrap();
    }
    p.reset_all(Reseed::Keep).unwrap();
    let mut acted = 0usize;
    for t in 0..150 {
        let mut want = vec![0; usize::try_from(n).unwrap() * 2 * w];
        let mut acts = vec![0; usize::try_from(n).unwrap() * 2 * w];
        for i in 0..n {
            let base = usize::try_from(i).unwrap() * 2 * w;
            let header = p.header(i).unwrap();
            let s = p.state(i).unwrap();
            acts[base..base + w].copy_from_slice(&learner(&c, &s));
            // Garbage in the heuristic slot's row, out of range.
            for a in &mut acts[base + w..base + 2 * w] {
                *a = -7;
            }
            if tessera_rl_core::RecordLayout::engine(header)
                .pending_reset
                .get()
            {
                continue;
            }
            let rows = tessera_rl_core::RecordLayout::agents(header);
            if rows[1].alive.get() && rows[1].acting.get() {
                want[base + w..base + 2 * w].copy_from_slice(&act_info(&c, &s, 1));
                acted += 1;
            }
        }
        p.step(ActionInput {
            discrete: &acts,
            continuous: &[],
        })
        .unwrap();
        assert_eq!(p.heuristic_actions().unwrap(), want.as_slice(), "t {t}");
    }
    assert!(acted > 0);
    (0..n).map(|i| p.state_hash(i).unwrap().get()).collect()
}

#[test]
fn test_info_heuristic_slots_equal_act_info_at_every_worker_count_and_schedule() {
    let inline = run(None, Schedule::Static);
    for k in [1, 2, 4] {
        assert_eq!(run(Some(k), Schedule::Static), inline);
        assert_eq!(run(Some(k), Schedule::Dynamic), inline);
    }
}
