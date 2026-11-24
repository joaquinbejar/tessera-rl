// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tag's window baseline (ADR-0079, #437 option (a)) and Tag heuristic
//! slots: the baseline reads only the agent's window (noninterference),
//! equals the old omniscient chase when the window covers the grid, falls
//! back to the lowest legal action with no opponent in view, and a
//! heuristic slot of a policy map plays `act_info` at any worker count.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_classic::tag::MAX_RADIUS;
use tessera_rl_classic::tag::grid::{Cell, distance, moved};
use tessera_rl_classic::{Tag, TagConfig, TagState};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic, ImperfectInformation, InfoHeuristic, PolicyGroup};
use tessera_rl_engine::{
    ActionInput, EngineError, EnvPool, Masked, PoolOptions, Reseed, Schedule, SyncOptions,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

const A: usize = 8;

fn rng() -> StepRng {
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS)
}

fn slot(agent: usize) -> AgentSlot {
    AgentSlot::new(u16::try_from(agent).unwrap(), A).unwrap()
}

fn act(cfg: &TagConfig, state: &TagState, agent: usize) -> i32 {
    let mut discrete = [0];
    Tag::act(
        cfg,
        state,
        slot(agent),
        &mut rng(),
        &mut ActionWriter::new(&mut discrete, &mut []),
    );
    discrete[0]
}

fn act_info(cfg: &TagConfig, state: &TagState, agent: usize) -> i32 {
    let mut discrete = [0];
    let info = Tag::info_state(cfg, state, slot(agent));
    Tag::act_info(
        cfg,
        &info,
        slot(agent),
        &mut rng(),
        &mut ActionWriter::new(&mut discrete, &mut []),
    );
    discrete[0]
}

/// The omniscient rule Tag's baseline had before ADR-0079: the nearest
/// active opponent anywhere on the grid.
fn omniscient(cfg: &TagConfig, state: &TagState, me: usize) -> i32 {
    let from = state.pos[me];
    let pursuer = cfg.is_pursuer(me);
    let nearest = |cell: Cell| {
        (0..cfg.agents())
            .filter(|&j| !state.caught[j].get() && cfg.is_pursuer(j) != pursuer)
            .map(|j| distance(cell, state.pos[j]))
            .min()
            .unwrap_or(0)
    };
    let mut best = (0, nearest(from));
    for action in 1..=4 {
        let Some(to) = moved(from, action, cfg.width, cfg.height) else {
            continue;
        };
        let d = nearest(to);
        if (pursuer && d < best.1) || (!pursuer && d > best.1) {
            best = (action, d);
        }
    }
    best.0
}

fn options(n: u32, seed: u64, schedule: Schedule) -> PoolOptions {
    PoolOptions {
        chunk_size: NonZeroU32::new(4).unwrap(),
        max_episode_steps: NonZeroU32::new(60),
        schedule,
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed))
    }
}

/// Random legal moves for every slot of every instance.
fn moves(p: &EnvPool<Tag, Masked>, t: usize) -> Vec<i32> {
    let masks = p.outputs().unwrap().masks();
    let rows = masks.len() / 5;
    (0..rows)
        .map(|row| {
            let pick = i32::try_from((row * 7 + t * 3) % 5).unwrap();
            if masks[row * 5 + usize::try_from(pick).unwrap()].get() {
                pick
            } else {
                0
            }
        })
        .collect()
}

#[test]
fn test_tag_window_baseline_equals_the_omniscient_rule_when_the_window_covers_the_grid() {
    let cfg = TagConfig {
        pursuers: 3,
        evaders: 3,
        radius: MAX_RADIUS,
        ..TagConfig::default()
    };
    let n = 8;
    let mut p =
        EnvPool::<Tag, Masked>::with_masks(cfg.clone(), options(n, 5, Schedule::Static)).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let mut checked = 0;
    for t in 0..40 {
        for i in 0..n {
            let state = p.state(i).unwrap();
            for a in 0..cfg.agents() {
                if cfg.is_pursuer(a) || !state.caught[a].get() {
                    assert_eq!(act(&cfg, state, a), omniscient(&cfg, state, a), "t {t}");
                    checked += 1;
                }
            }
        }
        let acts = moves(&p, t);
        p.step(ActionInput {
            discrete: &acts,
            continuous: &[],
        })
        .unwrap();
    }
    assert!(checked > 1000);
}

#[test]
fn test_tag_heuristic_noninterference_hidden_opponents_give_equal_actions() {
    let cfg = TagConfig {
        width: 12,
        height: 12,
        pursuers: 1,
        evaders: 2,
        radius: 2,
        ..TagConfig::default()
    };
    // The pursuer at [5, 5] sees evader 1 at [6, 6]; evader 2 is outside
    // its window, at [0, 11] or [11, 0].
    let mut a = TagState::default();
    a.pos[0] = [5, 5];
    a.pos[1] = [6, 6];
    a.pos[2] = [0, 11];
    let mut b = a;
    b.pos[2] = [11, 0];
    assert_ne!(a, b);
    assert_eq!(
        Tag::info_state(&cfg, &a, slot(0)),
        Tag::info_state(&cfg, &b, slot(0))
    );
    assert_eq!(act_info(&cfg, &a, 0), act_info(&cfg, &b, 0));
    assert_eq!(act(&cfg, &a, 0), act(&cfg, &b, 0));
    assert_eq!(act(&cfg, &a, 0), act_info(&cfg, &a, 0));
    // It chases the visible evader: south or east.
    assert!([2, 3].contains(&act(&cfg, &a, 0)));
}

#[test]
fn test_tag_heuristic_no_visible_opponent_plays_the_lowest_legal_action() {
    let cfg = TagConfig {
        width: 12,
        height: 12,
        pursuers: 1,
        evaders: 1,
        radius: 1,
        ..TagConfig::default()
    };
    let mut s = TagState::default();
    s.pos[0] = [0, 0];
    s.pos[1] = [9, 9];
    // The omniscient rule would chase; the window rule stays (action 0,
    // always legal), for the pursuer and the evader alike.
    assert_ne!(omniscient(&cfg, &s, 0), 0);
    assert_eq!(act(&cfg, &s, 0), 0);
    assert_eq!(act(&cfg, &s, 1), 0);
}

#[test]
fn test_tag_heuristic_slots_require_info_heuristic_or_privileged() {
    let cfg = TagConfig::default();
    let build = || EnvPool::<Tag, Masked>::with_masks(cfg.clone(), options(8, 1, Schedule::Static));
    assert!(matches!(
        build().unwrap().with_heuristic_slots(&[1]),
        Err(EngineError::PolicyMap { .. })
    ));
    let info = build().unwrap().with_info_heuristic_slots(&[1, 2]).unwrap();
    assert_eq!(info.policy_groups()[1], PolicyGroup::Heuristic);
    let privileged = build()
        .unwrap()
        .with_privileged_heuristic_slots(&[3])
        .unwrap();
    assert_eq!(
        privileged.policy_groups()[3],
        PolicyGroup::PrivilegedHeuristic
    );
    assert_eq!(privileged.policy_groups()[0], PolicyGroup::Learner);
}

/// Runs Tag with every evader a heuristic slot and garbage in their rows;
/// checks each step's heuristic actions against `act_info` on the states
/// before the step and returns the final state hashes.
fn run(workers: Option<usize>, schedule: Schedule) -> Vec<u64> {
    let cfg = TagConfig {
        pursuers: 2,
        evaders: 3,
        ..TagConfig::default()
    };
    let n = 12;
    let evaders: Vec<usize> = (0..cfg.agents()).filter(|&a| !cfg.is_pursuer(a)).collect();
    let mut p = EnvPool::<Tag, Masked>::with_masks(cfg.clone(), options(n, 9, schedule))
        .unwrap()
        .with_info_heuristic_slots(&evaders)
        .unwrap();
    if let Some(w) = workers {
        p = p
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap();
    }
    p.reset_all(Reseed::Keep).unwrap();
    for t in 0..80 {
        let mut want = vec![0; usize::try_from(n).unwrap() * A];
        for i in 0..n {
            let header = p.header(i).unwrap();
            if tessera_rl_core::RecordLayout::engine(header)
                .pending_reset
                .get()
            {
                continue;
            }
            let rows = tessera_rl_core::RecordLayout::agents(header);
            for &e in &evaders {
                if rows[e].alive.get() && rows[e].acting.get() {
                    want[usize::try_from(i).unwrap() * A + e] =
                        act_info(&cfg, p.state(i).unwrap(), e);
                }
            }
        }
        let mut acts = moves(&p, t);
        for (k, a) in acts.iter_mut().enumerate() {
            if evaders.contains(&(k % A)) {
                *a = -3;
            }
        }
        p.step(ActionInput {
            discrete: &acts,
            continuous: &[],
        })
        .unwrap();
        assert_eq!(p.heuristic_actions().unwrap(), want.as_slice(), "t {t}");
    }
    (0..n).map(|i| p.state_hash(i).unwrap().get()).collect()
}

#[test]
fn test_tag_heuristic_slots_equal_act_info_at_every_worker_count_and_schedule() {
    let inline = run(None, Schedule::Static);
    for w in [1, 2, 7] {
        assert_eq!(run(Some(w), Schedule::Static), inline);
        assert_eq!(run(Some(w), Schedule::Dynamic), inline);
    }
}
