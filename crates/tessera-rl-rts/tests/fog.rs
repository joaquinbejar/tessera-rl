// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fog: visibility against a brute-force reference, sight edges, explored
//! at reset, explored monotone and covering visibility at every decision,
//! explored carried by snapshot and restore, and the records that break
//! the explored rules refused (`rts-design.md` section 14, rulings 1, 2).

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    reason = "test fixtures"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Bool8, Reserved};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EnvPool, IllegalActionPolicy, Masked, PoolOptions,
    Reseed,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_rts::bots::Bot;
use tessera_rl_rts::env::{HEADS_PER_UNIT, SIDE_SLOTS};
use tessera_rl_rts::fog::{explored, side_bit, visible};
use tessera_rl_rts::units::{Kind, Unit};
use tessera_rl_rts::world::{Grid, MAX_CELLS, Team};
use tessera_rl_rts::{Rts, RtsConfig, RtsState};

type Pool = EnvPool<Rts, Masked>;

const TEAMS: [Team; 2] = [Team::Zero, Team::One];

fn cfg(size: u16, units: u16) -> RtsConfig {
    RtsConfig {
        size,
        decision_ticks: 4,
        max_units: units,
        step_limit: 600,
        seat_swap: false,
        opponent: 0,
    }
}

fn rng(seed: u64) -> StepRng {
    let identity = RngIdentity::new(Seed::new(seed), LogicalId::new(0));
    StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS)
}

fn pool(c: &RtsConfig, seed: u64) -> Pool {
    let options = PoolOptions {
        auto_reset: AutoReset::NextStep,
        illegal_action: IllegalActionPolicy::Noop,
        chunk_size: NonZeroU32::new(1).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(1).unwrap(), Seed::new(seed))
    };
    let mut p: Pool = EnvPool::with_masks(c.clone(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

/// Rush against Prospector: one decision's action rows.
fn bot_actions(c: &RtsConfig, s: &RtsState, r: &mut [StepRng; 2]) -> Vec<i32> {
    let mut a = Vec::new();
    for (agent, bot) in [(0u16, Bot::Rush), (1, Bot::Prospector)] {
        let mut heads = vec![0; usize::from(c.max_units) * HEADS_PER_UNIT];
        let mut w = ActionWriter::new(&mut heads, &mut []);
        bot.act(
            c,
            s,
            AgentSlot::new(agent, 2).unwrap(),
            &mut r[usize::from(agent)],
            &mut w,
        );
        a.extend(heads);
    }
    a
}

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

/// Every cell against every living own unit, by Chebyshev distance.
fn reference(c: &RtsConfig, s: &RtsState, team: Team) -> Vec<bool> {
    let g = Grid::new(c.size).unwrap();
    let base = if team == Team::Zero { 0 } else { SIDE_SLOTS };
    let own = &s.units[base..base + usize::from(c.max_units)];
    (0..MAX_CELLS)
        .map(|cell| {
            let Ok(cell) = u16::try_from(cell) else {
                return false;
            };
            cell < g.cells()
                && own.iter().any(|u| {
                    u.alive.get()
                        && Kind::from_byte(u.kind).is_some_and(|k| {
                            g.chebyshev(u.cell, cell) <= u16::from(k.stats().sight)
                        })
                })
        })
        .collect()
}

#[test]
fn test_fog_visibility_matches_a_brute_force_reference_in_play() {
    for (size, seed) in [(10, 1), (16, 2), (32, 3)] {
        let c = cfg(size, 16);
        let mut p = pool(&c, seed);
        let mut r = [rng(1), rng(2)];
        for _ in 0..150 {
            let s = *p.state(0).unwrap();
            for team in TEAMS {
                assert_eq!(visible(&c, &s, team).to_vec(), reference(&c, &s, team));
            }
            p.step(input(&bot_actions(&c, &s, &mut r))).unwrap();
        }
    }
}

fn unit(kind: Kind, cell: u16, finished: bool) -> Unit {
    Unit {
        cell,
        kind: kind as u8,
        hp: kind.stats().hp,
        alive: Bool8::TRUE,
        finished: Bool8::new(finished),
        carry: 0,
        counter: 0,
        progress: 0,
        producing: 0,
        pad: Reserved::default(),
        generation: 1,
    }
}

#[test]
fn test_fog_sight_edge_unfinished_and_dead_units() {
    let c = cfg(16, 4);
    let g = Grid::new(16).unwrap();
    let at = |x, y| g.index(x, y).unwrap();
    let mut s = RtsState::default();
    s.units[0] = unit(Kind::Core, at(0, 0), true);
    s.units[SIDE_SLOTS] = unit(Kind::Core, at(15, 15), true);
    // A Drone (sight 3) at (8, 8): (11, 8) seen, (12, 8) not.
    s.units[1] = unit(Kind::Drone, at(8, 8), true);
    let sight = Kind::Drone.stats().sight;
    let edge = at(8 + u16::from(sight), 8);
    let past = at(9 + u16::from(sight), 8);
    let v = visible(&c, &s, Team::Zero);
    assert!(v[usize::from(edge)] && !v[usize::from(past)]);
    // An unfinished Foundry sees.
    s.units[2] = unit(Kind::Foundry, at(3, 12), false);
    assert!(visible(&c, &s, Team::Zero)[usize::from(at(3, 12))]);
    // A dead unit does not.
    s.units[1].alive = Bool8::FALSE;
    assert!(!visible(&c, &s, Team::Zero)[usize::from(at(8, 8))]);
    // The other side sees none of it.
    assert!(!visible(&c, &s, Team::One)[usize::from(at(3, 12))]);
}

#[test]
fn test_fog_reset_explores_exactly_what_each_side_sees() {
    for seed in 0..12 {
        let c = cfg(16, 8);
        let s = *pool(&c, seed).state(0).unwrap();
        for team in TEAMS {
            let v = visible(&c, &s, team);
            for cell in 0..MAX_CELLS {
                let e = explored(&s, team, u16::try_from(cell).unwrap());
                assert_eq!(e, v[cell], "seed {seed} {team:?} cell {cell}");
            }
        }
    }
}

#[test]
fn test_fog_explored_grows_and_covers_visibility_at_every_decision() {
    let c = cfg(16, 16);
    let mut p = pool(&c, 14);
    let mut r = [rng(1), rng(2)];
    let mut prev = p.state(0).unwrap().explored;
    let mut grew = false;
    for k in 0..600 {
        let s = *p.state(0).unwrap();
        if s.decisions == 0 && k > 0 {
            prev = s.explored; // a new episode
        }
        for cell in 0..MAX_CELLS {
            assert_eq!(
                s.explored[cell] & prev[cell],
                prev[cell],
                "a bit cleared at {k}"
            );
        }
        grew |= s.explored != prev;
        for team in TEAMS {
            let v = visible(&c, &s, team);
            for (cell, &seen) in v.iter().enumerate() {
                assert!(!seen || s.explored[cell] & side_bit(team) != 0);
            }
        }
        prev = s.explored;
        p.step(input(&bot_actions(&c, &s, &mut r))).unwrap();
    }
    assert!(grew, "explored never grew");
}

#[test]
fn test_fog_explored_is_carried_by_snapshot_and_restore() {
    let c = cfg(16, 16);
    let mut p = pool(&c, 3);
    let mut r = [rng(1), rng(2)];
    for _ in 0..40 {
        let s = *p.state(0).unwrap();
        p.step(input(&bot_actions(&c, &s, &mut r))).unwrap();
    }
    let snap = p.snapshot(0).unwrap();
    let saved = r.clone();
    let run = |p: &mut Pool, r: &mut [StepRng; 2]| {
        for _ in 0..40 {
            let s = *p.state(0).unwrap();
            p.step(input(&bot_actions(&c, &s, r))).unwrap();
        }
        p.state(0).unwrap().explored
    };
    let first = run(&mut p, &mut r);
    p.restore(0, &snap).unwrap();
    let mut r2 = saved;
    assert_eq!(run(&mut p, &mut r2), first);
}

/// An edit of a record's state, given the Core's cell and the cell count.
type Edit = fn(&mut RtsState, u16, u16);

#[test]
fn test_fog_records_breaking_the_explored_rules_are_refused() {
    let c = cfg(12, 4);
    let mut p = pool(&c, 1);
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let header = *good.header();
    let base = *good.state();
    let g = Grid::new(12).unwrap();
    let core = base.units[0].cell;
    let cases: [(&str, Edit); 3] = [
        ("stray bit", |s, _, _| s.explored[0] |= 4),
        ("off the grid", |s, _, cells| {
            s.explored[usize::from(cells)] = 1;
        }),
        ("visible unexplored", |s, core, _| {
            s.explored[usize::from(core)] &= !side_bit(Team::Zero);
        }),
    ];
    for (name, edit) in cases {
        let mut s = base;
        edit(&mut s, core, g.cells());
        let rec = DecodedRecord::from_parts(header, s).unwrap();
        assert!(p.import(&[0], &[rec]).is_err(), "{name}");
    }
    let rec = DecodedRecord::from_parts(header, base).unwrap();
    assert!(p.import(&[0], &[rec]).is_ok());
}
