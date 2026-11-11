// SPDX-License-Identifier: MIT OR Apache-2.0

//! Noninterference under fog (`rts-design.md` section 14): for each side,
//! two states that differ only in facts hidden from that side give
//! byte-identical observations and masks for it. The hidden facts are
//! those of the stage 3 plan r2 (R3); a negative control makes each edit
//! visible and checks that the outputs then change, so the property is
//! not vacuous.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::unnecessary_wraps,
    reason = "test fixtures; every edit shares one signature"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::{ActionWriter, MaskWriter, ObsWriter};
use tessera_rl_core::{ActionMask, AgentSlot, Bool8, Environment};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, IllegalActionPolicy, Masked, PoolOptions, Reseed,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{ObsRng, RngAuthority, StepRng};
use tessera_rl_rts::bots::Bot;
use tessera_rl_rts::env::{CHANNELS, GLOBALS, HEADS_PER_UNIT, MASK_PER_UNIT, SIDE_SLOTS};
use tessera_rl_rts::fog::{seen, side_bit, visible};
use tessera_rl_rts::units::{Kind, Unit};
use tessera_rl_rts::world::{Grid, Team, Terrain};
use tessera_rl_rts::{Rts, RtsConfig, RtsState};

type Pool = EnvPool<Rts, Masked>;

fn cfg(size: u16) -> RtsConfig {
    RtsConfig {
        size,
        decision_ticks: 4,
        max_units: 16,
        step_limit: 600,
        seat_swap: false,
    }
}

fn identity(seed: u64) -> RngIdentity {
    RngIdentity::new(Seed::new(seed), LogicalId::new(0))
}

fn rng(seed: u64) -> StepRng {
    StepRng::new(
        &RngAuthority::engine_only(),
        &identity(seed),
        StreamId::DYNAMICS,
    )
}

fn agent(team: Team) -> AgentSlot {
    AgentSlot::new(u16::from(team == Team::One), 2).unwrap()
}

/// Everything `team` is given: its observation leaves and its mask.
fn view(c: &RtsConfig, s: &RtsState, team: Team) -> (Vec<u8>, Vec<bool>) {
    let cells = usize::from(c.size) * usize::from(c.size);
    let mut planes = vec![0u8; usize::try_from(CHANNELS).unwrap() * cells];
    let mut globals = vec![0u8; 2 * usize::try_from(GLOBALS).unwrap()];
    {
        let mut leaves = [&mut planes[..], &mut globals[..]];
        let mut w = ObsWriter::new(&mut leaves);
        let mut r = ObsRng::new(&RngAuthority::engine_only(), &identity(0));
        Rts::observe(c, s, agent(team), &mut r, &mut w);
    }
    let mut mask = vec![Bool8::FALSE; usize::from(c.max_units) * MASK_PER_UNIT];
    <Rts as ActionMask>::action_mask(c, s, agent(team), &mut MaskWriter::new(&mut mask));
    planes.extend(globals);
    (planes, mask.iter().map(|b| b.get()).collect())
}

fn enemy_slots(c: &RtsConfig, team: Team) -> core::ops::Range<usize> {
    let base = if team == Team::Zero { SIDE_SLOTS } else { 0 };
    base..base + usize::from(c.max_units)
}

fn occupied(s: &RtsState, cell: u16) -> bool {
    s.units.iter().any(|u| u.alive.get() && u.cell == cell)
}

/// One hidden-fact edit of `s` for `team`, or `None` when the state has
/// nothing to edit. `hidden(cell)` says whether `team` cannot see a cell.
type Edit = fn(&RtsConfig, &RtsState, Team, &dyn Fn(u16) -> bool) -> Option<RtsState>;

/// An enemy unit on a hidden cell, moved to another hidden empty open cell.
fn move_hidden(
    c: &RtsConfig,
    s: &RtsState,
    team: Team,
    hidden: &dyn Fn(u16) -> bool,
) -> Option<RtsState> {
    let g = Grid::new(c.size).unwrap();
    let i = enemy_slots(c, team).find(|&i| s.units[i].alive.get() && hidden(s.units[i].cell))?;
    let to = (0..g.cells()).find(|&n| {
        hidden(n) && !occupied(s, n) && s.terrain[usize::from(n)] == Terrain::Open as u8
    })?;
    let mut m = *s;
    m.units[i].cell = to;
    Some(m)
}

/// An enemy Striker added on a hidden empty open cell.
fn add_hidden(
    c: &RtsConfig,
    s: &RtsState,
    team: Team,
    hidden: &dyn Fn(u16) -> bool,
) -> Option<RtsState> {
    let g = Grid::new(c.size).unwrap();
    let i = enemy_slots(c, team).find(|&i| !s.units[i].alive.get())?;
    let to = (0..g.cells()).rev().find(|&n| {
        hidden(n) && !occupied(s, n) && s.terrain[usize::from(n)] == Terrain::Open as u8
    })?;
    let mut m = *s;
    m.units[i] = Unit {
        cell: to,
        kind: Kind::Striker as u8,
        hp: Kind::Striker.stats().hp,
        alive: Bool8::TRUE,
        finished: Bool8::TRUE,
        generation: s.units[i].generation + 1,
        ..Unit::default()
    };
    Some(m)
}

/// An enemy unit on a hidden cell removed.
fn remove_hidden(
    c: &RtsConfig,
    s: &RtsState,
    team: Team,
    hidden: &dyn Fn(u16) -> bool,
) -> Option<RtsState> {
    let i = enemy_slots(c, team).find(|&i| s.units[i].alive.get() && hidden(s.units[i].cell))?;
    let mut m = *s;
    m.units[i] = Unit {
        generation: s.units[i].generation,
        ..Unit::default()
    };
    Some(m)
}

/// Hit points of an enemy unit on a hidden cell.
fn hp_hidden(
    c: &RtsConfig,
    s: &RtsState,
    team: Team,
    hidden: &dyn Fn(u16) -> bool,
) -> Option<RtsState> {
    let i = enemy_slots(c, team).find(|&i| s.units[i].alive.get() && hidden(s.units[i].cell))?;
    let mut m = *s;
    m.units[i].hp = if s.units[i].hp > 1 { 1 } else { 2 };
    Some(m)
}

/// Terrain of an unexplored cell with no unit: open and wall swapped.
fn terrain_unexplored(
    c: &RtsConfig,
    s: &RtsState,
    team: Team,
    _: &dyn Fn(u16) -> bool,
) -> Option<RtsState> {
    let g = Grid::new(c.size).unwrap();
    let cell = (0..g.cells())
        .find(|&n| s.explored[usize::from(n)] & side_bit(team) == 0 && !occupied(s, n))?;
    let mut m = *s;
    let t = &mut m.terrain[usize::from(cell)];
    *t = if *t == Terrain::Wall as u8 {
        Terrain::Open as u8
    } else {
        Terrain::Wall as u8
    };
    Some(m)
}

/// The enemy's explored bits, all flipped on the grid.
fn enemy_explored(
    c: &RtsConfig,
    s: &RtsState,
    team: Team,
    _: &dyn Fn(u16) -> bool,
) -> Option<RtsState> {
    let g = Grid::new(c.size).unwrap();
    let other = if team == Team::Zero {
        Team::One
    } else {
        Team::Zero
    };
    let mut m = *s;
    for cell in 0..usize::from(g.cells()) {
        m.explored[cell] ^= side_bit(other);
    }
    Some(m)
}

/// The enemy's slots permuted (a rotation by one), generations included.
fn permute_enemy(
    c: &RtsConfig,
    s: &RtsState,
    team: Team,
    _: &dyn Fn(u16) -> bool,
) -> Option<RtsState> {
    let r = enemy_slots(c, team);
    let mut m = *s;
    m.units[r.clone()].rotate_left(1);
    for i in r {
        m.units[i].generation = m.units[i].generation.wrapping_add(7);
    }
    Some(m)
}

/// Enemy facts never shown, even where visible: carry, progress,
/// production, counter, whether it is finished, and the enemy's ore.
fn enemy_unshown(
    c: &RtsConfig,
    s: &RtsState,
    team: Team,
    _: &dyn Fn(u16) -> bool,
) -> Option<RtsState> {
    let mut m = *s;
    for i in enemy_slots(c, team) {
        let u = &mut m.units[i];
        if u.alive.get() {
            u.carry ^= 1;
            u.progress = u.progress.wrapping_add(3);
            u.producing = Kind::Lancer as u8;
            u.counter = u.counter.wrapping_add(1);
            u.finished = Bool8::new(!u.finished.get());
        }
    }
    let e = usize::from(team == Team::Zero);
    m.ore[e] = m.ore[e].wrapping_add(1000);
    Some(m)
}

/// Every unexplored cell with no unit walled: paths through the unknown
/// would change if a field read hidden terrain.
fn wall_unexplored(
    c: &RtsConfig,
    s: &RtsState,
    team: Team,
    _: &dyn Fn(u16) -> bool,
) -> Option<RtsState> {
    let g = Grid::new(c.size).unwrap();
    let mut m = *s;
    let mut any = false;
    for n in 0..g.cells() {
        if s.explored[usize::from(n)] & side_bit(team) == 0 && !occupied(s, n) {
            m.terrain[usize::from(n)] = Terrain::Wall as u8;
            any = true;
        }
    }
    any.then_some(m)
}

const EDITS: [(&str, Edit); 9] = [
    ("move a hidden enemy", move_hidden),
    ("add a hidden enemy", add_hidden),
    ("remove a hidden enemy", remove_hidden),
    ("hit points of a hidden enemy", hp_hidden),
    ("terrain of an unexplored cell", terrain_unexplored),
    ("walls on every unexplored cell", wall_unexplored),
    ("the enemy's explored bits", enemy_explored),
    ("the enemy's slot numbering", permute_enemy),
    ("enemy facts never shown", enemy_unshown),
];

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

fn bot_row(c: &RtsConfig, s: &RtsState, team: Team, bot: Bot, r: &mut StepRng) -> Vec<i32> {
    let mut heads = vec![0; usize::from(c.max_units) * HEADS_PER_UNIT];
    let mut w = ActionWriter::new(&mut heads, &mut []);
    bot.act(c, s, agent(team), r, &mut w);
    heads
}

#[test]
fn test_noninterference_observation_and_mask_ignore_every_hidden_fact() {
    let mut applied = [0usize; EDITS.len()];
    for (size, seed) in [(12, 1), (16, 2), (16, 14), (24, 3)] {
        let c = cfg(size);
        let mut p = pool(&c, seed);
        let mut r = [rng(1), rng(2)];
        for k in 0..240 {
            let s = *p.state(0).unwrap();
            if k % 6 == 0 {
                for team in [Team::Zero, Team::One] {
                    let seen = visible(&c, &s, team);
                    let hidden = |cell: u16| !seen[usize::from(cell)];
                    let base = view(&c, &s, team);
                    for (e, (name, edit)) in EDITS.iter().enumerate() {
                        if let Some(m) = edit(&c, &s, team, &hidden) {
                            applied[e] += 1;
                            assert_eq!(
                                view(&c, &m, team),
                                base,
                                "{name}, {team:?}, size {size} step {k}"
                            );
                        }
                    }
                }
            }
            let a = [
                bot_row(&c, &s, Team::Zero, Bot::Rush, &mut r[0]),
                bot_row(&c, &s, Team::One, Bot::Prospector, &mut r[1]),
            ]
            .concat();
            p.step(ActionInput {
                discrete: &a,
                continuous: &[],
            })
            .unwrap();
        }
    }
    for (e, (name, _)) in EDITS.iter().enumerate() {
        assert!(applied[e] > 0, "never applied: {name}");
    }
}

/// The negative control: the same edits on cells the side sees, or of
/// facts it is shown, change its view.
#[test]
fn test_noninterference_control_visible_edits_change_the_view() {
    let c = cfg(16);
    let mut changed = [false; 4];
    // Seed 14: Rush reaches Prospector, so enemies come into sight.
    let mut p = pool(&c, 14);
    let mut r = [rng(1), rng(2)];
    for _ in 0..600 {
        let s = *p.state(0).unwrap();
        for team in [Team::Zero, Team::One] {
            let seen = visible(&c, &s, team);
            let shown = |cell: u16| seen[usize::from(cell)];
            let base = view(&c, &s, team);
            // A visible edit is the hidden edit with "hidden" replaced by "seen".
            for (k, edit) in [move_hidden as Edit, add_hidden, remove_hidden, hp_hidden]
                .iter()
                .enumerate()
            {
                if let Some(m) = edit(&c, &s, team, &shown) {
                    changed[k] |= view(&c, &m, team) != base;
                }
            }
        }
        let a = [
            bot_row(&c, &s, Team::Zero, Bot::Rush, &mut r[0]),
            bot_row(&c, &s, Team::One, Bot::Prospector, &mut r[1]),
        ]
        .concat();
        p.step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
    }
    assert_eq!(changed, [true; 4]);
}

/// The scripted bots read only their side's view: their action rows are
/// identical under every hidden-fact edit (stage 3 plan r2, PR 2).
#[test]
fn test_noninterference_bots_ignore_every_hidden_fact() {
    let mut applied = [0usize; EDITS.len()];
    for (size, seed) in [(12, 1), (16, 9), (24, 3)] {
        let c = cfg(size);
        let mut p = pool(&c, seed);
        let mut r = [rng(1), rng(2)];
        for k in 0..240 {
            let s = *p.state(0).unwrap();
            if k % 8 == 0 {
                for team in [Team::Zero, Team::One] {
                    let seen = visible(&c, &s, team);
                    let hidden = |cell: u16| !seen[usize::from(cell)];
                    for (e, (name, edit)) in EDITS.iter().enumerate() {
                        let Some(m) = edit(&c, &s, team, &hidden) else {
                            continue;
                        };
                        applied[e] += 1;
                        for bot in [Bot::Rush, Bot::Prospector, Bot::RandomLegal] {
                            assert_eq!(
                                bot_row(&c, &m, team, bot, &mut rng(5)),
                                bot_row(&c, &s, team, bot, &mut rng(5)),
                                "{bot:?}: {name}, {team:?}, size {size} step {k}"
                            );
                        }
                    }
                }
            }
            let a = [
                bot_row(&c, &s, Team::Zero, Bot::Rush, &mut r[0]),
                bot_row(&c, &s, Team::One, Bot::Prospector, &mut r[1]),
            ]
            .concat();
            p.step(ActionInput {
                discrete: &a,
                continuous: &[],
            })
            .unwrap();
        }
    }
    for (e, (name, _)) in EDITS.iter().enumerate() {
        assert!(applied[e] > 0, "never applied: {name}");
    }
}

/// The side view's own contract (`fog::seen`): byte-identical under every
/// hidden-fact edit, so every reader of the view inherits noninterference.
#[test]
fn test_noninterference_side_view_ignores_every_hidden_fact() {
    let mut applied = [0usize; EDITS.len()];
    for (size, seed) in [(12, 1), (16, 9), (24, 3)] {
        let c = cfg(size);
        let mut p = pool(&c, seed);
        let mut r = [rng(1), rng(2)];
        for k in 0..240 {
            let s = *p.state(0).unwrap();
            if k % 8 == 0 {
                for team in [Team::Zero, Team::One] {
                    let sees = visible(&c, &s, team);
                    let hidden = |cell: u16| !sees[usize::from(cell)];
                    let base = seen(&c, &s, team);
                    for (e, (name, edit)) in EDITS.iter().enumerate() {
                        if let Some(m) = edit(&c, &s, team, &hidden) {
                            applied[e] += 1;
                            assert!(
                                seen(&c, &m, team) == base,
                                "{name}, {team:?}, size {size} step {k}"
                            );
                        }
                    }
                }
            }
            let a = [
                bot_row(&c, &s, Team::Zero, Bot::Rush, &mut r[0]),
                bot_row(&c, &s, Team::One, Bot::Prospector, &mut r[1]),
            ]
            .concat();
            p.step(ActionInput {
                discrete: &a,
                continuous: &[],
            })
            .unwrap();
        }
    }
    for (e, (name, _)) in EDITS.iter().enumerate() {
        assert!(applied[e] > 0, "never applied: {name}");
    }
}
