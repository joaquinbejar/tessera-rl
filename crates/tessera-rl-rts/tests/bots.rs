// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scripted bots: legal orders only, deterministic, mirror-symmetric,
//! walking corridors shared with own units, and a bot-driven golden.
//! Win rates are reported by the ignored test at the end, never asserted.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    reason = "test fixtures"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Bool8, Heuristic, Reserved};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EnvPool, IllegalActionPolicy, Masked, PoolOptions,
    Reseed,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_rts::bots::Bot;
use tessera_rl_rts::env::{HEADS_PER_UNIT, SIDE_SLOTS, UNIT_SLOTS, move_index, order_type};
use tessera_rl_rts::units::{Kind, Unit};
use tessera_rl_rts::world::{Dir, Grid, Team, Terrain};
use tessera_rl_rts::{Rts, RtsConfig, RtsState};

type Pool = EnvPool<Rts, Masked>;

const BOTS: [Bot; 3] = [Bot::Rush, Bot::Prospector, Bot::RandomLegal];

fn cfg(size: u16, units: u16) -> RtsConfig {
    RtsConfig {
        size,
        decision_ticks: 4,
        max_units: units,
        step_limit: 600,
        seat_swap: false,
    }
}

fn rng(seed: u64) -> StepRng {
    let identity = RngIdentity::new(Seed::new(seed), LogicalId::new(0));
    StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS)
}

/// `bot`'s action row for `agent` in `s`.
fn row(c: &RtsConfig, s: &RtsState, agent: u16, bot: Bot, r: &mut StepRng) -> Vec<i32> {
    let mut heads = vec![0; usize::from(c.max_units) * HEADS_PER_UNIT];
    let mut w = ActionWriter::new(&mut heads, &mut []);
    bot.act(c, s, AgentSlot::new(agent, 2).unwrap(), r, &mut w);
    heads
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

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

/// Plays `bots` against each other for `n` decisions; every action row is
/// checked legal first. Returns the episodes' end rewards of agent 0.
fn play(c: &RtsConfig, bots: [Bot; 2], seed: u64, n: usize) -> Vec<f32> {
    let mut p = pool(c, seed);
    let mut r = [rng(seed), rng(seed + 1)];
    let mut ends = Vec::new();
    for _ in 0..n {
        let s = *p.state(0).unwrap();
        let a = [
            row(c, &s, 0, bots[0], &mut r[0]),
            row(c, &s, 1, bots[1], &mut r[1]),
        ]
        .concat();
        assert_eq!(p.check_actions(input(&a)), Ok(()));
        let out = p.step(input(&a)).unwrap();
        if out.episode_terminated()[0].get() || out.episode_truncated()[0].get() {
            ends.push(out.rewards()[0]);
        }
    }
    let record = DecodedRecord::from(p.snapshot(0).unwrap());
    assert_eq!(record.header().agents[0].illegal_actions, 0);
    assert_eq!(record.header().agents[1].illegal_actions, 0);
    ends
}

#[test]
fn test_bots_give_only_legal_orders_in_every_pairing() {
    for c in [cfg(10, 8), cfg(16, 16)] {
        for a in BOTS {
            for b in BOTS {
                let _ = play(&c, [a, b], 7, 250);
            }
        }
    }
}

#[test]
fn test_scripted_bots_are_deterministic_and_draw_nothing() {
    let c = cfg(12, 8);
    let mut p = pool(&c, 3);
    let mut r = [rng(1), rng(2)];
    for _ in 0..80 {
        let s = *p.state(0).unwrap();
        for bot in [Bot::Rush, Bot::Prospector] {
            for agent in [0, 1] {
                let x = row(&c, &s, agent, bot, &mut rng(11));
                let y = row(&c, &s, agent, bot, &mut rng(999));
                assert_eq!(x, y);
            }
        }
        let a = [
            row(&c, &s, 0, Bot::Rush, &mut r[0]),
            row(&c, &s, 1, Bot::Prospector, &mut r[1]),
        ]
        .concat();
        p.step(input(&a)).unwrap();
    }
}

/// `s` seen from the other side (as in the environment's tests).
fn mirrored(c: &RtsConfig, s: &RtsState) -> RtsState {
    let g = Grid::new(c.size).unwrap();
    let mut m = *s;
    for i in 0..UNIT_SLOTS {
        let mut u = s.units[i];
        if u.alive.get() {
            u.cell = g.mirror(u.cell);
        }
        m.units[(i + SIDE_SLOTS) % UNIT_SLOTS] = u;
    }
    m.ore = [s.ore[1], s.ore[0]];
    for cell in 0..g.cells() {
        m.terrain[usize::from(g.mirror(cell))] = s.terrain[usize::from(cell)];
        // The sides swap, so their explored bits swap too.
        let e = s.explored[usize::from(cell)];
        m.explored[usize::from(g.mirror(cell))] = ((e & 1) << 1) | ((e >> 1) & 1);
    }
    m
}

#[test]
fn test_scripted_bots_act_as_mirror_images_on_either_side() {
    let c = cfg(12, 8);
    let mut p = pool(&c, 5);
    let mut r = [rng(1), rng(2)];
    for k in 0..150 {
        let s = *p.state(0).unwrap();
        let m = mirrored(&c, &s);
        for bot in [Bot::Rush, Bot::Prospector] {
            // Windows are in the side's frame: identical indices.
            assert_eq!(
                row(&c, &s, 1, bot, &mut rng(0)),
                row(&c, &m, 0, bot, &mut rng(0)),
                "{bot:?} step {k}"
            );
            assert_eq!(
                row(&c, &s, 0, bot, &mut rng(0)),
                row(&c, &m, 1, bot, &mut rng(0)),
                "{bot:?} step {k}"
            );
        }
        let a = [
            row(&c, &s, 0, Bot::Rush, &mut r[0]),
            row(&c, &s, 1, Bot::Prospector, &mut r[1]),
        ]
        .concat();
        p.step(input(&a)).unwrap();
    }
}

fn unit(kind: Kind, cell: u16) -> Unit {
    Unit {
        cell,
        kind: kind as u8,
        hp: kind.stats().hp,
        alive: Bool8::TRUE,
        finished: Bool8::TRUE,
        carry: 0,
        counter: 0,
        progress: 0,
        producing: 0,
        pad: Reserved::default(),
        generation: 1,
    }
}

#[test]
fn test_rush_strikers_both_advance_down_a_shared_corridor() {
    // A one-cell corridor along row 5; two own Strikers in it, one behind
    // the other. Mobile units are not obstacles, so both walk east.
    let c = cfg(12, 4);
    let g = Grid::new(12).unwrap();
    let at = |x, y| g.index(x, y).unwrap();
    let mut s = RtsState::default();
    for cell in 0..g.cells() {
        let (_, y) = g.coords(cell);
        if y != 5 {
            s.terrain[usize::from(cell)] = Terrain::Wall as u8;
        }
    }
    s.units[0] = unit(Kind::Core, at(0, 5));
    s.units[1] = unit(Kind::Striker, at(2, 5));
    s.units[2] = unit(Kind::Striker, at(3, 5));
    s.units[SIDE_SLOTS] = unit(Kind::Core, at(11, 5));
    let a = row(&c, &s, 0, Bot::Rush, &mut rng(0));
    let east = i32::from(move_index(Team::Zero, Dir::East));
    for k in [1usize, 2] {
        let base = k * HEADS_PER_UNIT;
        assert_eq!((a[base], a[base + 1]), (1, east), "striker {k}");
    }
}

fn fnv(hash: u64, word: u64) -> u64 {
    word.to_le_bytes().iter().fold(hash, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Rush against Prospector, then Rush against `RandomLegal`, each for 600
/// decisions at size 16 with 16 slots: action rows, rewards and state
/// hashes folded. Both games reach combat (asserted), so the fighters'
/// orders are in the hash. Pinned on macOS arm64; any change to a bot, the
/// rules or the masks moves it.
const BOT_GOLDEN: u64 = 0xf917_a49f_18a8_a2a4;
/// A seed where Rush beats Prospector inside 600 decisions.
const GOLDEN_SEED: u64 = 14;

#[test]
fn test_bots_golden_is_pinned() {
    let c = cfg(16, 16);
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for opponent in [Bot::Prospector, Bot::RandomLegal] {
        let mut p = pool(&c, GOLDEN_SEED);
        let mut r = [rng(1), rng(2)];
        let mut damage = false;
        for _ in 0..600 {
            let s = *p.state(0).unwrap();
            damage |= s.units.iter().any(|u| {
                u.alive.get()
                    && u.finished.get()
                    && Kind::from_byte(u.kind).is_some_and(|k| u.hp < k.stats().hp)
            });
            let a = [
                row(&c, &s, 0, Bot::Rush, &mut r[0]),
                row(&c, &s, 1, opponent, &mut r[1]),
            ]
            .concat();
            hash = a
                .iter()
                .fold(hash, |h, &v| fnv(h, u64::from(v.cast_unsigned())));
            let out = p.step(input(&a)).unwrap();
            for x in out.rewards() {
                hash = fnv(hash, u64::from(x.to_bits()));
            }
            hash = fnv(hash, p.state_hash(0).unwrap().get());
        }
        assert!(damage, "Rush against {opponent:?}: no combat");
    }
    assert_eq!(hash, BOT_GOLDEN, "{hash:#x}");
}

/// Measurement, not a claim: the pairings' results over seeds, both seats.
/// Run with `--ignored --nocapture`.
#[test]
#[ignore = "a report, run on demand"]
#[expect(
    clippy::print_stdout,
    reason = "an ignored on-demand report whose output is the table; never runs in CI"
)]
fn report_bot_results() {
    let c = cfg(16, 16);
    for a in BOTS {
        for b in BOTS {
            let (mut win, mut loss, mut draw) = (0, 0, 0);
            for seed in 0..20 {
                for r in play(&c, [a, b], seed, 600) {
                    match r {
                        x if x > 0.0 => win += 1,
                        x if x < 0.0 => loss += 1,
                        _ => draw += 1,
                    }
                }
            }
            println!("{a:?} vs {b:?}: {win} won, {loss} lost, {draw} drawn or truncated");
        }
    }
}

/// Progress, not balance: in Rush against Rush the economy delivers ore
/// and a Foundry is finished. Two Drones meeting head on (a refused swap)
/// or two Drones splitting the ore between two Foundries (refused whole,
/// rule 11) would stall both, as they did before the yield and builder
/// rules.
#[test]
fn test_rush_economy_delivers_ore_and_finishes_a_foundry() {
    let c = cfg(16, 16);
    for seed in 0..6 {
        let mut p = pool(&c, seed);
        let mut r = [rng(1), rng(2)];
        let (mut delivered, mut foundry, mut prev) = (false, false, u32::MAX);
        for _ in 0..200 {
            let s = *p.state(0).unwrap();
            // Spending only lowers ore; a rise is a deposit.
            delivered |= s.ore[0] > prev;
            prev = s.ore[0];
            foundry |= s.units[..SIDE_SLOTS]
                .iter()
                .any(|u| u.alive.get() && u.kind == Kind::Foundry as u8 && u.finished.get());
            let a = [
                row(&c, &s, 0, Bot::Rush, &mut r[0]),
                row(&c, &s, 1, Bot::Rush, &mut r[1]),
            ]
            .concat();
            p.step(input(&a)).unwrap();
        }
        assert!(delivered, "seed {seed}: no ore delivered");
        assert!(foundry, "seed {seed}: no Foundry finished");
    }
}

/// An open 12-by-12 map with both Cores in opposite corners.
fn open_state() -> (RtsConfig, Grid, RtsState) {
    let c = cfg(12, 8);
    let g = Grid::new(12).unwrap();
    let mut s = RtsState::default();
    s.units[0] = unit(Kind::Core, g.index(0, 0).unwrap());
    s.units[SIDE_SLOTS] = unit(Kind::Core, g.index(11, 11).unwrap());
    (c, g, s)
}

/// Unit `u`'s order in an action row.
fn order(a: &[i32], u: usize) -> (i32, i32, i32) {
    let b = u * HEADS_PER_UNIT;
    (a[b], a[b + 1], a[b + 2])
}

#[test]
fn test_fighters_strike_an_adjacent_enemy_in_their_sides_frame() {
    let (c, g, mut s) = open_state();
    s.units[1] = unit(Kind::Striker, g.index(5, 5).unwrap());
    s.units[SIDE_SLOTS + 1] = unit(Kind::Striker, g.index(6, 5).unwrap());
    for bot in [Bot::Rush, Bot::Prospector] {
        // Side Zero's enemy is east on the map, side One's is west.
        let zero = row(&c, &s, 0, bot, &mut rng(0));
        let one = row(&c, &s, 1, bot, &mut rng(0));
        let east = i32::from(move_index(Team::Zero, Dir::East));
        let west = i32::from(move_index(Team::One, Dir::West));
        assert_eq!(order(&zero, 1), (order_type::ATTACK, east, 0), "{bot:?}");
        assert_eq!(order(&one, 1), (order_type::ATTACK, west, 0), "{bot:?}");
    }
}

#[test]
fn test_rush_foundry_orders_strikers_and_prospector_lancers() {
    let (c, g, mut s) = open_state();
    s.units[1] = unit(Kind::Foundry, g.index(3, 3).unwrap());
    s.ore = [50, 50];
    for (bot, product) in [(Bot::Rush, 1), (Bot::Prospector, 2)] {
        let (t, _, p) = order(&row(&c, &s, 0, bot, &mut rng(0)), 1);
        assert_eq!((t, p), (order_type::BUILD, product), "{bot:?}");
    }
}

#[test]
fn test_prospector_fighters_hold_at_three_and_advance_at_four() {
    for army in [3u16, 4] {
        let (c, g, mut s) = open_state();
        for k in 0..army {
            s.units[1 + usize::from(k)] = unit(Kind::Lancer, g.index(2 + k, 0).unwrap());
        }
        let a = row(&c, &s, 0, Bot::Prospector, &mut rng(0));
        for u in 1..=usize::from(army) {
            let want = if army < 4 {
                order_type::NOOP
            } else {
                order_type::MOVE
            };
            assert_eq!(order(&a, u).0, want, "army {army}, lancer {u}");
        }
    }
}

#[test]
fn test_a_carrier_goes_first_past_a_lower_slot_drone() {
    // A one-cell corridor along row 5. Drone 1 (empty) walks east to the
    // ore, Drone 2 (carrying) walks west to the Core; they meet head on and
    // the carrier goes first although its slot is higher: Drone 1 backs
    // off to the only free cell, west, and Drone 2 follows.
    let c = cfg(12, 4);
    let g = Grid::new(12).unwrap();
    let at = |x, y| g.index(x, y).unwrap();
    let mut s = RtsState::default();
    for cell in 0..g.cells() {
        if g.coords(cell).1 != 5 {
            s.terrain[usize::from(cell)] = Terrain::Wall as u8;
        }
    }
    s.terrain[usize::from(at(9, 5))] = Terrain::Ore as u8;
    s.units[0] = unit(Kind::Core, at(0, 5));
    s.units[1] = unit(Kind::Drone, at(4, 5));
    s.units[2] = Unit {
        carry: 1,
        ..unit(Kind::Drone, at(5, 5))
    };
    s.units[SIDE_SLOTS] = unit(Kind::Core, at(11, 5));
    let a = row(&c, &s, 0, Bot::Rush, &mut rng(0));
    let west = i32::from(move_index(Team::Zero, Dir::West));
    assert_eq!(order(&a, 1), (order_type::MOVE, west, 0));
    assert_eq!(order(&a, 2), (order_type::MOVE, west, 0));
}

#[test]
fn test_scripted_bots_leave_the_rng_untouched() {
    let c = cfg(12, 8);
    let mut p = pool(&c, 4);
    let mut r = [rng(1), rng(2)];
    for _ in 0..60 {
        let s = *p.state(0).unwrap();
        for bot in [Bot::Rush, Bot::Prospector] {
            for agent in [0, 1] {
                let mut used = rng(11);
                let _ = row(&c, &s, agent, bot, &mut used);
                let mut fresh = rng(11);
                let n = NonZeroU32::MAX;
                for _ in 0..4 {
                    assert_eq!(used.below_u32(n), fresh.below_u32(n), "{bot:?}");
                }
            }
        }
        let a = [
            row(&c, &s, 0, Bot::Rush, &mut r[0]),
            row(&c, &s, 1, Bot::RandomLegal, &mut r[1]),
        ]
        .concat();
        p.step(input(&a)).unwrap();
    }
}

#[test]
fn test_random_legal_rows_depend_on_the_seed() {
    let c = cfg(12, 8);
    let s = *pool(&c, 6).state(0).unwrap();
    let rows: Vec<_> = (0..8)
        .map(|k| row(&c, &s, 0, Bot::RandomLegal, &mut rng(k)))
        .collect();
    assert!(rows.iter().any(|x| *x != rows[0]));
}

#[test]
fn test_heuristic_opponent_is_rush() {
    let c = cfg(12, 8);
    let mut p = pool(&c, 8);
    let mut r = [rng(1), rng(2)];
    for _ in 0..80 {
        let s = *p.state(0).unwrap();
        for agent in [0, 1] {
            let mut heads = vec![0; usize::from(c.max_units) * HEADS_PER_UNIT];
            let mut w = ActionWriter::new(&mut heads, &mut []);
            <Rts as Heuristic>::act(
                &c,
                &s,
                AgentSlot::new(agent, 2).unwrap(),
                &mut rng(0),
                &mut w,
            );
            assert_eq!(heads, row(&c, &s, agent, Bot::Rush, &mut rng(0)));
        }
        let a = [
            row(&c, &s, 0, Bot::Prospector, &mut r[0]),
            row(&c, &s, 1, Bot::Rush, &mut r[1]),
        ]
        .concat();
        p.step(input(&a)).unwrap();
    }
}

#[test]
fn test_bots_write_a_noop_for_every_empty_slot() {
    let c = cfg(12, 8);
    let s = *pool(&c, 3).state(0).unwrap();
    for bot in BOTS {
        let mut heads = vec![7; usize::from(c.max_units) * HEADS_PER_UNIT];
        let mut w = ActionWriter::new(&mut heads, &mut []);
        bot.act(&c, &s, AgentSlot::new(0, 2).unwrap(), &mut rng(0), &mut w);
        for (u, unit) in s.units[..usize::from(c.max_units)].iter().enumerate() {
            if !unit.alive.get() {
                assert_eq!(
                    order(&heads, u),
                    (order_type::NOOP, 0, 0),
                    "{bot:?} slot {u}"
                );
            }
        }
    }
}
