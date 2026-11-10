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
use tessera_rl_core::{AgentSlot, Bool8, Reserved};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EnvPool, IllegalActionPolicy, Masked, PoolOptions,
    Reseed,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_rts::bots::Bot;
use tessera_rl_rts::env::{HEADS_PER_UNIT, SIDE_SLOTS, UNIT_SLOTS, move_index};
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

/// Rush against Prospector for 300 decisions: state hash, rewards and
/// action rows folded. Pinned on macOS arm64; any change to a bot, the
/// rules or the masks moves it.
const BOT_GOLDEN: u64 = 0x611b_7438_4117_04eb;

#[test]
fn test_bots_golden_is_pinned() {
    let c = cfg(12, 8);
    let mut p = pool(&c, 2026);
    let mut r = [rng(1), rng(2)];
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for _ in 0..300 {
        let s = *p.state(0).unwrap();
        let a = [
            row(&c, &s, 0, Bot::Rush, &mut r[0]),
            row(&c, &s, 1, Bot::Prospector, &mut r[1]),
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
    assert_eq!(hash, BOT_GOLDEN, "{hash:#x}");
}

/// Measurement, not a claim: the pairings' results over seeds, both seats.
/// Run with `--ignored --nocapture`.
#[test]
#[ignore = "a report, run on demand"]
#[expect(clippy::print_stdout, reason = "the report")]
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
