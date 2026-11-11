// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scripted opponent option (RTS stage 5c, plan r2 O1 to O4): with an
//! opponent the pool has one agent, the learner in slot 0; the opponent's
//! orders are its bot's on its own side view, so the learner's game equals a
//! two-agent pool whose slot 1 is driven by the same bot; nothing of the
//! scripted seat is surfaced; snapshot and restore continue the game.

#![expect(
    clippy::unwrap_used,
    clippy::many_single_char_names,
    reason = "test fixtures"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Environment};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, IllegalActionPolicy, Masked, PoolOptions, Reseed,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_rts::bots::Bot;
use tessera_rl_rts::env::HEADS_PER_UNIT;
use tessera_rl_rts::{Rts, RtsConfig, RtsState};

type Pool = EnvPool<Rts, Masked>;

fn cfg(opponent: u8, swap: bool) -> RtsConfig {
    RtsConfig {
        size: 12,
        decision_ticks: 4,
        max_units: 8,
        step_limit: 300,
        seat_swap: swap,
        opponent,
    }
}

fn pool(c: &RtsConfig, seed: u64) -> Pool {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        illegal_action: IllegalActionPolicy::Noop,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(seed))
    };
    let mut p: Pool = EnvPool::with_masks(c.clone(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn row(c: &RtsConfig, s: &RtsState, agent: u16, bot: Bot, r: &mut StepRng) -> Vec<i32> {
    let mut heads = vec![0; usize::from(c.max_units) * HEADS_PER_UNIT];
    let mut w = ActionWriter::new(&mut heads, &mut []);
    bot.act(c, s, AgentSlot::new(agent, 2).unwrap(), r, &mut w);
    heads
}

fn learner_rng() -> StepRng {
    let id = RngIdentity::new(Seed::new(77), LogicalId::new(0));
    StepRng::new(&RngAuthority::engine_only(), &id, StreamId::OPPONENT)
}

/// Slot 0's row of a per-agent output of a one-instance pool (two agent
/// rows, slot 0 first).
fn slot0<T: Clone>(rows: &[T]) -> Vec<T> {
    rows.get(..rows.len() / 2).unwrap().to_vec()
}

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

#[test]
fn test_opponent_one_agent_and_none_keeps_two() {
    assert_eq!(Rts::initial_agents(&cfg(0, false)), 2);
    for code in 1..=3 {
        assert_eq!(Rts::initial_agents(&cfg(code, false)), 1);
    }
}

/// The learner (Random-legal on its own generator) against a deterministic
/// opponent equals a two-agent game whose slot 1 plays that bot: state,
/// rewards, and the learner's observation leaves and mask row.
#[test]
fn test_opponent_game_equals_a_two_agent_pool_driven_by_the_bot() {
    for (code, bot) in [(1u8, Bot::Rush), (2, Bot::Prospector)] {
        for swap in [false, true] {
            let c1 = cfg(code, swap);
            let c2 = cfg(0, swap);
            let mut one = pool(&c1, 5);
            let mut two = pool(&c2, 5);
            let (mut r1, mut r2) = (learner_rng(), learner_rng());
            let n = usize::from(c1.max_units) * HEADS_PER_UNIT;
            for k in 0..120 {
                let s1 = *one.state(0).unwrap();
                let s2 = *two.state(0).unwrap();
                assert_eq!(s1.units, s2.units, "{bot:?} swap {swap} step {k}");
                assert_eq!(s1.ore, s2.ore);
                let learner = row(&c1, &s1, 0, Bot::RandomLegal, &mut r1);
                let _ = row(&c2, &s2, 0, Bot::RandomLegal, &mut r2);
                let mut a1 = learner.clone();
                a1.resize(2 * n, 0); // slot 1's row is ignored: not an agent
                let mut a2 = learner;
                a2.extend(row(&c2, &s2, 1, bot, &mut learner_rng()));
                let o1 = one.step(input(&a1)).unwrap();
                let rew1 = o1.rewards()[0];
                let done1 = o1.episode_terminated()[0].get() || o1.episode_truncated()[0].get();
                let obs1: Vec<Vec<u8>> =
                    (0..2).map(|l| slot0(o1.observation(l).unwrap())).collect();
                let mask1 = slot0(o1.masks());
                let o2 = two.step(input(&a2)).unwrap();
                assert_eq!(rew1, o2.rewards()[0]);
                // R2: the learner's own outputs, observation leaves and
                // mask row, equal the two-agent pool's slot 0 at every
                // step: the opponent changes nothing the learner observes
                // beyond what the same bot in slot 1 does (the stage 3
                // noninterference property, carried to the opponent case).
                for (l, want) in obs1.iter().enumerate() {
                    assert_eq!(
                        *want,
                        slot0(o2.observation(l).unwrap()),
                        "leaf {l} step {k}"
                    );
                }
                assert_eq!(mask1, slot0(o2.masks()), "mask step {k}");
                if done1 {
                    break;
                }
            }
        }
    }
}

#[test]
fn test_opponent_seat_is_never_alive_nor_rewarded() {
    let c = cfg(1, false);
    let mut p = pool(&c, 3);
    let n = usize::from(c.max_units) * HEADS_PER_UNIT;
    for _ in 0..200 {
        let out = p.step(input(&vec![0; 2 * n])).unwrap();
        assert!(!out.alive()[1].get());
        assert_eq!(out.rewards()[1], 0.0);
        if out.episode_terminated()[0].get() || out.episode_truncated()[0].get() {
            break;
        }
    }
}

#[test]
fn test_opponent_game_continues_through_snapshot_and_restore() {
    let c = cfg(3, false);
    let mut p = pool(&c, 9);
    let n = usize::from(c.max_units) * HEADS_PER_UNIT;
    let mut r = learner_rng();
    let play = |p: &mut Pool, r: &mut StepRng, k: usize| {
        for _ in 0..k {
            let s = *p.state(0).unwrap();
            let mut a = row(&c, &s, 0, Bot::RandomLegal, r);
            a.resize(2 * n, 0);
            p.step(input(&a)).unwrap();
        }
        p.state_hash(0).unwrap()
    };
    play(&mut p, &mut r, 30);
    let snap = p.snapshot(0).unwrap();
    let saved = r.clone();
    let first = play(&mut p, &mut r, 30);
    p.restore(0, &snap).unwrap();
    let mut r2 = saved;
    assert_eq!(play(&mut p, &mut r2, 30), first);
}

/// O1: a file persisted before `opponent` existed (written by main at
/// b395ee6 with size 12, 8 units, limit 300, seed 5) is refused by its
/// config hash, typed, and never misread; the same pool's own file decodes.
#[test]
fn test_opponent_pre_field_persisted_file_is_refused_by_config_hash() {
    let c = cfg(0, false);
    let p = pool(&c, 5);
    let old = include_bytes!("fixtures/rts_pre_opponent.tessera");
    let got = p.decode(old, tessera_rl_engine::DecodeLimits::default());
    assert!(
        matches!(
            got,
            Err(tessera_rl_engine::PersistError::Incompatible {
                field: "config_hash"
            })
        ),
        "{got:?}"
    );
    let own = p.encode(&[0]).unwrap();
    assert!(
        p.decode(&own, tessera_rl_engine::DecodeLimits::default())
            .is_ok()
    );
}

fn fnv(hash: u64, word: u64) -> u64 {
    word.to_le_bytes().iter().fold(hash, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn protocol(swap: bool) -> RtsConfig {
    protocol_against(swap, 1)
}

fn protocol_against(swap: bool, opponent: u8) -> RtsConfig {
    RtsConfig {
        size: 16,
        decision_ticks: 4,
        max_units: 16,
        step_limit: 600,
        seat_swap: swap,
        opponent,
    }
}

/// Plays the learner as `bot` (Random-legal on its own generator, or a
/// scripted bot) against the Rush opponent at the protocol config; returns
/// the fold of rewards and state hashes and the learner's final reward sum.
fn versus_rush(swap: bool, seed: u64, bot: Bot) -> (u64, f32) {
    versus(&protocol(swap), seed, bot)
}

#[expect(
    clippy::indexing_slicing,
    reason = "test fixture: one instance, slot 0"
)]
fn versus(c: &RtsConfig, seed: u64, bot: Bot) -> (u64, f32) {
    let mut p = pool(c, seed);
    let n = usize::from(c.max_units) * HEADS_PER_UNIT;
    let mut r = learner_rng();
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut total = 0.0f32;
    for _ in 0..600 {
        let s = *p.state(0).unwrap();
        let mut a = row(c, &s, 0, bot, &mut r);
        a.resize(2 * n, 0);
        let out = p.step(input(&a)).unwrap();
        total += out.rewards()[0];
        let done = out.episode_terminated()[0].get() || out.episode_truncated()[0].get();
        hash = fnv(hash, u64::from(out.rewards()[0].to_bits()));
        hash = fnv(hash, p.state_hash(0).unwrap().get());
        if done {
            break;
        }
    }
    (hash, total)
}

/// O6: a fixed learner stream (Random-legal on a fixed generator) against
/// Rush at the protocol config, from both sides, folded. Pinned on macOS
/// arm64 after the stage 5a freeze; needs the Linux x86-64 check (C6).
const OPPONENT_GOLDEN: u64 = 0x00b0_d6a1_5b81_ced4;

#[test]
fn test_opponent_golden_is_pinned() {
    let (a, _) = versus_rush(false, 9, Bot::RandomLegal);
    let (b, _) = versus_rush(true, 9, Bot::RandomLegal);
    let got = fnv(a, b);
    assert_eq!(got, OPPONENT_GOLDEN, "{got:#x}");
}

/// Documentation for the protocol (TR 70057, item 1), not an input to its
/// N: with a deterministic, seat-symmetric learner (a scripted bot), the
/// seat-swapped game on one seed is the exact mirror of the first, so the
/// learner's outcome is the same from either side. The protocol therefore
/// uses one game per evaluation seed.
#[test]
fn test_opponent_seat_swapped_game_of_a_symmetric_learner_is_a_mirror() {
    for seed in [1u64, 2, 3, 9] {
        let (_, zero) = versus_rush(false, seed, Bot::Prospector);
        let (_, one) = versus_rush(true, seed, Bot::Prospector);
        assert_eq!(zero.to_bits(), one.to_bits(), "seed {seed}");
    }
}

/// O4 and R1: the Random-legal opponent (code 3) draws from the step
/// stream once per decision before the transition; the same seed gives the
/// same game, and a scripted learner against it is pinned from both sides.
const OPPONENT_RANDOM_GOLDEN: u64 = 0xf6ed_2423_7340_3f43;

#[test]
fn test_opponent_random_legal_is_deterministic_and_pinned() {
    let a = versus(&protocol_against(false, 3), 9, Bot::Rush);
    let b = versus(&protocol_against(true, 3), 9, Bot::Rush);
    assert_eq!(a, versus(&protocol_against(false, 3), 9, Bot::Rush));
    assert_eq!(b, versus(&protocol_against(true, 3), 9, Bot::Rush));
    assert_ne!(a.0, versus(&protocol_against(false, 3), 10, Bot::Rush).0);
    let got = fnv(a.0, b.0);
    assert_eq!(got, OPPONENT_RANDOM_GOLDEN, "{got:#x}");
}
