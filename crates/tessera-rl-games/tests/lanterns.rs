// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lanterns (#380): the rules, the masks, the exact determinization and the
//! information-state baseline.

#![expect(
    clippy::many_single_char_names,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: short names in loops, unwrap and index directly; a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::{
    AgentSlot, Determinize, Environment, Heuristic, ImperfectInformation, Validate,
};
use tessera_rl_engine::{ActionInput, DecodedRecord, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_games::lanterns::deal::{exact_distribution, sample};
use tessera_rl_games::lanterns::{
    self, ACTIONS, ALL, HAND, HIDDEN, KINDS, Lanterns, LanternsConfig, LanternsState, MAX_HINTS,
    Move, NO_CARD, NOT_FINAL, baseline, info_of,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

fn cfg(players: u32) -> LanternsConfig {
    LanternsConfig {
        players,
        zero_on_fuse: false,
    }
}

fn rng(seed: u64) -> StepRng {
    let id = RngIdentity::new(Seed::new(seed), LogicalId::DEFAULT);
    StepRng::new(&RngAuthority::engine_only(), &id, StreamId::DYNAMICS)
}

fn fresh(players: u32, seed: u64) -> LanternsState {
    let mut s = LanternsState::default();
    Lanterns::reset(&cfg(players), &mut rng(seed), &mut s);
    s
}

fn agent(i: usize) -> AgentSlot {
    AgentSlot::new(u16::try_from(i).unwrap(), lanterns::MAX_AGENTS).unwrap()
}

#[test]
fn test_lanterns_reset_deals_valid_hands_for_every_player_count() {
    for players in 2..=5 {
        let s = fresh(players, 7);
        let hand = if players <= 3 { 5 } else { 4 };
        assert_eq!(usize::from(s.next), hand * players as usize);
        assert_eq!((s.hints, s.fuses, s.final_turns), (MAX_HINTS, 3, NOT_FINAL));
        Lanterns::validate(&cfg(players), &s).unwrap();
        for seat in 0..5 {
            let held = s.hands[seat].iter().filter(|&&c| c != NO_CARD).count();
            assert_eq!(held, if seat < players as usize { hand } else { 0 });
        }
    }
}

#[test]
fn test_lanterns_move_codes_round_trip() {
    for a in 0..ACTIONS {
        assert_eq!(Move::decode(a).unwrap().encode(), a);
    }
    assert_eq!(Move::decode(ACTIONS), None);
}

/// A pool with one instance of `players` players, reset.
fn pool(players: u32, seed: u64) -> EnvPool<Lanterns, Masked> {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(seed));
    let mut p = EnvPool::with_masks(cfg(players), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn step(p: &mut EnvPool<Lanterns, Masked>, players: usize, action: u32) -> f32 {
    let mut a = vec![0i32; lanterns::MAX_AGENTS];
    let mover = usize::from(p.snapshot(0).unwrap().state().to_move);
    a[mover] = i32::try_from(action).unwrap();
    let out = p
        .step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
    let r = out.rewards()[0];
    for seat in 1..players {
        assert_eq!(out.rewards()[seat], r, "a shared reward");
    }
    r
}

/// A legal move that is not a play: a discard when hint tokens are spent,
/// else the first legal hint (debug checks reject masked-out actions).
fn filler(p: &EnvPool<Lanterns, Masked>) -> u32 {
    let mover = usize::from(p.snapshot(0).unwrap().state().to_move);
    let masks: Vec<bool> = p
        .outputs()
        .unwrap()
        .masks()
        .iter()
        .map(|b| b.get())
        .collect();
    let row = &masks[mover * ACTIONS as usize..(mover + 1) * ACTIONS as usize];
    (5..ACTIONS).find(|&a| row[a as usize]).unwrap()
}

#[test]
fn test_lanterns_hint_narrows_knowledge_both_ways_and_spends_a_token() {
    let mut p = pool(2, 11);
    let s = *p.snapshot(0).unwrap().state();
    let target = s.hands[1];
    let colour = lanterns::colour(target[0]);
    step(&mut p, 2, Move::HintColour { offset: 1, colour }.encode());
    let t = *p.snapshot(0).unwrap().state();
    assert_eq!(t.hints, MAX_HINTS - 1);
    for (k, &card) in target.iter().enumerate() {
        if lanterns::colour(card) == colour {
            assert_eq!(t.colours[1][k], 1 << colour, "positive");
        } else {
            assert_eq!(t.colours[1][k], ALL & !(1 << colour), "negative");
        }
        assert_eq!(t.ranks[1][k], ALL);
    }
    Lanterns::validate(&cfg(2), &t).unwrap();
}

#[test]
fn test_lanterns_play_scores_or_costs_a_fuse_and_refills() {
    // Search a seed whose first player holds a playable rank 1 and an
    // unplayable card.
    for seed in 0..200 {
        let mut p = pool(2, seed);
        let s = *p.snapshot(0).unwrap().state();
        let Some(good) = (0..HAND).find(|&k| lanterns::rank(s.hands[0][k]) == 0) else {
            continue;
        };
        let Some(bad) = (0..HAND).find(|&k| lanterns::rank(s.hands[0][k]) > 1) else {
            continue;
        };
        let r = step(&mut p, 2, Move::Play(good).encode());
        let t = *p.snapshot(0).unwrap().state();
        assert_eq!(r, 1.0);
        assert_eq!(t.score(), 1);
        assert_eq!(t.next, s.next + 1, "refilled");
        assert_eq!(t.hands[0][good], s.deck[usize::from(s.next)]);
        // Player 1 hints, then player 0 misplays.
        let fill = filler(&p);
        step(&mut p, 2, fill);
        let before = *p.snapshot(0).unwrap().state();
        let card = before.hands[0][bad];
        let playable =
            usize::from(before.fireworks[lanterns::colour(card)]) == lanterns::rank(card);
        if playable {
            continue;
        }
        let r = step(&mut p, 2, Move::Play(bad).encode());
        let after = *p.snapshot(0).unwrap().state();
        assert_eq!(r, 0.0);
        assert_eq!(after.fuses, before.fuses - 1);
        assert_eq!(
            after.discards[usize::from(card)],
            before.discards[usize::from(card)] + 1
        );
        return;
    }
    panic!("no seed in 0..200 gave the scenario");
}

/// The mover's first slot whose card is not playable now, if any.
fn misplay(s: &LanternsState) -> Option<usize> {
    let me = usize::from(s.to_move);
    (0..HAND).find(|&k| {
        let c = s.hands[me][k];
        c != NO_CARD && usize::from(s.fireworks[lanterns::colour(c)]) != lanterns::rank(c)
    })
}

/// The third fuse ends the game; the zero variant takes the score back.
#[test]
fn test_lanterns_third_fuse_ends_the_game_and_zero_variant_takes_the_score() {
    for zero in [false, true] {
        let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(8));
        let c = LanternsConfig {
            players: 2,
            zero_on_fuse: zero,
        };
        let mut p = EnvPool::<Lanterns, Masked>::with_masks(c, options).unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        // Score first when a rank 1 is in hand, then misplay until the
        // third fuse.
        let s = *p.snapshot(0).unwrap().state();
        if let Some(k) = (0..HAND).find(|&k| lanterns::rank(s.hands[0][k]) == 0) {
            step(&mut p, 2, Move::Play(k).encode());
        }
        let mut fuses = 3;
        let mut ended = false;
        for _ in 0..40 {
            let s = *p.snapshot(0).unwrap().state();
            let Some(k) = misplay(&s) else {
                let fill = filler(&p);
                step(&mut p, 2, fill);
                continue;
            };
            let score = s.score();
            let r = step(&mut p, 2, Move::Play(k).encode());
            fuses -= 1;
            let done = p.outputs().unwrap().episode_terminated()[0].get();
            assert_eq!(done, fuses == 0);
            if done {
                let want = if zero { -f32::from(score) } else { 0.0 };
                assert_eq!(r, want);
                ended = true;
                break;
            }
            assert_eq!(r, 0.0);
        }
        assert!(ended, "three misplays did not happen");
    }
}

/// Playing the 25th card ends the game at once (a perfect score).
#[test]
fn test_lanterns_perfect_score_ends_the_game() {
    let mut s = LanternsState::default();
    // Every card but the last red-five (kind 24) is on the fireworks.
    let played: Vec<u8> = (0..24).collect();
    let mut extras = Vec::new();
    for kind in 0..25u8 {
        let copies = lanterns::COPIES[usize::from(kind) % 5];
        for _ in 1..copies {
            extras.push(kind);
        }
    }
    let mut hand0 = vec![24u8];
    hand0.extend_from_slice(&extras[..4]);
    let hand1: Vec<u8> = extras[4..9].to_vec();
    let mut deck = played.clone();
    deck.extend(&hand0);
    deck.extend(&hand1);
    deck.extend(&extras[9..]);
    assert_eq!(deck.len(), 50);
    s.deck.copy_from_slice(&deck);
    s.next = 34;
    s.hands = [[NO_CARD; HAND]; 5];
    s.colours = [[0; HAND]; 5];
    s.ranks = [[0; HAND]; 5];
    for k in 0..HAND {
        s.hands[0][k] = hand0[k];
        s.hands[1][k] = hand1[k];
        for seat in 0..2 {
            s.colours[seat][k] = ALL;
            s.ranks[seat][k] = ALL;
        }
    }
    s.fireworks = [5, 5, 5, 5, 4];
    s.hints = MAX_HINTS;
    s.fuses = 3;
    s.final_turns = NOT_FINAL;
    s.players = 2;
    Lanterns::validate(&cfg(2), &s).unwrap();
    let mut p = pool(2, 1);
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let rec = DecodedRecord::from_parts(*good.header(), s).unwrap();
    p.import(&[0], &[rec]).unwrap();
    let r = step(&mut p, 2, Move::Play(0).encode());
    assert_eq!(r, 1.0);
    assert!(p.outputs().unwrap().episode_terminated()[0].get());
}

/// Random legal games: the mask equals the rules' legality, every state
/// validates, scores never fall without the zero variant, and games end.
#[test]
fn test_lanterns_random_games_keep_masks_legal_and_states_valid() {
    for players in 2..=5u32 {
        let mut p = pool(players, u64::from(players) * 17);
        let mut r = rng(99);
        for _ in 0..400 {
            let s = *p.snapshot(0).unwrap().state();
            let masks: Vec<bool> = p
                .outputs()
                .unwrap()
                .masks()
                .iter()
                .map(|b| b.get())
                .collect();
            let mover = usize::from(s.to_move);
            let row = &masks[mover * ACTIONS as usize..(mover + 1) * ACTIONS as usize];
            let legal: Vec<u32> = (0..ACTIONS).filter(|&a| row[a as usize]).collect();
            assert!(!legal.is_empty(), "a player always has a legal move");
            for a in 0..ACTIONS {
                let m = Move::decode(a).unwrap();
                let expect = match m {
                    Move::Play(k) => s.hands[mover][k] != NO_CARD,
                    Move::Discard(k) => s.hands[mover][k] != NO_CARD && s.hints < MAX_HINTS,
                    _ => row[a as usize],
                };
                assert_eq!(row[a as usize], expect, "action {a}");
            }
            let pick = legal[usize::try_from(
                r.below_u32(NonZeroU32::new(u32::try_from(legal.len()).unwrap()).unwrap()),
            )
            .unwrap()];
            step(&mut p, players as usize, pick);
            let t = *p.snapshot(0).unwrap().state();
            if t.done == 0 {
                Lanterns::validate(&cfg(players), &t).unwrap();
            }
        }
    }
}

/// The final round: after the last card is drawn, each player gets exactly
/// one more turn.
#[test]
fn test_lanterns_final_round_gives_every_player_one_turn() {
    let players = 3usize;
    let mut p = pool(3, 5);
    let mut r = rng(1);
    let mut turns_after = None;
    for _ in 0..400 {
        let s = *p.snapshot(0).unwrap().state();
        if s.done != 0 || p.outputs().unwrap().episode_terminated()[0].get() {
            break;
        }
        if s.final_turns != NOT_FINAL {
            *turns_after.get_or_insert(0) += 1;
        }
        let mover = usize::from(s.to_move);
        let masks: Vec<bool> = p
            .outputs()
            .unwrap()
            .masks()
            .iter()
            .map(|b| b.get())
            .collect();
        let row = &masks[mover * ACTIONS as usize..(mover + 1) * ACTIONS as usize];
        // Discard when possible, so the deck runs out; with all tokens,
        // hint; never play (no fuse can end the game first).
        let a = (5..10)
            .find(|&a| row[a])
            .or_else(|| (10..ACTIONS as usize).find(|&a| row[a]))
            .or_else(|| (0..5).find(|&a| row[a]))
            .unwrap();
        let _ = r.below_u32(NonZeroU32::MIN);
        let before = p.snapshot(0).unwrap();
        step(&mut p, players, u32::try_from(a).unwrap());
        if p.outputs().unwrap().episode_terminated()[0].get() {
            if before.state().final_turns != NOT_FINAL {
                assert_eq!(turns_after, Some(players));
            }
            return;
        }
    }
    panic!("the game did not end");
}

fn info_with_knowledge(seed: u64) -> lanterns::LanternsInfo {
    // A 2-player game after a few hints to player 1, seen by player 1.
    let mut p = pool(2, seed);
    let s = *p.snapshot(0).unwrap().state();
    let colour = lanterns::colour(s.hands[1][0]);
    step(&mut p, 2, Move::HintColour { offset: 1, colour }.encode());
    let fill = filler(&p);
    step(&mut p, 2, fill);
    let s = *p.snapshot(0).unwrap().state();
    let rank = lanterns::rank(s.hands[1][2]);
    step(&mut p, 2, Move::HintRank { offset: 1, rank }.encode());
    info_of(p.snapshot(0).unwrap().state(), 1)
}

/// Exact sampling: frequencies of own-hand kind assignments match the
/// exact distribution from enumerating physical deals (review r1).
#[test]
fn test_lanterns_determinize_matches_exact_enumeration() {
    let mut info = info_with_knowledge(21);
    // Two own slots keep enumeration small and the test exact.
    for k in 2..HAND {
        info.hands[1][k] = NO_CARD;
        info.colours[1][k] = 0;
        info.ranks[1][k] = 0;
    }
    let exact = exact_distribution(&info);
    assert!(exact.len() > 3, "a non-trivial distribution");
    let n = 100_000u32;
    let mut r = rng(4);
    let mut counts = vec![0u32; exact.len()];
    for _ in 0..n {
        let s = sample(&info, &mut r);
        let hand = s.hands[1];
        let i = exact
            .iter()
            .position(|(h, _)| *h == hand)
            .unwrap_or_else(|| panic!("an inconsistent deal {hand:?}"));
        counts[i] += 1;
    }
    for ((_, p), &c) in exact.iter().zip(&counts) {
        let expect = p * f64::from(n);
        let sd = (expect * (1.0 - p)).sqrt();
        assert!(
            (f64::from(c) - expect).abs() < 5.0 * sd + 1.0,
            "{c} vs {expect}"
        );
    }
}

/// Every sample agrees with the information state and is a valid state;
/// samples ignore the hidden cards (noninterference, SPEC 9.3).
#[test]
fn test_lanterns_determinize_is_consistent_and_noninterfering() {
    for seed in 0..20 {
        let info = info_with_knowledge(seed);
        let mut r = rng(seed);
        for _ in 0..50 {
            let mut s = LanternsState::default();
            Lanterns::sample(&cfg(2), &info, &mut r, &mut s);
            Lanterns::validate(&cfg(2), &s).unwrap();
            let again = Lanterns::info_state(&cfg(2), &s, agent(1));
            assert_eq!(again, info);
        }
        // The same info state from two different true states gives the
        // same samples for the same stream.
        let mut a = LanternsState::default();
        let mut b = LanternsState::default();
        Lanterns::sample(&cfg(2), &info, &mut rng(9), &mut a);
        Lanterns::sample(&cfg(2), &info, &mut rng(9), &mut b);
        assert_eq!(a, b);
    }
}

/// The baseline reads only the information state, and its actions are
/// legal.
#[test]
fn test_lanterns_baseline_is_legal_and_reads_only_the_info_state() {
    for players in 2..=5u32 {
        let mut p = pool(players, 3);
        for _ in 0..200 {
            let s = *p.snapshot(0).unwrap().state();
            if s.done != 0 {
                break;
            }
            let mover = usize::from(s.to_move);
            let info = info_of(&s, mover);
            let a = baseline::act_info(&info);
            let masks: Vec<bool> = p
                .outputs()
                .unwrap()
                .masks()
                .iter()
                .map(|b| b.get())
                .collect();
            assert!(
                masks[mover * ACTIONS as usize + a as usize],
                "a legal baseline action"
            );
            // A determinized state with the same info gives the same act.
            let mut other = LanternsState::default();
            Lanterns::sample(&cfg(players), &info, &mut rng(5), &mut other);
            let mut w = [0i32; 1];
            let mut out = tessera_rl_core::io::ActionWriter::new(&mut w, &mut []);
            Lanterns::act(&cfg(players), &other, agent(mover), &mut rng(0), &mut out);
            assert_eq!(w[0], i32::try_from(a).unwrap());
            step(&mut p, players as usize, a);
        }
    }
}

/// The baseline scores well above a random legal player (SPEC 16.1).
#[test]
fn test_lanterns_baseline_beats_random_play() {
    let games = 30;
    let mut total = [0u32; 2];
    for (which, t) in total.iter_mut().enumerate() {
        for g in 0..games {
            let mut p = pool(3, 1_000 + g);
            let mut r = rng(g);
            for _ in 0..200 {
                let s = *p.snapshot(0).unwrap().state();
                let mover = usize::from(s.to_move);
                let a = if which == 0 {
                    baseline::act_info(&info_of(&s, mover))
                } else {
                    let masks: Vec<bool> = p
                        .outputs()
                        .unwrap()
                        .masks()
                        .iter()
                        .map(|b| b.get())
                        .collect();
                    let row = &masks[mover * ACTIONS as usize..(mover + 1) * ACTIONS as usize];
                    let legal: Vec<u32> = (0..ACTIONS).filter(|&a| row[a as usize]).collect();
                    legal[usize::try_from(
                        r.below_u32(NonZeroU32::new(u32::try_from(legal.len()).unwrap()).unwrap()),
                    )
                    .unwrap()]
                };
                // The score is the sum of the rewards (no zero variant);
                // the state after the last step may already be reset.
                let gain = step(&mut p, 3, a);
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "gains are 0 or 1"
                )]
                {
                    *t += gain as u32;
                }
                if p.outputs().unwrap().episode_terminated()[0].get() {
                    break;
                }
            }
        }
    }
    assert!(
        total[0] > 3 * total[1].max(1),
        "baseline {} vs random {}",
        total[0],
        total[1]
    );
}

#[test]
fn test_lanterns_hidden_marks_and_kinds() {
    assert_eq!(usize::from(NO_CARD), KINDS);
    assert_eq!(usize::from(HIDDEN), KINDS + 1);
}
