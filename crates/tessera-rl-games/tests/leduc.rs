// SPDX-License-Identifier: MIT OR Apache-2.0

//! Leduc hold'em through the engine: the deal, betting rounds and raise
//! limits, the public card drawn at step time from the cards left,
//! folds and showdowns, the mask and the no-op, the information state and
//! its sampler, import validation and the scripted baseline.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::{AgentSlot, Determinize, ImperfectInformation};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, Masked, PoolOptions, Reseed,
};
use tessera_rl_games::leduc::{CALL, FOLD, MAX_CONTRIBUTION, NO_CARD, RAISE, draw};
use tessera_rl_games::{Leduc, LeducConfig, LeducState};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

const CFG: LeducConfig = LeducConfig {};

fn pool(seed: u64) -> EnvPool<Leduc, Masked> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(seed))
    };
    let mut p = EnvPool::with_masks(CFG, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

/// Plays `action` for the side to move; the other slot sends check or call.
fn play(p: &mut EnvPool<Leduc, Masked>, action: u32) {
    let me = p.state(0).unwrap().mover();
    let mut act = [1, 1];
    act[me] = i32::try_from(action).unwrap();
    p.step(ActionInput {
        discrete: &act,
        continuous: &[],
    })
    .unwrap();
}

fn rewards(p: &EnvPool<Leduc, Masked>) -> [f32; 2] {
    let r = p.outputs().rewards();
    [r[0], r[1]]
}

fn mask(p: &EnvPool<Leduc, Masked>, agent: usize) -> Vec<bool> {
    p.outputs().masks()[agent * 3..agent * 3 + 3]
        .iter()
        .map(|f| f.get())
        .collect()
}

fn terminated(p: &EnvPool<Leduc, Masked>) -> bool {
    p.outputs().episode_terminated()[0].get()
}

fn agent(i: u16) -> AgentSlot {
    AgentSlot::new(i, 2).unwrap()
}

fn rng(seed: u64, salt_word: u64) -> StepRng {
    let id = RngIdentity::from_parts(
        Seed::new(seed),
        tessera_rl_num::identity::BranchSalt::new(salt_word),
        LogicalId::DEFAULT,
        0,
        0,
    );
    StepRng::new(&RngAuthority::engine_only(), &id, StreamId::DYNAMICS)
}

#[test]
fn test_leduc_reset_deals_two_private_cards_and_antes_one_each() {
    for seed in 0..64 {
        let p = pool(seed);
        let s = *p.state(0).unwrap();
        assert!(s.cards.iter().all(|&c| c < 3));
        assert_eq!(
            (s.public, s.round, s.contributions, s.to_move),
            (NO_CARD, 0, [1, 1], 0)
        );
        assert!(p.outputs().acting()[0].get());
        assert!(!p.outputs().acting()[1].get());
    }
}

#[test]
fn test_leduc_deal_frequencies_follow_the_deck() {
    // Two copies each: a pair of private cards has probability 1/5.
    let mut pairs = 0;
    for seed in 0..2000 {
        let s = *pool(seed).state(0).unwrap();
        pairs += u32::from(s.cards[0] == s.cards[1]);
    }
    assert!((300..500).contains(&pairs), "{pairs}");
}

#[test]
fn test_leduc_draw_never_exceeds_two_copies_of_a_rank() {
    let mut r = rng(4, 9);
    for _ in 0..500 {
        assert_ne!(draw(&mut r, &[2, 2]), 2);
    }
    let mut seen = [0_u32; 3];
    for _ in 0..3000 {
        seen[draw(&mut r, &[0, 1]) as usize] += 1;
    }
    // One jack, one queen and two kings left.
    assert!((600..900).contains(&seen[0]) && (600..900).contains(&seen[1]));
    assert!((1300..1700).contains(&seen[2]), "{seen:?}");
}

#[test]
fn test_leduc_check_check_deals_the_public_card_from_the_cards_left() {
    for seed in 0..64 {
        let mut p = pool(seed);
        play(&mut p, CALL);
        assert_eq!(p.state(0).unwrap().round, 0);
        play(&mut p, CALL);
        let s = *p.state(0).unwrap();
        assert_eq!((s.round, s.to_move, s.raises), (1, 0, 0));
        assert!(s.public < 3);
        let copies = s.cards.iter().filter(|&&c| c == s.public).count();
        assert!(copies < 2, "a third copy of a rank");
        assert!(!terminated(&p));
    }
}

#[test]
fn test_leduc_raise_sizes_are_two_then_four_and_two_raises_per_round() {
    let mut p = pool(3);
    play(&mut p, RAISE);
    assert_eq!(p.state(0).unwrap().contributions, [3, 1]);
    assert_eq!(mask(&p, 1), vec![true, true, true]);
    play(&mut p, RAISE);
    assert_eq!(p.state(0).unwrap().contributions, [3, 5]);
    assert_eq!(mask(&p, 0), vec![true, true, false], "two raises made");
    play(&mut p, CALL);
    let s = *p.state(0).unwrap();
    assert_eq!((s.round, s.contributions), (1, [5, 5]));
    assert_eq!(mask(&p, 0), vec![false, true, true], "no bet to answer");
    play(&mut p, RAISE);
    assert_eq!(p.state(0).unwrap().contributions, [9, 5]);
    play(&mut p, RAISE);
    play(&mut p, CALL);
    assert!(terminated(&p));
    assert_eq!(p.state(0).unwrap().contributions, [MAX_CONTRIBUTION; 2]);
}

#[test]
fn test_leduc_fold_loses_the_folders_contribution() {
    let mut p = pool(5);
    play(&mut p, RAISE);
    play(&mut p, FOLD);
    assert!(terminated(&p));
    assert_eq!(rewards(&p), [1.0 / 13.0, -1.0 / 13.0]);
}

#[test]
fn test_leduc_noop_is_check_or_call_and_fold_needs_a_bet() {
    use tessera_rl_core::ActionMask;
    use tessera_rl_core::io::ActionWriter;
    let mut p = pool(5);
    assert_eq!(mask(&p, 0), vec![false, true, true]);
    let mut a = [7_i32];
    Leduc::noop(
        &CFG,
        p.state(0).unwrap(),
        agent(0),
        &mut ActionWriter::new(&mut a, &mut []),
    );
    assert_eq!(a, [i32::try_from(CALL).unwrap()]);
    play(&mut p, CALL);
    assert_eq!(p.state(0).unwrap().history, [0b10, 0], "a check");
    assert_eq!(mask(&p, 1), vec![false, true, true]);
}

#[test]
fn test_leduc_showdown_pays_the_pair_then_the_higher_rank() {
    let base = LeducState {
        public: 1,
        round: 1,
        contributions: [5, 5],
        ..LeducState::default()
    };
    let with = |cards| LeducState { cards, ..base };
    assert_eq!(with([1, 2]).showdown(), 1, "a pair beats a king");
    assert_eq!(with([2, 0]).showdown(), 1);
    assert_eq!(with([0, 2]).showdown(), -1);
    assert_eq!(with([2, 2]).showdown(), 0);
    for seed in 0..64 {
        let mut p = pool(seed);
        for _ in 0..4 {
            play(&mut p, CALL);
        }
        assert!(terminated(&p));
        let s = *p.state(0).unwrap();
        let expected = f32::from(i8::try_from(s.showdown()).unwrap()) / 13.0;
        assert_eq!(rewards(&p), [expected, -expected]);
    }
}

#[test]
fn test_leduc_info_state_hides_only_the_opponents_card() {
    let mut p = pool(8);
    play(&mut p, RAISE);
    play(&mut p, CALL);
    let s = *p.state(0).unwrap();
    for me in 0..2_u16 {
        let info = Leduc::info_state(&CFG, &s, agent(me));
        assert_eq!(info.card, s.cards[usize::from(me)]);
        assert_eq!(info.me, u32::from(me));
        let mut other = s;
        other.cards[1 - usize::from(me)] = (s.cards[1 - usize::from(me)] + 1) % 3;
        assert_eq!(Leduc::info_state(&CFG, &other, agent(me)), info);
    }
}

#[test]
fn test_leduc_sample_keeps_the_info_state_and_draws_from_the_unseen_cards() {
    let mut p = pool(8);
    play(&mut p, CALL);
    play(&mut p, CALL);
    let s = *p.state(0).unwrap();
    let info = Leduc::info_state(&CFG, &s, agent(0));
    for word in 0..200 {
        let mut out = LeducState::default();
        Leduc::sample(&CFG, &info, &mut rng(1, word), &mut out);
        assert_eq!(Leduc::info_state(&CFG, &out, agent(0)), info);
        // The draw is the first draw of the same generator over the cards
        // agent 0 cannot see.
        assert_eq!(
            out.cards[1],
            draw(&mut rng(1, word), &[info.card, info.public])
        );
        assert_eq!(out.cards[0], s.cards[0]);
    }
}

fn import(p: &mut EnvPool<Leduc, Masked>, state: LeducState) -> Result<(), EngineError> {
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let rec = DecodedRecord::from_parts(*good.header(), state).unwrap();
    p.import(&[0], &[rec])
}

fn invalid(r: &Result<(), EngineError>) -> bool {
    matches!(
        r,
        Err(EngineError::Invariant { index: 0, .. } | EngineError::InvalidRecord { index: 0, .. })
    )
}

#[test]
fn test_leduc_import_rejects_impossible_decks_and_counters() {
    let mut p = pool(2);
    let base = *p.state(0).unwrap();
    assert!(import(&mut p, base).is_ok());
    let triple = LeducState {
        cards: [1, 1],
        public: 1,
        round: 1,
        ..base
    };
    assert!(invalid(&import(&mut p, triple)));
    assert!(invalid(&import(&mut p, LeducState { public: 0, ..base })));
    assert!(invalid(&import(&mut p, LeducState { raises: 3, ..base })));
    assert!(invalid(&import(
        &mut p,
        LeducState {
            cards: [3, 0],
            ..base
        }
    )));
    assert!(invalid(&import(
        &mut p,
        LeducState {
            contributions: [0, 1],
            ..base
        }
    )));
    assert!(invalid(&import(
        &mut p,
        LeducState {
            history_len: [1, 0],
            history: [0b1000, 0],
            ..base
        }
    )));
    assert!(invalid(&import(&mut p, LeducState { done: 1, ..base })));
}

#[test]
fn test_leduc_heuristic_plays_whole_hands_to_the_end() {
    use tessera_rl_core::Heuristic;
    use tessera_rl_core::io::ActionWriter;
    for seed in 0..64 {
        let mut p = pool(seed);
        let mut steps = 0;
        while !terminated(&p) {
            let s = *p.state(0).unwrap();
            let mut a = [0_i32];
            let mut w = ActionWriter::new(&mut a, &mut []);
            Leduc::act(
                &CFG,
                &s,
                agent(u16::try_from(s.mover()).unwrap()),
                &mut rng(0, 0),
                &mut w,
            );
            play(&mut p, u32::try_from(a[0]).unwrap());
            steps += 1;
        }
        assert!(steps <= 8);
    }
}

#[test]
fn test_leduc_exact_chip_outcome_maps_to_chips_over_thirteen() {
    use tessera_rl_games::leduc::MAX_CONTRIBUTION;
    // Raise, raise, call; raise, call: each player has 1 + 4 + 4 = 9 in.
    // Find a seed where slot 0 wins the showdown outright.
    for seed in 0..512 {
        let mut p = pool(seed);
        for a in [RAISE, RAISE, CALL, RAISE, CALL] {
            play(&mut p, a);
        }
        assert!(terminated(&p));
        let s = *p.state(0).unwrap();
        assert_eq!(s.contributions, [9, 9]);
        if s.showdown() != 1 {
            continue;
        }
        let [r0, r1] = rewards(&p);
        assert_eq!(r0, 9.0 / 13.0);
        assert_eq!(r1, -9.0 / 13.0);
        assert_eq!(r0 + r1, 0.0, "zero-sum");
        assert_eq!(MAX_CONTRIBUTION, 13);
        assert_eq!((r0 * 13.0).round(), 9.0, "times 13 is chips");
        return;
    }
    panic!("no seed where slot 0 wins");
}
