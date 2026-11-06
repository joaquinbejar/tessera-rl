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
    let r = p.outputs().unwrap().rewards();
    [r[0], r[1]]
}

fn mask(p: &EnvPool<Leduc, Masked>, agent: usize) -> Vec<bool> {
    p.outputs().unwrap().masks()[agent * 3..agent * 3 + 3]
        .iter()
        .map(|f| f.get())
        .collect()
}

fn terminated(p: &EnvPool<Leduc, Masked>) -> bool {
    p.outputs().unwrap().episode_terminated()[0].get()
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
        assert!(p.outputs().unwrap().acting()[0].get());
        assert!(!p.outputs().unwrap().acting()[1].get());
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
    assert_eq!(p.state(0).unwrap().done, 1, "the hand is over");
    assert!(
        p.outputs().unwrap().acting().iter().all(|a| !a.get()),
        "nobody acts"
    );
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

/// The observation of `k`, from the rules: own rank, public rank, round,
/// whether `k` is to move, both contributions (own first), raises, the
/// histories and their lengths. Never the opponent's card.
fn expected_obs(s: &LeducState, k: usize) -> Vec<u8> {
    let to_move = u32::from(s.done == 0 && s.mover() == k);
    [
        s.cards[k],
        s.public,
        s.round,
        to_move,
        s.contributions[k],
        s.contributions[1 - k],
        s.raises,
        s.history[0],
        s.history[1],
        s.history_len[0],
        s.history_len[1],
    ]
    .iter()
    .map(|&v| u8::try_from(v).unwrap())
    .collect()
}

fn obs_of(bytes: &[u8], k: usize) -> Vec<u8> {
    bytes[k * OBS..(k + 1) * OBS].to_vec()
}

const OBS: usize = tessera_rl_games::leduc::OBS_LEN;

/// Hands covering a fold in each round, a showdown after raises and
/// check-check rounds.
const HANDS: [&[u32]; 5] = [
    &[CALL, CALL, CALL, CALL],
    &[RAISE, FOLD],
    &[CALL, RAISE, RAISE, CALL, RAISE, FOLD],
    &[RAISE, RAISE, CALL, RAISE, RAISE, CALL],
    &[CALL, CALL, RAISE, CALL],
];

/// Another rank for slot `j` that the deck allows, if any.
fn other_card(s: &LeducState, j: usize) -> Option<u32> {
    (0..3).find(|&c| {
        c != s.cards[j] && {
            let mut x = *s;
            x.cards[j] = c;
            let n = |r| x.cards.iter().filter(|&&v| v == r).count() + usize::from(x.public == r);
            (0..3).all(|r| n(r) <= 2)
        }
    })
}

fn import_state(p: &mut EnvPool<Leduc, Masked>, s: LeducState) -> Result<(), EngineError> {
    let h = *p.snapshot(0).unwrap().header();
    p.import(&[0], &[DecodedRecord::from_parts(h, s).unwrap()])
}

#[test]
fn test_leduc_observation_bytes_match_the_rules_and_hide_the_opponents_card() {
    for seed in 0..24 {
        for hand in HANDS {
            let mut p = pool(seed);
            let mut positions = 0;
            for step in 0..=hand.len() {
                let s = *p.state(0).unwrap();
                let bytes = p.outputs().unwrap().observation(0).unwrap().to_vec();
                for k in 0..2 {
                    assert_eq!(
                        obs_of(&bytes, k),
                        expected_obs(&s, k),
                        "seed {seed} step {step}"
                    );
                    // Only the opponent's card changes: k sees the same bytes.
                    if let Some(c) = other_card(&s, 1 - k) {
                        let mut q = pool(seed);
                        let mut moved = s;
                        moved.cards[1 - k] = c;
                        let h = *p.snapshot(0).unwrap().header();
                        q.import(&[0], &[DecodedRecord::from_parts(h, moved).unwrap()])
                            .unwrap();
                        let other = q.outputs().unwrap().observation(0).unwrap().to_vec();
                        assert_eq!(
                            obs_of(&other, k),
                            obs_of(&bytes, k),
                            "seed {seed} step {step}"
                        );
                        assert_ne!(obs_of(&other, 1 - k), obs_of(&bytes, 1 - k), "its own card");
                        positions += 1;
                    }
                }
                if let Some(&a) = hand.get(step) {
                    play(&mut p, a);
                }
            }
            assert!(
                terminated(&p),
                "the hand ends: the last position is terminal"
            );
            assert!(positions > 0);
        }
    }
}

#[test]
fn test_leduc_terminal_observation_matches_the_rules_and_hides_the_opponents_card() {
    let same_step = |seed| {
        let options = PoolOptions {
            auto_reset: AutoReset::SameStep,
            ..PoolOptions::new(NonZeroU32::MIN, Seed::new(seed))
        };
        let mut p = EnvPool::<Leduc, Masked>::with_masks(CFG, options).unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        p
    };
    for seed in 0..24 {
        for hand in HANDS {
            let (last, before) = hand.split_last().unwrap();
            let mut reference = pool(seed);
            for &a in hand {
                play(&mut reference, a);
            }
            let end = *reference.state(0).unwrap();
            let mut p = same_step(seed);
            for &a in before {
                play(&mut p, a);
            }
            let pre = *p.state(0).unwrap();
            let header = *p.snapshot(0).unwrap().header();
            play(&mut p, *last);
            let terminal = p
                .outputs()
                .unwrap()
                .terminal_observation(0)
                .unwrap()
                .to_vec();
            for k in 0..2 {
                assert_eq!(obs_of(&terminal, k), expected_obs(&end, k), "seed {seed}");
                let Some(c) = other_card(&pre, 1 - k) else {
                    continue;
                };
                // A fold keeps the cards; a showdown deals no card after
                // the last action, so the alternative stays valid.
                let mut moved = pre;
                moved.cards[1 - k] = c;
                let mut q = same_step(seed);
                q.import(&[0], &[DecodedRecord::from_parts(header, moved).unwrap()])
                    .unwrap();
                play(&mut q, *last);
                let other = q
                    .outputs()
                    .unwrap()
                    .terminal_observation(0)
                    .unwrap()
                    .to_vec();
                assert_eq!(obs_of(&other, k), obs_of(&terminal, k), "seed {seed}");
            }
        }
    }
}

#[test]
fn test_leduc_never_reset_record_round_trips_and_continues() {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(21))
    };
    let mut fresh = EnvPool::<Leduc, Masked>::with_masks(CFG, options).unwrap();
    let rec = fresh.snapshot(0).unwrap();
    let bytes = rec.to_bytes();
    let decoded: DecodedRecord<Leduc> =
        tessera_rl_core::encoding::decode_from_slice(&bytes).unwrap();
    let mut q = EnvPool::<Leduc, Masked>::with_masks(CFG, options).unwrap();
    q.import(&[0], &[decoded]).unwrap();
    assert_eq!(*q.state(0).unwrap(), LeducState::default());
    // The continuation equals a pool that never left the initial value.
    fresh.reset_all(Reseed::Keep).unwrap();
    q.reset_all(Reseed::Keep).unwrap();
    for a in [RAISE, CALL, CALL, CALL] {
        play(&mut fresh, a);
        play(&mut q, a);
        assert_eq!(
            fresh.snapshot(0).unwrap().to_bytes(),
            q.snapshot(0).unwrap().to_bytes()
        );
    }
    // Before the first reset, only the initial value is accepted.
    let odd = LeducState {
        cards: [1, 0],
        ..LeducState::default()
    };
    let mut r = EnvPool::<Leduc, Masked>::with_masks(CFG, options).unwrap();
    let h = *r.snapshot(0).unwrap().header();
    assert!(invalid(
        &r.import(&[0], &[DecodedRecord::from_parts(h, odd).unwrap()])
    ));
}

#[test]
fn test_leduc_import_refuses_stakes_the_history_does_not_lead_to() {
    let mut p = pool(2);
    let before = p.snapshot(0).unwrap().to_bytes();
    let inflated = LeducState {
        cards: [2, 0],
        contributions: [13, 13],
        ..*p.state(0).unwrap()
    };
    assert!(invalid(&import_state(&mut p, inflated)));
    assert_eq!(
        p.snapshot(0).unwrap().to_bytes(),
        before,
        "the pool is unchanged"
    );
    // A legal history must match its betting fields exactly.
    let mut q = pool(2);
    play(&mut q, RAISE);
    let s = *q.state(0).unwrap();
    for wrong in [
        LeducState { raises: 0, ..s },
        LeducState { to_move: 0, ..s },
        LeducState {
            contributions: [5, 1],
            ..s
        },
        LeducState {
            history: [0b10, 0],
            ..s
        },
    ] {
        assert!(invalid(&import_state(&mut q, wrong)), "{wrong:?}");
    }
    assert!(import_state(&mut q, s).is_ok());

    // A full first round (four actions): no bit above them may be set.
    let mut full = pool(2);
    for a in [CALL, RAISE, RAISE, CALL] {
        play(&mut full, a);
    }
    let s = *full.state(0).unwrap();
    assert_eq!(s.history_len[0], 4);
    let mut high = s;
    high.history[0] |= 1 << 20;
    assert!(invalid(&import_state(&mut full, high)));
    assert!(import_state(&mut full, s).is_ok());
}

/// Histories whose betting fields are exactly what the actions lead to,
/// so only the legality of an action refuses them.
#[test]
fn test_leduc_validate_refuses_illegal_actions_in_a_consistent_history() {
    use tessera_rl_core::Validate;
    let initial = LeducState {
        cards: [0, 1],
        public: NO_CARD,
        contributions: [1, 1],
        ..LeducState::default()
    };
    // Raise, raise, raise: 3, then 5, then 7 for slot 0.
    let third_raise = LeducState {
        history: [0b11_11_11, 0],
        history_len: [3, 0],
        contributions: [7, 5],
        raises: 3,
        to_move: 1,
        ..initial
    };
    assert!(Leduc::validate(&CFG, &third_raise).is_err());
    let two_raises = LeducState {
        history: [0b11_11, 0],
        history_len: [2, 0],
        contributions: [3, 5],
        raises: 2,
        to_move: 0,
        ..initial
    };
    assert!(Leduc::validate(&CFG, &two_raises).is_ok());
    // A fold with no bet to answer, which would end the hand.
    let fold = LeducState {
        history: [0b01, 0],
        history_len: [1, 0],
        done: 1,
        ..initial
    };
    assert!(Leduc::validate(&CFG, &fold).is_err());
    let raise_fold = LeducState {
        history: [0b01_11, 0],
        history_len: [2, 0],
        contributions: [3, 1],
        raises: 1,
        to_move: 1,
        done: 1,
        ..initial
    };
    assert!(Leduc::validate(&CFG, &raise_fold).is_ok());
}

/// Every legal continuation of `p`'s instance keeps rewards in [-1, 1] and
/// contributions within `MAX_CONTRIBUTION`.
fn explore(p: &mut EnvPool<Leduc, Masked>, depth: u32) -> u32 {
    let at = p.snapshot(0).unwrap();
    let s = *at.state();
    let mut leaves = 0;
    for a in [FOLD, CALL, RAISE] {
        let legal = match a {
            FOLD => s.facing_bet(),
            RAISE => s.raises < 2,
            _ => true,
        };
        if !legal {
            continue;
        }
        p.restore(0, &at).unwrap();
        play(p, a);
        let after = *p.state(0).unwrap();
        assert!(
            after.contributions.iter().all(|&c| c <= MAX_CONTRIBUTION),
            "{after:?}"
        );
        assert!(
            rewards(p).iter().all(|r| r.abs() <= 1.0),
            "{:?} {after:?}",
            rewards(p)
        );
        if terminated(p) {
            leaves += 1;
        } else {
            assert!(depth < 8, "a hand longer than 8 actions");
            leaves += explore(p, depth + 1);
        }
    }
    p.restore(0, &at).unwrap();
    leaves
}

/// The acting row of the header for `s`.
fn with_acting(
    mut h: <Leduc as tessera_rl_core::Environment>::Header,
    s: &LeducState,
) -> <Leduc as tessera_rl_core::Environment>::Header {
    use tessera_rl_core::{Bool8, Environment, RecordLayout};
    for (k, row) in h.agents_mut().iter_mut().enumerate() {
        row.acting = Bool8::from(Leduc::acting(&CFG, s, agent(u16::try_from(k).unwrap())));
    }
    h
}

#[test]
fn test_leduc_every_accepted_import_keeps_rewards_and_stakes_in_bounds() {
    use tessera_rl_core::Validate;
    let mut accepted = 0;
    let mut draws = rng(77, 3);
    let mut candidates = Vec::new();
    // Reachable states: every position of every hand, with every deal.
    let mut reachable = Vec::new();
    for seed in 0..32 {
        for hand in HANDS {
            let mut p = pool(seed);
            for &a in hand {
                reachable.push(*p.state(0).unwrap());
                play(&mut p, a);
            }
        }
    }
    // Random perturbations of them, of any field, kept when Validate
    // accepts them: most are refused.
    let w = |g: &mut StepRng, n: u32| g.below_u32(NonZeroU32::new(n).unwrap());
    for _ in 0..200_000 {
        let mut s = reachable[w(&mut draws, u32::try_from(reachable.len()).unwrap()) as usize];
        match w(&mut draws, 8) {
            0 => s.cards[w(&mut draws, 2) as usize] = w(&mut draws, 4),
            1 => s.public = w(&mut draws, 4),
            2 => s.round = w(&mut draws, 3),
            3 => s.to_move = w(&mut draws, 3),
            4 => s.raises = w(&mut draws, 4),
            5 => s.contributions[w(&mut draws, 2) as usize] = w(&mut draws, 15),
            6 => s.history[w(&mut draws, 2) as usize] ^= 1 << w(&mut draws, 32),
            _ => s.history_len[w(&mut draws, 2) as usize] = w(&mut draws, 6),
        }
        if Leduc::validate(&CFG, &s).is_ok() && !reachable.contains(&s) {
            candidates.push(s);
        }
    }
    let random = candidates.len();
    assert!(random >= 20, "{random} random states accepted");
    candidates.extend(reachable);
    let mut p = pool(1);
    let base = *p.snapshot(0).unwrap().header();
    for s in candidates {
        if s.done != 0 {
            continue;
        }
        let rec = DecodedRecord::from_parts(with_acting(base, &s), s).unwrap();
        p.import(&[0], &[rec]).unwrap();
        assert!(explore(&mut p, 0) > 0);
        accepted += 1;
    }
    assert!(accepted > 160, "{accepted}");
}
