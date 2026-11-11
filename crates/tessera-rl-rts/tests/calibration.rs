// SPDX-License-Identifier: MIT OR Apache-2.0

//! The stage 5a calibration tool (`examples/rts_calibration`): a game is a
//! function of (config, bots, seed), a pairing's summary is deterministic,
//! both seats are played, and the Wilson interval is right on known values.

#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::float_arithmetic,
    reason = "test of an offline tool"
)]

#[path = "../examples/rts_calibration/games.rs"]
mod games;

use games::{Outcome, Summary, column, independent_counts, outcome_of, pairing, play, rng, wilson};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_rts::RtsConfig;
use tessera_rl_rts::bots::Bot;

fn small() -> RtsConfig {
    RtsConfig {
        size: 10,
        decision_ticks: 4,
        max_units: 8,
        step_limit: 300,
        seat_swap: false,
    }
}

#[test]
fn test_calibration_game_is_a_function_of_config_bots_and_seed() {
    let c = small();
    for seed in 0..4 {
        let a = play(&c, [Bot::Rush, Bot::RandomLegal], seed);
        assert_eq!(a, play(&c, [Bot::Rush, Bot::RandomLegal], seed));
        assert!(a.decisions >= 1 && a.decisions <= c.step_limit);
        if a.outcome == Outcome::Truncated {
            assert_eq!(a.decisions, c.step_limit);
        }
    }
}

#[test]
fn test_calibration_pairing_plays_both_seats_and_counts_every_game() {
    let c = small();
    let s = pairing(&c, Bot::Rush, Bot::Prospector, 0..5);
    for seat in s.by_seat {
        assert_eq!(seat.iter().sum::<u32>(), 5);
    }
    assert_eq!(s.lengths.len(), 10);
    let truncated = s.by_seat[0][3] + s.by_seat[1][3];
    assert_eq!(
        s.decided_lengths.len(),
        10 - usize::try_from(truncated).expect("small")
    );
    assert_eq!(s, pairing(&c, Bot::Rush, Bot::Prospector, 0..5));
}

/// The summary of a small fixed run, pinned so that any change to a bot,
/// the rules or the tool's seeding shows here (macOS arm64 and Linux
/// x86-64, like the other RTS goldens).
#[test]
fn test_calibration_small_summary_is_pinned() {
    let s = pairing(&small(), Bot::Rush, Bot::RandomLegal, 0..8);
    let got = (
        s.by_seat,
        s.lengths.iter().map(|&l| u64::from(l)).sum::<u64>(),
    );
    assert_eq!(got, ([[8, 0, 0, 0], [8, 0, 0, 0]], 748), "{got:?}");
}

#[test]
fn test_calibration_wilson_matches_known_values() {
    let close =
        |(a, b): (f64, f64), (c, d): (f64, f64)| (a - c).abs() < 1e-4 && (b - d).abs() < 1e-4;
    assert!(close(wilson(0, 10), (0.0, 0.2775)));
    assert!(close(wilson(5, 10), (0.2366, 0.7634)));
    assert!(close(wilson(240, 400), (0.5512, 0.6468)));
    assert_eq!(wilson(0, 0), (0.0, 1.0));
}

#[test]
fn test_calibration_outcome_and_column_tables() {
    let t = |term, trunc, r| outcome_of(term, trunc, r);
    assert_eq!(t(false, false, [0.0, 0.0]), None);
    assert_eq!(t(false, true, [0.0, 0.0]), Some(Outcome::Truncated));
    assert_eq!(t(true, false, [1.0, -1.0]), Some(Outcome::ZeroWins));
    assert_eq!(t(true, false, [-1.0, 1.0]), Some(Outcome::OneWins));
    assert_eq!(t(true, false, [0.0, 0.0]), Some(Outcome::Draw));
    assert_eq!(t(true, false, [-1.0, -1.0]), Some(Outcome::Draw));
    for (seat, outcome, col) in [
        (0, Outcome::ZeroWins, 0),
        (0, Outcome::OneWins, 1),
        (1, Outcome::OneWins, 0),
        (1, Outcome::ZeroWins, 1),
        (0, Outcome::Draw, 2),
        (1, Outcome::Draw, 2),
        (0, Outcome::Truncated, 3),
        (1, Outcome::Truncated, 3),
    ] {
        assert_eq!(column(seat, outcome), col, "{seat} {outcome:?}");
    }
}

#[test]
fn test_calibration_bot_generators_never_equal_the_environments() {
    // The pool's instance generator is (Seed(seed), LogicalId(0), DYNAMICS),
    // which also draws the map at reset; no bot generator may equal it.
    let n = core::num::NonZeroU32::new(1 << 30).expect("nonzero");
    for seed in [0u64, 1, 7, 1_000_000] {
        let env = |logical| {
            let id = RngIdentity::new(Seed::new(seed), LogicalId::new(logical));
            StepRng::new(&RngAuthority::engine_only(), &id, StreamId::DYNAMICS)
        };
        let draws = |mut r: StepRng| -> Vec<u32> { (0..4).map(|_| r.below_u32(n)).collect() };
        for seat in 0..2 {
            let bot = draws(rng(seed, seat));
            assert_ne!(bot, draws(env(0)), "seed {seed} seat {seat}");
            assert_ne!(bot, draws(env(seat)), "seed {seed} seat {seat}");
        }
    }
}

/// R1 on the calibration record: two scripted bots give mirror games, so
/// their intervals count one game per seed; with Random-legal both seats
/// count. The choice follows the bots, not the results.
#[test]
fn test_calibration_independent_games_count_one_per_seed_for_two_scripted_bots() {
    let c = small();
    let rp = pairing(&c, Bot::Rush, Bot::Prospector, 0..6);
    assert_eq!(rp.by_seat[0], rp.by_seat[1]);
    let u = independent_counts(&rp, Bot::Rush, Bot::Prospector);
    assert_eq!(u, rp.by_seat[0]);
    assert_eq!(u.iter().sum::<u32>(), 6);
    let rr = pairing(&c, Bot::Rush, Bot::RandomLegal, 0..6);
    let u = independent_counts(&rr, Bot::Rush, Bot::RandomLegal);
    assert_eq!(u.iter().sum::<u32>(), 12);
}

#[test]
#[should_panic(expected = "mirror games")]
fn test_calibration_independent_counts_refuse_scripted_seats_that_disagree() {
    let s = Summary {
        by_seat: [[3, 1, 0, 0], [2, 2, 0, 0]],
        ..Summary::default()
    };
    let _ = independent_counts(&s, Bot::Rush, Bot::Prospector);
}
