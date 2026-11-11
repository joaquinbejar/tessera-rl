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

use games::{Outcome, pairing, play, wilson};
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
    assert_eq!(got, ([[2, 0, 0, 6], [2, 0, 0, 6]], 3761), "{got:?}");
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
