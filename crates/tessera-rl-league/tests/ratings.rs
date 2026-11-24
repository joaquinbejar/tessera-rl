// SPDX-License-Identifier: MIT OR Apache-2.0

//! Elo and `TrueSkill` against reference values (ADR-0088 items 4.3, 4.4).

use tessera_rl_league::trueskill::{self, Params, Rating};
use tessera_rl_league::{LN10, Outcome, elo};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn test_ln10_is_pinned_to_the_f64_of_ln_10() {
    assert_eq!(LN10.to_bits(), 0x4002_6BB1_BBB5_5516);
    assert_eq!(LN10, core::f64::consts::LN_10);
}

#[test]
fn test_elo_equal_ratings_win_moves_half_k() {
    assert_eq!(elo::expected(1200.0, 1200.0), 0.5);
    let (a, b) = elo::update(1200.0, 1200.0, Outcome::Win, 32.0);
    assert_eq!((a, b), (1216.0, 1184.0));
    let (a, b) = elo::update(1200.0, 1200.0, Outcome::Draw, 32.0);
    assert_eq!((a, b), (1200.0, 1200.0));
}

#[test]
fn test_elo_expected_score_matches_the_logistic_formula() {
    // 400 points: 10:1 odds, expected 10/11.
    assert!(close(elo::expected(1600.0, 1200.0), 10.0 / 11.0, 1e-15));
    assert!(close(elo::expected(1200.0, 1600.0), 1.0 / 11.0, 1e-15));
    // A 200-point favourite who loses: k * (0 - 0.7597...).
    let (a, _) = elo::update(1400.0, 1200.0, Outcome::Loss, 32.0);
    assert!(close(a, 1400.0 - 32.0 * 0.759_746_926_647_957_1, 1e-9));
}

#[test]
fn test_trueskill_cdf_and_pdf_match_known_values() {
    assert_eq!(trueskill::cdf(0.0), 0.5);
    assert!(close(trueskill::cdf(1.96), 0.975_002_104_851_780, 1e-12));
    assert!(close(
        trueskill::cdf(-3.0),
        0.001_349_898_031_630_094_6,
        1e-15
    ));
    assert!(close(trueskill::pdf(0.0), 0.398_942_280_401_432_7, 1e-16));
}

/// The reference values of the `trueskill` Python package's documentation:
/// two new players, default environment (draw probability 0.1).
#[test]
#[expect(
    clippy::many_single_char_names,
    reason = "winner, loser, a, b and params p"
)]
fn test_trueskill_one_vs_one_matches_the_reference_package() {
    let p = Params::default();
    let (w, l) = trueskill::update(Rating::initial(), Rating::initial(), Outcome::Win, p);
    assert!(
        close(w.mu, 29.396, 5e-4) && close(w.sigma, 7.171, 5e-4),
        "{w:?}"
    );
    assert!(
        close(l.mu, 20.604, 5e-4) && close(l.sigma, 7.171, 5e-4),
        "{l:?}"
    );
    let (a, b) = trueskill::update(Rating::initial(), Rating::initial(), Outcome::Draw, p);
    assert!(
        close(a.mu, 25.0, 5e-4) && close(a.sigma, 6.458, 5e-4),
        "{a:?}"
    );
    assert!(
        close(b.mu, 25.0, 5e-4) && close(b.sigma, 6.458, 5e-4),
        "{b:?}"
    );
}

#[test]
fn test_trueskill_loss_is_the_mirror_of_a_win() {
    let p = Params::default();
    let a = Rating {
        mu: 27.0,
        sigma: 5.0,
    };
    let b = Rating {
        mu: 22.0,
        sigma: 7.0,
    };
    let (a1, b1) = trueskill::update(a, b, Outcome::Loss, p);
    let (b2, a2) = trueskill::update(b, a, Outcome::Win, p);
    assert_eq!((a1, b1), (a2, b2));
}

#[test]
fn test_trueskill_far_tails_stay_finite() {
    let p = Params::default();
    let strong = Rating {
        mu: 1.0e4,
        sigma: 1.0,
    };
    let weak = Rating {
        mu: -1.0e4,
        sigma: 1.0,
    };
    for o in [Outcome::Win, Outcome::Draw, Outcome::Loss] {
        let (a, b) = trueskill::update(strong, weak, o, p);
        assert!(
            a.mu.is_finite() && a.sigma.is_finite() && b.mu.is_finite() && b.sigma.is_finite(),
            "{o:?}"
        );
        assert!(a.sigma >= 0.0 && b.sigma >= 0.0);
    }
}
