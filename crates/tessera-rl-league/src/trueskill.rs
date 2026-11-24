// SPDX-License-Identifier: MIT OR Apache-2.0

//! `TrueSkill` for two teams of one player, with a draw margin (ADR-0088
//! item 4.4; Herbrich, Minka and Graepel, "`TrueSkill`: A Bayesian Skill
//! Rating System", 2006).
//!
//! The standard normal cdf is `0.5 * erfc(-x / sqrt(2))`, with the frozen
//! `erfc` of `tessera_rl_num::math`; the pdf uses the frozen `exp`. `sqrt` is
//! IEEE correctly rounded. Callers run updates inside
//! `tessera_rl_num::fp::with_canonical`.

use tessera_rl_num::math;

use crate::Outcome;

/// A rating: the mean and standard deviation of a skill belief.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rating {
    /// The mean skill.
    pub mu: f64,
    /// The standard deviation of the skill.
    pub sigma: f64,
}

/// The model parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// The performance noise.
    pub beta: f64,
    /// The dynamics noise added to each sigma before a game.
    pub tau: f64,
    /// The draw margin, in performance units.
    pub draw_margin: f64,
}

/// The initial mean of the reference parameters.
pub const MU: f64 = 25.0;
/// The initial standard deviation of the reference parameters.
pub const SIGMA: f64 = 25.0 / 3.0;

impl Default for Params {
    /// The reference parameters: beta = sigma / 2, tau = sigma / 100, and the
    /// draw margin of a 10% draw probability between two new players,
    /// `sqrt(2) * beta * Phi^-1(0.55)` = `0.740_466_587_452_148_1` (computed offline and
    /// pinned, so no inverse cdf is needed).
    fn default() -> Self {
        Self {
            beta: SIGMA / 2.0,
            tau: SIGMA / 100.0,
            draw_margin: f64::from_bits(0x3FE7_B1E6_FC1C_680A),
        }
    }
}

impl Rating {
    /// The reference initial rating, (25, 25/3).
    #[must_use]
    pub const fn initial() -> Self {
        Self {
            mu: MU,
            sigma: SIGMA,
        }
    }
}

const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
const FRAC_1_SQRT_2: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// The standard normal density.
#[must_use]
pub fn pdf(x: f64) -> f64 {
    FRAC_1_SQRT_2PI * math::exp(-0.5 * x * x)
}

/// The standard normal cumulative distribution.
#[must_use]
pub fn cdf(x: f64) -> f64 {
    0.5 * math::erfc(-x * FRAC_1_SQRT_2)
}

/// Denominators below this use the asymptotic forms, as reference
/// implementations do, so far tails stay finite.
const TINY: f64 = 2.222_758_749e-162;

/// `v` and `w` of a decisive game, for `t` the scaled mean difference and
/// `e` the scaled draw margin.
fn v_w_win(t: f64, e: f64) -> (f64, f64) {
    let x = t - e;
    let denom = cdf(x);
    let v = if denom < TINY { -x } else { pdf(x) / denom };
    (v, v * (v + x))
}

/// `v` and `w` of a draw.
#[expect(
    clippy::many_single_char_names,
    reason = "the notation of the TrueSkill paper"
)]
fn v_w_draw(t: f64, e: f64) -> (f64, f64) {
    let a = e - t;
    let b = -e - t;
    let denom = cdf(a) - cdf(b);
    if denom < TINY {
        // Far tail: the limit of the truncated Gaussian moments.
        let v = if t < 0.0 { -t - e } else { -t + e };
        return (v, 1.0);
    }
    let v = (pdf(b) - pdf(a)) / denom;
    let w = v * v + (a * pdf(a) - b * pdf(b)) / denom;
    (v, w)
}

/// Both ratings after one game whose outcome is `outcome` for A.
#[must_use]
#[expect(
    clippy::many_single_char_names,
    reason = "the notation of the TrueSkill paper"
)]
pub fn update(a: Rating, b: Rating, outcome: Outcome, p: Params) -> (Rating, Rating) {
    // Dynamics first: the skills may have drifted since the last game.
    let va = a.sigma * a.sigma + p.tau * p.tau;
    let vb = b.sigma * b.sigma + p.tau * p.tau;
    let c2 = 2.0 * p.beta * p.beta + va + vb;
    let c = c2.sqrt();
    let e = p.draw_margin / c;
    // Orient the game from the winner's side; a draw keeps A first.
    let (wv, lv, wmu, lmu, swapped) = match outcome {
        Outcome::Win | Outcome::Draw => (va, vb, a.mu, b.mu, false),
        Outcome::Loss => (vb, va, b.mu, a.mu, true),
    };
    let t = (wmu - lmu) / c;
    let (v, w) = if outcome == Outcome::Draw {
        v_w_draw(t, e)
    } else {
        v_w_win(t, e)
    };
    let new = |mu: f64, var: f64, sign: f64| Rating {
        mu: mu + sign * var / c * v,
        sigma: (var * (1.0 - var / c2 * w).max(0.0)).sqrt(),
    };
    let winner = new(wmu, wv, 1.0);
    let loser = new(lmu, lv, -1.0);
    if swapped {
        (loser, winner)
    } else {
        (winner, loser)
    }
}
