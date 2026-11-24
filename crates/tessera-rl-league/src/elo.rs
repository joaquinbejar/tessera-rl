// SPDX-License-Identifier: MIT OR Apache-2.0

//! Elo ratings (ADR-0088 item 4.3).
//!
//! The expected score of A against B is `1 / (1 + 10^((Rb - Ra) / 400))`,
//! computed as `1 / (1 + exp(LN10 * (Rb - Ra) / 400))` with the frozen
//! `exp` of `tessera_rl_num::math` and the pinned [`crate::LN10`]. Callers
//! run updates inside `tessera_rl_num::fp::with_canonical`.

use tessera_rl_num::math;

use crate::{LN10, Outcome};

/// The usual initial rating.
pub const INITIAL: f64 = 1200.0;

/// The expected score of a player rated `ra` against one rated `rb`.
#[must_use]
pub fn expected(ra: f64, rb: f64) -> f64 {
    1.0 / (1.0 + math::exp(LN10 * (rb - ra) / 400.0))
}

/// Both ratings after one game whose outcome is `outcome` for A, with
/// update factor `k`. Returns `(ra', rb')`; B's change is the negative of
/// A's.
#[must_use]
pub fn update(ra: f64, rb: f64, outcome: Outcome, k: f64) -> (f64, f64) {
    let delta = k * (outcome.score() - expected(ra, rb));
    (ra + delta, rb - delta)
}
