// SPDX-License-Identifier: MIT OR Apache-2.0

//! The pre-registered P2-G1 openings (ADR-0042): positions reached by `K`
//! uniformly random legal plies from a `SplitMix64` seed, kept when they are
//! new up to mirror symmetry and neither side has a forced win within `L`
//! plies (exact solver null-window tests).

use crate::board::{CELLS, Position, WIDTH};
use crate::rng::SplitMix64;
use crate::solver::Solver;
use std::collections::BTreeSet;

/// Random plies per opening.
pub const K: u32 = 8;
/// Horizon of the forced-win filter, in plies from the opening position.
pub const L: u32 = 8;

/// One opening: its move sequence (1-based columns) and the seed that made it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opening {
    /// Move sequence as 1-based column digits.
    pub moves: String,
    /// The `SplitMix64` seed.
    pub seed: u64,
}

/// The score bound of the filter: a forced win within `L` plies for the
/// side to move has a score of at least this (and a forced loss at most its
/// negation). Scores follow the solver's convention.
#[must_use]
pub fn bound(p: &Position) -> i32 {
    (CELLS as i32 + 2 - (p.moves() + L) as i32) / 2
}

/// The opening of `seed`, if it passes every filter except novelty.
pub fn candidate(seed: u64, solver: &mut Solver) -> Option<(Opening, Position)> {
    let mut rng = SplitMix64::new(seed);
    let mut p = Position::new();
    let mut moves = String::new();
    for _ in 0..K {
        let legal: Vec<u32> = (0..WIDTH).filter(|&c| p.can_play(c)).collect();
        let col = *legal.get(rng.below(legal.len()))?;
        if p.is_winning_col(col) {
            return None;
        }
        p.play_col(col);
        moves.push(char::from_digit(col + 1, 10)?);
    }
    if p.can_win_next() {
        return None;
    }
    let b = bound(&p);
    if solver.score_at_least(&p, b) || solver.score_at_most(&p, -b) {
        return None;
    }
    Some((Opening { moves, seed }, p))
}

/// `count` openings from seeds `base, base + 1, ...`, skipping any position
/// (up to mirror symmetry) in `exclude` or already taken.
pub fn generate(
    base: u64,
    count: usize,
    exclude: &BTreeSet<u64>,
    solver: &mut Solver,
) -> Vec<Opening> {
    let mut seen = exclude.clone();
    let mut out = Vec::with_capacity(count);
    let mut next_seed = base;
    while out.len() < count {
        if let Some((opening, pos)) = candidate(next_seed, solver)
            && seen.insert(pos.canonical_key())
        {
            out.push(opening);
        }
        next_seed = next_seed.wrapping_add(1);
    }
    out
}
