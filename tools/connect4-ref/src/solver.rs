// SPDX-License-Identifier: MIT OR Apache-2.0

//! Perfect Connect Four solver: negamax with alpha-beta, a transposition
//! table of upper bounds, threat-based move ordering and iterative
//! null-window search.
//!
//! The score of a position is from the side to move's view: 0 is a draw, a
//! positive score `s` means the side to move wins, and a larger `s` means a
//! faster win (`s = (CELLS + 1 - n) / 2`, `n` the stones played when the
//! winning stone is placed, counted as in the classic solver).

use crate::board::{CELLS, CENTRE_ORDER, Position, WIDTH, column_mask};

/// Lowest possible score.
pub const MIN_SCORE: i32 = -(CELLS as i32) / 2 + 3;

/// The solver, with its transposition table.
pub struct Solver {
    keys: Vec<u64>,
    vals: Vec<u8>,
    /// Nodes searched (diagnostics).
    pub nodes: u64,
}

const TABLE: usize = 4_194_301; // prime

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver {
    /// A solver with an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            keys: vec![0; TABLE],
            vals: vec![0; TABLE],
            nodes: 0,
        }
    }

    fn slot(key: u64) -> usize {
        usize::try_from(key % TABLE as u64).unwrap_or(0)
    }

    fn get(&self, key: u64) -> u8 {
        let i = Self::slot(key);
        match (self.keys.get(i), self.vals.get(i)) {
            (Some(&k), Some(&v)) if k == key => v,
            _ => 0,
        }
    }

    fn put(&mut self, key: u64, val: u8) {
        let i = Self::slot(key);
        if let (Some(k), Some(v)) = (self.keys.get_mut(i), self.vals.get_mut(i)) {
            *k = key;
            *v = val;
        }
    }

    /// Negamax over `[alpha, beta]`; requires that the side to move cannot
    /// win at once.
    pub fn negamax(&mut self, p: &Position, mut alpha: i32, mut beta: i32) -> i32 {
        self.nodes += 1;
        let moves = i32::try_from(p.moves()).unwrap_or(0);
        let cells = CELLS as i32;
        let next = p.non_losing_moves();
        if next == 0 {
            return -(cells - moves) / 2;
        }
        if moves >= cells - 2 {
            return 0;
        }
        let min = -(cells - 2 - moves) / 2;
        if alpha < min {
            alpha = min;
            if alpha >= beta {
                return alpha;
            }
        }
        let mut max = (cells - 1 - moves) / 2;
        let stored = self.get(p.key());
        if stored != 0 {
            max = i32::from(stored) + MIN_SCORE - 1;
        }
        if beta > max {
            beta = max;
            if alpha >= beta {
                return beta;
            }
        }
        // Children ordered by threats created, then centre first.
        let mut order: [(u32, usize, u64); 7] = [(0, 0, 0); 7];
        let mut n = 0;
        for (i, &col) in CENTRE_ORDER.iter().enumerate() {
            let mv = next & column_mask(col);
            if mv != 0
                && let Some(slot) = order.get_mut(n)
            {
                *slot = (p.move_score(mv), i, mv);
                n += 1;
            }
        }
        let children = order.get_mut(..n).unwrap_or_default();
        children.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        for &(_, _, mv) in children.iter() {
            let mut child = *p;
            child.play_bit(mv);
            let score = -self.negamax(&child, -beta, -alpha);
            if score >= beta {
                return score;
            }
            if score > alpha {
                alpha = score;
            }
        }
        self.put(p.key(), u8::try_from(alpha - MIN_SCORE + 1).unwrap_or(0));
        alpha
    }

    /// The exact score of `p`.
    pub fn solve(&mut self, p: &Position) -> i32 {
        let cells = CELLS as i32;
        let moves = i32::try_from(p.moves()).unwrap_or(0);
        if p.can_win_next() {
            return (cells + 1 - moves) / 2;
        }
        let mut min = -(cells - moves) / 2;
        let mut max = (cells + 1 - moves) / 2;
        while min < max {
            let mut med = min + (max - min) / 2;
            if med <= 0 && min / 2 < med {
                med = min / 2;
            } else if med >= 0 && max / 2 > med {
                med = max / 2;
            }
            let r = self.negamax(p, med, med + 1);
            if r <= med {
                max = r;
            } else {
                min = r;
            }
        }
        min
    }

    /// Whether the score of `p` is at least `t` (null-window test).
    pub fn score_at_least(&mut self, p: &Position, t: i32) -> bool {
        if p.can_win_next() {
            return self.solve(p) >= t;
        }
        self.negamax(p, t - 1, t) >= t
    }

    /// Whether the score of `p` is at most `t` (null-window test).
    pub fn score_at_most(&mut self, p: &Position, t: i32) -> bool {
        if p.can_win_next() {
            return self.solve(p) <= t;
        }
        self.negamax(p, t, t + 1) <= t
    }

    /// The score of every legal column, from the side to move's view.
    pub fn column_scores(&mut self, p: &Position) -> [Option<i32>; 7] {
        let cells = CELLS as i32;
        let moves = i32::try_from(p.moves()).unwrap_or(0);
        let mut out = [None; 7];
        for col in 0..WIDTH {
            if !p.can_play(col) {
                continue;
            }
            let score = if p.is_winning_col(col) {
                (cells + 1 - moves) / 2
            } else {
                let mut child = *p;
                child.play_col(col);
                -self.solve(&child)
            };
            if let Some(slot) = out.get_mut(col as usize) {
                *slot = Some(score);
            }
        }
        out
    }

    /// The perfect move: the best score, ties broken centre first.
    pub fn best_col(&mut self, p: &Position) -> Option<u32> {
        let scores = self.column_scores(p);
        let mut best: Option<(i32, u32)> = None;
        for &col in &CENTRE_ORDER {
            if let Some(Some(s)) = scores.get(col as usize)
                && best.is_none_or(|(b, _)| *s > b)
            {
                best = Some((*s, col));
            }
        }
        best.map(|(_, c)| c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plain minimax with no pruning or table: the independent reference.
    fn brute(p: &Position) -> i32 {
        let cells = CELLS as i32;
        let moves = i32::try_from(p.moves()).unwrap_or(0);
        if moves == cells {
            return 0;
        }
        let mut best = i32::MIN;
        for col in 0..WIDTH {
            if !p.can_play(col) {
                continue;
            }
            let v = if p.is_winning_col(col) {
                (cells + 1 - moves) / 2
            } else {
                let mut c = *p;
                c.play_col(col);
                -brute(&c)
            };
            best = best.max(v);
        }
        best
    }

    #[test]
    fn test_solver_matches_brute_force_on_late_positions() {
        let mut rng = crate::rng::SplitMix64::new(7);
        let mut checked = 0;
        let mut s = Solver::new();
        while checked < 300 {
            let mut p = Position::new();
            let mut ok = true;
            while p.moves() < 32 {
                let legal: Vec<u32> = (0..WIDTH).filter(|&c| p.can_play(c)).collect();
                let col = *legal.get(rng.below(legal.len())).unwrap_or(&0);
                if p.is_winning_col(col) {
                    ok = false;
                    break;
                }
                p.play_col(col);
            }
            if ok {
                assert_eq!(s.solve(&p), brute(&p));
                checked += 1;
            }
        }
    }

    #[test]
    fn test_solver_immediate_win_is_found() {
        let p = Position::from_moves("121212").unwrap_or_default();
        assert_eq!(Solver::new().best_col(&p), Some(0));
    }

    #[test]
    fn test_solver_null_window_tests_agree_with_solve() {
        let mut rng = crate::rng::SplitMix64::new(11);
        let mut s = Solver::new();
        for _ in 0..50 {
            let mut p = Position::new();
            while p.moves() < 30 {
                let legal: Vec<u32> = (0..WIDTH)
                    .filter(|&c| p.can_play(c) && !p.is_winning_col(c))
                    .collect();
                let Some(&col) = legal.get(rng.below(legal.len().max(1))) else {
                    break;
                };
                p.play_col(col);
            }
            let v = s.solve(&p);
            assert!(s.score_at_least(&p, v) && !s.score_at_least(&p, v + 1));
            assert!(s.score_at_most(&p, v) && !s.score_at_most(&p, v - 1));
        }
    }
}
