// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scripted P2-G1 opponent (ADR-0042): deterministic depth-limited
//! negamax with alpha-beta over a fixed window heuristic, centre-first move
//! order and a seeded tie break among equally scored root moves.
//!
//! Every parameter is a constant here and is pinned in ADR-0042 by the
//! source hash of this file; changing any of them is a new bot.

use crate::board::{CELLS, CENTRE_ORDER, HEIGHT, Position, WIDTH, column_mask};
use crate::rng::SplitMix64;

/// Search depth in plies.
pub const DEPTH: u32 = 4;
/// Weight of an open window holding two of the side's stones.
pub const W2: i32 = 2;
/// Weight of an open window holding three of the side's stones.
pub const W3: i32 = 5;
/// Weight of each stone in the centre column.
pub const WC: i32 = 3;
/// Value of a win (minus the plies played, so faster wins score higher).
pub const WIN: i32 = 1_000_000;

/// The 69 four-cell windows of the board, as bit masks.
fn windows() -> Vec<u64> {
    let bit = |c: u32, r: u32| 1u64 << (c * (HEIGHT + 1) + r);
    let mut out = Vec::with_capacity(69);
    for c in 0..WIDTH {
        for r in 0..HEIGHT {
            for (dc, dr) in [(1i32, 0i32), (0, 1), (1, 1), (1, -1)] {
                let (ec, er) = (c as i32 + 3 * dc, r as i32 + 3 * dr);
                if ec < 0 || ec >= WIDTH as i32 || er < 0 || er >= HEIGHT as i32 {
                    continue;
                }
                let mut m = 0;
                for k in 0..4 {
                    let (cc, rr) = (c as i32 + k * dc, r as i32 + k * dr);
                    m |= bit(cc.unsigned_abs(), rr.unsigned_abs());
                }
                out.push(m);
            }
        }
    }
    out
}

/// The scripted bot.
pub struct Bot {
    windows: Vec<u64>,
    depth: u32,
    tie_seed: u64,
}

impl Bot {
    /// The pinned bot with tie-break seed `tie_seed` (ADR-0042: the
    /// tie-break seed stream is separate from the evaluation seeds).
    #[must_use]
    pub fn new(tie_seed: u64) -> Self {
        Self::with_depth(DEPTH, tie_seed)
    }

    /// A bot of another depth (calibration baselines only, not P2-G1).
    #[must_use]
    pub fn with_depth(depth: u32, tie_seed: u64) -> Self {
        Self {
            windows: windows(),
            depth,
            tie_seed,
        }
    }

    /// The static evaluation from the side to move's view.
    #[must_use]
    pub fn eval(&self, p: &Position) -> i32 {
        let (me, them) = (p.current(), p.opponent());
        let mut v = 0;
        for &w in &self.windows {
            let a = (me & w).count_ones();
            let b = (them & w).count_ones();
            if b == 0 {
                v += match a {
                    2 => W2,
                    3 => W3,
                    _ => 0,
                };
            } else if a == 0 {
                v -= match b {
                    2 => W2,
                    3 => W3,
                    _ => 0,
                };
            }
        }
        let centre = column_mask(3);
        v + WC * ((me & centre).count_ones() as i32 - (them & centre).count_ones() as i32)
    }

    fn negamax(&self, p: &Position, depth: u32, mut alpha: i32, beta: i32) -> i32 {
        let moves = p.moves() as i32;
        if p.can_win_next() {
            return WIN - moves - 1;
        }
        if p.moves() >= CELLS {
            return 0;
        }
        if depth == 0 {
            return self.eval(p);
        }
        let mut best = i32::MIN + 1;
        for &col in &CENTRE_ORDER {
            if !p.can_play(col) {
                continue;
            }
            let mut c = *p;
            c.play_col(col);
            let v = -self.negamax(&c, depth - 1, -beta, -alpha);
            best = best.max(v);
            alpha = alpha.max(v);
            if alpha >= beta {
                break;
            }
        }
        if best == i32::MIN + 1 { 0 } else { best }
    }

    /// The root scores of every legal column (full window, so ties are exact).
    #[must_use]
    pub fn root_scores(&self, p: &Position) -> [Option<i32>; 7] {
        let mut out = [None; 7];
        for col in 0..WIDTH {
            if !p.can_play(col) {
                continue;
            }
            let v = if p.is_winning_col(col) {
                WIN - p.moves() as i32
            } else {
                let mut c = *p;
                c.play_col(col);
                -self.negamax(&c, self.depth.saturating_sub(1), -i32::MAX, i32::MAX)
            };
            if let Some(slot) = out.get_mut(col as usize) {
                *slot = Some(v);
            }
        }
        out
    }

    /// The bot's move: a best root score; among ties, the pick of a
    /// `SplitMix64` keyed by the tie seed and the position key, over the
    /// tied columns in centre-first order.
    #[must_use]
    pub fn choose(&self, p: &Position) -> Option<u32> {
        let scores = self.root_scores(p);
        let best = scores.iter().flatten().copied().max()?;
        let tied: Vec<u32> = CENTRE_ORDER
            .iter()
            .copied()
            .filter(|&c| scores.get(c as usize).copied().flatten() == Some(best))
            .collect();
        let mut rng = SplitMix64::new(self.tie_seed ^ p.key());
        tied.get(rng.below(tied.len())).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(seq: &str) -> Position {
        Position::from_moves(seq).unwrap_or_default()
    }

    #[test]
    fn test_bot_has_69_windows() {
        assert_eq!(windows().len(), 69);
    }

    #[test]
    fn test_bot_takes_a_win_in_one() {
        // Side to move (first player) has three in column 1.
        assert_eq!(Bot::new(0).choose(&pos("121212")), Some(0));
    }

    #[test]
    fn test_bot_blocks_a_win_in_one() {
        // First player threatens column 1; second player must block.
        assert_eq!(Bot::new(0).choose(&pos("12121")), Some(0));
    }

    #[test]
    fn test_bot_takes_a_horizontal_win_in_one() {
        // First player holds columns 1 to 3 on the bottom row.
        assert_eq!(Bot::new(0).choose(&pos("172737")), Some(3));
    }

    #[test]
    fn test_bot_blocks_a_horizontal_win_in_one() {
        // Second player to move must take column 4.
        assert_eq!(Bot::new(0).choose(&pos("17273")), Some(3));
    }

    #[test]
    fn test_bot_is_deterministic_per_tie_seed() {
        let p = pos("");
        assert_eq!(Bot::new(5).choose(&p), Bot::new(5).choose(&p));
    }
}
