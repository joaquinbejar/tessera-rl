// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scripted P2-G1 opponent (ADR-0042), ported from the pinned
//! `tools/connect4-ref/src/bot.rs` without changing its play:
//! deterministic depth-limited negamax with alpha-beta over a fixed
//! window heuristic, centre-first move order and a seeded tie break among
//! equally scored root moves.
//!
//! The port allocates nothing (the windows are a constant table and the
//! tied columns a fixed array), so it can run inside a step. The tests
//! check that it chooses the same column and the same root scores as the
//! reference on every position of the calibration and on random games,
//! and they reproduce the ADR-0042 calibration.

use super::Connect4State;
use super::board::{CELLS, CENTRE_ORDER, HEIGHT, WIDTH, can_play, column, has_four, landing};

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
/// Seed base of the tie-break stream (ADR-0042: game or opening `j` uses
/// `TIE_BASE + j`), disjoint from the opening seeds.
pub const TIE_BASE: u64 = 0x7e55_e2a0_0000_0000;

const H1: u32 = HEIGHT + 1;
const WINDOW_COUNT: usize = 69;

/// The 69 four-cell windows, in the reference's order (column, row,
/// then direction).
const WINDOWS: [u64; WINDOW_COUNT] = windows();

#[expect(
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    reason = "a compile-time table: an index out of range fails the build"
)]
const fn windows() -> [u64; WINDOW_COUNT] {
    let dirs: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];
    let (w, h) = (WIDTH.cast_signed(), HEIGHT.cast_signed());
    let mut out = [0u64; WINDOW_COUNT];
    let mut n = 0;
    let mut c = 0;
    while c < w {
        let mut r = 0;
        while r < h {
            let mut d = 0;
            while d < dirs.len() {
                let (dc, dr) = dirs[d];
                let (ec, er) = (c + 3 * dc, r + 3 * dr);
                if ec >= 0 && ec < w && er >= 0 && er < h && n < WINDOW_COUNT {
                    let mut m = 0;
                    let mut k = 0;
                    while k < 4 {
                        let cc = (c + k * dc).unsigned_abs();
                        let rr = (r + k * dr).unsigned_abs();
                        m |= 1u64 << (cc * H1 + rr);
                        k += 1;
                    }
                    out[n] = m;
                    n += 1;
                }
                d += 1;
            }
            r += 1;
        }
        c += 1;
    }
    out
}

/// A search position: the side to move's stones, the occupied cells and
/// the plies played (the reference's representation).
#[derive(Clone, Copy)]
struct Pos {
    current: u64,
    mask: u64,
    moves: u32,
}

const BOARD: u64 = super::board::BOARD;

/// Empty cells that would complete four for `pos`.
const fn winning_cells(pos: u64, mask: u64) -> u64 {
    /// The cells completing a line of direction `s` (horizontal and the
    /// diagonals).
    const fn along(pos: u64, s: u32) -> u64 {
        let p = (pos << s) & (pos << (2 * s));
        let q = (pos >> s) & (pos >> (2 * s));
        (p & (pos << (3 * s))) | (p & (pos >> s)) | (q & (pos << s)) | (q & (pos >> (3 * s)))
    }
    let vertical = (pos << 1) & (pos << 2) & (pos << 3);
    let r = vertical | along(pos, H1) | along(pos, HEIGHT) | along(pos, HEIGHT + 2);
    r & (BOARD ^ mask)
}

const fn bottom_row() -> u64 {
    let mut b = 0;
    let mut c = 0;
    while c < WIDTH {
        b |= 1u64 << (c * H1);
        c += 1;
    }
    b
}

const BOTTOM: u64 = bottom_row();

impl Pos {
    fn of(state: &Connect4State) -> Self {
        let current = state.stones.get(state.to_move()).copied().unwrap_or(0);
        Self {
            current,
            mask: state.occupied(),
            moves: state.moves,
        }
    }

    const fn opponent(&self) -> u64 {
        self.current ^ self.mask
    }

    fn play(&mut self, col: u32) {
        self.current ^= self.mask;
        self.mask |= landing(self.mask, col);
        self.moves += 1;
    }

    fn wins_with(&self, col: u32) -> bool {
        has_four(self.current | landing(self.mask, col))
    }

    const fn can_win_next(&self) -> bool {
        winning_cells(self.current, self.mask) & ((self.mask + BOTTOM) & BOARD) != 0
    }

    const fn key(&self) -> u64 {
        self.current + self.mask
    }

    fn plies(&self) -> i32 {
        i32::try_from(self.moves).unwrap_or(0)
    }
}

/// `SplitMix64`, the tie-break generator of the reference tools.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in `[0, n)` by multiply-shift.
    fn below(&mut self, n: usize) -> usize {
        let n64 = u64::try_from(n).unwrap_or(0);
        usize::try_from((u128::from(self.next_u64()) * u128::from(n64)) >> 64).unwrap_or(0)
    }
}

fn count(stones: u64) -> i32 {
    i32::try_from(stones.count_ones()).unwrap_or(0)
}

/// The scripted bot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bot {
    depth: u32,
    tie_seed: u64,
}

impl Bot {
    /// The pinned bot with tie-break seed `tie_seed`.
    #[must_use]
    pub const fn new(tie_seed: u64) -> Self {
        Self::with_depth(DEPTH, tie_seed)
    }

    /// A bot of another depth (calibration baselines only, not P2-G1).
    #[must_use]
    pub const fn with_depth(depth: u32, tie_seed: u64) -> Self {
        Self { depth, tie_seed }
    }

    /// Search depth in plies.
    #[must_use]
    pub const fn depth(&self) -> u32 {
        self.depth
    }

    /// Tie-break seed.
    #[must_use]
    pub const fn tie_seed(&self) -> u64 {
        self.tie_seed
    }

    /// The static evaluation from the side to move's view.
    #[must_use]
    pub fn eval(&self, state: &Connect4State) -> i32 {
        eval(&Pos::of(state))
    }

    /// The root score of every column, `None` for a full one (full
    /// window, so ties are exact).
    #[must_use]
    pub fn root_scores(&self, state: &Connect4State) -> [Option<i32>; 7] {
        root_scores(&Pos::of(state), self.depth)
    }

    /// The bot's column: a best root score; among ties, the pick of a
    /// `SplitMix64` keyed by the tie seed and the position key, over the
    /// tied columns in centre-first order. `None` on a full board.
    #[must_use]
    pub fn choose(&self, state: &Connect4State) -> Option<u32> {
        let p = Pos::of(state);
        let scores = root_scores(&p, self.depth);
        let best = scores.iter().flatten().copied().max()?;
        let mut tied = [0u32; 7];
        let mut n = 0;
        for c in CENTRE_ORDER {
            if scores.get(c as usize).copied().flatten() == Some(best)
                && let Some(slot) = tied.get_mut(n)
            {
                *slot = c;
                n += 1;
            }
        }
        let mut rng = SplitMix64(self.tie_seed ^ p.key());
        tied.get(..n)?.get(rng.below(n)).copied()
    }
}

fn eval(p: &Pos) -> i32 {
    let (me, them) = (p.current, p.opponent());
    let mut v = 0;
    for w in WINDOWS {
        let a = (me & w).count_ones();
        let b = (them & w).count_ones();
        let weight = |k| match k {
            2 => W2,
            3 => W3,
            _ => 0,
        };
        if b == 0 {
            v += weight(a);
        } else if a == 0 {
            v -= weight(b);
        }
    }
    let centre = column(3);
    v + WC * (count(me & centre) - count(them & centre))
}

fn negamax(p: &Pos, depth: u32, mut alpha: i32, beta: i32) -> i32 {
    if p.can_win_next() {
        return WIN - p.plies() - 1;
    }
    if p.moves >= CELLS {
        return 0;
    }
    if depth == 0 {
        return eval(p);
    }
    let mut best = i32::MIN + 1;
    for col in CENTRE_ORDER {
        if !can_play(p.mask, col) {
            continue;
        }
        let mut c = *p;
        c.play(col);
        let v = -negamax(&c, depth - 1, -beta, -alpha);
        best = best.max(v);
        alpha = alpha.max(v);
        if alpha >= beta {
            break;
        }
    }
    if best == i32::MIN + 1 { 0 } else { best }
}

fn root_scores(p: &Pos, depth: u32) -> [Option<i32>; 7] {
    let mut out = [None; 7];
    for col in 0..WIDTH {
        if !can_play(p.mask, col) {
            continue;
        }
        let v = if p.wins_with(col) {
            WIN - p.plies()
        } else {
            let mut c = *p;
            c.play(col);
            -negamax(&c, depth.saturating_sub(1), -i32::MAX, i32::MAX)
        };
        if let Some(slot) = out.get_mut(col as usize) {
            *slot = Some(v);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bot_window_table_has_69_distinct_four_cell_windows() {
        for (i, w) in WINDOWS.iter().enumerate() {
            assert_eq!(w.count_ones(), 4);
            assert_eq!(w & !BOARD, 0);
            assert!(!WINDOWS.get(..i).is_some_and(|prev| prev.contains(w)));
        }
    }
}
