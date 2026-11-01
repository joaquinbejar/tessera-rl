// SPDX-License-Identifier: MIT OR Apache-2.0

//! Connect Four position as two bitboards (7 columns of 6 rows plus one
//! sentinel bit each; bit `col * 7 + row`, row 0 at the bottom).
//!
//! The representation and the threat computations follow the classic
//! bitboard solver design (current player's stones and the occupied mask).

/// Board width.
pub const WIDTH: u32 = 7;
/// Board height.
pub const HEIGHT: u32 = 6;
/// Cells on the board.
pub const CELLS: u32 = WIDTH * HEIGHT;

const H1: u32 = HEIGHT + 1;

/// Columns from the centre outwards: the fixed move order.
pub const CENTRE_ORDER: [u32; 7] = [3, 2, 4, 1, 5, 0, 6];

const fn bottom_mask_col(col: u32) -> u64 {
    1u64 << (col * H1)
}

const fn top_mask_col(col: u32) -> u64 {
    1u64 << (HEIGHT - 1 + col * H1)
}

/// The cells of column `col`.
#[must_use]
pub const fn column_mask(col: u32) -> u64 {
    ((1u64 << HEIGHT) - 1) << (col * H1)
}

const fn bottom() -> u64 {
    let mut b = 0;
    let mut c = 0;
    while c < WIDTH {
        b |= bottom_mask_col(c);
        c += 1;
    }
    b
}

const BOTTOM: u64 = bottom();
const BOARD: u64 = BOTTOM * ((1u64 << HEIGHT) - 1);

/// Cells that would complete four for `pos` (empty cells only).
#[must_use]
pub fn winning_cells(pos: u64, mask: u64) -> u64 {
    // Vertical.
    let mut r = (pos << 1) & (pos << 2) & (pos << 3);
    for s in [H1, HEIGHT, HEIGHT + 2] {
        let p = (pos << s) & (pos << (2 * s));
        r |= p & (pos << (3 * s));
        r |= p & (pos >> s);
        let p = (pos >> s) & (pos >> (2 * s));
        r |= p & (pos << s);
        r |= p & (pos >> (3 * s));
    }
    r & (BOARD ^ mask)
}

/// Whether `pos` holds four in a row.
#[must_use]
pub fn has_four(pos: u64) -> bool {
    for s in [H1, HEIGHT, HEIGHT + 2, 1] {
        let m = pos & (pos >> s);
        if m & (m >> (2 * s)) != 0 {
            return true;
        }
    }
    false
}

/// A position: the side to move's stones, the occupied cells, plies played.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Position {
    current: u64,
    mask: u64,
    moves: u32,
}

impl Position {
    /// The empty board.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            current: 0,
            mask: 0,
            moves: 0,
        }
    }

    /// Plies played.
    #[must_use]
    pub const fn moves(&self) -> u32 {
        self.moves
    }

    /// The side to move's stones.
    #[must_use]
    pub const fn current(&self) -> u64 {
        self.current
    }

    /// The opponent's stones.
    #[must_use]
    pub const fn opponent(&self) -> u64 {
        self.current ^ self.mask
    }

    /// Whether column `col` (0-based) has room.
    #[must_use]
    pub const fn can_play(&self, col: u32) -> bool {
        col < WIDTH && self.mask & top_mask_col(col) == 0
    }

    /// Plays column `col`; the caller checked `can_play`.
    pub fn play_col(&mut self, col: u32) {
        self.play_bit((self.mask + bottom_mask_col(col)) & column_mask(col));
    }

    /// Plays the move given as its single cell bit.
    pub fn play_bit(&mut self, mv: u64) {
        self.current ^= self.mask;
        self.mask |= mv;
        self.moves += 1;
    }

    /// Whether playing `col` wins at once.
    #[must_use]
    pub fn is_winning_col(&self, col: u32) -> bool {
        has_four(self.current | ((self.mask + bottom_mask_col(col)) & column_mask(col)))
    }

    /// Playable cells (one per non-full column).
    #[must_use]
    pub const fn possible(&self) -> u64 {
        (self.mask + BOTTOM) & BOARD
    }

    /// Whether the side to move can win with its next stone.
    #[must_use]
    pub fn can_win_next(&self) -> bool {
        winning_cells(self.current, self.mask) & self.possible() != 0
    }

    /// Moves that do not hand the opponent an immediate win (0 if all lose).
    #[must_use]
    pub fn non_losing_moves(&self) -> u64 {
        let mut possible = self.possible();
        let opp_win = winning_cells(self.opponent(), self.mask);
        let forced = possible & opp_win;
        if forced != 0 {
            if forced & (forced - 1) != 0 {
                return 0;
            }
            possible = forced;
        }
        possible & !(opp_win >> 1)
    }

    /// Threats created by playing `mv` (move ordering).
    #[must_use]
    pub fn move_score(&self, mv: u64) -> u32 {
        winning_cells(self.current | mv, self.mask).count_ones()
    }

    /// A unique key of the position.
    #[must_use]
    pub const fn key(&self) -> u64 {
        self.current + self.mask
    }

    /// The key of the left-right mirror, for de-duplication.
    #[must_use]
    pub fn mirror_key(&self) -> u64 {
        let mirror = |b: u64| {
            let mut out = 0;
            for c in 0..WIDTH {
                let colbits = (b >> (c * H1)) & ((1u64 << H1) - 1);
                out |= colbits << ((WIDTH - 1 - c) * H1);
            }
            out
        };
        mirror(self.current) + mirror(self.mask)
    }

    /// The canonical (mirror-invariant) key.
    #[must_use]
    pub fn canonical_key(&self) -> u64 {
        self.key().min(self.mirror_key())
    }

    /// Plays a sequence of 1-based column digits; `None` if a move is
    /// illegal or a game is already won before the last move.
    #[must_use]
    pub fn from_moves(seq: &str) -> Option<Self> {
        let mut p = Self::new();
        for ch in seq.chars() {
            let col = ch.to_digit(10)?.checked_sub(1)?;
            if !p.can_play(col) || p.is_winning_col(col) {
                return None;
            }
            p.play_col(col);
        }
        Some(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_vertical_four_is_won() {
        let mut p = Position::new();
        for c in [0, 1, 0, 1, 0, 1] {
            p.play_col(c);
        }
        assert!(p.is_winning_col(0));
        assert!(!p.is_winning_col(2));
    }

    #[test]
    fn test_board_from_moves_rejects_full_column() {
        assert!(Position::from_moves("1111111").is_none());
        assert!(Position::from_moves("444444").is_some());
    }

    #[test]
    fn test_board_mirror_of_mirror_is_identity() {
        let p = Position::from_moves("1234").unwrap_or_default();
        let q = Position::from_moves("7654").unwrap_or_default();
        assert_eq!(p.mirror_key(), q.key());
        assert_eq!(p.canonical_key(), q.canonical_key());
    }
}
