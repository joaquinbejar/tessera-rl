// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Othello board as bitboards: bit `row * 8 + col` is a square, with
//! row 0 the top rank and col 0 the left file, so `a1` is bit 0 and `h8`
//! bit 63. Directional fills give the legal squares and the flips without
//! tables or allocation.

/// Squares on a side of the board.
pub const SIDE: u32 = 8;
/// Squares on the board.
pub const SQUARES: u32 = SIDE * SIDE;
/// The four centre squares (`d4`, `e4`, `d5`, `e5`): occupied from the
/// start, so a placement never lands there.
pub const CENTRE: u64 = square(3, 3) | square(4, 3) | square(3, 4) | square(4, 4);
/// White's starting discs (`d4`, `e5`).
pub const START_WHITE: u64 = square(3, 3) | square(4, 4);
/// Black's starting discs (`e4`, `d5`).
pub const START_BLACK: u64 = square(4, 3) | square(3, 4);
/// The directions of a line on the board.
pub const DIRECTIONS: usize = 8;

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = 0x8080_8080_8080_8080;
/// The longest run of opposing discs a line can bracket.
const MAX_RUN: u32 = SIDE - 2;

/// The bit of the square at `col`, `row`.
#[inline]
#[must_use]
pub const fn square(col: u32, row: u32) -> u64 {
    1u64 << (row * SIDE + col)
}

/// Moves every square of `b` one step in direction `dir` (0 to 7),
/// dropping squares that leave the board.
#[inline]
#[must_use]
pub const fn shift(b: u64, dir: usize) -> u64 {
    match dir {
        0 => (b & !FILE_H) << 1,
        1 => (b & !FILE_A) >> 1,
        2 => b << SIDE,
        3 => b >> SIDE,
        4 => (b & !FILE_H) << (SIDE + 1),
        5 => (b & !FILE_A) >> (SIDE + 1),
        6 => (b & !FILE_A) << (SIDE - 1),
        _ => (b & !FILE_H) >> (SIDE - 1),
    }
}

/// The run of `opp` discs starting next to `from` in direction `dir`.
#[inline]
const fn run(from: u64, opp: u64, dir: usize) -> u64 {
    let mut x = shift(from, dir) & opp;
    let mut k = 1;
    while k < MAX_RUN {
        x |= shift(x, dir) & opp;
        k += 1;
    }
    x
}

/// The squares where a side holding `me` may place against `opp`.
#[must_use]
pub const fn legal_squares(me: u64, opp: u64) -> u64 {
    let empty = !(me | opp);
    let mut legal = 0;
    let mut dir = 0;
    while dir < DIRECTIONS {
        legal |= shift(run(me, opp, dir), dir) & empty;
        dir += 1;
    }
    legal
}

/// The `opp` discs flipped by placing on the square `at` while holding
/// `me`.
#[must_use]
pub const fn flips(me: u64, opp: u64, at: u64) -> u64 {
    let mut flipped = 0;
    let mut dir = 0;
    while dir < DIRECTIONS {
        let x = run(at, opp, dir);
        if shift(x, dir) & me != 0 {
            flipped |= x;
        }
        dir += 1;
    }
    flipped
}

/// `b` and every square next to it, in all eight directions.
#[must_use]
pub const fn with_neighbours(b: u64) -> u64 {
    let mut out = b;
    let mut dir = 0;
    while dir < DIRECTIONS {
        out |= shift(b, dir);
        dir += 1;
    }
    out
}

/// Whether the squares of `b` are one 8-connected group.
#[must_use]
pub const fn is_connected(b: u64) -> bool {
    if b == 0 {
        return true;
    }
    let mut reached = b & b.wrapping_neg();
    loop {
        let next = with_neighbours(reached) & b;
        if next == reached {
            return reached == b;
        }
        reached = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_shift_never_wraps_around_an_edge() {
        for dir in 0..DIRECTIONS {
            for sq in 0..SQUARES {
                let (col, row) = (sq % SIDE, sq / SIDE);
                let moved = shift(1u64 << sq, dir);
                assert!(moved.count_ones() <= 1);
                if moved != 0 {
                    let to = moved.trailing_zeros();
                    assert!((to % SIDE).abs_diff(col) <= 1 && (to / SIDE).abs_diff(row) <= 1);
                }
            }
        }
    }

    #[test]
    fn test_board_opening_squares_are_the_four_published_ones() {
        let expected = square(3, 2) | square(2, 3) | square(5, 4) | square(4, 5);
        assert_eq!(legal_squares(START_BLACK, START_WHITE), expected);
    }

    #[test]
    fn test_board_connectivity_splits_on_a_gap() {
        assert!(is_connected(CENTRE));
        assert!(!is_connected(CENTRE | square(0, 0)));
    }
}
