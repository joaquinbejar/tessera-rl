// SPDX-License-Identifier: MIT OR Apache-2.0

//! Connect Four bitboards: 7 columns of 6 rows plus one sentinel bit each,
//! cell `(col, row)` at bit `col * 7 + row`, row 0 at the bottom. The
//! layout is the one of the M0 reference tools (`tools/connect4-ref`).

/// Board width (columns).
pub const WIDTH: u32 = 7;
/// Board height (rows).
pub const HEIGHT: u32 = 6;
/// Cells on the board.
pub const CELLS: u32 = WIDTH * HEIGHT;

const H1: u32 = HEIGHT + 1;

/// Columns from the centre outwards.
pub const CENTRE_ORDER: [u32; 7] = [3, 2, 4, 1, 5, 0, 6];

/// The bit of cell `(col, row)`.
#[must_use]
pub const fn cell(col: u32, row: u32) -> u64 {
    1u64 << (col * H1 + row)
}

/// The bottom cell of column `col`.
#[must_use]
const fn bottom_col(col: u32) -> u64 {
    1u64 << (col * H1)
}

/// The top cell of column `col`.
#[must_use]
const fn top_col(col: u32) -> u64 {
    1u64 << (HEIGHT - 1 + col * H1)
}

/// The cells of column `col`.
#[must_use]
pub const fn column(col: u32) -> u64 {
    ((1u64 << HEIGHT) - 1) << (col * H1)
}

const fn bottom_row() -> u64 {
    let mut b = 0;
    let mut c = 0;
    while c < WIDTH {
        b |= bottom_col(c);
        c += 1;
    }
    b
}

const BOTTOM: u64 = bottom_row();
/// Every cell of the board (no sentinel bits).
pub const BOARD: u64 = BOTTOM * ((1u64 << HEIGHT) - 1);

/// Whether `stones` holds four in a row.
#[must_use]
pub const fn has_four(stones: u64) -> bool {
    // Vertical, horizontal and the two diagonals.
    const fn line(stones: u64, s: u32) -> bool {
        let m = stones & (stones >> s);
        m & (m >> (2 * s)) != 0
    }
    line(stones, 1) || line(stones, H1) || line(stones, HEIGHT) || line(stones, HEIGHT + 2)
}

/// Whether column `col` has room, given the occupied cells `occupied`.
#[must_use]
pub const fn can_play(occupied: u64, col: u32) -> bool {
    col < WIDTH && occupied & top_col(col) == 0
}

/// The cell a stone dropped into column `col` lands on (0 if full).
#[must_use]
pub const fn landing(occupied: u64, col: u32) -> u64 {
    (occupied + bottom_col(col)) & column(col)
}

/// Whether `occupied` obeys gravity: each column is filled from the
/// bottom without gaps, and no sentinel bit is set.
#[must_use]
pub const fn is_stacked(occupied: u64) -> bool {
    if occupied & !BOARD != 0 {
        return false;
    }
    let mut c = 0;
    while c < WIDTH {
        // The column's bits shifted down are `2^h - 1` for some height h.
        let bits = (occupied & column(c)) >> (c * H1);
        if bits & (bits + 1) != 0 {
            return false;
        }
        c += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_four_in_every_direction_is_found() {
        let line = |dc: u32, dr: u32, c0: u32, r0: u32| {
            (0..4).fold(0, |b, k| b | cell(c0 + k * dc, r0 + k * dr))
        };
        assert!(has_four(line(0, 1, 2, 1)));
        assert!(has_four(line(1, 0, 3, 5)));
        assert!(has_four(line(1, 1, 0, 0)));
        let anti = (0..4).fold(0, |b, k| b | cell(k + 1, 4 - k));
        assert!(has_four(anti));
        assert!(!has_four(line(0, 1, 0, 0) & !cell(0, 3)));
    }

    #[test]
    fn test_board_four_does_not_wrap_across_columns() {
        // Top two of column 0 and bottom two of column 1 are adjacent bits
        // only through the sentinel, which stays empty.
        let b = cell(0, 4) | cell(0, 5) | cell(1, 0) | cell(1, 1);
        assert!(!has_four(b));
    }

    #[test]
    fn test_board_stacking_rejects_gaps_and_sentinels() {
        assert!(is_stacked(0));
        assert!(is_stacked(cell(3, 0) | cell(3, 1)));
        assert!(!is_stacked(cell(3, 1)));
        assert!(!is_stacked(1u64 << HEIGHT));
        assert!(is_stacked(BOARD));
    }

    #[test]
    fn test_board_landing_fills_bottom_up() {
        let mut occ = 0;
        for row in 0..HEIGHT {
            assert!(can_play(occ, 2));
            let at = landing(occ, 2);
            assert_eq!(at, cell(2, row));
            occ |= at;
        }
        assert!(!can_play(occ, 2));
        assert!(!can_play(occ, WIDTH));
    }
}
