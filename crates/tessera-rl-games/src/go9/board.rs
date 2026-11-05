// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Go 9x9 board as 81-bit sets in a `u128`: bit `row * 9 + col` is a
//! point. Groups are found by flood fill inside a set, without tables or
//! allocation.

/// Points on a side of the board.
pub const SIDE: u32 = 9;
/// Points on the board.
pub const POINTS: u32 = SIDE * SIDE;
/// Every point of the board.
pub const BOARD: u128 = (1u128 << POINTS) - 1;

const fn left_column() -> u128 {
    let mut b = 0;
    let mut row = 0;
    while row < SIDE {
        b |= 1u128 << (row * SIDE);
        row += 1;
    }
    b
}

const LEFT: u128 = left_column();
const RIGHT: u128 = LEFT << (SIDE - 1);

/// The bit of the point at `col`, `row`.
#[inline]
#[must_use]
pub const fn point(col: u32, row: u32) -> u128 {
    1u128 << (row * SIDE + col)
}

/// The points next to any point of `b`, which may include `b` itself.
#[inline]
#[must_use]
pub const fn neighbours(b: u128) -> u128 {
    (((b & !RIGHT) << 1) | ((b & !LEFT) >> 1) | (b << SIDE) | (b >> SIDE)) & BOARD
}

/// The group of `seed` inside `within`: `seed` must lie in `within`.
#[must_use]
pub const fn group(seed: u128, within: u128) -> u128 {
    let mut g = seed;
    loop {
        let next = (g | neighbours(g)) & within;
        if next == g {
            return g;
        }
        g = next;
    }
}

/// The result of placing a stone on an empty point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placed {
    /// The mover's stones afterwards.
    pub me: u128,
    /// The opponent's stones afterwards.
    pub opp: u128,
}

/// Places `at` for the side holding `me`: opposing groups left without a
/// liberty go first, then the mover's own group needs one. `None` is
/// suicide. `at` must be empty.
#[must_use]
pub const fn place(me: u128, opp: u128, at: u128) -> Option<Placed> {
    let mine = me | at;
    let mut theirs = opp;
    let mut touching = neighbours(at) & opp;
    while touching != 0 {
        let g = group(touching & touching.wrapping_neg(), opp);
        touching &= !g;
        if neighbours(g) & !(mine | opp) & BOARD == 0 {
            theirs &= !g;
        }
    }
    if neighbours(group(at, mine)) & !(mine | theirs) & BOARD == 0 {
        return None;
    }
    Some(Placed {
        me: mine,
        opp: theirs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_neighbours_never_wrap_around_an_edge() {
        assert_eq!(
            neighbours(point(8, 0)),
            point(7, 0) | point(8, 1),
            "top right corner"
        );
        assert_eq!(
            neighbours(point(0, 8)),
            point(0, 7) | point(1, 8),
            "bottom left corner"
        );
    }

    #[test]
    fn test_board_place_captures_before_checking_suicide() {
        let me = point(1, 0);
        let opp = point(0, 0);
        let placed = place(me, opp, point(0, 1));
        assert_eq!(
            placed,
            Some(Placed {
                me: me | point(0, 1),
                opp: 0
            })
        );
    }

    #[test]
    fn test_board_place_refuses_suicide() {
        let opp = point(1, 0) | point(0, 1);
        assert_eq!(place(0, opp, point(0, 0)), None);
    }
}
