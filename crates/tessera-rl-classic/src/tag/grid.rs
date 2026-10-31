// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tag grid geometry: moves, bounds and distinct placement.

use core::num::NonZeroU32;

use tessera_rl_num::rng::StepRng;

/// Number of actions: stay, north, south, east, west.
pub const ACTIONS: u32 = 5;
/// The no-op action.
pub const STAY: i32 = 0;

/// A cell `[x, y]`; `y` grows southwards.
pub type Cell = [i16; 2];

/// The displacement of `action`; unknown actions stay.
#[must_use]
#[inline]
pub const fn delta(action: i32) -> [i32; 2] {
    match action {
        1 => [0, -1],
        2 => [0, 1],
        3 => [1, 0],
        4 => [-1, 0],
        _ => [0, 0],
    }
}

/// `cell` moved by `action`, or `None` if that leaves the `width` by
/// `height` grid.
#[must_use]
#[inline]
pub fn moved(cell: Cell, action: i32, width: u32, height: u32) -> Option<Cell> {
    let [dx, dy] = delta(action);
    let x = i32::from(cell[0]) + dx;
    let y = i32::from(cell[1]) + dy;
    let inside = |v: i32, n: u32| u32::try_from(v).is_ok_and(|v| v < n);
    if !(inside(x, width) && inside(y, height)) {
        return None;
    }
    Some([i16::try_from(x).ok()?, i16::try_from(y).ok()?])
}

/// Places `cells.len()` agents on distinct cells of the grid, uniformly at
/// random, in slot order. Needs `cells.len() <= width * height`.
pub fn place(cells: &mut [Cell], width: u32, height: u32, rng: &mut StepRng) {
    let total = width * height;
    for i in 0..cells.len() {
        let (placed, rest) = cells.split_at_mut(i);
        let Some(free) = u32::try_from(i)
            .ok()
            .and_then(|i| total.checked_sub(i))
            .and_then(NonZeroU32::new)
        else {
            return;
        };
        // The k-th free cell in row-major order.
        let mut k = rng.below_u32(free);
        for index in 0..total {
            let cell = [
                i16::try_from(index % width).unwrap_or(0),
                i16::try_from(index / width).unwrap_or(0),
            ];
            if placed.contains(&cell) {
                continue;
            }
            if k == 0 {
                if let Some(slot) = rest.first_mut() {
                    *slot = cell;
                }
                break;
            }
            k -= 1;
        }
    }
}

/// Manhattan distance.
#[must_use]
#[inline]
pub fn distance(a: Cell, b: Cell) -> u32 {
    (i32::from(a[0]) - i32::from(b[0])).unsigned_abs()
        + (i32::from(a[1]) - i32::from(b[1])).unsigned_abs()
}

#[cfg(test)]
mod tests {
    use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
    use tessera_rl_num::rng::{RngAuthority, StepRng};

    use super::{distance, moved, place};

    #[test]
    fn test_moved_off_the_grid_is_none() {
        assert_eq!(moved([0, 0], 1, 3, 3), None);
        assert_eq!(moved([0, 0], 4, 3, 3), None);
        assert_eq!(moved([2, 2], 2, 3, 3), None);
        assert_eq!(moved([2, 2], 3, 3, 3), None);
        assert_eq!(moved([1, 1], 1, 3, 3), Some([1, 0]));
        assert_eq!(moved([1, 1], 0, 3, 3), Some([1, 1]));
    }

    #[test]
    fn test_place_fills_every_cell_distinctly() {
        let identity = RngIdentity::new(Seed::new(7), LogicalId::new(0));
        let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
        let mut cells = [[-1, -1]; 9];
        place(&mut cells, 3, 3, &mut rng);
        for (i, a) in cells.iter().enumerate() {
            assert!((0..3).contains(&a[0]) && (0..3).contains(&a[1]));
            assert!(cells.iter().skip(i + 1).all(|b| b != a));
        }
        assert_eq!(distance([0, 0], [2, 1]), 3);
    }
}
