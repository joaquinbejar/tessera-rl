// SPDX-License-Identifier: MIT OR Apache-2.0

//! Map generation, integers only (DESIGN-r1 section 5).
//!
//! 1. An integer value-noise field. Every lattice node (each 8 cells, one
//!    more than the map in each direction) gets a byte from the instance's
//!    `StepRng`, row by row. Each cell interpolates its four nodes
//!    bilinearly in Q8 fixed point (integer multiply and shift), giving a
//!    value in `0..=255`.
//! 2. Tile classes from the field's order statistics, by counting: the
//!    lowest values become water (`water_permille` of the cells), the
//!    highest stone (`stone_permille`), the next highest forest
//!    (`forest_permille`), the rest grass. A threshold value is one class
//!    as a whole, so the shares are met up to the cells that share the
//!    threshold value.
//! 3. Connectivity: a flood fill over passable tiles (grass and forest)
//!    from the passable tile nearest the centre. Unreached passable tiles
//!    become stone, so every passable tile is reachable from every other.
//!
//! Everything runs on fixed-size arrays: reset allocates nothing.

use core::num::NonZeroU32;

use tessera_rl_num::rng::StepRng;

use super::{FOREST, GRASS, MAX_SIZE, STONE, WATER};

/// The lattice spacing, in cells.
pub const CELL: usize = 8;
/// Lattice nodes per side, for the largest map.
const NODES: usize = MAX_SIZE / CELL + 2;

/// The tile shares of a map, per mille.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shares {
    /// Water.
    pub water: u32,
    /// Forest.
    pub forest: u32,
    /// Stone.
    pub stone: u32,
}

/// Writes a `size` x `size` map into `tiles`. Cells beyond `size` are left
/// as they were (zero after a reset).
pub fn generate(
    rng: &mut StepRng,
    size: usize,
    shares: Shares,
    tiles: &mut [[u8; MAX_SIZE]; MAX_SIZE],
) {
    let size = size.min(MAX_SIZE);
    let mut nodes = [[0u8; NODES]; NODES];
    let per_side = size / CELL + 2;
    for row in nodes.iter_mut().take(per_side) {
        for node in row.iter_mut().take(per_side) {
            // Byte from the top bits of a uniform draw.
            *node = u8::try_from(rng.below_u32(NonZeroU32::new(256).unwrap_or(NonZeroU32::MIN)))
                .unwrap_or(0);
        }
    }
    let mut field = [[0u8; MAX_SIZE]; MAX_SIZE];
    let mut counts = [0u32; 256];
    for (y, field_row) in field.iter_mut().enumerate().take(size) {
        for (x, cell) in field_row.iter_mut().enumerate().take(size) {
            let v = interpolate(&nodes, x, y);
            *cell = v;
            if let Some(c) = counts.get_mut(usize::from(v)) {
                *c += 1;
            }
        }
    }
    let total = super::cells(size);
    // At most 9,216 cells times 1,000 per mille: no overflow (asserted in
    // `super::cells`).
    let share = |p: u32| total * p / 1000;
    // Water: the lowest values up to the water share.
    let water_top = threshold_low(&counts, share(shares.water));
    // Stone, then forest: the highest values.
    let stone_low = threshold_high(&counts, share(shares.stone));
    let forest_low = threshold_high(&counts, share(shares.stone) + share(shares.forest));
    for (row, field_row) in tiles.iter_mut().zip(field.iter()).take(size) {
        for (tile, &v) in row.iter_mut().zip(field_row.iter()).take(size) {
            let v = u16::from(v);
            *tile = if water_top.is_some_and(|t| v <= t) {
                WATER
            } else if stone_low.is_some_and(|t| v >= t) {
                STONE
            } else if forest_low.is_some_and(|t| v >= t) {
                FOREST
            } else {
                GRASS
            };
        }
    }
    connect(tiles, size);
}

/// The Q8 bilinear interpolation of cell `(x, y)` from its four lattice
/// nodes.
fn interpolate(nodes: &[[u8; NODES]; NODES], x: usize, y: usize) -> u8 {
    let (gx, gy) = (x / CELL, y / CELL);
    // Q8 weights: 0, 32, ..., 224 of 256.
    let wx = u32::try_from((x % CELL) * (256 / CELL)).unwrap_or(0);
    let wy = u32::try_from((y % CELL) * (256 / CELL)).unwrap_or(0);
    let at = |i: usize, j: usize| -> u32 {
        u32::from(nodes.get(j).and_then(|r| r.get(i)).copied().unwrap_or(0))
    };
    let top = at(gx, gy) * (256 - wx) + at(gx + 1, gy) * wx;
    let bottom = at(gx, gy + 1) * (256 - wx) + at(gx + 1, gy + 1) * wx;
    // (top * (256 - wy) + bottom * wy) is at most 255 * 2^16: fits in u32.
    let v = (top * (256 - wy) + bottom * wy) >> 16;
    u8::try_from(v).unwrap_or(u8::MAX)
}

/// The largest value `t` such that the cells with values at most `t`
/// number at most `want`, or `None` when even the lowest value has more.
fn threshold_low(counts: &[u32; 256], want: u32) -> Option<u16> {
    let mut seen = 0u32;
    let mut best = None;
    for (v, &c) in counts.iter().enumerate() {
        seen += c;
        if seen > want {
            break;
        }
        if c > 0 {
            best = u16::try_from(v).ok();
        }
    }
    best
}

/// The smallest value `t` such that the cells with values at least `t`
/// number at most `want`, or `None` when even the highest value has more.
fn threshold_high(counts: &[u32; 256], want: u32) -> Option<u16> {
    let mut seen = 0u32;
    let mut best = None;
    for (v, &c) in counts.iter().enumerate().rev() {
        seen += c;
        if seen > want {
            break;
        }
        if c > 0 {
            best = u16::try_from(v).ok();
        }
    }
    best
}

/// Whether a tile can be stood on.
#[must_use]
pub const fn passable(tile: u8) -> bool {
    tile == GRASS || tile == FOREST
}

/// Turns passable tiles unreachable from the centre's region into stone.
fn connect(tiles: &mut [[u8; MAX_SIZE]; MAX_SIZE], size: usize) {
    let tile = |t: &[[u8; MAX_SIZE]; MAX_SIZE], x: usize, y: usize| {
        t.get(y).and_then(|r| r.get(x)).copied().unwrap_or(STONE)
    };
    // The passable tile nearest the centre, scanning rings outward.
    let c = size / 2;
    let mut start = None;
    'rings: for r in 0..size {
        // The ring of radius r around c, clipped to the map.
        let (low, high) = (c - r.min(c), (c + r).min(size - 1));
        for y in low..=high {
            for x in low..=high {
                if passable(tile(tiles, x, y)) {
                    start = Some((x, y));
                    break 'rings;
                }
            }
        }
    }
    let mut seen = [[false; MAX_SIZE]; MAX_SIZE];
    // A stack of cells, at most every cell once, as two coordinate arrays
    // (each below the 16 KB local-array limit).
    let mut stack_x = [0u8; MAX_SIZE * MAX_SIZE];
    let mut stack_y = [0u8; MAX_SIZE * MAX_SIZE];
    let mut top = 0usize;
    if let Some((x, y)) = start
        && let (Some(sx), Some(sy), Some(cell)) = (
            stack_x.get_mut(0),
            stack_y.get_mut(0),
            seen.get_mut(y).and_then(|r| r.get_mut(x)),
        )
    {
        *sx = u8::try_from(x).unwrap_or(0);
        *sy = u8::try_from(y).unwrap_or(0);
        *cell = true;
        top = 1;
    }
    while top > 0 {
        top -= 1;
        let x = usize::from(stack_x.get(top).copied().unwrap_or(0));
        let y = usize::from(stack_y.get(top).copied().unwrap_or(0));
        let neighbours = [
            x.checked_sub(1).map(|nx| (nx, y)),
            Some((x + 1, y)),
            y.checked_sub(1).map(|ny| (x, ny)),
            Some((x, y + 1)),
        ];
        for (nx, ny) in neighbours.into_iter().flatten() {
            if nx >= size || ny >= size || !passable(tile(tiles, nx, ny)) {
                continue;
            }
            let Some(cell) = seen.get_mut(ny).and_then(|r| r.get_mut(nx)) else {
                continue;
            };
            if *cell {
                continue;
            }
            *cell = true;
            if let (Some(sx), Some(sy)) = (stack_x.get_mut(top), stack_y.get_mut(top)) {
                *sx = u8::try_from(nx).unwrap_or(0);
                *sy = u8::try_from(ny).unwrap_or(0);
                top += 1;
            }
        }
    }
    for (row, seen_row) in tiles.iter_mut().zip(seen.iter()).take(size) {
        for (t, &reached) in row.iter_mut().zip(seen_row.iter()).take(size) {
            if passable(*t) && !reached {
                *t = STONE;
            }
        }
    }
}
