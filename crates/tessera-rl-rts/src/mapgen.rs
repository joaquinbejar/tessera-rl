// SPDX-License-Identifier: MIT OR Apache-2.0

//! The seeded map generator (design section 2, rule 8).
//!
//! Each attempt places the two Cores at fixed mirrored cells, then wall
//! pairs and ore pairs at drawn cells and their 180-degree images, keeping
//! a clear zone around each Core. An attempt is accepted when it placed
//! exactly [`ORE_PAIRS`] ore pairs and a flood fill from Core 0 reaches
//! Core 1 and a 4-neighbour of every ore node, so every map has the same
//! economy. After
//! [`MAP_RETRIES`] rejected attempts the generator returns the fixed open
//! fallback map, which passes the same checks.
//!
//! The draws come from the caller (the reset stream, in the environment),
//! and every attempt consumes the same number of them, whatever it places.

#![deny(clippy::float_arithmetic)]

use core::num::NonZeroU32;

use crate::world::{Dir, Grid, MAX_CELLS, Team, Terrain};

/// Attempts before the fallback map.
pub const MAP_RETRIES: u32 = 16;
/// Ore nodes per team, on every map, the fallback included.
pub const ORE_PAIRS: u32 = 3;
/// Draws spent on ore per attempt; unplaceable draws are skipped.
pub const ORE_DRAWS: u32 = 8 * ORE_PAIRS;
/// Cells within this Chebyshev distance of a Core stay open.
pub const CLEAR_RADIUS: u16 = 2;
/// Column and row of Core 0; Core 1 is its mirror.
pub const CORE_XY: (u16, u16) = (2, 2);

/// A generated map: terrain and the Core cells.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Map {
    /// [`Terrain`] bytes, row-major; cells past the grid are open.
    pub terrain: [u8; MAX_CELLS],
    /// The Core cell of team 0 and of team 1.
    pub cores: [u16; 2],
    /// Attempts made, 1 to [`MAP_RETRIES`].
    pub attempts: u32,
    /// Whether this is the fallback map.
    pub fallback: bool,
}

/// Wall draws per attempt: an eighth of the cells.
#[must_use]
pub const fn wall_draws(grid: Grid) -> u32 {
    grid.cells() as u32 / 8
}

/// Generates a map of `grid` from `draw`, which returns a value below its
/// bound (the environment passes the reset stream).
#[must_use]
pub fn generate(grid: Grid, mut draw: impl FnMut(NonZeroU32) -> u32) -> Map {
    let cores = core_cells(grid);
    for attempt in 1..=MAP_RETRIES {
        let (terrain, placed) = attempt_terrain(grid, cores, &mut draw);
        if placed == ORE_PAIRS && accepts(grid, &terrain, cores) {
            return Map {
                terrain,
                cores,
                attempts: attempt,
                fallback: false,
            };
        }
    }
    Map {
        terrain: fallback_terrain(grid, cores),
        cores,
        attempts: MAP_RETRIES,
        fallback: true,
    }
}

/// The Core cells: [`CORE_XY`] and its mirror.
#[must_use]
pub fn core_cells(grid: Grid) -> [u16; 2] {
    let zero = grid.index(CORE_XY.0, CORE_XY.1).unwrap_or(0);
    [zero, grid.mirror(zero)]
}

fn clear(grid: Grid, cores: [u16; 2], cell: u16) -> bool {
    cores
        .iter()
        .any(|&c| grid.chebyshev(c, cell) <= CLEAR_RADIUS)
}

fn set_pair(grid: Grid, terrain: &mut [u8; MAX_CELLS], cell: u16, t: Terrain) {
    for c in [cell, grid.mirror(cell)] {
        if let Some(slot) = terrain.get_mut(usize::from(c)) {
            *slot = t as u8;
        }
    }
}

fn open_at(terrain: &[u8; MAX_CELLS], cell: u16) -> bool {
    terrain.get(usize::from(cell)) == Some(&(Terrain::Open as u8))
}

fn attempt_terrain(
    grid: Grid,
    cores: [u16; 2],
    draw: &mut impl FnMut(NonZeroU32) -> u32,
) -> ([u8; MAX_CELLS], u32) {
    let mut terrain = [Terrain::Open as u8; MAX_CELLS];
    let bound = NonZeroU32::new(u32::from(grid.cells())).unwrap_or(NonZeroU32::MIN);
    for _ in 0..wall_draws(grid) {
        let cell = u16::try_from(draw(bound)).unwrap_or(0);
        if cell < grid.cells() && !clear(grid, cores, cell) {
            set_pair(grid, &mut terrain, cell, Terrain::Wall);
        }
    }
    let mut placed = 0;
    for _ in 0..ORE_DRAWS {
        let cell = u16::try_from(draw(bound)).unwrap_or(0);
        let mirror = grid.mirror(cell);
        if placed < ORE_PAIRS
            && cell < grid.cells()
            && cell != mirror
            && !clear(grid, cores, cell)
            && open_at(&terrain, cell)
            && open_at(&terrain, mirror)
        {
            set_pair(grid, &mut terrain, cell, Terrain::Ore);
            placed += 1;
        }
    }
    (terrain, placed)
}

/// The fixed open map: no walls, [`ORE_PAIRS`] ore nodes per team at
/// `(6, 1)`, `(7, 1)` and `(7, 2)` and their mirrors, distinct and outside
/// the clear zones at every size.
#[must_use]
pub fn fallback_terrain(grid: Grid, cores: [u16; 2]) -> [u8; MAX_CELLS] {
    let mut terrain = [Terrain::Open as u8; MAX_CELLS];
    for (x, y) in [(6, 1), (7, 1), (7, 2)] {
        if let Some(cell) = grid.index(x, y)
            && !clear(grid, cores, cell)
        {
            set_pair(grid, &mut terrain, cell, Terrain::Ore);
        }
    }
    terrain
}

/// The reachability half of the acceptance check: from Core 0, over open
/// cells, Core 1 and a 4-neighbour of every ore node are reached. The ore
/// count is checked by [`generate`].
#[must_use]
pub fn accepts(grid: Grid, terrain: &[u8; MAX_CELLS], cores: [u16; 2]) -> bool {
    let reached = flood(grid, terrain, cores[0]);
    let hit = |c: u16| reached.get(usize::from(c)).copied().unwrap_or(false);
    if !hit(cores[1]) {
        return false;
    }
    (0..grid.cells())
        .filter(|&c| terrain.get(usize::from(c)) == Some(&(Terrain::Ore as u8)))
        .all(|c| {
            grid.neighbours(c, Team::Zero)
                .iter()
                .flatten()
                .any(|&n| hit(n))
        })
}

/// Cells reached from `start` over open terrain, 4-connected.
#[must_use]
pub fn flood(grid: Grid, terrain: &[u8; MAX_CELLS], start: u16) -> [bool; MAX_CELLS] {
    let mut reached = [false; MAX_CELLS];
    let mut queue = [0u16; MAX_CELLS];
    let (mut head, mut tail) = (0usize, 0usize);
    if open_at(terrain, start)
        && let (Some(r), Some(q)) = (reached.get_mut(usize::from(start)), queue.get_mut(0))
    {
        *r = true;
        *q = start;
        tail = 1;
    }
    while head < tail {
        let Some(&cell) = queue.get(head) else { break };
        head += 1;
        for dir in Dir::order(Team::Zero) {
            let Some(n) = grid.step(cell, dir) else {
                continue;
            };
            if !open_at(terrain, n) {
                continue;
            }
            if let Some(r) = reached.get_mut(usize::from(n))
                && !*r
            {
                *r = true;
                if let Some(q) = queue.get_mut(tail) {
                    *q = n;
                    tail += 1;
                }
            }
        }
    }
    reached
}
