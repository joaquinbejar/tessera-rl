// SPDX-License-Identifier: MIT OR Apache-2.0

//! Flow fields: integer breadth-first distances to a goal class, and the
//! step a unit takes down them (`rts-design.md`, section 5).
//!
//! A field is derived each step and never stored in the state. It lives in
//! fixed arrays sized for the largest grid, so building one never
//! allocates. Distances are global: a field shared by both teams bakes in
//! no order. Only the step choice is ordered, by the moving unit's team
//! through [`Dir::order`] (rule 9).

#![deny(clippy::float_arithmetic)]

use crate::world::{Dir, Grid, MAX_CELLS, Team};

/// The distance of a cell no goal can be reached from.
pub const UNREACHABLE: u16 = u16::MAX;

/// One goal class's distances over the grid, with the queue the search
/// needs. Reused across steps; [`Field::build`] overwrites it.
#[derive(Clone)]
pub struct Field {
    /// Cells of the grid of the last build; reads past it are unreachable.
    cells: u16,
    dist: [u16; MAX_CELLS],
    queue: [u16; MAX_CELLS],
}

impl Default for Field {
    fn default() -> Self {
        Self {
            cells: 0,
            dist: [UNREACHABLE; MAX_CELLS],
            queue: [0; MAX_CELLS],
        }
    }
}

impl Field {
    /// Recomputes the field: every goal cell at distance 0, then 4-neighbour
    /// steps through cells where `passable` holds. A goal cell need not be
    /// passable (an enemy structure, an ore node): it is a source, not a
    /// cell anyone enters. Goals outside the grid are ignored.
    pub fn build(
        &mut self,
        grid: Grid,
        goals: impl IntoIterator<Item = u16>,
        passable: impl Fn(u16) -> bool,
    ) {
        self.cells = grid.cells();
        let cells = usize::from(self.cells);
        if let Some(live) = self.dist.get_mut(..cells) {
            live.fill(UNREACHABLE);
        }
        let mut tail = 0usize;
        for goal in goals {
            if goal < grid.cells() && self.visit(goal, 0, tail) {
                tail += 1;
            }
        }
        let mut head = 0usize;
        while let Some(&cell) = self.queue.get(head).filter(|_| head < tail) {
            head += 1;
            // Every queued distance is below `cells`, at most MAX_CELLS.
            let next = self.distance(cell) + 1;
            // Distances are order-independent, so the map's own order serves.
            for dir in [Dir::North, Dir::East, Dir::South, Dir::West] {
                if let Some(n) = grid.step(cell, dir)
                    && self.distance(n) == UNREACHABLE
                    && passable(n)
                    && self.visit(n, next, tail)
                {
                    tail += 1;
                }
            }
        }
    }

    /// Sets an unvisited `cell` to `d` and queues it at `tail`; false if it
    /// was already visited. Each cell is queued at most once, so `tail`
    /// stays below `MAX_CELLS`.
    fn visit(&mut self, cell: u16, d: u16, tail: usize) -> bool {
        match (
            self.dist.get_mut(usize::from(cell)),
            self.queue.get_mut(tail),
        ) {
            (Some(slot), Some(q)) if *slot == UNREACHABLE => {
                *slot = d;
                *q = cell;
                true
            }
            _ => false,
        }
    }

    /// The distance of `cell`, [`UNREACHABLE`] if no goal is reachable or
    /// `cell` is outside the grid of the last build.
    #[must_use]
    pub fn distance(&self, cell: u16) -> u16 {
        if cell >= self.cells {
            return UNREACHABLE;
        }
        self.dist
            .get(usize::from(cell))
            .copied()
            .unwrap_or(UNREACHABLE)
    }

    /// The step a unit of `team` at `cell` takes down the field: the
    /// neighbour of lowest distance strictly below its own, ties broken by
    /// the team's fixed order. `None` when the cell is unreachable or no
    /// neighbour is closer, so a unit never moves toward an unreachable goal.
    ///
    /// The unit's own distance is read, so the `passable` of the last build
    /// must hold for every cell a mover stands on (an occupied cell counted
    /// as an obstacle would read as unreachable and the unit would stay).
    #[must_use]
    pub fn descend(&self, grid: Grid, cell: u16, team: Team) -> Option<Dir> {
        let here = self.distance(cell);
        // At distance 1 the only closer cell is a goal, which is not entered.
        if here == UNREACHABLE || here <= 1 {
            return None;
        }
        let mut best: Option<(u16, Dir)> = None;
        for dir in Dir::order(team) {
            if let Some(n) = grid.step(cell, dir) {
                let d = self.distance(n);
                if d < here && best.is_none_or(|(b, _)| d < b) {
                    best = Some((d, dir));
                }
            }
        }
        best.map(|(_, dir)| dir)
    }
}
