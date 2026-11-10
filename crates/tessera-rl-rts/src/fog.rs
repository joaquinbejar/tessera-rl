// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fog of war (`rts-design.md` section 14, rulings 1 and 2).
//!
//! Visibility is derived from the state and never stored: a cell is
//! visible to a side when it lies within the Chebyshev `sight` of a living
//! unit of that side, finished or not. Explored is the one stored memory:
//! one bit per side of the map in [`RtsState::explored`], set at reset and
//! at the end of every decision, never cleared.

#![deny(clippy::float_arithmetic)]

use crate::env::{RtsConfig, RtsState, grid, slots, unit_kind};
use crate::world::{MAX_CELLS, Team};

/// Which cells a side sees, row-major; cells past the grid are `false`.
pub type Visibility = [bool; MAX_CELLS];

/// The explored bit of `team` in [`RtsState::explored`].
#[must_use]
pub const fn side_bit(team: Team) -> u8 {
    match team {
        Team::Zero => 1,
        Team::One => 2,
    }
}

/// The cells `team` sees now.
#[must_use]
pub fn visible(cfg: &RtsConfig, s: &RtsState, team: Team) -> Visibility {
    let g = grid(cfg);
    let mut out = [false; MAX_CELLS];
    for slot in slots(cfg, team) {
        let Some(u) = s.units.get(slot) else { continue };
        let Some(kind) = unit_kind(u) else { continue };
        let sight = u16::from(kind.stats().sight);
        let (ux, uy) = g.coords(u.cell);
        let last = g.size().saturating_sub(1);
        let xs = ux.saturating_sub(sight)..=ux.saturating_add(sight).min(last);
        let ys = uy.saturating_sub(sight)..=uy.saturating_add(sight).min(last);
        for cy in ys {
            for cx in xs.clone() {
                if let Some(c) = g.index(cx, cy)
                    && let Some(v) = out.get_mut(usize::from(c))
                {
                    *v = true;
                }
            }
        }
    }
    out
}

/// Whether `team` has explored `cell`.
#[must_use]
pub fn explored(s: &RtsState, team: Team, cell: u16) -> bool {
    s.explored
        .get(usize::from(cell))
        .is_some_and(|&b| b & side_bit(team) != 0)
}

/// Folds what each side sees now into its explored bits.
pub(crate) fn explore(cfg: &RtsConfig, s: &mut RtsState) {
    for team in [Team::Zero, Team::One] {
        let seen = visible(cfg, s, team);
        for (bits, &v) in s.explored.iter_mut().zip(seen.iter()) {
            if v {
                *bits |= side_bit(team);
            }
        }
    }
}

/// The explored rules a record must keep: only the two side bits, nothing
/// past the grid, and every visible cell explored.
pub(crate) fn check_explored(cfg: &RtsConfig, s: &RtsState) -> Result<(), &'static str> {
    let cells = usize::from(grid(cfg).cells());
    for (cell, &bits) in s.explored.iter().enumerate() {
        if bits > 3 || (cell >= cells && bits != 0) {
            return Err("an explored byte with a stray bit, or off the grid");
        }
    }
    for team in [Team::Zero, Team::One] {
        let seen = visible(cfg, s, team);
        if seen
            .iter()
            .zip(s.explored.iter())
            .any(|(&v, &b)| v && b & side_bit(team) == 0)
        {
            return Err("a visible cell that is not explored");
        }
    }
    Ok(())
}

/// The state as `team` knows it: the side view the scripted bots read
/// (`rts-design.md` section 14, ruling 5). Enemy units are kept only on
/// cells `team` sees now, without the facts the observation never shows
/// (carry, counter, progress, production); terrain is kept only where
/// `team` has explored, open elsewhere; the enemy's ore and explored bits
/// are cleared. A plain copy on the stack: no allocation.
#[must_use]
pub fn seen(cfg: &RtsConfig, s: &RtsState, team: Team) -> RtsState {
    let sees = visible(cfg, s, team);
    let mut v = *s;
    for (slot, u) in v.units.iter_mut().enumerate() {
        if crate::env::side_of_slot(slot) == team || !u.alive.get() {
            continue;
        }
        if sees.get(usize::from(u.cell)).copied().unwrap_or(false) {
            u.carry = 0;
            u.counter = 0;
            u.progress = 0;
            u.producing = 0;
        } else {
            *u = crate::units::Unit::default();
        }
        u.generation = 0;
    }
    let own = side_bit(team);
    for (cell, (t, e)) in v.terrain.iter_mut().zip(v.explored.iter_mut()).enumerate() {
        if *e & own == 0 && cell < usize::from(grid(cfg).cells()) {
            *t = crate::world::Terrain::Open as u8;
        }
        *e &= own;
    }
    if let Some(o) = v.ore.get_mut(1 - crate::env::index_of(team)) {
        *o = 0;
    }
    v
}
