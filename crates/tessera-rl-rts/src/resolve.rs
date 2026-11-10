// SPDX-License-Identifier: MIT OR Apache-2.0

//! Move resolution for one tick (rule 1 of `rts-design.md` section 12).
//!
//! Every mover claims one cell. The resolution only ever refuses claims,
//! in three steps:
//!
//! 1. every claim on a cell claimed more than once is refused;
//! 2. swaps and cycles are refused: from each mover, follow the chain of
//!    "the unit holding my target, if it is a mover still claiming"; a
//!    chain that returns to a mover already on it closes a cycle, and every
//!    mover on the cycle is refused;
//! 3. a claim on a cell held by a unit that is not leaving (not a mover, or
//!    a mover whose claim was refused) is refused, repeated until nothing
//!    changes.
//!
//! Each step only removes claims and step 3 is iterated to its fixpoint, so
//! the result does not depend on the order units are visited. A refused
//! mover gave a legal order and is merely blocked: it is not an illegal
//! action. Nothing allocates.

#![deny(clippy::float_arithmetic)]

use crate::world::MAX_CELLS;

/// Unit slots of both teams at the largest configuration.
pub const MAX_UNITS: usize = 256;
/// No unit, no claim.
pub const NONE: u16 = u16::MAX;

/// The occupancy and claims of one tick. `holder[c]` is the unit on cell
/// `c` or [`NONE`]; `target[u]` is the cell unit `u` claims or [`NONE`].
/// A claim on the unit's own cell is not a move and is refused.
#[derive(Debug, Clone)]
pub struct Claims {
    /// Unit on each cell.
    pub holder: [u16; MAX_CELLS],
    /// Cell each unit claims.
    pub target: [u16; MAX_UNITS],
}

impl Default for Claims {
    fn default() -> Self {
        Self {
            holder: [NONE; MAX_CELLS],
            target: [NONE; MAX_UNITS],
        }
    }
}

fn at<T: Copy>(slice: &[T], i: usize, fallback: T) -> T {
    slice.get(i).copied().unwrap_or(fallback)
}

impl Claims {
    /// Which claims stand: `true` for a mover that moves this tick.
    #[must_use]
    pub fn resolve(&self) -> [bool; MAX_UNITS] {
        let mut active = [false; MAX_UNITS];
        let mut count = [0u8; MAX_CELLS];
        for (u, &t) in self.target.iter().enumerate() {
            if t == NONE {
                continue;
            }
            // Its own cell is not a move.
            if at(&self.holder, usize::from(t), NONE) == u16::try_from(u).unwrap_or(NONE) {
                continue;
            }
            if let (Some(a), Some(c)) = (active.get_mut(u), count.get_mut(usize::from(t))) {
                *a = true;
                *c = c.saturating_add(1);
            }
        }
        // Step 1: contested cells.
        for (u, a) in active.iter_mut().enumerate() {
            let t = at(&self.target, u, NONE);
            if *a && at(&count, usize::from(t), 0) > 1 {
                *a = false;
            }
        }
        // Step 2: swaps and cycles.
        self.refuse_cycles(&mut active);
        // Step 3: holders that stay, to the fixpoint.
        loop {
            let mut changed = false;
            for u in 0..MAX_UNITS {
                if !at(&active, u, false) {
                    continue;
                }
                let h = at(&self.holder, usize::from(at(&self.target, u, NONE)), NONE);
                if h != NONE && !at(&active, usize::from(h), false) {
                    if let Some(a) = active.get_mut(u) {
                        *a = false;
                    }
                    changed = true;
                }
            }
            if !changed {
                return active;
            }
        }
    }

    /// The next mover on `u`'s chain: the active mover holding its target.
    fn next(&self, active: &[bool; MAX_UNITS], u: usize) -> Option<usize> {
        let h = at(&self.holder, usize::from(at(&self.target, u, NONE)), NONE);
        let h = usize::from(h);
        (h < MAX_UNITS && at(active, h, false)).then_some(h)
    }

    fn refuse_cycles(&self, active: &mut [bool; MAX_UNITS]) {
        // 0 unseen, 1 on the current path, 2 finished.
        let mut state = [0u8; MAX_UNITS];
        let mut on_cycle = [false; MAX_UNITS];
        let mut path = [0usize; MAX_UNITS];
        for start in 0..MAX_UNITS {
            if !at(active, start, false) || at(&state, start, 0) != 0 {
                continue;
            }
            let mut len = 0;
            let mut cur = Some(start);
            while let Some(u) = cur {
                match at(&state, u, 0) {
                    0 => {
                        if let (Some(s), Some(p)) = (state.get_mut(u), path.get_mut(len)) {
                            *s = 1;
                            *p = u;
                            len += 1;
                        }
                        cur = self.next(active, u);
                    }
                    1 => {
                        // `u` is on the current path: the cycle runs from it.
                        let from = path.iter().take(len).position(|&p| p == u).unwrap_or(len);
                        for &p in path.iter().take(len).skip(from) {
                            if let Some(c) = on_cycle.get_mut(p) {
                                *c = true;
                            }
                        }
                        cur = None;
                    }
                    _ => cur = None,
                }
            }
            for &p in path.iter().take(len) {
                if let Some(s) = state.get_mut(p) {
                    *s = 2;
                }
            }
        }
        for (a, &c) in active.iter_mut().zip(on_cycle.iter()) {
            if c {
                *a = false;
            }
        }
    }
}
