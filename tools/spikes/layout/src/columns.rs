// SPDX-License-Identifier: MIT OR Apache-2.0

//! Column (SoA) storage of the classic states, written by hand: the spike
//! measures the layout, not a derive.

use tessera_rl_classic::tag::MAX_AGENTS;
use tessera_rl_classic::{CartPoleState, TagState};
use tessera_rl_core::Bool8;

/// A state that can live in per-field columns.
pub trait Columnar: Sized {
    /// The columns for some number of instances.
    type Columns: core::fmt::Debug;
    /// Columns for `n` instances.
    fn columns(n: usize) -> Self::Columns;
    /// Instance `i` assembled from the columns.
    fn gather(columns: &Self::Columns, i: usize) -> Self;
    /// Writes `self` into the columns at `i`.
    fn scatter(&self, columns: &mut Self::Columns, i: usize);
    /// Copies every column of `src` into `dst` (same size).
    fn copy_columns(src: &Self::Columns, dst: &mut Self::Columns);
}

/// CartPole columns: one per physical component.
#[derive(Debug)]
pub struct CartPoleColumns {
    /// `x`, `x_dot`, `theta`, `theta_dot`.
    pub s: [Vec<f64>; 4],
}

impl Columnar for CartPoleState {
    type Columns = CartPoleColumns;

    fn columns(n: usize) -> CartPoleColumns {
        CartPoleColumns {
            s: core::array::from_fn(|_| vec![0.0; n]),
        }
    }

    #[inline]
    fn gather(c: &CartPoleColumns, i: usize) -> Self {
        Self {
            s: core::array::from_fn(|k| {
                c.s.get(k)
                    .and_then(|col| col.get(i))
                    .copied()
                    .unwrap_or(0.0)
            }),
        }
    }

    fn copy_columns(src: &CartPoleColumns, dst: &mut CartPoleColumns) {
        for (d, s) in dst.s.iter_mut().zip(&src.s) {
            d.copy_from_slice(s);
        }
    }

    #[inline]
    fn scatter(&self, c: &mut CartPoleColumns, i: usize) {
        for (col, v) in c.s.iter_mut().zip(self.s) {
            if let Some(slot) = col.get_mut(i) {
                *slot = v;
            }
        }
    }
}

/// Tag columns: one per agent coordinate and per caught flag.
#[derive(Debug)]
pub struct TagColumns {
    /// `[agent][axis]` columns of positions.
    pub pos: [[Vec<i16>; 2]; MAX_AGENTS],
    /// Per-agent caught columns.
    pub caught: [Vec<Bool8>; MAX_AGENTS],
}

impl Columnar for TagState {
    type Columns = TagColumns;

    fn columns(n: usize) -> TagColumns {
        TagColumns {
            pos: core::array::from_fn(|_| core::array::from_fn(|_| vec![0; n])),
            caught: core::array::from_fn(|_| vec![Bool8::FALSE; n]),
        }
    }

    #[inline]
    fn gather(c: &TagColumns, i: usize) -> Self {
        let get = |col: &Vec<i16>| col.get(i).copied().unwrap_or(0);
        Self {
            pos: core::array::from_fn(|a| {
                c.pos.get(a).map_or([0; 2], |p| [get(&p[0]), get(&p[1])])
            }),
            caught: core::array::from_fn(|a| {
                c.caught
                    .get(a)
                    .and_then(|col| col.get(i))
                    .copied()
                    .unwrap_or(Bool8::FALSE)
            }),
        }
    }

    fn copy_columns(src: &TagColumns, dst: &mut TagColumns) {
        for (d, s) in dst.pos.iter_mut().zip(&src.pos) {
            for (d, s) in d.iter_mut().zip(s) {
                d.copy_from_slice(s);
            }
        }
        for (d, s) in dst.caught.iter_mut().zip(&src.caught) {
            d.copy_from_slice(s);
        }
    }

    #[inline]
    fn scatter(&self, c: &mut TagColumns, i: usize) {
        for (cols, cell) in c.pos.iter_mut().zip(self.pos) {
            for (col, v) in cols.iter_mut().zip(cell) {
                if let Some(slot) = col.get_mut(i) {
                    *slot = v;
                }
            }
        }
        for (col, v) in c.caught.iter_mut().zip(self.caught) {
            if let Some(slot) = col.get_mut(i) {
                *slot = v;
            }
        }
    }
}
