// SPDX-License-Identifier: MIT OR Apache-2.0

//! `GridLanes`' step on eight instances at once, on integer and fixed-point
//! lanes (ADR-0090 stage 2). Every lane does what `GridLanes::step` does:
//! the move with Grid's `checked_*().unwrap_or(old)` (the overflow mask
//! keeps the old value), the wall rule, the saturating step count, then the
//! off-grid and goal tests on the floor of the position (`to_raw() >> 16`),
//! and the end: terminal before truncation.

use tessera_rl_core::{EpisodeEnd, LANES, LaneOut};
use tessera_rl_num::Fx32;
use tessera_rl_num::math::lane_int::{Fx32Lanes, I32Lanes};
use tessera_rl_simd::{LaneKernel, Lanes, Resolved};

use super::{GridOutcome, ONE, outcome};
use crate::grid::GridConfig;

/// The group's columns.
pub(super) struct Columns<'a> {
    pub(super) col: &'a mut [Fx32; LANES],
    pub(super) row: &'a mut [Fx32; LANES],
    pub(super) goal_col: &'a mut [i32; LANES],
    pub(super) goal_row: &'a mut [i32; LANES],
    pub(super) steps: &'a mut [u32; LANES],
    pub(super) outcome: &'a mut [GridOutcome; LANES],
}

/// Steps the group at `level`.
pub(super) fn step(
    level: Resolved,
    cfg: &GridConfig,
    columns: &mut Columns<'_>,
    actions: &[i32; LANES],
    out: &mut LaneOut,
) {
    let flags = level.dispatch_lanes(StepKernel {
        cfg,
        col: columns.col,
        row: columns.row,
        steps: columns.steps,
        goal_col: columns.goal_col,
        goal_row: columns.goal_row,
        actions,
    });
    for l in 0..LANES {
        let won = flags.won >> l & 1 == 1;
        let off = flags.off >> l & 1 == 1;
        let truncated = flags.truncated >> l & 1 == 1;
        if let Some(o) = columns.outcome.get_mut(l) {
            *o = outcome(won, off);
        }
        let (reward, end) = if off || won {
            (if won { 1.0 } else { -1.0 }, Some(EpisodeEnd::Terminated))
        } else if truncated {
            (0.0, Some(EpisodeEnd::Truncated))
        } else {
            (0.0, None)
        };
        if let (Some(r), Some(e)) = (out.reward.get_mut(l), out.end.get_mut(l)) {
            *r = reward;
            *e = end;
        }
    }
}

/// One bit per lane: on the goal, off the grid, at `max_steps`.
struct Flags {
    won: u8,
    off: u8,
    truncated: u8,
}

struct StepKernel<'a> {
    cfg: &'a GridConfig,
    col: &'a mut [Fx32; LANES],
    row: &'a mut [Fx32; LANES],
    steps: &'a mut [u32; LANES],
    goal_col: &'a [i32; LANES],
    goal_row: &'a [i32; LANES],
    actions: &'a [i32; LANES],
}

impl LaneKernel for StepKernel<'_> {
    type Output = Flags;

    #[expect(
        clippy::inline_always,
        reason = "LaneKernel::run must inline into the target_feature context"
    )]
    #[inline(always)]
    fn run<S: Lanes>(self) -> Flags {
        let one = S::Fx32::splat(ONE);
        let col = S::Fx32::from_array(*self.col);
        let row = S::Fx32::from_array(*self.row);
        let action = S::I32::from_array(*self.actions);
        let north = action.eq(S::I32::splat(0));
        let south = action.eq(S::I32::splat(1));
        let east = action.eq(S::I32::splat(2));

        // Grid's `checked_*().unwrap_or(old)`: where the add overflows, the
        // old value stays.
        let (up, up_over) = row.overflowing_sub(one);
        let up = S::Fx32::select(up_over, row, up);
        let (down, down_over) = row.overflowing_add(one);
        let down = S::Fx32::select(down_over, row, down);
        let (right, right_over) = col.overflowing_add(one);
        let right = S::Fx32::select(right_over, col, right);
        let (left, left_over) = col.overflowing_sub(one);
        let left = S::Fx32::select(left_over, col, left);

        // Action 0 north, 1 south, 2 east, anything else west (Grid's `_`).
        let vertical = S::I32::mask_or(north, south);
        let next_row = S::Fx32::select(north, up, S::Fx32::select(south, down, row));
        let next_col = S::Fx32::select(east, right, S::Fx32::select(vertical, col, left));

        // The wall rule (`layout::enter`): a lookup per lane on the floor of
        // the target cell; a wall keeps the agent where it was.
        let target_col = next_col.to_raw().shr_arith::<16>().to_array();
        let target_row = next_row.to_raw().shr_arith::<16>().to_array();
        let layout = self.cfg.layout.get();
        let mut walls = [false; LANES];
        for ((w, &c), &r) in walls.iter_mut().zip(&target_col).zip(&target_row) {
            *w = layout.is_wall(c, r);
        }
        let wall = S::I32::mask_from_array(walls);
        let col = S::Fx32::select(wall, col, next_col);
        let row = S::Fx32::select(wall, row, next_row);
        *self.col = col.to_array();
        *self.row = row.to_array();

        // The step count saturates at `u32::MAX` (its bits as `i32`: -1).
        let steps = S::I32::from_array(self.steps.map(u32::cast_signed));
        let full = steps.eq(S::I32::splat(-1));
        let steps = S::I32::select(full, steps, steps.wrapping_add(S::I32::splat(1)));
        *self.steps = steps.to_array().map(i32::cast_unsigned);

        let cell_col = col.to_raw().shr_arith::<16>();
        let cell_row = row.to_raw().shr_arith::<16>();
        let zero = S::I32::splat(0);
        let size = S::I32::splat(i32::from(self.cfg.size));
        let off = S::I32::mask_or(
            S::I32::mask_or(cell_col.lt(zero), cell_col.ge(size)),
            S::I32::mask_or(cell_row.lt(zero), cell_row.ge(size)),
        );
        let won = S::I32::mask_and(
            cell_col.eq(S::I32::from_array(*self.goal_col)),
            cell_row.eq(S::I32::from_array(*self.goal_row)),
        );
        let max_steps = S::I32::splat(i32::from(self.cfg.max_steps));
        let truncated = steps.ge_u(max_steps);
        Flags {
            won: S::I32::mask_bits(won),
            off: S::I32::mask_bits(off),
            truncated: S::I32::mask_bits(truncated),
        }
    }
}
