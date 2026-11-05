// SPDX-License-Identifier: MIT OR Apache-2.0

//! Import validation and the scripted baseline of the reference grid world,
//! apart from the environment itself so that it stays the short example of
//! SPEC section 10.

use core::num::NonZeroU32;

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, HeaderView, Heuristic, InvariantViolation, Phase, Validate};
use tessera_rl_num::rng::StepRng;

use super::{Grid, GridConfig, GridState};

impl GridState {
    /// The agent's column and row as integers.
    #[must_use]
    pub const fn cell(&self) -> (i32, i32) {
        (self.pos[0].to_raw() >> 16, self.pos[1].to_raw() >> 16)
    }

    /// Whether the cell at `col`, `row` is on in `channel`, on a grid whose
    /// last row and column are `last`.
    pub(super) fn lit(&self, channel: u32, col: i32, row: i32, last: i32) -> bool {
        match channel {
            0 => (col, row) == self.cell(),
            1 => (col, row) == (i32::from(self.goal[0]), i32::from(self.goal[1])),
            _ => row == 0 || col == 0 || row == last || col == last,
        }
    }
}

/// A goal uniform on every cell but the top left one.
pub(super) fn draw_goal(cells: u16, rng: &mut StepRng) -> [i16; 2] {
    let n = u32::from(cells);
    let others = NonZeroU32::new(n * n - 1).unwrap_or(NonZeroU32::MIN);
    let k = rng.below_u32(others) + 1;
    let coord = |v: u32| i16::try_from(v).unwrap_or(0);
    [coord(k % n), coord(k / n)]
}

/// Imported states: a goal on the grid other than the start, a whole-cell
/// position on the grid or one cell off it, flags that say where the agent
/// stands, and a step count the walk from the start allows (at least the
/// distance, same parity, at most `max_steps`).
impl Validate for Grid {
    fn validate(cfg: &GridConfig, s: &GridState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let size = i32::from(cfg.size);
        let goal = (i32::from(s.goal[0]), i32::from(s.goal[1]));
        let on = |v: i32| (0..size).contains(&v);
        if !on(goal.0) || !on(goal.1) || goal == (0, 0) {
            return invalid("a goal off the grid or on the start");
        }
        if s.pos.iter().any(|p| p.to_raw() & 0xffff != 0) {
            return invalid("a position that is not a whole cell");
        }
        let (col, row) = s.cell();
        let inside = on(col) && on(row);
        let one_off =
            (col == -1 || col == size) && on(row) || (row == -1 || row == size) && on(col);
        if !inside && !one_off {
            return invalid("a position beyond one cell off the grid");
        }
        if s.out_of_bounds.get() == inside {
            return invalid("an out-of-bounds flag that disagrees with the position");
        }
        if s.reached_goal.get() != ((col, row) == goal) {
            return invalid("a reached flag that disagrees with the position");
        }
        let distance = col.abs() + row.abs();
        let steps = i32::from(s.steps);
        if steps > i32::from(cfg.max_steps) || steps < distance || (steps - distance) % 2 != 0 {
            return invalid("a step count that the walk does not allow");
        }
        Ok(())
    }

    /// ADR-0040: before the first reset the state is exactly the initial
    /// value; once reset, the slot is alive, and a running episode has not
    /// ended.
    fn validate_record(
        cfg: &GridConfig,
        header: &HeaderView<'_>,
        s: &GridState,
    ) -> Result<(), InvariantViolation> {
        let phase = header.phase();
        if phase == Phase::NeverReset {
            if *s != GridState::default() {
                return Err(InvariantViolation {
                    code: "a state other than the initial value before the first reset",
                });
            }
            return Ok(());
        }
        Self::validate(cfg, s)?;
        if !header.agent(0).is_some_and(|v| v.alive) {
            return Err(InvariantViolation {
                code: "the agent is not alive",
            });
        }
        if phase == Phase::Active
            && (s.reached_goal.get() || s.out_of_bounds.get() || s.steps >= cfg.max_steps)
        {
            return Err(InvariantViolation {
                code: "a running episode that has ended",
            });
        }
        Ok(())
    }
}

/// Scripted baseline: walk the rows, then the columns, to the goal.
impl Heuristic for Grid {
    fn act(
        _: &GridConfig,
        s: &GridState,
        _: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let (col, row) = s.cell();
        let (goal_col, goal_row) = (i32::from(s.goal[0]), i32::from(s.goal[1]));
        let action = if row > goal_row {
            0
        } else if row < goal_row {
            1
        } else if col < goal_col {
            2
        } else {
            3
        };
        out.set_discrete(0, action);
    }
}
