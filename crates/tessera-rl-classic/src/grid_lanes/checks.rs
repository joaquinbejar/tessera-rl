// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persistence, registration, import validation and the scripted baseline
//! of `tessera/GridLanes-v1`: Grid's rules on the lane-wise state.

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, HeaderView, Heuristic, InvariantViolation, Phase, Validate};
use tessera_rl_num::rng::StepRng;

use super::{GridLanes, GridLanesState, GridOutcome, outcome};
use crate::grid::GridConfig;

impl tessera_rl_core::Persist for GridLanes {
    const ENV_ID: &'static str = super::ENV_ID;
    const ENV_VERSION: u32 = super::ENV_VERSION;
}

// Registration under the versioned id (SPEC 10, ADR-0054).
tessera_rl_core::register!(GridLanes, id = "tessera/GridLanes-v1");

/// Imported states, as Grid's: a goal on the grid other than the start, a
/// whole-cell position on the grid or one cell off it, an outcome that says
/// where the agent stands, and a step count the walk from the start allows.
impl Validate for GridLanes {
    fn validate(cfg: &GridConfig, s: &GridLanesState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let size = i32::from(cfg.size);
        let goal = (s.goal_col, s.goal_row);
        let on = |v: i32| (0..size).contains(&v);
        if !on(goal.0) || !on(goal.1) || goal == (0, 0) {
            return invalid("a goal off the grid or on the start");
        }
        if s.col.to_raw() & 0xffff != 0 || s.row.to_raw() & 0xffff != 0 {
            return invalid("a position that is not a whole cell");
        }
        let (col, row) = s.cell();
        let inside = on(col) && on(row);
        let one_off =
            (col == -1 || col == size) && on(row) || (row == -1 || row == size) && on(col);
        if !inside && !one_off {
            return invalid("a position beyond one cell off the grid");
        }
        if s.outcome != outcome((col, row) == goal, !inside) {
            return invalid("an outcome that disagrees with the position");
        }
        let distance = i64::from(col.abs()) + i64::from(row.abs());
        let steps = i64::from(s.steps);
        if steps > i64::from(cfg.max_steps) || steps < distance || (steps - distance) % 2 != 0 {
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
        s: &GridLanesState,
    ) -> Result<(), InvariantViolation> {
        let phase = header.phase();
        if phase == Phase::NeverReset {
            if *s != GridLanesState::default() {
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
            && (s.outcome != GridOutcome::Running || s.steps >= u32::from(cfg.max_steps))
        {
            return Err(InvariantViolation {
                code: "a running episode that has ended",
            });
        }
        Ok(())
    }
}

/// Scripted baseline, as Grid's: walk the rows, then the columns, to the
/// goal.
impl Heuristic for GridLanes {
    fn act(
        _: &GridConfig,
        s: &GridLanesState,
        _: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let (col, row) = s.cell();
        let action = if row > s.goal_row {
            0
        } else if row < s.goal_row {
            1
        } else if col < s.goal_col {
            2
        } else {
            3
        };
        out.set_discrete(0, action);
    }
}
