// SPDX-License-Identifier: MIT OR Apache-2.0

//! `tessera/GridLanes-v1`: the reference grid world on a lane-wise state, the
//! first shipped fixed-point `LaneEnvironment` (ADR-0090 stage 2, #345).
//!
//! Its dynamics are [`Grid`](crate::grid::Grid)'s, step for step: the same moves,
//! walls, rewards and episode ends, and the same observation bytes. Only the
//! state differs: one column per field (Struct-of-Arrays), so groups of
//! eight instances step through [`LaneEnvironment::step_lanes`] on
//! integer and fixed-point lanes. Because the state is lane-wise, a pool of
//! it refuses Async and heuristic slots and needs `chunk_size % 8 == 0`
//! (ADR-0090 decisions 4.5 and 4.7).

mod checks;
mod kernel;
#[cfg(test)]
mod tests;

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, LANES, LaneEnvHooks, LaneEnvironment, LaneGroup, LaneOut,
    NoAdaptive, Observation, RecordHeader, SpaceError, StepOut, Tier, lane_env_hooks,
};
use tessera_rl_macros::{EnvState, StateEnum};
use tessera_rl_num::Fx32;
use tessera_rl_num::rng::{ObsRng, StepRng};
use tessera_rl_simd::Resolved;

use crate::grid::{GridConfig, GridObs, layout, pixels};

/// Environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/GridLanes-v1";
/// Environment version.
pub const ENV_VERSION: u32 = 1;
const ONE: Fx32 = Fx32::from_raw(1 << 16);

/// Where the agent stands after a step. Grid's two flags, `reached_goal` and
/// `out_of_bounds`, never hold together (the goal is on the grid), so one
/// value carries both.
#[derive(StateEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum GridOutcome {
    /// On the grid, not on the goal.
    #[default]
    Running = 0,
    /// On the goal.
    ReachedGoal = 1,
    /// Off the grid.
    OutOfBounds = 2,
}

/// The grid state as six lane fields: 24 bytes, no padding.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
#[state(lanes)]
pub struct GridLanesState {
    /// Agent column, always a whole cell.
    pub col: Fx32,
    /// Agent row, always a whole cell.
    pub row: Fx32,
    /// Goal column.
    pub goal_col: i32,
    /// Goal row.
    pub goal_row: i32,
    /// Steps taken in this episode. It saturates at `u32::MAX`, which an
    /// episode never reaches (`max_steps` is at most 10,000).
    pub steps: u32,
    /// Where the agent stands.
    pub outcome: GridOutcome,
}

impl GridLanesState {
    /// The agent's column and row as integers: the floor of the position,
    /// as Grid's `GridState::cell`.
    #[must_use]
    pub const fn cell(&self) -> (i32, i32) {
        (self.col.to_raw() >> 16, self.row.to_raw() >> 16)
    }
}

/// The reference grid world on a lane-wise state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GridLanes;

impl Environment for GridLanes {
    type State = GridLanesState;
    type Config = GridConfig;
    type Obs = GridObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;
    const LANE_ENV: Option<LaneEnvHooks<Self>> = Some(lane_env_hooks::<Self>());

    fn observation_space(cfg: &GridConfig) -> Result<Space, SpaceError> {
        GridObs::space(cfg)
    }

    fn action_space(_: &GridConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 4 })
    }

    /// The agent starts at the top left; the goal is uniform on the rest.
    fn reset(cfg: &GridConfig, rng: &mut StepRng, s: &mut GridLanesState) {
        let [goal_col, goal_row] = layout::goal(cfg, rng);
        *s = GridLanesState {
            goal_col: i32::from(goal_col),
            goal_row: i32::from(goal_row),
            ..GridLanesState::default()
        };
    }

    fn step(
        cfg: &GridConfig,
        s: &mut GridLanesState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let Some(agent) = AgentSlot::new(0, 1) else {
            return;
        };
        match transition(cfg, s, actions.discrete(agent, 0)) {
            Ending::Terminated(reward) => {
                out.set_reward(agent, reward);
                out.end_episode(EpisodeEnd::Terminated);
            }
            Ending::Truncated => out.end_episode(EpisodeEnd::Truncated),
            Ending::None => {}
        }
    }

    fn observe(
        cfg: &GridConfig,
        s: &GridLanesState,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, GridObs>,
    ) {
        let Some(mut leaf) = out.leaf_mut(GridObs::PIXELS) else {
            return;
        };
        pixels::write(cfg, s.cell(), (s.goal_col, s.goal_row), &mut leaf);
    }
}

/// How a step ended the episode, if it did.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Ending {
    /// Not ended.
    None,
    /// Ended on the goal (reward 1) or off the grid (reward -1).
    Terminated(f32),
    /// Ended at `max_steps`.
    Truncated,
}

/// One instance's step, as Grid's: the move (an overflow keeps the old
/// value), the wall rule, the step count, the outcome and the end, terminal
/// before truncation. `GridLanes::step` and the lane kernel's reference.
fn transition(cfg: &GridConfig, s: &mut GridLanesState, action: Option<i32>) -> Ending {
    let (col, row) = (s.col, s.row);
    let next = match action {
        Some(0) => [col, row.checked_sub(ONE).unwrap_or(row)],
        Some(1) => [col, row.checked_add(ONE).unwrap_or(row)],
        Some(2) => [col.checked_add(ONE).unwrap_or(col), row],
        _ => [col.checked_sub(ONE).unwrap_or(col), row],
    };
    [s.col, s.row] = layout::enter(cfg, [col, row], next);
    s.steps = s.steps.saturating_add(1);
    let (col, row) = s.cell();
    let size = i32::from(cfg.size);
    let off = !(0..size).contains(&col) || !(0..size).contains(&row);
    let won = (col, row) == (s.goal_col, s.goal_row);
    s.outcome = outcome(won, off);
    if off || won {
        Ending::Terminated(if won { 1.0 } else { -1.0 })
    } else if s.steps >= u32::from(cfg.max_steps) {
        Ending::Truncated
    } else {
        Ending::None
    }
}

/// The state's outcome from Grid's two flags; the goal wins, as Grid's
/// reward does.
const fn outcome(won: bool, off: bool) -> GridOutcome {
    if won {
        GridOutcome::ReachedGoal
    } else if off {
        GridOutcome::OutOfBounds
    } else {
        GridOutcome::Running
    }
}

/// Groups stepped through `step_lanes`, counted in test builds only, so a
/// test can tell the lane path ran (the scalar fallback gives the same
/// bytes).
#[cfg(test)]
pub static LANE_STEPS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

impl LaneEnvironment for GridLanes {
    const STEP_DRAWS_RNG: bool = false;
    const OBSERVE_DRAWS_RNG: bool = false;

    fn step_lanes(
        cfg: &GridConfig,
        level: Resolved,
        lanes: &mut LaneGroup<'_, Self>,
        actions: &[i32; LANES],
        out: &mut LaneOut,
    ) {
        #[cfg(test)]
        LANE_STEPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let (col, row, goal_col, goal_row, steps, outcomes) = lanes;
        kernel::step(
            level,
            cfg,
            &mut kernel::Columns {
                col,
                row,
                goal_col,
                goal_row,
                steps,
                outcome: outcomes,
            },
            actions,
            out,
        );
    }

    fn observe_lanes(cfg: &GridConfig, lanes: &LaneGroup<'_, Self>, leaves: &mut [&mut [u8]]) {
        let (col, row, goal_col, goal_row, _, _) = lanes;
        let Some(rows) = leaves.first_mut() else {
            return;
        };
        let len = 3 * usize::from(cfg.size) * usize::from(cfg.size);
        for (l, dst) in rows.chunks_exact_mut(len).take(LANES).enumerate() {
            let (Some(c), Some(r), Some(gc), Some(gr)) =
                (col.get(l), row.get(l), goal_col.get(l), goal_row.get(l))
            else {
                return;
            };
            let agent = (c.to_raw() >> 16, r.to_raw() >> 16);
            pixels::write(cfg, agent, (*gc, *gr), dst);
        }
    }
}

const _: () = assert!(<GridLanesState as tessera_rl_core::EnvState>::LANES);
const _: () = assert!(size_of::<GridLanesState>() == 24);
