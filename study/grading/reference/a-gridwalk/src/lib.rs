//! `study-ref-a/StudyRefA-v1`: the reference implementation of rule sheet
//! T1-A, the grid walk (study/tasks/t1-a-gridwalk.md, #639). It began as
//! the `cargo tessera new --kind env` scaffold. Graders use it to record
//! study/grading/golden/a.txt. It is not handed to participants.
//!
//! The rules (the sheet is normative):
//! - a 5 x 5 grid, walls at (1, 1), (1, 3), (3, 1) and (3, 3), the goal at
//!   (4, 4), as (row, column);
//! - actions: 0 stay, 1 up, 2 right, 3 down, 4 left. A move off the grid
//!   or into a wall stays. Every action is legal;
//! - reset draws one `rng.next_u32()` and starts on
//!   `FREE[draw % FREE.len()]`, where `FREE` lists the cells that are
//!   neither a wall nor the goal, in row-major order;
//! - reaching the goal gives reward 1 and terminates; every other step
//!   gives 0; step `episode` (default 20) truncates;
//! - the observation is one `Box(i32, [4])` leaf: agent row, agent column,
//!   goal row, goal column.

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Bool8, Environment, EpisodeEnd, Heuristic, InvariantViolation,
    NoAdaptive, ObsLeaf, Observation, Persist, RecordHeader, Reserved, SpaceError, StepOut, Tier,
    Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

/// The versioned environment id (ADR-0023).
pub const ENV_ID: &str = "study-ref-a/StudyRefA-v1";

/// The environment version.
pub const ENV_VERSION: u32 = 1;

/// Agent slots.
pub const AGENTS: usize = 1;

/// Choices per agent: stay, up, right, down, left.
pub const CHOICES: u32 = 5;

/// The bound of every observation value.
pub const BOUND: i32 = 4;

/// The grid's side.
const SIDE: u8 = 5;

/// The walls, as (row, column).
const WALLS: [(u8, u8); 4] = [(1, 1), (1, 3), (3, 1), (3, 3)];

/// The goal, as (row, column).
const GOAL: (u8, u8) = (4, 4);

/// The cells that are neither a wall nor the goal, in row-major order.
const FREE: [(u8, u8); 20] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 0),
    (1, 2),
    (1, 4),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 0),
    (3, 2),
    (3, 4),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
];

/// The configuration, shared by every instance of a pool.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct GridConfig {
    /// Steps per episode; the episode is truncated at this step.
    #[config(default = 20, range = 1..=100_000)]
    pub episode: u32,
}

/// One instance's state: 8 bytes, no padding (ADR-0029).
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct GridState {
    /// Steps taken in this episode.
    pub step: u32,
    /// The agent's row.
    pub row: u8,
    /// The agent's column.
    pub col: u8,
    /// Whether the agent reached the goal.
    pub reached: Bool8,
    /// Padding made explicit.
    pub reserved: Reserved<1>,
}

/// The observation of the agent.
#[derive(Observation)]
pub struct GridObs {
    /// Agent row, agent column, goal row, goal column.
    #[space(box(low = 0, high = 4, shape = [4]))]
    pub values: ObsLeaf<i32>,
}

/// The grid walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GridWalk;

/// The environment the study's grader runs (the sheet's grading name).
pub type StudyEnv = GridWalk;

/// Whether `cell` is a wall.
fn wall(cell: (u8, u8)) -> bool {
    WALLS.contains(&cell)
}

/// The cell after `action` from `(row, col)`.
fn moved(row: u8, col: u8, action: i32) -> (u8, u8) {
    let next = match action {
        1 => row.checked_sub(1).map(|r| (r, col)),
        2 => Some((row, col.saturating_add(1))).filter(|&(_, c)| c < SIDE),
        3 => Some((row.saturating_add(1), col)).filter(|&(r, _)| r < SIDE),
        4 => col.checked_sub(1).map(|c| (row, c)),
        _ => None,
    };
    next.filter(|&c| !wall(c)).unwrap_or((row, col))
}

impl Environment for GridWalk {
    type State = GridState;
    type Config = GridConfig;
    type Obs = GridObs;
    type Action = ();
    type Header = RecordHeader<AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = AGENTS;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &GridConfig) -> Result<Space, SpaceError> {
        GridObs::space(cfg)
    }

    /// One discrete head: the move.
    fn action_space(_: &GridConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: CHOICES })
    }

    /// One draw picks the start among the free cells.
    fn reset(_: &GridConfig, rng: &mut StepRng, s: &mut GridState) {
        let draw = usize::try_from(rng.next_u32()).unwrap_or(0);
        let (row, col) = FREE.get(draw % FREE.len()).copied().unwrap_or((0, 0));
        *s = GridState {
            row,
            col,
            ..GridState::default()
        };
    }

    fn step(
        cfg: &GridConfig,
        s: &mut GridState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        // Bounded by the episode length (at most 100_000).
        s.step = s.step.saturating_add(1);
        let Some(agent) = AgentSlot::new(0, AGENTS) else {
            return;
        };
        let action = actions.discrete(agent, 0).unwrap_or(0);
        let (row, col) = moved(s.row, s.col, action);
        s.row = row;
        s.col = col;
        s.reached = Bool8::from((row, col) == GOAL);
        if s.reached.get() {
            out.set_reward(agent, 1.0);
            out.end_episode(EpisodeEnd::Terminated);
        } else if s.step >= cfg.episode {
            out.end_episode(EpisodeEnd::Truncated);
        }
    }

    fn observe(
        _: &GridConfig,
        s: &GridState,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, GridObs>,
    ) {
        let Some(mut leaf) = out.leaf_mut(GridObs::VALUES) else {
            return;
        };
        let values = [s.row, s.col, GOAL.0, GOAL.1];
        for (k, v) in values.into_iter().enumerate() {
            leaf.set(k, i32::from(v));
        }
    }
}

/// Every move is legal; staying is the no-op.
impl ActionMask for GridWalk {
    fn action_mask(_: &GridConfig, _: &GridState, _: AgentSlot, out: &mut MaskWriter<'_>) {
        for a in 0..CHOICES {
            out.set(usize::try_from(a).unwrap_or(0), true);
        }
    }

    fn noop(_: &GridConfig, _: &GridState, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, 0);
    }
}

/// The scripted baseline: down while possible, then right.
impl Heuristic for GridWalk {
    fn act(
        _: &GridConfig,
        s: &GridState,
        _: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let down = moved(s.row, s.col, 3) != (s.row, s.col);
        out.set_discrete(0, if down { 3 } else { 2 });
    }
}

/// The invariants an imported state must hold (ADR-0040).
impl Validate for GridWalk {
    fn validate(cfg: &GridConfig, s: &GridState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        if s.row >= SIDE || s.col >= SIDE {
            return invalid("a cell off the grid");
        }
        if wall((s.row, s.col)) {
            return invalid("an agent inside a wall");
        }
        if s.step > cfg.episode {
            return invalid("a step past the episode length");
        }
        if s.reached.get() != ((s.row, s.col) == GOAL) {
            return invalid("a reached flag that disagrees with the cell");
        }
        Ok(())
    }
}

/// The id and version persisted records carry (SPEC 9).
impl Persist for GridWalk {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
