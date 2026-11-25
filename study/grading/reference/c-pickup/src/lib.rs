//! `study-ref-c/StudyRefC-v1`: the reference implementation of rule sheet
//! T1-C, simultaneous pickup (study/tasks/t1-c-pickup.md, #639). It began
//! as the `cargo tessera new --kind env --agents 2` scaffold. Graders use
//! it to record study/grading/golden/c.txt. It is not handed to
//! participants.
//!
//! The rules (the sheet is normative):
//! - a 5 x 5 grid; agent 0 starts at (0, 0), agent 1 at (4, 4), as
//!   (row, column); both act every step (simultaneous turns);
//! - three items. Reset draws one `rng.next_u32()` per item, in item order,
//!   and places item k on `CELLS[draw % 23]`, where `CELLS` lists every
//!   cell but the two starts in row-major order. A cell that already holds
//!   an item moves on to the next cell of `CELLS`, wrapping;
//! - actions: 0 stay, 1 up, 2 right, 3 down, 4 left. A move off the grid is
//!   illegal (masked); agents may share a cell;
//! - both agents move (slot 0 first), then each item not yet taken goes to
//!   agent 0 if it stands on it, else to agent 1 if it does: reward 1 to
//!   the taker. Taking the last item terminates; step `episode` (default
//!   30) truncates;
//! - partial observation, one `Box(i32, [11])` leaf per agent: its own row
//!   and column, then the 3 x 3 window around it in row-major order, each
//!   cell 0 empty, 1 item, 2 the other agent, 3 off the grid (the centre
//!   is the agent's own cell: 1 if an item lies there, else 0). The sheet
//!   leaves the observation free; only rewards, flags and masks are graded.

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Bool8, Environment, EpisodeEnd, Heuristic, InvariantViolation,
    NoAdaptive, ObsLeaf, Observation, Persist, RecordHeader, Reserved, SpaceError, StepOut, Tier,
    Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation, StateEnum};
use tessera_rl_num::rng::{ObsRng, StepRng};

/// The versioned environment id (ADR-0023).
pub const ENV_ID: &str = "study-ref-c/StudyRefC-v1";

/// The environment version.
pub const ENV_VERSION: u32 = 1;

/// Agent slots.
pub const AGENTS: usize = 2;

/// Items.
pub const ITEMS: usize = 3;

/// Choices per agent: stay, up, right, down, left.
pub const CHOICES: u32 = 5;

/// The bound of every observation value.
pub const BOUND: i32 = 4;

/// Values per observation: row, column and the 3 x 3 window.
pub const OBS_LEN: usize = 11;

/// The grid's side.
const SIDE: u8 = 5;

/// The agents' start cells.
const STARTS: [(u8, u8); AGENTS] = [(0, 0), (4, 4)];

/// The configuration, shared by every instance of a pool.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct PickupConfig {
    /// Steps per episode; the episode is truncated at this step.
    #[config(default = 30, range = 1..=100_000)]
    pub episode: u32,
}

/// The phase of an episode (`#[derive(StateEnum)]`).
#[derive(StateEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Phase {
    /// Items remain.
    #[default]
    Collecting = 1,
    /// Every item is taken.
    Done = 2,
}

/// One instance's state: 20 bytes, no padding (ADR-0029).
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct PickupState {
    /// Steps taken in this episode.
    pub step: u32,
    /// The agents' rows.
    pub rows: [u8; AGENTS],
    /// The agents' columns.
    pub cols: [u8; AGENTS],
    /// The items' rows.
    pub item_rows: [u8; ITEMS],
    /// The items' columns.
    pub item_cols: [u8; ITEMS],
    /// Whether each item is taken.
    pub taken: [Bool8; ITEMS],
    /// The phase, from `taken`.
    pub phase: Phase,
    /// Padding made explicit.
    pub reserved: Reserved<2>,
}

/// The observation of one agent.
#[derive(Observation)]
pub struct PickupObs {
    /// Own row and column, then the 3 x 3 window.
    #[space(box(low = 0, high = 4, shape = [11]))]
    pub values: ObsLeaf<i32>,
}

/// Simultaneous pickup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pickup;

/// The environment the study's grader runs (the sheet's grading name).
pub type StudyEnv = Pickup;

/// The cell after `action` from `(row, col)`, or `None` off the grid.
fn moved(row: u8, col: u8, action: u32) -> Option<(u8, u8)> {
    match action {
        0 => Some((row, col)),
        1 => row.checked_sub(1).map(|r| (r, col)),
        2 => Some((row, col.saturating_add(1))).filter(|&(_, c)| c < SIDE),
        3 => Some((row.saturating_add(1), col)).filter(|&(r, _)| r < SIDE),
        4 => col.checked_sub(1).map(|c| (row, c)),
        _ => None,
    }
}

/// Every cell but the two starts, in row-major order.
const CELLS: [(u8, u8); 23] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
];

/// The agent at `slot`'s cell.
fn at(s: &PickupState, slot: usize) -> Option<(u8, u8)> {
    Some((*s.rows.get(slot)?, *s.cols.get(slot)?))
}

/// Whether an untaken item lies on `cell`.
fn item_on(s: &PickupState, cell: (u8, u8)) -> bool {
    (0..ITEMS).any(|k| {
        let item = (s.item_rows.get(k).copied(), s.item_cols.get(k).copied());
        item == (Some(cell.0), Some(cell.1)) && !s.taken.get(k).is_some_and(|t| t.get())
    })
}

impl Environment for Pickup {
    type State = PickupState;
    type Config = PickupConfig;
    type Obs = PickupObs;
    type Action = ();
    type Header = RecordHeader<AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = AGENTS;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &PickupConfig) -> Result<Space, SpaceError> {
        PickupObs::space(cfg)
    }

    /// One discrete head: the move.
    fn action_space(_: &PickupConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: CHOICES })
    }

    /// One draw per item, in item order.
    fn reset(_: &PickupConfig, rng: &mut StepRng, s: &mut PickupState) {
        *s = PickupState::default();
        for (k, start) in STARTS.into_iter().enumerate() {
            if let (Some(r), Some(c)) = (s.rows.get_mut(k), s.cols.get_mut(k)) {
                *r = start.0;
                *c = start.1;
            }
        }
        for k in 0..ITEMS {
            let draw = usize::try_from(rng.next_u32()).unwrap_or(0);
            let mut i = draw % CELLS.len();
            // At most ITEMS - 1 cells are taken, so this ends.
            while CELLS.get(i).is_some_and(|&c| item_on(s, c)) {
                i = (i + 1) % CELLS.len();
            }
            let cell = CELLS.get(i).copied().unwrap_or((2, 2));
            if let (Some(r), Some(c)) = (s.item_rows.get_mut(k), s.item_cols.get_mut(k)) {
                *r = cell.0;
                *c = cell.1;
            }
        }
    }

    fn step(
        cfg: &PickupConfig,
        s: &mut PickupState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        // Bounded by the episode length (at most 100_000).
        s.step = s.step.saturating_add(1);
        for i in 0..AGENTS {
            let Some(agent) = u16::try_from(i)
                .ok()
                .and_then(|i| AgentSlot::new(i, AGENTS))
            else {
                continue;
            };
            let Some((row, col)) = at(s, i) else {
                continue;
            };
            // The mask forbids moves off the grid; staying replaces them.
            let action = actions
                .discrete(agent, 0)
                .and_then(|a| u32::try_from(a).ok())
                .unwrap_or(0);
            let (row, col) = moved(row, col, action).unwrap_or((row, col));
            if let (Some(r), Some(c)) = (s.rows.get_mut(i), s.cols.get_mut(i)) {
                *r = row;
                *c = col;
            }
        }
        for k in 0..ITEMS {
            if s.taken.get(k).is_some_and(|t| t.get()) {
                continue;
            }
            let item = (s.item_rows.get(k).copied(), s.item_cols.get(k).copied());
            let taker =
                (0..AGENTS).find(|&i| at(s, i).is_some_and(|(r, c)| (Some(r), Some(c)) == item));
            let Some(i) = taker else {
                continue;
            };
            if let Some(t) = s.taken.get_mut(k) {
                *t = Bool8::from(true);
            }
            if let Some(agent) = u16::try_from(i)
                .ok()
                .and_then(|i| AgentSlot::new(i, AGENTS))
            {
                out.set_reward(agent, 1.0);
            }
        }
        let done = s.taken.iter().all(|t| t.get());
        s.phase = if done { Phase::Done } else { Phase::Collecting };
        if done {
            out.end_episode(EpisodeEnd::Terminated);
        } else if s.step >= cfg.episode {
            out.end_episode(EpisodeEnd::Truncated);
        }
    }

    fn observe(
        _: &PickupConfig,
        s: &PickupState,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, PickupObs>,
    ) {
        let Some(mut leaf) = out.leaf_mut(PickupObs::VALUES) else {
            return;
        };
        let me = agent.index();
        let Some((row, col)) = at(s, me) else {
            return;
        };
        let other = at(s, 1 - me.min(1));
        leaf.set(0, i32::from(row));
        leaf.set(1, i32::from(col));
        let mut k = 2;
        for dr in -1_i32..=1 {
            for dc in -1_i32..=1 {
                let r = u8::try_from(i32::from(row) + dr).ok().filter(|&r| r < SIDE);
                let c = u8::try_from(i32::from(col) + dc).ok().filter(|&c| c < SIDE);
                let value = match (r, c) {
                    (Some(r), Some(c)) if item_on(s, (r, c)) => 1,
                    (Some(r), Some(c)) if (dr, dc) != (0, 0) && other == Some((r, c)) => 2,
                    (Some(_), Some(_)) => 0,
                    _ => 3,
                };
                leaf.set(k, value);
                k += 1;
            }
        }
    }
}

/// Moves that stay on the grid are legal; staying is the no-op.
impl ActionMask for Pickup {
    fn action_mask(_: &PickupConfig, s: &PickupState, agent: AgentSlot, out: &mut MaskWriter<'_>) {
        let cell = at(s, agent.index());
        for a in 0..CHOICES {
            let legal = cell.is_some_and(|(r, c)| moved(r, c, a).is_some());
            out.set(usize::try_from(a).unwrap_or(0), legal);
        }
    }

    fn noop(_: &PickupConfig, _: &PickupState, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, 0);
    }
}

/// The scripted baseline: stay.
impl Heuristic for Pickup {
    fn act(
        _: &PickupConfig,
        _: &PickupState,
        _: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        out.set_discrete(0, 0);
    }
}

/// The invariants an imported state must hold (ADR-0040).
impl Validate for Pickup {
    fn validate(cfg: &PickupConfig, s: &PickupState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let on_grid = |v: &[u8]| v.iter().all(|&x| x < SIDE);
        if !on_grid(&s.rows)
            || !on_grid(&s.cols)
            || !on_grid(&s.item_rows)
            || !on_grid(&s.item_cols)
        {
            return invalid("a cell off the grid");
        }
        if s.step > cfg.episode {
            return invalid("a step past the episode length");
        }
        let done = s.taken.iter().all(|t| t.get());
        if (s.phase == Phase::Done) != done {
            return invalid("a phase that disagrees with the items");
        }
        Ok(())
    }
}

/// The id and version persisted records carry (SPEC 9).
impl Persist for Pickup {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
