// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tag (`tessera/Tag-v1`): stochastic pursuit-evasion on a grid,
//! cross-platform tier, integer state.
//!
//! - Agents: slots `0..pursuers` are pursuers, the next `evaders` slots
//!   are evaders; 2 to 8 in total (`initial_agents`). Unused slots stay
//!   empty.
//! - Moves are simultaneous: `Discrete(5)`, stay, north, south, east,
//!   west. With probability `slip` per mille an agent's move is replaced by
//!   a uniformly random one. A move off the grid stays in place; the mask
//!   marks those moves illegal and the no-op is stay.
//! - Capture: after the moves, an evader on a cell with at least one
//!   pursuer dies (`agent_died`), with reward -1; every pursuer on that cell
//!   gets +1. Agents only meet on cells: swapping places is not a capture.
//! - Ends: terminated when every evader is caught; truncated after
//!   `max_steps`.
//! - Observation: an egocentric `(2r+1) x (2r+1)` window around the agent,
//!   `Box(u8, [3, 2r+1, 2r+1])`: channel 0 marks cells outside the grid,
//!   channels 1 and 2 count the pursuers and the live evaders on each cell
//!   (the observer included).

pub mod grid;

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Bool8, Environment, EpisodeEnd, Heuristic, InvariantViolation,
    NoAdaptive, ObsLeaf, Observation, RecordHeader, Reserved, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

use grid::{ACTIONS, Cell, STAY, distance, moved, place};

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/Tag-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;
/// Agent slots.
pub const MAX_AGENTS: usize = 8;
/// Largest observation radius.
pub const MAX_RADIUS: u32 = 7;

const MAX_SIDE: usize = 2 * MAX_RADIUS as usize + 1;
const PER_MILLE: core::num::NonZeroU32 = core::num::NonZeroU32::new(1000).unwrap();
const MOVES: core::num::NonZeroU32 = core::num::NonZeroU32::new(ACTIONS).unwrap();

/// Tag configuration. `pursuers + evaders` must not exceed 8; the engine
/// checks it at construction as `initial_agents`.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct TagConfig {
    /// Grid width in cells.
    #[config(default = 8, range = 3..=64)]
    pub width: u32,
    /// Grid height in cells.
    #[config(default = 8, range = 3..=64)]
    pub height: u32,
    /// Number of pursuers (the first slots).
    #[config(default = 1, range = 1..=7)]
    pub pursuers: u32,
    /// Number of evaders (the slots after the pursuers).
    #[config(default = 3, range = 1..=7)]
    pub evaders: u32,
    /// Probability, per mille, that a move is replaced by a random one.
    #[config(default = 100, range = 0..=1000)]
    pub slip: u32,
    /// Observation radius `r`: the window is `2r+1` cells wide.
    #[config(default = 2, range = 0..=7)]
    pub radius: u32,
    /// Steps after which the episode is truncated.
    #[config(default = 200, range = 1..=1_000_000)]
    pub max_steps: u32,
}

impl TagConfig {
    /// Total agents: pursuers then evaders.
    #[must_use]
    pub fn agents(&self) -> usize {
        usize::try_from(self.pursuers + self.evaders).unwrap_or(usize::MAX)
    }

    /// Whether `slot` is a pursuer.
    #[must_use]
    pub fn is_pursuer(&self, slot: usize) -> bool {
        u32::try_from(slot).is_ok_and(|s| s < self.pursuers)
    }

    fn side(&self) -> usize {
        usize::try_from(2 * self.radius + 1).unwrap_or(MAX_SIDE)
    }
}

/// Tag state.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct TagState {
    /// Cell of each slot, `[x, y]`.
    pub pos: [[i16; 2]; MAX_AGENTS],
    /// Evaders caught in this episode.
    pub caught: [Bool8; MAX_AGENTS],
    /// Steps taken in this episode.
    pub steps: u32,
    /// Padding to a multiple of 8 bytes.
    pub reserved: Reserved<4>,
}

impl TagState {
    fn active(&self, cfg: &TagConfig, slot: usize) -> bool {
        slot < cfg.agents() && !self.caught.get(slot).is_some_and(|c| c.get())
    }
}

/// Tag observation: the egocentric window.
#[derive(Observation)]
pub struct TagObs {
    /// `[channel, row, column]`, row-major; the observer is at the centre.
    #[space(box(low = 0, high = 255, dtype = u8, shape = |c: &TagConfig| [3, 2 * c.radius + 1, 2 * c.radius + 1]))]
    pub window: ObsLeaf<u8>,
}

/// The Tag environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Tag;

fn slot(index: usize) -> Option<AgentSlot> {
    AgentSlot::new(u16::try_from(index).ok()?, MAX_AGENTS)
}

impl Environment for Tag {
    type State = TagState;
    type Config = TagConfig;
    type Obs = TagObs;
    type Action = ();
    type Header = RecordHeader<MAX_AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = MAX_AGENTS;
    const TIER: Tier = Tier::CrossPlatform;
    const PERFECT_INFORMATION: bool = false;

    fn observation_space(cfg: &TagConfig) -> Result<Space, SpaceError> {
        TagObs::space(cfg)
    }

    fn action_space(_: &TagConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: ACTIONS })
    }

    fn initial_agents(cfg: &TagConfig) -> usize {
        cfg.agents()
    }

    fn reset(cfg: &TagConfig, rng: &mut StepRng, state: &mut TagState) {
        let n = cfg.agents().min(MAX_AGENTS);
        if let Some(cells) = state.pos.get_mut(..n) {
            place(cells, cfg.width, cfg.height, rng);
        }
        state.caught = [Bool8::FALSE; MAX_AGENTS];
        state.steps = 0;
    }

    fn step(
        cfg: &TagConfig,
        state: &mut TagState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        // Simultaneous moves, drawn in slot order.
        let mut next = state.pos;
        for (i, cell) in next.iter_mut().enumerate() {
            let Some(agent) = slot(i).filter(|_| state.active(cfg, i)) else {
                continue;
            };
            let mut action = actions.discrete(agent, 0).unwrap_or(STAY);
            if cfg.slip > 0 && rng.below_u32(PER_MILLE) < cfg.slip {
                action = i32::try_from(rng.below_u32(MOVES)).unwrap_or(STAY);
            }
            if let Some(to) = moved(*cell, action, cfg.width, cfg.height) {
                *cell = to;
            }
        }
        state.pos = next;

        let pursuers = usize::try_from(cfg.pursuers).unwrap_or(0);
        let mut remaining = 0;
        for e in pursuers..cfg.agents().min(MAX_AGENTS) {
            if !state.active(cfg, e) {
                continue;
            }
            let at = state.pos.get(e).copied();
            let mut hunters = 0;
            for p in 0..pursuers {
                if state.pos.get(p).copied() == at
                    && let Some(agent) = slot(p)
                {
                    out.add_reward(agent, 1.0);
                    hunters += 1;
                }
            }
            if hunters == 0 {
                remaining += 1;
                continue;
            }
            if let (Some(agent), Some(caught)) = (slot(e), state.caught.get_mut(e)) {
                *caught = Bool8::TRUE;
                out.set_reward(agent, -1.0);
                out.agent_died(agent);
            }
        }

        // Bounded by max_steps, checked below.
        state.steps += 1;
        if remaining == 0 {
            out.end_episode(EpisodeEnd::Terminated);
        } else if state.steps >= cfg.max_steps {
            out.end_episode(EpisodeEnd::Truncated);
        }
    }

    fn observe(
        cfg: &TagConfig,
        state: &TagState,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, TagObs>,
    ) {
        let side = cfg.side();
        let plane = side * side;
        let mut window = [0u8; 3 * MAX_SIDE * MAX_SIDE];
        let Some(&[cx, cy]) = state.pos.get(agent.index()) else {
            return;
        };
        let r = i32::try_from(cfg.radius).unwrap_or(0);
        let cell_index = |x: i32, y: i32| {
            let col = usize::try_from(x - i32::from(cx) + r).ok()?;
            let row = usize::try_from(y - i32::from(cy) + r).ok()?;
            (col < side && row < side).then_some(row * side + col)
        };
        // Channel 0: cells outside the grid.
        for row in 0..side {
            for col in 0..side {
                let x = i32::from(cx) + i32::try_from(col).unwrap_or(0) - r;
                let y = i32::from(cy) + i32::try_from(row).unwrap_or(0) - r;
                let inside = u32::try_from(x).is_ok_and(|x| x < cfg.width)
                    && u32::try_from(y).is_ok_and(|y| y < cfg.height);
                if let Some(v) = window.get_mut(row * side + col) {
                    *v = u8::from(!inside);
                }
            }
        }
        // Channels 1 and 2: pursuers and live evaders.
        for (j, &[x, y]) in state.pos.iter().enumerate() {
            if !state.active(cfg, j) {
                continue;
            }
            let channel = if cfg.is_pursuer(j) { 1 } else { 2 };
            if let Some(v) = cell_index(i32::from(x), i32::from(y))
                .and_then(|k| window.get_mut(channel * plane + k))
            {
                // At most 8 agents per cell.
                *v += 1;
            }
        }
        if let Some(mut leaf) = out.leaf_mut(TagObs::WINDOW) {
            for (k, v) in window.iter().take(3 * plane).enumerate() {
                leaf.set(k, *v);
            }
        }
    }
}

/// Moves off the grid are illegal; stay is the no-op.
impl ActionMask for Tag {
    fn action_mask(cfg: &TagConfig, state: &TagState, agent: AgentSlot, out: &mut MaskWriter<'_>) {
        let Some(&cell) = state.pos.get(agent.index()) else {
            out.fill(true);
            return;
        };
        for action in 0..ACTIONS {
            let legal = i32::try_from(action)
                .is_ok_and(|a| moved(cell, a, cfg.width, cfg.height).is_some());
            out.set(usize::try_from(action).unwrap_or(0), legal);
        }
    }

    fn noop(_: &TagConfig, _: &TagState, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, STAY);
    }
}

/// Imported states: used slots on the grid, unused slots zeroed, only
/// evaders caught, and within one episode.
impl Validate for Tag {
    fn validate(cfg: &TagConfig, state: &TagState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        if state.steps > cfg.max_steps {
            return invalid("steps beyond max_steps");
        }
        for (i, &[x, y]) in state.pos.iter().enumerate() {
            let used = i < cfg.agents();
            if !used && [x, y] != [0, 0] {
                return invalid("an unused slot with a position");
            }
            let inside = u32::try_from(x).is_ok_and(|x| x < cfg.width)
                && u32::try_from(y).is_ok_and(|y| y < cfg.height);
            if used && !inside {
                return invalid("an agent outside the grid");
            }
            let caught = state.caught.get(i).is_some_and(|c| c.get());
            if caught && (!used || cfg.is_pursuer(i)) {
                return invalid("a caught slot that is not an evader");
            }
        }
        Ok(())
    }
}

/// Scripted baseline: pursuers step towards the nearest live evader,
/// evaders step to the legal cell farthest from the nearest pursuer. Ties
/// go to the lowest slot and the lowest action.
impl Heuristic for Tag {
    fn act(
        cfg: &TagConfig,
        state: &TagState,
        agent: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let me = agent.index();
        let Some(&from) = state.pos.get(me) else {
            out.set_discrete(0, STAY);
            return;
        };
        let pursuer = cfg.is_pursuer(me);
        let nearest = |cell: Cell| {
            state
                .pos
                .iter()
                .enumerate()
                .filter(|&(j, _)| state.active(cfg, j) && cfg.is_pursuer(j) != pursuer)
                .map(|(_, &other)| distance(cell, other))
                .min()
                .unwrap_or(0)
        };
        let mut best = (STAY, nearest(from));
        for action in 1..=4 {
            let Some(to) = moved(from, action, cfg.width, cfg.height) else {
                continue;
            };
            let d = nearest(to);
            if (pursuer && d < best.1) || (!pursuer && d > best.1) {
                best = (action, d);
            }
        }
        out.set_discrete(0, best.0);
    }
}
