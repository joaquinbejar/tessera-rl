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
//! - Ends: terminated when every evader is caught. Truncation is the
//!   engine's step limit (`PoolOptions::max_episode_steps`, SPEC 5.4).
//! - Observation: an egocentric `(2r+1) x (2r+1)` window around the agent,
//!   `Box(u8, [3, 2r+1, 2r+1])`: channel 0 marks cells outside the grid,
//!   channels 1 and 2 count the pursuers and the live evaders on each cell
//!   (the observer included).

pub mod grid;
mod window;

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Bool8, Environment, EpisodeEnd, Heuristic, ImperfectInformation,
    InfoHeuristic, InvariantViolation, NoAdaptive, ObsLeaf, Observation, RecordHeader, Reserved,
    SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

use grid::{ACTIONS, STAY, delta, distance, moved, place};

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/Tag-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;

// Registration under the versioned id (SPEC 10, ADR-0054).
tessera_rl_core::register!(Tag, id = "tessera/Tag-v1");
/// Agent slots.
pub const MAX_AGENTS: usize = 8;
/// Largest observation radius.
pub const MAX_RADIUS: u32 = 7;

const MAX_SIDE: usize = 2 * MAX_RADIUS as usize + 1;
/// Bytes of the largest window, `[3, 15, 15]`.
pub const WINDOW_BYTES: usize = 3 * MAX_SIDE * MAX_SIDE;
const PER_MILLE: core::num::NonZeroU32 = core::num::NonZeroU32::new(1000).unwrap();
const MOVES: core::num::NonZeroU32 = core::num::NonZeroU32::new(ACTIONS).unwrap();

fn check_agents(cfg: &TagConfig) -> Result<(), tessera_rl_core::ConfigError> {
    if cfg.agents() > MAX_AGENTS {
        return Err(tessera_rl_core::ConfigError {
            field: "evaders",
            reason: "pursuers + evaders must not exceed 8",
        });
    }
    Ok(())
}

/// Tag configuration. `pursuers + evaders` must not exceed 8.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
#[config(validate = check_agents)]
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

/// Values of the egocentric window of slot `me`, `[3, side, side]`
/// row-major in the first `3 * side * side` bytes, zero after them. It
/// reads only `me`'s position and the cells inside its window.
fn egocentric_window(cfg: &TagConfig, state: &TagState, me: usize) -> [u8; WINDOW_BYTES] {
    let side = cfg.side();
    let plane = side * side;
    let mut window = [0u8; WINDOW_BYTES];
    let Some(&[cx, cy]) = state.pos.get(me) else {
        return window;
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
        if let Some(v) =
            cell_index(i32::from(x), i32::from(y)).and_then(|k| window.get_mut(channel * plane + k))
        {
            // At most 8 agents per cell.
            *v += 1;
        }
    }
    window
}

/// The Tag environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Tag;

fn slot(index: usize) -> Option<AgentSlot> {
    AgentSlot::new(u16::try_from(index).ok()?, MAX_AGENTS)
}

/// The role of a pursuer slot.
pub const PURSUER: &str = "pursuer";
/// The role of an evader slot.
pub const EVADER: &str = "evader";

/// The role of `slot` by the config's split: [`PURSUER`] for the first
/// `pursuers` slots, [`EVADER`] for the next `evaders`, `""` for unused
/// slots.
#[must_use]
fn role_of(cfg: &TagConfig, slot: AgentSlot) -> &'static str {
    if cfg.is_pursuer(slot.index()) {
        PURSUER
    } else if slot.index() < cfg.agents() {
        EVADER
    } else {
        ""
    }
}

impl Environment for Tag {
    /// Parallel natively and AEC through cycle buffering (SPEC 12, ADR-0080).
    const ADAPTERS: tessera_rl_core::Adapters =
        tessera_rl_core::Adapters::PZ_PARALLEL.union(tessera_rl_core::Adapters::PZ_AEC);
    type State = TagState;
    type Config = TagConfig;
    type Obs = TagObs;
    type Action = ();
    type Header = RecordHeader<MAX_AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = MAX_AGENTS;
    const TIER: Tier = Tier::CrossPlatform;
    /// Agents see only their window: two states that differ outside it
    /// give the same observation.
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

    /// Team 0 the pursuers, team 1 the evaders; unused slots keep the
    /// default, 0.
    fn team(cfg: &TagConfig, slot: AgentSlot) -> u16 {
        match role_of(cfg, slot) {
            EVADER => 1,
            _ => 0,
        }
    }

    /// `"pursuer"` or `"evader"` by the config's split; unused slots keep
    /// the default, `""`.
    fn role(cfg: &TagConfig, slot: AgentSlot) -> &'static str {
        role_of(cfg, slot)
    }

    fn reset(cfg: &TagConfig, rng: &mut StepRng, state: &mut TagState) {
        let n = cfg.agents().min(MAX_AGENTS);
        if let Some(cells) = state.pos.get_mut(..n) {
            place(cells, cfg.width, cfg.height, rng);
        }
        state.caught = [Bool8::FALSE; MAX_AGENTS];
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

        if remaining == 0 {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        cfg: &TagConfig,
        state: &TagState,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, TagObs>,
    ) {
        // The rows are zero on entry (SPEC 10): only the nonzero bytes are
        // written (#771). `egocentric_window` builds the same bytes for
        // `info_state`.
        if let Some(mut leaf) = out.leaf_mut(TagObs::WINDOW) {
            window::write_window(cfg, state, agent.index(), |k, v| {
                leaf.set(k, v);
            });
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
/// evaders caught.
impl Validate for Tag {
    fn validate(cfg: &TagConfig, state: &TagState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
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

    /// ADR-0040: in a record that has been reset, a used slot is alive
    /// exactly when its agent is not caught, and an unused slot is neither
    /// alive nor acting. A never-reset record has no episode yet.
    fn validate_record(
        cfg: &TagConfig,
        header: &tessera_rl_core::HeaderView<'_>,
        state: &TagState,
    ) -> Result<(), InvariantViolation> {
        Self::validate(cfg, state)?;
        if header.phase() == tessera_rl_core::Phase::NeverReset {
            return Ok(());
        }
        for slot in 0..header.agents() {
            let Some(view) = header.agent(slot) else {
                continue;
            };
            if slot < cfg.agents() {
                let caught = state.caught.get(slot).is_some_and(|c| c.get());
                if view.alive == caught {
                    return Err(InvariantViolation {
                        code: "a used slot alive exactly when its agent is caught",
                    });
                }
            } else if view.alive || view.acting {
                return Err(InvariantViolation {
                    code: "an unused slot that is alive or acting",
                });
            }
        }
        Ok(())
    }
}

/// The window baseline (ADR-0079, #437 option (a)), from the agent's
/// window alone: a pursuer steps towards the nearest live evader it sees,
/// an evader to the visible cell farthest from the nearest pursuer it
/// sees, by Manhattan distance inside the window. A move is a candidate
/// only when its target cell is inside the window and on the grid. Ties go
/// to the lowest action. With no opponent in the window, the lowest legal
/// action under the mask: stay (action 0), which is always legal.
fn window_act(cfg: &TagConfig, window: &[u8; WINDOW_BYTES], me: usize) -> i32 {
    let side = cfg.side();
    let plane = side * side;
    let r = i32::try_from(cfg.radius).unwrap_or(0);
    let pursuer = cfg.is_pursuer(me);
    // Opponents: evaders (channel 2) for a pursuer, pursuers (channel 1)
    // for an evader.
    let channel = if pursuer { 2 } else { 1 };
    // The window cell of offset `(dx, dy)` from the observer, if inside it.
    let cell = |dx: i32, dy: i32| {
        let col = usize::try_from(dx + r).ok()?;
        let row = usize::try_from(dy + r).ok()?;
        (col < side && row < side).then_some(row * side + col)
    };
    let on_grid = |dx: i32, dy: i32| {
        cell(dx, dy)
            .and_then(|k| window.get(k))
            .is_some_and(|&outside| outside == 0)
    };
    let nearest = |dx: i32, dy: i32| {
        let mut best: Option<u32> = None;
        for row in 0..side {
            for col in 0..side {
                let seen = window
                    .get(channel * plane + row * side + col)
                    .is_some_and(|&n| n > 0);
                if !seen {
                    continue;
                }
                let ox = i32::try_from(col).unwrap_or(0) - r;
                let oy = i32::try_from(row).unwrap_or(0) - r;
                let d = distance([coord(dx), coord(dy)], [coord(ox), coord(oy)]);
                best = Some(best.map_or(d, |b| b.min(d)));
            }
        }
        best
    };
    let Some(here) = nearest(0, 0) else {
        // No visible opponent: the lowest legal action, stay.
        return (0..ACTIONS)
            .filter_map(|a| i32::try_from(a).ok())
            .find(|&a| {
                let [dx, dy] = delta(a);
                on_grid(dx, dy)
            })
            .unwrap_or(STAY);
    };
    let mut best = (STAY, here);
    for action in 1..=4 {
        let [dx, dy] = delta(action);
        if !on_grid(dx, dy) {
            continue;
        }
        let Some(d) = nearest(dx, dy) else {
            continue;
        };
        if (pursuer && d < best.1) || (!pursuer && d > best.1) {
            best = (action, d);
        }
    }
    best.0
}

/// A window offset as a cell coordinate (offsets are at most 8 in size).
fn coord(v: i32) -> i16 {
    i16::try_from(v).unwrap_or(0)
}

/// The scripted baseline is the window baseline: `act` computes the
/// agent's information state and calls [`InfoHeuristic::act_info`], so the
/// baseline and a heuristic slot of a policy map agree, and states with
/// equal windows give equal actions.
impl Heuristic for Tag {
    fn act(
        cfg: &TagConfig,
        state: &TagState,
        agent: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let info = Self::info_state(cfg, state, agent);
        Self::act_info(cfg, &info, agent, rng, out);
    }
}

/// The window baseline through the information state (ADR-0079 item 4.2).
impl InfoHeuristic for Tag {
    fn act_info(
        cfg: &TagConfig,
        info: &TagInfo,
        agent: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        out.set_discrete(0, window_act(cfg, &info.window, agent.index()));
    }
}

/// What one agent knows: exactly its current egocentric window, the same
/// bytes as its observation, zero-padded to the largest radius. Its
/// absolute cell is not part of it: the observation does not reveal it
/// (the out-of-grid channel carries what the agent sees of the edges).
///
/// This is a current-observation information state: it does not track
/// history and is not a sufficient statistic of the observation history
/// (an agent may remember where it saw an evader). A history-tracking
/// information state would be a different version of the environment.
/// Phase 2 (`Determinize`, validated with Leduc) must revisit it. The
/// episode step is not part of it: it lives in the engine header.
#[derive(EnvState, Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TagInfo {
    /// The window, `[3, 2r+1, 2r+1]` then zeros.
    pub window: [u8; WINDOW_BYTES],
    /// Padding to an even size.
    pub reserved: Reserved<1>,
}

impl Default for TagInfo {
    fn default() -> Self {
        Self {
            window: [0; WINDOW_BYTES],
            reserved: Reserved::default(),
        }
    }
}

/// Reads only the agent's own window (noninterference by construction).
/// No `Determinize` in phase 1a.
impl ImperfectInformation for Tag {
    type InfoState = TagInfo;

    fn info_state(cfg: &TagConfig, state: &TagState, agent: AgentSlot) -> TagInfo {
        TagInfo {
            window: egocentric_window(cfg, state, agent.index()),
            reserved: Reserved::default(),
        }
    }
}

impl tessera_rl_core::Persist for Tag {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
