// SPDX-License-Identifier: MIT OR Apache-2.0

//! Key and door (`tessera/GridKeyDoor-v1`): a grid world generated at each
//! reset, seen from each agent's own position and facing, for 1 to 4 agents.
//! Cross-platform tier, integer state. Validates the Grid worlds row of SPEC 16.
//!
//! - World: a room of `size` x `size` cells split by one vertical wall with one
//!   door, closed at the start. A key and the agents start on the left, the goal
//!   lies on the right. All are drawn at reset from the instance's `StepRng`,
//!   in a fixed order; nothing is drawn afterwards.
//! - Action: `Discrete(5)`: turn left, turn right, forward, interact, wait.
//!   Turning and waiting are always legal; waiting is the no-op. Forward needs a
//!   free front cell: walls, the closed door, the key on the floor and agents
//!   that have not finished all block. Interact takes the key in front, or opens
//!   the closed door in front for the agent holding the key. A finished agent may
//!   only wait.
//! - Turns: simultaneous, resolved in slot order inside a step, each action
//!   against the world as earlier slots left it. A move or an interaction that
//!   an earlier slot made impossible does nothing. The mask is read before the
//!   step, so a cell occupied then stays refused even if its agent leaves.
//! - Finishing: an agent on the goal is finished and stays there; several may
//!   stand on it.
//! - Rewards and ends: +1 to every agent, and terminated, on the step when the
//!   last one finishes; otherwise 0. Truncation is the engine's step limit.
//! - Observation: `view`, `Box(u8, [5, 7, 7])`, rotated to the facing with the
//!   agent at the bottom centre looking up. Channels: wall or outside, closed
//!   door, key on the floor, goal, other agents. `carrying`, `Box(u8, [1])`.
//!   Unlike MiniGrid-style vision there is no occlusion: an agent sees through
//!   walls inside its window.

mod checks;

use core::num::NonZeroU32;

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Bool8, Environment, EpisodeEnd, NoAdaptive, ObsLeaf, Observation,
    RecordHeader, SpaceError, StepOut, Tier,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/GridKeyDoor-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;

// Registration under the versioned id (SPEC 10, ADR-0054).
tessera_rl_core::register!(KeyDoor, id = "tessera/GridKeyDoor-v1");
/// Agent slots.
pub const MAX_AGENTS: usize = 4;
/// Turn left.
pub const LEFT: i32 = 0;
/// Turn right.
pub const RIGHT: i32 = 1;
/// Move to the front cell.
pub const FORWARD: i32 = 2;
/// Take the key or open the door in front.
pub const INTERACT: i32 = 3;
/// Do nothing; the no-op.
pub const WAIT: i32 = 4;
/// Choices of the action head.
pub const ACTIONS: u32 = 5;
/// `key_holder` when nobody holds the key.
pub const NO_HOLDER: u8 = u8::MAX;
/// Cells on a side of the view.
pub const VIEW: i32 = 7;

/// Key and door configuration.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct KeyDoorConfig {
    /// Cells on a side of the room.
    #[config(default = 8, range = 6..=16)]
    pub size: u32,
    /// Number of agents.
    #[config(default = 1, range = 1..=4)]
    pub agents: u32,
}

/// Key and door state: 24 bytes, no padding. Unused slots hold zero.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct KeyDoorState {
    /// Column and row of each agent.
    pub pos: [[u8; 2]; MAX_AGENTS],
    /// Facing of each agent: 0 north, 1 east, 2 south, 3 west.
    pub facing: [u8; MAX_AGENTS],
    /// Whether each agent stands on the goal.
    pub finished: [Bool8; MAX_AGENTS],
    /// Column and row of the key's floor cell; it stops mattering once held.
    pub key: [u8; 2],
    /// Column and row of the goal.
    pub goal: [u8; 2],
    /// Column of the wall.
    pub wall_col: u8,
    /// Row of the door in the wall.
    pub door_row: u8,
    /// Whether the door is open.
    pub door_open: Bool8,
    /// Slot holding the key, or [`NO_HOLDER`].
    pub key_holder: u8,
}

/// Column and row steps of a facing.
#[must_use]
pub const fn step_of(facing: u8) -> (i32, i32) {
    match facing {
        0 => (0, -1),
        1 => (1, 0),
        2 => (0, 1),
        _ => (-1, 0),
    }
}

impl KeyDoorState {
    fn cell(&self, slot: usize) -> Option<(i32, i32)> {
        self.pos
            .get(slot)
            .map(|p| (i32::from(p[0]), i32::from(p[1])))
    }

    /// The agent's facing, 0 for a slot out of range.
    fn facing_of(&self, slot: usize) -> u8 {
        self.facing.get(slot).copied().unwrap_or(0)
    }

    /// The cell in front of the agent.
    fn front(&self, slot: usize) -> Option<(i32, i32)> {
        let (c, r) = self.cell(slot)?;
        let (dc, dr) = step_of(self.facing_of(slot));
        Some((c + dc, r + dr))
    }

    /// Whether the agent has finished.
    #[must_use]
    pub fn is_finished(&self, slot: usize) -> bool {
        self.finished.get(slot).is_some_and(|f| f.get())
    }

    const fn is_door(&self, col: i32, row: i32) -> bool {
        col == self.wall_col as i32 && row == self.door_row as i32
    }

    const fn is_wall(&self, col: i32, row: i32) -> bool {
        col == self.wall_col as i32 && row != self.door_row as i32
    }

    /// Whether the key lies on the floor at `col`, `row`.
    #[must_use]
    pub const fn key_at(&self, col: i32, row: i32) -> bool {
        self.key_holder == NO_HOLDER && col == self.key[0] as i32 && row == self.key[1] as i32
    }

    /// Whether an unfinished agent other than `except` stands at `col`, `row`.
    fn agent_blocks(&self, cfg: &KeyDoorConfig, col: i32, row: i32, except: usize) -> bool {
        (0..cfg.agents as usize)
            .any(|k| k != except && !self.is_finished(k) && self.cell(k) == Some((col, row)))
    }

    /// Whether slot `slot` could move to `col`, `row`.
    #[must_use]
    pub fn can_enter(&self, cfg: &KeyDoorConfig, col: i32, row: i32, slot: usize) -> bool {
        let n = cfg.size.cast_signed();
        (0..n).contains(&col)
            && (0..n).contains(&row)
            && !self.is_wall(col, row)
            && (!self.is_door(col, row) || self.door_open.get())
            && !self.key_at(col, row)
            && !self.agent_blocks(cfg, col, row, slot)
    }

    /// Whether `action` is legal for `slot` now.
    #[must_use]
    pub fn is_legal(&self, cfg: &KeyDoorConfig, slot: usize, action: i32) -> bool {
        if action == WAIT {
            return true;
        }
        if self.is_finished(slot) {
            return false;
        }
        let Some((col, row)) = self.front(slot) else {
            return false;
        };
        match action {
            LEFT | RIGHT => true,
            FORWARD => self.can_enter(cfg, col, row, slot),
            INTERACT => {
                self.key_at(col, row)
                    || (self.is_door(col, row)
                        && !self.door_open.get()
                        && usize::from(self.key_holder) == slot)
            }
            _ => false,
        }
    }

    /// Applies `action` for `slot`; one that is not legal now does nothing.
    pub fn apply(&mut self, cfg: &KeyDoorConfig, slot: usize, action: i32) {
        if !self.is_legal(cfg, slot, action) {
            return;
        }
        let front = self.front(slot);
        let f = self.facing_of(slot);
        match action {
            LEFT | RIGHT => {
                if let Some(v) = self.facing.get_mut(slot) {
                    *v = (f + if action == LEFT { 3 } else { 1 }) % 4;
                }
            }
            FORWARD => {
                let Some((col, row)) = front else {
                    return;
                };
                let goal = (i32::from(self.goal[0]), i32::from(self.goal[1]));
                if let (Some(p), Ok(c), Ok(r)) =
                    (self.pos.get_mut(slot), u8::try_from(col), u8::try_from(row))
                {
                    *p = [c, r];
                }
                if (col, row) == goal
                    && let Some(done) = self.finished.get_mut(slot)
                {
                    *done = Bool8::TRUE;
                }
            }
            INTERACT => {
                if self.key_holder == NO_HOLDER {
                    self.key_holder = u8::try_from(slot).unwrap_or(NO_HOLDER);
                } else {
                    self.door_open = Bool8::TRUE;
                }
            }
            _ => {}
        }
    }

    /// Whether every agent has finished.
    #[must_use]
    pub fn all_finished(&self, cfg: &KeyDoorConfig) -> bool {
        (0..cfg.agents as usize).all(|k| self.is_finished(k))
    }
}

/// Key and door observation.
#[derive(Observation)]
pub struct KeyDoorObs {
    /// `[channel, row, column]`, row-major; the agent is at row 6, column 3.
    #[space(box(low = 0, high = 1, dtype = u8, shape = [5, 7, 7]))]
    pub view: ObsLeaf<u8>,
    /// Whether the agent holds the key.
    #[space(box(low = 0, high = 1, dtype = u8, shape = [1]))]
    pub carrying: ObsLeaf<u8>,
}

/// The key and door environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeyDoor;

fn below(rng: &mut StepRng, bound: u32) -> u32 {
    rng.below_u32(NonZeroU32::new(bound).unwrap_or(NonZeroU32::MIN))
}

/// The `nth` cell, column by column, among the columns before `wall` of a room
/// of side `n`, skipping `taken`.
fn nth_free(wall: u8, n: u8, taken: &[[u8; 2]], nth: u32) -> [u8; 2] {
    let mut left = nth;
    for col in 0..wall {
        for row in 0..n {
            if taken.contains(&[col, row]) {
                continue;
            }
            if left == 0 {
                return [col, row];
            }
            left -= 1;
        }
    }
    [0, 0]
}

impl Environment for KeyDoor {
    type State = KeyDoorState;
    type Config = KeyDoorConfig;
    type Obs = KeyDoorObs;
    type Action = ();
    type Header = RecordHeader<MAX_AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = MAX_AGENTS;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &KeyDoorConfig) -> Result<Space, SpaceError> {
        KeyDoorObs::space(cfg)
    }

    fn action_space(_: &KeyDoorConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: ACTIONS })
    }

    fn initial_agents(cfg: &KeyDoorConfig) -> usize {
        cfg.agents as usize
    }

    /// Draws, in order: the wall, the door, the key, the goal, then each
    /// agent's cell and facing.
    fn reset(cfg: &KeyDoorConfig, rng: &mut StepRng, s: &mut KeyDoorState) {
        let side = cfg.size;
        let n = u8::try_from(side).unwrap_or(0);
        *s = KeyDoorState {
            key_holder: NO_HOLDER,
            ..KeyDoorState::default()
        };
        let wall = 2 + below(rng, side - 4);
        s.wall_col = u8::try_from(wall).unwrap_or(2);
        s.door_row = u8::try_from(below(rng, side)).unwrap_or(0);
        let cells = wall * side;
        s.key = nth_free(s.wall_col, n, &[], below(rng, cells));
        let goal_col = wall + 1 + below(rng, side - wall - 1);
        s.goal = [
            u8::try_from(goal_col).unwrap_or(0),
            u8::try_from(below(rng, side)).unwrap_or(0),
        ];
        let mut taken = [[0u8; 2]; MAX_AGENTS + 1];
        if let Some(first) = taken.first_mut() {
            *first = s.key;
        }
        for k in 0..cfg.agents as usize {
            let used = taken.get(..=k).unwrap_or(&[]);
            let free = cells - u32::try_from(k).unwrap_or(0) - 1;
            let at = nth_free(s.wall_col, n, used, below(rng, free));
            if let (Some(p), Some(t), Some(f)) =
                (s.pos.get_mut(k), taken.get_mut(k + 1), s.facing.get_mut(k))
            {
                *p = at;
                *t = at;
                *f = u8::try_from(below(rng, 4)).unwrap_or(0);
            }
        }
    }

    fn step(
        cfg: &KeyDoorConfig,
        state: &mut KeyDoorState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        for k in 0..cfg.agents as usize {
            let Some(agent) = u16::try_from(k)
                .ok()
                .and_then(|i| AgentSlot::new(i, MAX_AGENTS))
            else {
                continue;
            };
            state.apply(cfg, k, actions.discrete(agent, 0).unwrap_or(WAIT));
        }
        if state.all_finished(cfg) {
            for k in 0..cfg.agents as usize {
                if let Some(agent) = u16::try_from(k)
                    .ok()
                    .and_then(|i| AgentSlot::new(i, MAX_AGENTS))
                {
                    out.set_reward(agent, 1.0);
                }
            }
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        cfg: &KeyDoorConfig,
        s: &KeyDoorState,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, KeyDoorObs>,
    ) {
        let me = agent.index();
        if let Some(mut leaf) = out.leaf_mut(KeyDoorObs::CARRYING) {
            leaf.set(0, u8::from(usize::from(s.key_holder) == me));
        }
        let Some(mut leaf) = out.leaf_mut(KeyDoorObs::VIEW) else {
            return;
        };
        let Some((ac, ar)) = s.cell(me) else {
            return;
        };
        let (dc, dr) = step_of(s.facing_of(me));
        // To the agent's right: its forward step turned a quarter clockwise.
        let (rc, rr) = (-dr, dc);
        let n = cfg.size.cast_signed();
        let side = VIEW as usize;
        for vr in 0..VIEW {
            for vc in 0..VIEW {
                let (ahead, right) = (VIEW - 1 - vr, vc - VIEW / 2);
                let col = ac + ahead * dc + right * rc;
                let row = ar + ahead * dr + right * rr;
                let inside = (0..n).contains(&col) && (0..n).contains(&row);
                let goal = (i32::from(s.goal[0]), i32::from(s.goal[1]));
                let other =
                    (0..cfg.agents as usize).any(|k| k != me && s.cell(k) == Some((col, row)));
                let values = [
                    !inside || s.is_wall(col, row),
                    inside && s.is_door(col, row) && !s.door_open.get(),
                    s.key_at(col, row),
                    (col, row) == goal,
                    other,
                ];
                let at = usize::try_from(vr).unwrap_or(0) * side + usize::try_from(vc).unwrap_or(0);
                for (channel, on) in values.iter().enumerate() {
                    leaf.set(channel * side * side + at, u8::from(*on));
                }
            }
        }
    }
}

/// Each agent's own legal actions; waiting is the no-op.
impl ActionMask for KeyDoor {
    fn action_mask(
        cfg: &KeyDoorConfig,
        state: &KeyDoorState,
        agent: AgentSlot,
        out: &mut MaskWriter<'_>,
    ) {
        for action in LEFT..=WAIT {
            out.set(
                usize::try_from(action).unwrap_or(0),
                state.is_legal(cfg, agent.index(), action),
            );
        }
    }

    fn noop(_: &KeyDoorConfig, _: &KeyDoorState, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, WAIT);
    }
}

impl tessera_rl_core::Persist for KeyDoor {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
