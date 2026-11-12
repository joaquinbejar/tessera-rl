// SPDX-License-Identifier: MIT OR Apache-2.0

//! Snake (`tessera/Snake-v1`): one or two agents grow on a square board by
//! eating, and die on collision. Cross-platform tier, integer state. Validates
//! the pixel observations of SPEC 16's row. Original rules text and rendering.
//!
//! - Action: `Discrete(4)`, an absolute direction: north, east, south, west.
//!   The mask forbids reversing; the no-op is the current direction, and the
//!   kernel reads any action the mask refuses as that too.
//! - Step: simultaneous, with no slot priority. Every target comes from the old
//!   state, before anything commits.
//!   1. An agent eats if its target is the food and no other agent targets it.
//!   2. An agent dies if its target is off the board; if two agents target one
//!      cell (both die, food or not); or if its target is occupied in the old
//!      state, not counting the tail of any agent that does not eat this step,
//!      even if that agent dies this step. Swapped heads die by this rule, as a
//!      head of a body of length 3 or more is never a tail.
//!   3. Survivors move, eaters keep their tail, dead bodies leave the board.
//!   4. If food was eaten, new food is uniform among free cells; with no free
//!      cell there is none, and the episode terminates.
//! - Ends: terminated when every agent is dead or the board has no free cell. A
//!   survivor plays on alone. Truncation is the engine's step limit.
//! - Rewards: +1 on eating, -1 on dying, which cannot coincide; otherwise 0.
//! - Observation: one leaf, seen from the observer, chosen by `obs`:
//!   - [`OBS_PIXELS`]: `Box(u8, [1, 84, 84])`, grayscale, integer drawing. A
//!     cell is `84 / size` pixels square, the board centred.
//!   - [`OBS_STATE`]: `Box(f32, [5 * size * size], 0, 1)`, flags only, so no
//!     float arithmetic: planes of own head, own body, other head, other body
//!     and food, each row-major. A dead agent's planes are zero.

mod baseline;
mod validate;

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
pub const ENV_ID: &str = "tessera/Snake-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;

// Registration under the versioned id (SPEC 10, ADR-0054).
tessera_rl_core::register!(Snake, id = "tessera/Snake-v1");
/// Agent slots.
pub const MAX_AGENTS: usize = 2;
/// Choices of the action head.
pub const ACTIONS: u32 = 4;
/// Largest board side.
pub const MAX_SIZE: u32 = 32;
/// Capacity of a body ring: the cells of the largest board.
pub const RING: usize = 1024;
/// `food` when there is none.
pub const NO_FOOD: u16 = u16::MAX;
/// Length of a body at reset.
pub const SPAWN_LEN: u16 = 3;
/// Pixels on a side of the `pixels` observation.
pub const PIXELS: u32 = 84;
/// `obs`: grayscale pixels.
pub const OBS_PIXELS: u32 = 0;
/// `obs`: flat flags.
pub const OBS_STATE: u32 = 1;
/// Planes of the `state` observation.
pub const PLANES: u32 = 5;

/// Shade outside the board.
pub const SHADE_OUTSIDE: u8 = 32;
/// Shade of a free cell.
pub const SHADE_FREE: u8 = 0;
/// Shade of the other agent's body.
pub const SHADE_OTHER_BODY: u8 = 96;
/// Shade of the other agent's head.
pub const SHADE_OTHER_HEAD: u8 = 128;
/// Shade of the food.
pub const SHADE_FOOD: u8 = 160;
/// Shade of the observer's body.
pub const SHADE_OWN_BODY: u8 = 208;
/// Shade of the observer's head.
pub const SHADE_OWN_HEAD: u8 = 255;

/// Snake configuration.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct SnakeConfig {
    /// Cells on a side of the board.
    #[config(default = 16, range = 6..=32)]
    pub size: u32,
    /// Number of agents.
    #[config(default = 1, range = 1..=2)]
    pub agents: u32,
    /// The observation: [`OBS_PIXELS`] or [`OBS_STATE`].
    #[config(default = 0, range = 0..=1)]
    pub obs: u32,
}

impl SnakeConfig {
    /// Cells of the board.
    #[must_use]
    pub const fn cells(&self) -> u32 {
        self.size * self.size
    }
}

/// Snake state: 4,110 bytes, no padding, whatever `size` is. A dead or unused
/// slot is all zero, and so is a ring cell outside its body.
#[derive(EnvState, Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct SnakeState {
    /// Ring of packed cells, `row * size + col`, for each agent.
    pub body: [[u16; RING]; MAX_AGENTS],
    /// Ring index of each head.
    pub head: [u16; MAX_AGENTS],
    /// Length of each body.
    pub len: [u16; MAX_AGENTS],
    /// Packed cell of the food, or [`NO_FOOD`].
    pub food: u16,
    /// Direction of each agent: 0 north, 1 east, 2 south, 3 west.
    pub facing: [u8; MAX_AGENTS],
    /// Whether each agent is alive.
    pub alive: [Bool8; MAX_AGENTS],
}

impl Default for SnakeState {
    fn default() -> Self {
        Self {
            body: [[0; RING]; MAX_AGENTS],
            head: [0; MAX_AGENTS],
            len: [0; MAX_AGENTS],
            food: 0,
            facing: [0; MAX_AGENTS],
            alive: [Bool8::FALSE; MAX_AGENTS],
        }
    }
}

/// Occupied cells, as bits, on the stack.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Cells([u64; RING / 64]);

impl Cells {
    fn split(cell: u16) -> (usize, u32) {
        (usize::from(cell / 64), u32::from(cell % 64))
    }

    pub(crate) fn add(&mut self, cell: u16) {
        let (word, bit) = Self::split(cell);
        if let Some(w) = self.0.get_mut(word) {
            *w |= 1 << bit;
        }
    }

    pub(crate) fn remove(&mut self, cell: u16) {
        let (word, bit) = Self::split(cell);
        if let Some(w) = self.0.get_mut(word) {
            *w &= !(1 << bit);
        }
    }

    pub(crate) fn has(&self, cell: u16) -> bool {
        let (word, bit) = Self::split(cell);
        self.0.get(word).is_some_and(|w| w >> bit & 1 == 1)
    }

    fn count(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }
}

/// The opposite of a direction.
#[must_use]
pub const fn opposite(direction: u8) -> u8 {
    (direction + 2) % 4
}

/// The row and column steps of a direction.
#[must_use]
pub const fn step_of(direction: u8) -> (i32, i32) {
    match direction {
        0 => (-1, 0),
        1 => (0, 1),
        2 => (1, 0),
        _ => (0, -1),
    }
}

/// The cell one step from `cell`, if it is on a board of side `size`.
#[must_use]
pub fn neighbour(size: u32, cell: u16, direction: u8) -> Option<u16> {
    let n = size.cast_signed();
    let at = i32::from(cell);
    let (dr, dc) = step_of(direction);
    let (row, col) = (at / n + dr, at % n + dc);
    if (0..n).contains(&row) && (0..n).contains(&col) {
        u16::try_from(row * n + col).ok()
    } else {
        None
    }
}

impl SnakeState {
    /// Whether the agent is alive.
    #[must_use]
    pub fn is_alive(&self, slot: usize) -> bool {
        self.alive.get(slot).is_some_and(|a| a.get())
    }

    /// The agent's cells from head to tail; none for a dead or unused slot.
    pub fn cells_of(&self, slot: usize) -> impl Iterator<Item = u16> + '_ {
        let len = usize::from(self.len.get(slot).copied().unwrap_or(0));
        let head = usize::from(self.head.get(slot).copied().unwrap_or(0));
        let ring = self.body.get(slot);
        (0..len).map(move |i| {
            ring.and_then(|r| r.get((head + RING - i) % RING))
                .copied()
                .unwrap_or(0)
        })
    }

    /// The agent's head, if alive.
    #[must_use]
    pub fn head_cell(&self, slot: usize) -> Option<u16> {
        self.cells_of(slot).next()
    }

    /// The agent's tail, if alive.
    #[must_use]
    pub fn tail_cell(&self, slot: usize) -> Option<u16> {
        self.cells_of(slot).last()
    }

    pub(crate) fn occupancy(&self) -> Cells {
        let mut cells = Cells::default();
        for k in 0..MAX_AGENTS {
            for cell in self.cells_of(k) {
                cells.add(cell);
            }
        }
        cells
    }

    fn facing_of(&self, slot: usize) -> u8 {
        self.facing.get(slot).copied().unwrap_or(0)
    }

    /// The direction an action means: a refused action is the current one.
    fn direction(&self, slot: usize, action: i32) -> u8 {
        let facing = self.facing_of(slot);
        match u8::try_from(action) {
            Ok(d) if d < 4 && d != opposite(facing) => d,
            _ => facing,
        }
    }

    /// Clears the agent's slot: its body leaves the board.
    fn clear(&mut self, slot: usize) {
        if let Some(ring) = self.body.get_mut(slot) {
            *ring = [0; RING];
        }
        for field in [self.head.get_mut(slot), self.len.get_mut(slot)]
            .into_iter()
            .flatten()
        {
            *field = 0;
        }
        if let Some(f) = self.facing.get_mut(slot) {
            *f = 0;
        }
        if let Some(a) = self.alive.get_mut(slot) {
            *a = Bool8::FALSE;
        }
    }

    /// Moves the agent's head to `cell`; an eater keeps its tail.
    fn advance(&mut self, slot: usize, cell: u16, direction: u8, eats: bool) {
        let head = usize::from(self.head.get(slot).copied().unwrap_or(0));
        let len = usize::from(self.len.get(slot).copied().unwrap_or(0));
        let next = (head + 1) % RING;
        if let Some(ring) = self.body.get_mut(slot) {
            if !eats && let Some(tail) = ring.get_mut((head + RING + 1 - len) % RING) {
                *tail = 0;
            }
            if let Some(h) = ring.get_mut(next) {
                *h = cell;
            }
        }
        if let Some(h) = self.head.get_mut(slot) {
            *h = u16::try_from(next).unwrap_or(0);
        }
        if eats && let Some(l) = self.len.get_mut(slot) {
            // At most `size * size` <= 1024 cells, as food needs a free cell.
            *l += 1;
        }
        if let Some(f) = self.facing.get_mut(slot) {
            *f = direction;
        }
    }

    /// Draws new food among the free cells; `false` if there is none.
    fn place_food(&mut self, cfg: &SnakeConfig, rng: &mut StepRng) -> bool {
        let taken = self.occupancy();
        let Some(free) = NonZeroU32::new(cfg.cells().saturating_sub(taken.count())) else {
            self.food = NO_FOOD;
            return false;
        };
        let mut left = rng.below_u32(free);
        for cell in 0..u16::try_from(cfg.cells()).unwrap_or(0) {
            if taken.has(cell) {
                continue;
            }
            if left == 0 {
                self.food = cell;
                return true;
            }
            left -= 1;
        }
        false
    }
}

/// Grayscale pixels, one leaf.
#[derive(Observation)]
pub struct SnakePixels {
    /// `[1, row, column]`.
    #[space(box(low = 0, high = 255, dtype = u8, shape = [1, PIXELS, PIXELS]))]
    pub obs: ObsLeaf<u8>,
}

/// Flat flags, one leaf.
#[derive(Observation)]
pub struct SnakeFlags {
    /// Five row-major planes.
    #[space(box(low = 0.0, high = 1.0, dtype = f32, shape = |c: &SnakeConfig| [PLANES * c.cells()]))]
    pub obs: ObsLeaf<f32>,
}

/// Snake observation: one leaf whose type `obs` selects, written as raw bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SnakeObs;

impl Observation<SnakeConfig> for SnakeObs {
    const LEAVES: usize = 1;

    fn space(cfg: &SnakeConfig) -> Result<Space, SpaceError> {
        if cfg.obs == OBS_STATE {
            SnakeFlags::space(cfg)
        } else {
            <SnakePixels as Observation<SnakeConfig>>::space(cfg)
        }
    }
}

/// The snake environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Snake;

/// The fixed spawn of a slot: its cells from tail to head and its direction.
/// Two agents are a half turn apart, each heading inward along its own row.
fn spawn(cfg: &SnakeConfig, slot: usize) -> ([u16; 3], u8) {
    let n = cfg.size;
    let packed = |row: u32, col: u32| u16::try_from(row * n + col).unwrap_or(0);
    if cfg.agents == 1 {
        let (row, col) = (n / 2, n / 2 - 2);
        return (
            [packed(row, col), packed(row, col + 1), packed(row, col + 2)],
            1,
        );
    }
    let row = n / 4;
    if slot == 0 {
        ([packed(row, 1), packed(row, 2), packed(row, 3)], 1)
    } else {
        let (row, col) = (n - 1 - row, n - 2);
        (
            [packed(row, col), packed(row, col - 1), packed(row, col - 2)],
            3,
        )
    }
}

fn slot(index: usize) -> Option<AgentSlot> {
    u16::try_from(index)
        .ok()
        .and_then(|i| AgentSlot::new(i, MAX_AGENTS))
}

fn draw_pixels(cfg: &SnakeConfig, s: &SnakeState, me: usize, bytes: &mut [u8]) {
    let side = PIXELS as usize;
    bytes.fill(SHADE_OUTSIDE);
    let n = cfg.size as usize;
    let scale = side / n;
    let margin = (side - scale * n) / 2;
    let mut paint = |cell: u16, shade: u8| {
        let at = usize::from(cell);
        let (top, left) = (margin + at / n * scale, margin + at % n * scale);
        for row in top..top + scale {
            if let Some(line) = bytes.get_mut(row * side + left..row * side + left + scale) {
                line.fill(shade);
            }
        }
    };
    for cell in 0..u16::try_from(cfg.cells()).unwrap_or(0) {
        paint(cell, SHADE_FREE);
    }
    if s.food != NO_FOOD {
        paint(s.food, SHADE_FOOD);
    }
    for (who, body, head) in [
        (1 - me, SHADE_OTHER_BODY, SHADE_OTHER_HEAD),
        (me, SHADE_OWN_BODY, SHADE_OWN_HEAD),
    ] {
        for (i, cell) in s.cells_of(who).enumerate() {
            paint(cell, if i == 0 { head } else { body });
        }
    }
}

/// `bytes` is zero on entry (ADR-0058): only the set flags are written.
fn draw_flags(cfg: &SnakeConfig, s: &SnakeState, me: usize, bytes: &mut [u8]) {
    let cells = cfg.cells() as usize;
    let mut flag = |plane: usize, cell: u16| {
        let at = (plane * cells + usize::from(cell)) * 4;
        if let Some(v) = bytes.get_mut(at..at + 4) {
            v.copy_from_slice(&1.0f32.to_le_bytes());
        }
    };
    for (who, first) in [(me, 0), (1 - me, 2)] {
        for (i, cell) in s.cells_of(who).enumerate() {
            flag(first + usize::from(i != 0), cell);
        }
    }
    if s.food != NO_FOOD {
        flag(4, s.food);
    }
}

impl Environment for Snake {
    type State = SnakeState;
    type Config = SnakeConfig;
    type Obs = SnakeObs;
    type Action = ();
    type Header = RecordHeader<MAX_AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = MAX_AGENTS;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &SnakeConfig) -> Result<Space, SpaceError> {
        SnakeObs::space(cfg)
    }

    fn action_space(_: &SnakeConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: ACTIONS })
    }

    fn initial_agents(cfg: &SnakeConfig) -> usize {
        cfg.agents as usize
    }

    /// Fixed spawns; the only draw is the food.
    fn reset(cfg: &SnakeConfig, rng: &mut StepRng, s: &mut SnakeState) {
        *s = SnakeState::default();
        for k in 0..cfg.agents as usize {
            let (cells, direction) = spawn(cfg, k);
            if let Some(ring) = s.body.get_mut(k)
                && let Some(part) = ring.get_mut(..cells.len())
            {
                part.copy_from_slice(&cells);
            }
            if let (Some(h), Some(l), Some(f), Some(a)) = (
                s.head.get_mut(k),
                s.len.get_mut(k),
                s.facing.get_mut(k),
                s.alive.get_mut(k),
            ) {
                *h = SPAWN_LEN - 1;
                *l = SPAWN_LEN;
                *f = direction;
                *a = Bool8::TRUE;
            }
        }
        s.place_food(cfg, rng);
    }

    fn step(
        cfg: &SnakeConfig,
        s: &mut SnakeState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let mut directions = [0u8; MAX_AGENTS];
        let mut targets = [None; MAX_AGENTS];
        let mut dies = [false; MAX_AGENTS];
        for k in 0..MAX_AGENTS {
            if !s.is_alive(k) {
                continue;
            }
            let action = slot(k).and_then(|a| actions.discrete(a, 0)).unwrap_or(-1);
            let direction = s.direction(k, action);
            let target = s
                .head_cell(k)
                .and_then(|head| neighbour(cfg.size, head, direction));
            if let (Some(d), Some(t), Some(dead)) =
                (directions.get_mut(k), targets.get_mut(k), dies.get_mut(k))
            {
                *d = direction;
                *t = target;
                *dead = target.is_none();
            }
        }
        let [t0, t1] = targets;
        let contested = t0.is_some() && t0 == t1;
        let eats = targets.map(|t| !contested && t == Some(s.food));

        let mut occupied = s.occupancy();
        for k in 0..MAX_AGENTS {
            if !eats.get(k).copied().unwrap_or(false)
                && let Some(tail) = s.tail_cell(k)
            {
                occupied.remove(tail);
            }
        }
        for k in 0..MAX_AGENTS {
            let target = targets.get(k).copied().flatten();
            if let (Some(t), Some(dead)) = (target, dies.get_mut(k)) {
                *dead = contested || occupied.has(t);
            }
        }

        let mut ate = false;
        for k in 0..MAX_AGENTS {
            if !s.is_alive(k) {
                continue;
            }
            let Some(agent) = slot(k) else { continue };
            if dies.get(k).copied().unwrap_or(true) {
                s.clear(k);
                out.set_reward(agent, -1.0);
                out.agent_died(agent);
            } else if let Some(t) = targets.get(k).copied().flatten() {
                let eater = eats.get(k).copied().unwrap_or(false);
                let direction = directions.get(k).copied().unwrap_or(0);
                s.advance(k, t, direction, eater);
                if eater {
                    ate = true;
                    out.set_reward(agent, 1.0);
                }
            }
        }
        let full = ate && !s.place_food(cfg, rng);
        if full || !s.alive.iter().any(|a| a.get()) {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        cfg: &SnakeConfig,
        s: &SnakeState,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, SnakeObs>,
    ) {
        let Some(bytes) = out.leaf(0) else {
            return;
        };
        if cfg.obs == OBS_STATE {
            draw_flags(cfg, s, agent.index(), bytes);
        } else {
            draw_pixels(cfg, s, agent.index(), bytes);
        }
    }
}

/// Every direction but the reverse; the no-op is the current direction.
impl ActionMask for Snake {
    fn action_mask(_: &SnakeConfig, s: &SnakeState, agent: AgentSlot, out: &mut MaskWriter<'_>) {
        let reverse = opposite(s.facing_of(agent.index()));
        for direction in 0..4u8 {
            out.set(usize::from(direction), direction != reverse);
        }
    }

    fn noop(_: &SnakeConfig, s: &SnakeState, agent: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, i32::from(s.facing_of(agent.index())));
    }
}

impl tessera_rl_core::Persist for Snake {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
