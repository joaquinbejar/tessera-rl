// SPDX-License-Identifier: MIT OR Apache-2.0

//! The survival world (`tessera/Survival-v1`): 16 to 128 agents forage,
//! drink and fight on a procedurally generated tile map, an original design
//! in the genre of many-agent survival worlds (SPEC 16, ADR-0034; design
//! `docs/reviews/382/DESIGN-r1.md` and `STAGE1-r1.md`). Cross-platform
//! tier, integer state, simultaneous moves.
//!
//! - **Map:** `size` x `size` tiles (grass, water, forest, stone), made at
//!   reset from the instance's `StepRng` (`map`). Grass and forest are
//!   passable, and every passable tile is reachable.
//! - **Agents:** `agents` slots, all alive at reset on distinct grass
//!   tiles, with health, food and water at 100.
//! - **Actions:** `MultiDiscrete([5, 7])`: move (stay, N, E, S, W) and act
//!   (none, forage, drink, attack N, E, S, W). A move or act that cannot
//!   happen is treated as stay or none.
//! - **Step:** every phase reads the state at its start and writes every
//!   agent at once, so slot order does not matter, apart from rule T5:
//!   1. **Moves:** a move into the map edge, water, stone, or a tile with a
//!      live agent at the start of the step fails (start-of-step occupancy,
//!      T3). Two or more agents moving into one free tile all stay (T1).
//!   2. **Attacks**, on the positions after the moves: an attack hits the
//!      live agent on that neighbour tile. Damage (`ATTACK`) is summed per
//!      target and applied at once, so mutual attacks both land. A kill
//!      adds 1 to the `kills` of every attacker that hit the victim.
//!   3. **Forage, drink, upkeep:** forage on a ready forest tile restores
//!      `FORAGE` food and starts its regrow timer; drink next to water
//!      restores `DRINK` water. Then food and water fall by 1, health falls
//!      by `STARVE` when either is at 0, and attack damage is applied.
//!      Health stays in `0..=MAX_HEALTH`, a game rule. Regrow timers count
//!      down.
//!   4. **Deaths and respawn:** an agent at 0 health dies. With
//!      `respawn > 0`, a dead agent respawns after `respawn` steps on a free
//!      grass tile, as a new agent (a new slot generation). Respawning
//!      agents draw their candidate tiles from `StepRng` and are placed in
//!      slot order, so a lower slot draws and places first and a later one
//!      never takes its tile (T5, the one slot-ordered rule).
//! - **Rewards:** +1 for every agent alive at the end of the step. Kills
//!   give no reward (review Q2); they are counted in the state.
//! - **End:** truncated at `steps` steps; terminated when no agent is alive
//!   and `respawn` is 0.
//! - **Observation:** `window`, `Box(u8, [4, 2r + 1, 2r + 1])` around the
//!   agent (terrain, ready forest, agent present, its health); `own`,
//!   `Box(i32, [6])` (health, food, water, x, y, step); `others`, a `Set`
//!   of up to 16 visible agents, nearest first, each `[dx, dy, health]`.
//! - **Masks:** blocked moves and impossible acts (`ActionMask`).
//!
//! Version 2 adds the window, the set and the masks (#382 stage 2); version
//! 1 had the `own` leaf only and was never released.

pub mod baseline;
#[doc(hidden)]
pub mod lanes;
pub mod map;

use core::num::NonZeroU32;

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Bool8, ConfigError, Environment, EpisodeEnd, HeaderView,
    ImperfectInformation, InvariantViolation, NoAdaptive, ObsLeaf, ObsSet, Observation, Phase,
    RecordHeader, SpaceError, StepOut, Tier, TurnStructure, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/Survival-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 2;

// Registration under the versioned id (SPEC 10, ADR-0054).
tessera_rl_core::register!(Survival, id = "tessera/Survival-v1");

/// Agent slots.
pub const MAX_AGENTS: usize = 128;
/// The largest map side.
pub const MAX_SIZE: usize = 96;
/// Grass.
pub const GRASS: u8 = 0;
/// Water: not passable; drink next to it.
pub const WATER: u8 = 1;
/// Forest: passable; forage on it when ready.
pub const FOREST: u8 = 2;
/// Stone: not passable.
pub const STONE: u8 = 3;
/// Health, food and water at spawn, and their caps.
pub const MAX_HEALTH: i16 = 100;
/// The food and water cap.
pub const MAX_STOCK: i16 = 100;
/// Food restored by foraging.
pub const FORAGE: i16 = 30;
/// Water restored by drinking.
pub const DRINK: i16 = 30;
/// Health lost per step at 0 food or 0 water.
pub const STARVE: i16 = 5;
/// Damage of one attack.
pub const ATTACK: i16 = 20;
/// Choices of the move head.
pub const MOVES: u32 = 5;
/// Choices of the act head.
pub const ACTS: u32 = 7;
/// Tries at a respawn tile before the fallback scan.
const RESPAWN_TRIES: usize = 64;

/// Survival world configuration.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
#[config(validate = check_config)]
pub struct SurvivalConfig {
    /// Agents, 16 to 128.
    #[config(default = 32, range = 16..=128)]
    pub agents: u32,
    /// Map side, 32 to 96.
    #[config(default = 64, range = 32..=96)]
    pub size: u32,
    /// Steps before the episode is truncated.
    #[config(default = 1024, range = 1..=4096)]
    pub steps: u32,
    /// The observation window radius (stage 2).
    #[config(default = 5, range = 2..=7)]
    pub view: u32,
    /// Steps dead before respawning; 0 is no respawn.
    #[config(default = 16, range = 0..=255)]
    pub respawn: u32,
    /// Steps a foraged forest takes to regrow.
    #[config(default = 32, range = 1..=255)]
    pub forest_regrow: u32,
    /// Target water share, per mille.
    #[config(default = 80, range = 0..=500)]
    pub water_permille: u32,
    /// Target forest share, per mille.
    #[config(default = 200, range = 0..=500)]
    pub forest_permille: u32,
    /// Target stone share, per mille.
    #[config(default = 60, range = 0..=500)]
    pub stone_permille: u32,
}

fn check_config(cfg: &SurvivalConfig) -> Result<(), ConfigError> {
    let sum = cfg.water_permille + cfg.forest_permille + cfg.stone_permille;
    if sum > 900 {
        return Err(ConfigError {
            field: "water_permille",
            reason: "water, forest and stone shares must sum to at most 900 per mille",
        });
    }
    Ok(())
}

impl SurvivalConfig {
    /// Agents, as a count.
    #[must_use]
    pub fn agents(&self) -> usize {
        usize::try_from(self.agents)
            .unwrap_or(16)
            .clamp(1, MAX_AGENTS)
    }

    /// The map side, as a count.
    #[must_use]
    pub fn size(&self) -> usize {
        usize::try_from(self.size).unwrap_or(32).clamp(1, MAX_SIZE)
    }
}

/// Survival state: the map and the agents, struct of arrays. The byte
/// fields come after the `i16` and `u16` ones and their total is even, so
/// the layout has no padding.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct SurvivalState {
    /// Agent columns.
    pub x: [i16; MAX_AGENTS],
    /// Agent rows.
    pub y: [i16; MAX_AGENTS],
    /// Health, `0..=MAX_HEALTH`.
    pub health: [i16; MAX_AGENTS],
    /// Food, `0..=MAX_STOCK`.
    pub food: [i16; MAX_AGENTS],
    /// Water, `0..=MAX_STOCK`.
    pub water: [i16; MAX_AGENTS],
    /// Steps since death (0 while alive).
    pub dead_for: [u16; MAX_AGENTS],
    /// Steps alive in the current life.
    pub age: [u16; MAX_AGENTS],
    /// Kills in the current life (no reward, review Q2).
    pub kills: [u16; MAX_AGENTS],
    /// The episode step.
    pub step: u16,
    /// Tiles, `[y][x]`; cells beyond `size` are zero.
    pub tiles: [[u8; MAX_SIZE]; MAX_SIZE],
    /// Forest regrow timers, `[y][x]`: 0 is ready.
    pub regrow: [[u8; MAX_SIZE]; MAX_SIZE],
    /// Whether each agent is alive.
    pub alive: [Bool8; MAX_AGENTS],
    /// The map side of this instance.
    pub size: u8,
    /// The agents of this instance (16 to 128).
    pub agents: u8,
}

/// The state's size: 8 x 256 bytes of agent fields, the step, 2 x 9,216
/// bytes of map, 128 alive flags and 2 counts, the same at every config. A
/// change here changes every snapshot (DESIGN-r1 section 2 budgets about
/// 21 KB).
pub const STATE_BYTES: usize = 8 * 2 * MAX_AGENTS + 2 + 2 * MAX_SIZE * MAX_SIZE + MAX_AGENTS + 2;
const _: () = assert!(core::mem::size_of::<SurvivalState>() == STATE_BYTES);
const _: () = assert!(STATE_BYTES == 20_612);

impl Default for SurvivalState {
    fn default() -> Self {
        Self {
            x: [0; MAX_AGENTS],
            y: [0; MAX_AGENTS],
            health: [0; MAX_AGENTS],
            food: [0; MAX_AGENTS],
            water: [0; MAX_AGENTS],
            dead_for: [0; MAX_AGENTS],
            age: [0; MAX_AGENTS],
            kills: [0; MAX_AGENTS],
            step: 0,
            tiles: [[0; MAX_SIZE]; MAX_SIZE],
            regrow: [[0; MAX_SIZE]; MAX_SIZE],
            alive: [Bool8::FALSE; MAX_AGENTS],
            size: 0,
            agents: 0,
        }
    }
}

/// Survival observation (stage 1).
#[derive(Observation)]
pub struct SurvivalObs {
    /// The tile window around the agent, `[4, 2r + 1, 2r + 1]` with `r =
    /// view`, rows north to south: terrain (`OUTSIDE` off the map), ready
    /// forest (1), a live agent present (1), that agent's health.
    #[space(box(low = 0, high = 255, dtype = u8, shape = |c: &SurvivalConfig| [4, 2 * c.view + 1, 2 * c.view + 1]))]
    pub window: ObsLeaf<u8>,
    /// The agent's health, food, water, x, y and the episode step.
    #[space(box(low = 0, high = 4096, shape = [6]))]
    pub own: ObsLeaf<i32>,
    /// Up to `MAX_VISIBLE` other live agents in the window, nearest first
    /// (squared distance, then `dx`, then `dy`: no slot order, T6), each
    /// `[dx, dy, health]` (ADR-0082).
    #[space(set(max = MAX_VISIBLE, inner(box(low = -7, high = 100, shape = [3]))))]
    pub others: ObsSet<i32>,
}

/// Terrain value of a window cell off the map.
pub const OUTSIDE: u8 = 255;
/// The most other agents in the `others` set.
pub const MAX_VISIBLE: usize = 16;
/// The largest view radius.
pub const MAX_VIEW: usize = 7;

/// The survival world environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Survival;

/// The cells of a `size` x `size` map, `size` clipped to `MAX_SIZE`.
#[must_use]
pub const fn cells(size: usize) -> u32 {
    let clipped = if size < MAX_SIZE { size } else { MAX_SIZE };
    #[expect(clippy::cast_possible_truncation, reason = "at most MAX_SIZE = 96")]
    let per_side = clipped as u32;
    per_side * per_side
}
// Per-mille shares of the largest map stay within u32: 9,216 x 1,000.
const _: () = assert!((MAX_SIZE * MAX_SIZE * 1000) < u32::MAX as usize);

fn slot(index: usize) -> Option<AgentSlot> {
    AgentSlot::new(u16::try_from(index).ok()?, MAX_AGENTS)
}

/// The cell one step from `(x, y)` in `dir` (1 N, 2 E, 3 S, 4 W), or
/// `(x, y)` for 0.
#[must_use]
pub const fn toward(x: i16, y: i16, dir: u32) -> (i16, i16) {
    match dir {
        1 => (x, y - 1),
        2 => (x + 1, y),
        3 => (x, y + 1),
        4 => (x - 1, y),
        _ => (x, y),
    }
}

impl SurvivalState {
    /// Agents, as a count.
    #[must_use]
    pub fn agent_count(&self) -> usize {
        usize::from(self.agents).min(MAX_AGENTS)
    }

    /// The map side.
    #[must_use]
    pub fn side(&self) -> usize {
        usize::from(self.size).min(MAX_SIZE)
    }

    /// The tile at `(x, y)`, or `None` off the map.
    #[must_use]
    pub fn tile(&self, x: i16, y: i16) -> Option<u8> {
        let (x, y) = (usize::try_from(x).ok()?, usize::try_from(y).ok()?);
        if x >= self.side() || y >= self.side() {
            return None;
        }
        self.tiles.get(y)?.get(x).copied()
    }

    /// Whether agent `a` is alive.
    #[must_use]
    pub fn is_alive(&self, a: usize) -> bool {
        a < self.agent_count() && self.alive.get(a).is_some_and(|f| f.get())
    }

    /// Agent `a`'s position.
    #[must_use]
    pub fn pos(&self, a: usize) -> (i16, i16) {
        (
            self.x.get(a).copied().unwrap_or(-1),
            self.y.get(a).copied().unwrap_or(-1),
        )
    }

    /// The live agent at `(x, y)`, if any.
    #[must_use]
    pub fn agent_at(&self, x: i16, y: i16) -> Option<usize> {
        (0..self.agent_count()).find(|&a| self.is_alive(a) && self.pos(a) == (x, y))
    }

    /// Whether `(x, y)` is passable grass or forest with no live agent.
    #[must_use]
    pub fn free(&self, x: i16, y: i16) -> bool {
        self.tile(x, y).is_some_and(map::passable) && self.agent_at(x, y).is_none()
    }

    /// Whether a forest tile at `(x, y)` is ready to forage.
    #[must_use]
    pub fn ready_forest(&self, x: i16, y: i16) -> bool {
        self.tile(x, y) == Some(FOREST)
            && usize::try_from(x)
                .ok()
                .zip(usize::try_from(y).ok())
                .is_some_and(|(cx, cy)| self.regrow.get(cy).and_then(|r| r.get(cx)) == Some(&0))
    }

    /// Whether `(x, y)` has a water tile among its 4 neighbours.
    #[must_use]
    pub fn by_water(&self, x: i16, y: i16) -> bool {
        (1..=4).any(|d| {
            let (nx, ny) = toward(x, y, d);
            self.tile(nx, ny) == Some(WATER)
        })
    }

    /// Places agent `a` alive at `(x, y)` with full stocks.
    fn spawn(&mut self, a: usize, x: i16, y: i16) {
        let set = |v: &mut [i16; MAX_AGENTS], value: i16| {
            if let Some(e) = v.get_mut(a) {
                *e = value;
            }
        };
        set(&mut self.x, x);
        set(&mut self.y, y);
        set(&mut self.health, MAX_HEALTH);
        set(&mut self.food, MAX_STOCK);
        set(&mut self.water, MAX_STOCK);
        for v in [&mut self.dead_for, &mut self.age, &mut self.kills] {
            if let Some(e) = v.get_mut(a) {
                *e = 0;
            }
        }
        if let Some(f) = self.alive.get_mut(a) {
            *f = Bool8::TRUE;
        }
    }

    /// A free grass tile for a spawn from the draw sequence `draw(k)`,
    /// preferring tiles at least `view` (Chebyshev) from every live agent;
    /// then the first free grass tile scanning from the last draw. `None`
    /// when no grass tile is free.
    fn spawn_tile(&self, view: i16, mut draw: impl FnMut(u64) -> u64) -> Option<(i16, i16)> {
        let side = self.side();
        let cells = u64::try_from(side * side).ok()?.max(1);
        let at = |v: u64| {
            let i = usize::try_from(v % cells).unwrap_or(0);
            (
                i16::try_from(i % side).unwrap_or(0),
                i16::try_from(i / side).unwrap_or(0),
            )
        };
        let far = |x: i16, y: i16| {
            (0..self.agent_count()).all(|b| {
                !self.is_alive(b) || {
                    let (bx, by) = self.pos(b);
                    (bx - x).abs().max((by - y).abs()) >= view
                }
            })
        };
        let mut last = 0;
        for k in 0..RESPAWN_TRIES {
            last = draw(u64::try_from(k).unwrap_or(0));
            let (x, y) = at(last);
            if self.tile(x, y) == Some(GRASS) && self.agent_at(x, y).is_none() && far(x, y) {
                return Some((x, y));
            }
        }
        let from = last % cells;
        (0..cells)
            .map(|i| at((from + i) % cells))
            .find(|&(x, y)| self.tile(x, y) == Some(GRASS) && self.agent_at(x, y).is_none())
    }
}

/// Writes agent `me`'s window, `[4, 2r + 1, 2r + 1]` with `r = view`, as
/// `set(index, value)` calls; unwritten cells stay zero. Shared by the
/// observation and the information state, so they are the same bytes.
fn fill_window(
    cfg: &SurvivalConfig,
    state: &SurvivalState,
    me: usize,
    mut set: impl FnMut(usize, u8),
) {
    let (x, y) = state.pos(me);
    let radius = i16::try_from(cfg.view).unwrap_or(2).clamp(2, 7);
    let width = usize::try_from(2 * radius + 1).unwrap_or(5);
    // The live agents in the window, by window cell (slot + 1), in one
    // pass over the agents: O(agents) per observation.
    let mut nearby = [[0u8; 2 * MAX_VIEW + 1]; 2 * MAX_VIEW + 1];
    for b in 0..state.agent_count() {
        if !state.is_alive(b) {
            continue;
        }
        let (bx, by) = state.pos(b);
        let (col, row) = (bx - x + radius, by - y + radius);
        if let (Ok(col), Ok(row)) = (usize::try_from(col), usize::try_from(row))
            && let Some(cell) = nearby.get_mut(row).and_then(|r| r.get_mut(col))
            && col < width
            && row < width
        {
            *cell = u8::try_from(b + 1).unwrap_or(0);
        }
    }
    {
        for (row, dy) in (-radius..=radius).enumerate() {
            for (col, dx) in (-radius..=radius).enumerate() {
                let (cx, cy) = (x + dx, y + dy);
                let at = |channel: usize| channel * width * width + row * width + col;
                let Some(tile) = state.tile(cx, cy) else {
                    set(at(0), OUTSIDE);
                    continue;
                };
                set(at(0), tile);
                set(at(1), u8::from(state.ready_forest(cx, cy)));
                let occupant = nearby
                    .get(row)
                    .and_then(|r| r.get(col))
                    .copied()
                    .unwrap_or(0);
                if let Some(b) = usize::from(occupant).checked_sub(1) {
                    let health = state.health.get(b).copied().unwrap_or(0);
                    set(at(2), 1);
                    set(at(3), u8::try_from(health).unwrap_or(0));
                }
            }
        }
    }
}

impl Environment for Survival {
    type State = SurvivalState;
    type Config = SurvivalConfig;
    type Obs = SurvivalObs;
    type Action = ();
    type Header = RecordHeader<MAX_AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = MAX_AGENTS;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Simultaneous;
    /// Partially observable: each agent sees its window (stage 3).
    const PERFECT_INFORMATION: bool = false;
    /// Parallel only: AEC cycle buffering is not designed for this many
    /// agents (DESIGN-r1 section 8).
    const ADAPTERS: tessera_rl_core::Adapters = tessera_rl_core::Adapters::PZ_PARALLEL;

    fn observation_space(cfg: &SurvivalConfig) -> Result<Space, SpaceError> {
        SurvivalObs::space(cfg)
    }

    fn action_space(_: &SurvivalConfig) -> Result<Space, SpaceError> {
        Ok(Space::MultiDiscrete {
            nvec: vec![MOVES, ACTS],
        })
    }

    fn initial_agents(cfg: &SurvivalConfig) -> usize {
        cfg.agents()
    }

    fn acting(_: &SurvivalConfig, state: &SurvivalState, agent: AgentSlot) -> bool {
        state.is_alive(agent.index())
    }

    fn reset(cfg: &SurvivalConfig, rng: &mut StepRng, state: &mut SurvivalState) {
        let (agents, size) = (cfg.agents(), cfg.size());
        *state = SurvivalState {
            size: u8::try_from(size).unwrap_or(32),
            agents: u8::try_from(agents).unwrap_or(16),
            ..SurvivalState::default()
        };
        let shares = map::Shares {
            water: cfg.water_permille,
            forest: cfg.forest_permille,
            stone: cfg.stone_permille,
        };
        map::generate(rng, size, shares, &mut state.tiles);
        // Starting tiles: per slot, in slot order, each from its own draws;
        // a later slot sees the earlier ones (T5).
        let view = i16::try_from(cfg.view).unwrap_or(2);
        let cells = NonZeroU32::new(cells(size)).unwrap_or(NonZeroU32::MIN);
        for a in 0..agents {
            let mut draws = [0u64; RESPAWN_TRIES];
            for d in &mut draws {
                *d = u64::from(rng.below_u32(cells));
            }
            let tile = state.spawn_tile(view, |k| {
                draws
                    .get(usize::try_from(k).unwrap_or(0))
                    .copied()
                    .unwrap_or(0)
            });
            if let Some((x, y)) = tile {
                state.spawn(a, x, y);
            }
        }
    }

    #[expect(clippy::too_many_lines, reason = "one phase per block, in order")]
    fn step(
        cfg: &SurvivalConfig,
        state: &mut SurvivalState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let n = state.agent_count();
        let start = *state;
        let head = |a: usize, h: usize| -> u32 {
            slot(a)
                .and_then(|s| actions.discrete(s, h))
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(0)
        };
        // 1. Moves: targets from the start-of-step state.
        let mut target = [(0i16, 0i16); MAX_AGENTS];
        for (a, t) in target.iter_mut().enumerate().take(n) {
            let (x, y) = start.pos(a);
            *t = (x, y);
            if !start.is_alive(a) {
                continue;
            }
            let (tx, ty) = toward(x, y, head(a, 0));
            // T3: occupied at the start of the step, or not passable.
            if (tx, ty) != (x, y) && start.free(tx, ty) {
                *t = (tx, ty);
            }
        }
        for a in 0..n {
            let Some(&t) = target.get(a) else { continue };
            if !start.is_alive(a) || t == start.pos(a) {
                continue;
            }
            // T1: another live mover with the same target.
            let contested = (0..n).any(|b| {
                b != a && start.is_alive(b) && target.get(b) == Some(&t) && t != start.pos(b)
            });
            let (x, y) = if contested { start.pos(a) } else { t };
            if let (Some(sx), Some(sy)) = (state.x.get_mut(a), state.y.get_mut(a)) {
                (*sx, *sy) = (x, y);
            }
        }
        // 2. Attacks on the post-move positions, summed per target.
        let moved = *state;
        let mut damage = [0i16; MAX_AGENTS];
        let mut hit_by = [[false; MAX_AGENTS]; MAX_AGENTS];
        for a in 0..n {
            let act = head(a, 1);
            if !moved.is_alive(a) || !(3..=6).contains(&act) {
                continue;
            }
            let (x, y) = moved.pos(a);
            let (tx, ty) = toward(x, y, act - 2);
            if let Some(v) = moved.agent_at(tx, ty) {
                if let Some(d) = damage.get_mut(v) {
                    *d += ATTACK;
                }
                if let Some(h) = hit_by.get_mut(v).and_then(|r| r.get_mut(a)) {
                    *h = true;
                }
            }
        }
        // 3. Forage, drink, upkeep, damage.
        let regrow = u8::try_from(cfg.forest_regrow).unwrap_or(1);
        // Timers count down before this step's foraging sets new ones.
        let side = state.side();
        for row in state.regrow.iter_mut().take(side) {
            for t in row.iter_mut().take(side) {
                if *t > 0 {
                    *t -= 1;
                }
            }
        }
        for a in 0..n {
            if !moved.is_alive(a) {
                continue;
            }
            let (x, y) = moved.pos(a);
            let act = head(a, 1);
            let mut food = moved.food.get(a).copied().unwrap_or(0);
            let mut water = moved.water.get(a).copied().unwrap_or(0);
            if act == 1 && moved.ready_forest(x, y) {
                food = (food + FORAGE).min(MAX_STOCK);
                if let (Ok(cx), Ok(cy)) = (usize::try_from(x), usize::try_from(y))
                    && let Some(t) = state.regrow.get_mut(cy).and_then(|r| r.get_mut(cx))
                {
                    *t = regrow;
                }
            }
            if act == 2 && moved.by_water(x, y) {
                water = (water + DRINK).min(MAX_STOCK);
            }
            food = (food - 1).max(0);
            water = (water - 1).max(0);
            let mut health = moved.health.get(a).copied().unwrap_or(0);
            if food == 0 || water == 0 {
                health -= STARVE;
            }
            health -= damage.get(a).copied().unwrap_or(0);
            // Game rule: health stays in 0..=MAX_HEALTH.
            let health = health.clamp(0, MAX_HEALTH);
            if let (Some(f), Some(w), Some(h)) = (
                state.food.get_mut(a),
                state.water.get_mut(a),
                state.health.get_mut(a),
            ) {
                (*f, *w, *h) = (food, water, health);
            }
        }
        // 4. Deaths (simultaneous), kill counts, then respawn.
        for v in 0..n {
            if !moved.is_alive(v) || state.health.get(v).copied().unwrap_or(0) > 0 {
                continue;
            }
            if let Some(f) = state.alive.get_mut(v) {
                *f = Bool8::FALSE;
            }
            if let Some(s) = slot(v) {
                out.agent_died(s);
            }
            for a in 0..n {
                // At most 4 kills a step for 4,096 steps: no overflow.
                if hit_by.get(v).and_then(|r| r.get(a)) == Some(&true)
                    && let Some(k) = state.kills.get_mut(a)
                {
                    *k += 1;
                }
            }
        }
        let respawn = u16::try_from(cfg.respawn).unwrap_or(0);
        let view = i16::try_from(cfg.view).unwrap_or(2);
        for a in 0..n {
            if state.is_alive(a) {
                // Age, `dead_for` and `step` are at most 4,096.
                if let Some(g) = state.age.get_mut(a) {
                    *g += 1;
                }
                continue;
            }
            if !moved.is_alive(a) {
                // Dead before this step: count towards the respawn.
                if let Some(d) = state.dead_for.get_mut(a) {
                    *d += 1;
                }
            }
            let dead_for = state.dead_for.get(a).copied().unwrap_or(0);
            if respawn > 0 && dead_for >= respawn {
                // StepRng only (ADR-0007): this agent's candidates, drawn in
                // slot order; T5 orders the placement as well.
                let cells = NonZeroU32::new(cells(state.side())).unwrap_or(NonZeroU32::MIN);
                let mut draws = [0u64; RESPAWN_TRIES];
                for d in &mut draws {
                    *d = u64::from(rng.below_u32(cells));
                }
                if let Some((x, y)) = state.spawn_tile(view, |k| {
                    draws
                        .get(usize::try_from(k).unwrap_or(0))
                        .copied()
                        .unwrap_or(0)
                }) {
                    state.spawn(a, x, y);
                    if let Some(s) = slot(a) {
                        out.agent_spawned(s);
                    }
                }
            }
        }
        // Rewards: survival only.
        for a in 0..n {
            if let Some(s) = slot(a) {
                out.set_reward(s, if state.is_alive(a) { 1.0 } else { 0.0 });
            }
        }
        state.step += 1;
        let none_alive = (0..n).all(|a| !state.is_alive(a));
        if none_alive && respawn == 0 {
            out.end_episode(EpisodeEnd::Terminated);
        } else if u32::from(state.step) >= cfg.steps {
            out.end_episode(EpisodeEnd::Truncated);
        }
    }

    fn observe(
        cfg: &SurvivalConfig,
        state: &SurvivalState,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, SurvivalObs>,
    ) {
        let me = agent.index();
        let (x, y) = state.pos(me);
        let get = |v: &[i16; MAX_AGENTS]| i32::from(v.get(me).copied().unwrap_or(0));
        let radius = i16::try_from(cfg.view).unwrap_or(2).clamp(2, 7);
        if let Some(mut leaf) = out.leaf_mut(SurvivalObs::WINDOW) {
            fill_window(cfg, state, me, |at, v| {
                leaf.set(at, v);
            });
        }
        if let Some(mut leaf) = out.leaf_mut(SurvivalObs::OWN) {
            let row = [
                get(&state.health),
                get(&state.food),
                get(&state.water),
                i32::from(x.max(0)),
                i32::from(y.max(0)),
                i32::from(state.step),
            ];
            for (f, v) in row.into_iter().enumerate() {
                leaf.set(f, v);
            }
        }
        // Visible agents, nearest first; the key never involves the slot.
        let mut seen = [(i32::MAX, 0i16, 0i16, 0i16); MAX_AGENTS];
        let mut count = 0;
        if state.is_alive(me) {
            for b in 0..state.agent_count() {
                if b == me || !state.is_alive(b) {
                    continue;
                }
                let (bx, by) = state.pos(b);
                let (dx, dy) = (bx - x, by - y);
                if dx.abs() > radius || dy.abs() > radius {
                    continue;
                }
                if let Some(e) = seen.get_mut(count) {
                    let d2 = i32::from(dx) * i32::from(dx) + i32::from(dy) * i32::from(dy);
                    *e = (d2, dx, dy, state.health.get(b).copied().unwrap_or(0));
                    count += 1;
                }
            }
        }
        let visible = seen.get_mut(..count).unwrap_or_default();
        visible.sort_unstable();
        if let Some(mut set) = out.set_mut(SurvivalObs::OTHERS) {
            for (i, &(_, dx, dy, health)) in visible.iter().take(MAX_VISIBLE).enumerate() {
                set.set(i, true);
                if let Some(mut item) = set.item(i) {
                    item.set(0, i32::from(dx));
                    item.set(1, i32::from(dy));
                    item.set(2, i32::from(health));
                }
            }
        }
    }
}

/// Masks (ADR-0014): the two heads' rows side by side, move (`0..5`) then
/// act (`5..12`). Stay and none are always legal, and they are the no-op.
/// A move is legal into a passable tile with no live agent at the start of
/// the step (T3); forage on a ready forest; drink next to water; attack a
/// live neighbour. A dead agent does not act, and its row is stay and none
/// only.
const _: () = assert!(MOVES == 5 && ACTS == 7, "the mask rows of ActionMask");

impl ActionMask for Survival {
    fn action_mask(
        _: &SurvivalConfig,
        state: &SurvivalState,
        agent: AgentSlot,
        out: &mut MaskWriter<'_>,
    ) {
        // The act head's first row.
        const ACT: usize = 5;
        let a = agent.index();
        out.set(0, true);
        out.set(ACT, true);
        if !state.is_alive(a) {
            return;
        }
        let (x, y) = state.pos(a);
        for (dir, i) in (1..=4u32).zip(1..=4usize) {
            let (tx, ty) = toward(x, y, dir);
            out.set(i, state.free(tx, ty));
            out.set(ACT + 2 + i, state.agent_at(tx, ty).is_some());
        }
        out.set(ACT + 1, state.ready_forest(x, y));
        out.set(ACT + 2, state.by_water(x, y));
    }

    fn noop(_: &SurvivalConfig, _: &SurvivalState, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, 0);
        out.set_discrete(1, 0);
    }
}

/// What one agent knows (`ImperfectInformation`): its `own` values and its
/// window, the observation's bytes, the window zero-padded to the largest
/// view. The `others` set is a function of the window, so it is not
/// repeated.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct SurvivalInfo {
    /// Health, food, water, x, y and the episode step.
    pub own: [i32; 6],
    /// The window, `[4, 2r + 1, 2r + 1]` for `r = view`, then zeros.
    pub window: [u8; WINDOW_BYTES],
}

/// The bytes of the largest window.
pub const WINDOW_BYTES: usize = 4 * (2 * MAX_VIEW + 1) * (2 * MAX_VIEW + 1);
const _: () = assert!(core::mem::size_of::<SurvivalInfo>() == 24 + WINDOW_BYTES);

impl Default for SurvivalInfo {
    fn default() -> Self {
        Self {
            own: [0; 6],
            window: [0; WINDOW_BYTES],
        }
    }
}

impl ImperfectInformation for Survival {
    type InfoState = SurvivalInfo;

    fn info_state(cfg: &SurvivalConfig, state: &SurvivalState, agent: AgentSlot) -> SurvivalInfo {
        let me = agent.index();
        let mut info = SurvivalInfo::default();
        fill_window(cfg, state, me, |at, v| {
            if let Some(cell) = info.window.get_mut(at) {
                *cell = v;
            }
        });
        let (x, y) = state.pos(me);
        let get = |v: &[i16; MAX_AGENTS]| i32::from(v.get(me).copied().unwrap_or(0));
        info.own = [
            get(&state.health),
            get(&state.food),
            get(&state.water),
            i32::from(x.max(0)),
            i32::from(y.max(0)),
            i32::from(state.step),
        ];
        info
    }
}

impl Validate for Survival {
    fn validate(cfg: &SurvivalConfig, s: &SurvivalState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        if *s == SurvivalState::default() {
            return Ok(());
        }
        let (n, side) = (cfg.agents(), cfg.size());
        if s.agent_count() != n || s.side() != side {
            return invalid("the instance differs from the config");
        }
        for (y, (row, timers)) in s.tiles.iter().zip(s.regrow.iter()).enumerate() {
            for (x, (&t, &r)) in row.iter().zip(timers.iter()).enumerate() {
                if (x >= side || y >= side) && (t != 0 || r != 0) {
                    return invalid("map data beyond the configured size");
                }
                if t > STONE || (r != 0 && t != FOREST) {
                    return invalid("an unknown tile, or a regrow timer off forest");
                }
            }
        }
        for a in 0..MAX_AGENTS {
            let alive = s.alive.get(a).is_some_and(|f| f.get());
            if a >= n {
                if alive {
                    return invalid("a live agent beyond the configured agents");
                }
                continue;
            }
            let stat =
                |v: &[i16; MAX_AGENTS], max: i16| v.get(a).is_some_and(|&e| (0..=max).contains(&e));
            if !stat(&s.health, MAX_HEALTH)
                || !stat(&s.food, MAX_STOCK)
                || !stat(&s.water, MAX_STOCK)
            {
                return invalid("a stat out of range");
            }
            if alive != (s.health.get(a).copied().unwrap_or(0) > 0) {
                return invalid("alive does not match health");
            }
            if alive {
                let (x, y) = s.pos(a);
                if !s.tile(x, y).is_some_and(map::passable) {
                    return invalid("a live agent off the passable tiles");
                }
                if (0..a).any(|b| s.is_alive(b) && s.pos(b) == (x, y)) {
                    return invalid("two live agents on one tile");
                }
            }
        }
        Ok(())
    }

    /// The state's alive flags match the header's agents (ADR-0040).
    fn validate_record(
        cfg: &SurvivalConfig,
        header: &HeaderView<'_>,
        state: &SurvivalState,
    ) -> Result<(), InvariantViolation> {
        Self::validate(cfg, state)?;
        if header.phase() == Phase::NeverReset {
            return Ok(());
        }
        for a in 0..cfg.agents() {
            let header_alive = header.agent(a).is_some_and(|h| h.alive);
            if header_alive != state.is_alive(a) {
                return Err(InvariantViolation {
                    code: "an alive flag that differs from the header",
                });
            }
        }
        Ok(())
    }
}

impl tessera_rl_core::Persist for Survival {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
