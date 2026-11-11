// SPDX-License-Identifier: MIT OR Apache-2.0

//! `tessera/RTS-v1`: the Environment (stage 3: fog).
//!
//! - Agents: two, one per side; `seat_swap` chooses which side agent slot 0
//!   plays (`rts-design.md` section 13, item 1). Turns are simultaneous.
//! - Action: per unit slot of the side, three heads (`type`, `target`,
//!   `product`), `target` and `product` conditional on `type`, declared as
//!   ADR-0051 entity groups of three, so an illegal unit order falls back
//!   alone.
//! - Masks: written from the state at the start of a decision (sections 6
//!   and 13).
//! - Transition: `decision_ticks` ticks per step, each in the fixed order
//!   of section 4 and section 13; integer only.
//! - Observation: `[11, size, size]` planes of `u8` in the side's frame and
//!   a `[2]` global leaf of `u16`, under fog: the enemy is read only on
//!   cells the side sees now, terrain only where it has explored
//!   ([`crate::fog`], `rts-design.md` section 14). The search pool refuses
//!   the environment (`PERFECT_INFORMATION` is `false`); no
//!   `ImperfectInformation` is implemented, since nothing requires it yet.
//! - Reward: +1 win, -1 loss, 0 draw, at the end only.

#![deny(clippy::float_arithmetic)]

use core::num::{NonZeroU16, NonZeroU32};

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Bool8, Environment, EpisodeEnd, HeaderView, InvariantViolation,
    NoAdaptive, ObsLeaf, Observation, Phase, RecordHeader, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

use crate::mapgen::generate;
use crate::resolve::{Claims, NONE};
use crate::units::{Kind, Unit, advance_counter};
use crate::world::{Dir, Grid, MAX_CELLS, Team, Terrain};

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/RTS-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 2;

// Registration under the versioned id (SPEC 10, ADR-0054).
tessera_rl_core::register!(Rts, id = "tessera/RTS-v1");
/// Agent slots: one per side.
pub const MAX_AGENTS: usize = 2;
/// State unit slots per side.
pub const SIDE_SLOTS: usize = 128;
/// State unit slots of both sides.
pub const UNIT_SLOTS: usize = 2 * SIDE_SLOTS;
/// Side of the target window, odd; centred on the unit.
pub const WINDOW: u16 = 9;
/// Cells of the target window.
pub const WINDOW_CELLS: u16 = WINDOW * WINDOW;
/// Choices of the `type` head.
pub const N_TYPES: u16 = 6;
/// Choices of the `product` head.
pub const N_PRODUCTS: u16 = 4;
/// Heads per unit slot: `type`, `target`, `product`.
pub const HEADS_PER_UNIT: usize = 3;
/// Mask bytes per unit slot: the `type` row and the conditional blocks.
pub const MASK_PER_UNIT: usize = (N_TYPES + N_TYPES * WINDOW_CELLS + N_TYPES * N_PRODUCTS) as usize;
/// Ore each side starts with (a calibration placeholder).
pub const START_ORE: u32 = 5;
/// Drones each side starts with, placed as a production would be.
pub const START_DRONES: usize = 1;
/// Observation channels.
pub const CHANNELS: u32 = 11;
/// Entries of the global observation leaf: own ore, decisions played.
pub const GLOBALS: u32 = 2;

/// The `type` head.
pub mod order_type {
    /// Nothing.
    pub const NOOP: i32 = 0;
    /// One step toward an adjacent window cell, as a direction.
    pub const MOVE: i32 = 1;
    /// Attack a window cell.
    pub const ATTACK: i32 = 2;
    /// Harvest an adjacent ore node.
    pub const HARVEST: i32 = 3;
    /// Deposit at an adjacent own Core.
    pub const DEPOSIT: i32 = 4;
    /// Build (a Drone) or produce (a structure).
    pub const BUILD: i32 = 5;
}

/// The `product` head.
const PRODUCTS: [Kind; 4] = [Kind::Drone, Kind::Striker, Kind::Lancer, Kind::Foundry];

/// RTS configuration.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct RtsConfig {
    /// Side of the grid.
    #[config(default = 16, range = 8..=32)]
    pub size: u16,
    /// Ticks per decision.
    #[config(default = 4, range = 1..=16)]
    pub decision_ticks: u16,
    /// Unit slots per side.
    /// At least the Core, the starting Drones and one free slot.
    #[config(default = 64, range = 3..=128)]
    pub max_units: u16,
    /// Decisions before a draw by truncation.
    #[config(default = 2000, range = 1..=100_000)]
    pub step_limit: u32,
    /// Whether agent slot 0 plays side One.
    #[config(default = false)]
    pub seat_swap: bool,
}

/// RTS state. The all-zero default is the state before the first reset.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct RtsState {
    /// Unit slots: side Zero's from 0, side One's from [`SIDE_SLOTS`].
    pub units: [Unit; UNIT_SLOTS],
    /// [`Terrain`] bytes, row-major; cells past the grid are open.
    pub terrain: [u8; MAX_CELLS],
    /// Explored cells, row-major: bit 0 for side Zero, bit 1 for side One
    /// (sides of the map, not agent slots); cells past the grid are 0.
    pub explored: [u8; MAX_CELLS],
    /// Ore of side Zero and side One.
    pub ore: [u32; 2],
    /// Decisions played.
    pub decisions: u32,
    /// Ticks played, modulo 2^32 (an explicit, documented wrap).
    pub ticks: u32,
}

impl Default for RtsState {
    fn default() -> Self {
        Self {
            units: [Unit::default(); UNIT_SLOTS],
            terrain: [0; MAX_CELLS],
            explored: [0; MAX_CELLS],
            ore: [0; 2],
            decisions: 0,
            ticks: 0,
        }
    }
}

/// A side's observation under fog (`rts-design.md` section 14, ruling 3).
#[derive(Observation)]
pub struct RtsObs {
    /// `[channel, row, column]` in the side's frame:
    /// 0 terrain where explored, else 0 (channel 2 tells an unexplored
    /// cell from open terrain);
    /// 1 visible now; 2 explored by this side;
    /// 3 own unit kind; 4 own hit points; 5 own carry; 6 own progress;
    /// 7 enemy unit kind and 8 enemy hit points, on visible cells only;
    /// 9 own unit finished; 10 own unit's production (its kind byte, 0 if
    /// none).
    #[space(box(low = 0, high = 255, dtype = u8, shape = |c: &RtsConfig| [CHANNELS, u32::from(c.size), u32::from(c.size)]))]
    pub planes: ObsLeaf<u8>,
    /// Own ore, clamped at 65535 (the observation only, never the state),
    /// and the decisions played as a share of the step limit, in units of
    /// 1/65535.
    #[space(box(low = 0, high = 65535, dtype = u16, shape = |_: &RtsConfig| [GLOBALS]))]
    pub globals: ObsLeaf<u16>,
}

/// The RTS environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rts;

/// A decoded order, in map coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Order {
    None,
    Move(Dir),
    Attack(u16),
    Harvest(u16),
    Deposit(u16),
    Build(u16),
    Produce(Kind),
}

pub(crate) const fn index_of(team: Team) -> usize {
    match team {
        Team::Zero => 0,
        Team::One => 1,
    }
}

pub(crate) const fn side_of_slot(slot: usize) -> Team {
    if slot < SIDE_SLOTS {
        Team::Zero
    } else {
        Team::One
    }
}

/// The side agent slot `agent` plays.
#[must_use]
pub fn side_of_agent(cfg: &RtsConfig, agent: usize) -> Team {
    let side = if agent == 0 { Team::Zero } else { Team::One };
    if cfg.seat_swap { side.other() } else { side }
}

/// The grid of `cfg`. `size` is validated to `MIN_SIZE..=MAX_SIZE` at
/// construction, so the fallback is never taken.
pub(crate) fn grid(cfg: &RtsConfig) -> Grid {
    Grid::new(cfg.size).unwrap_or(MIN_GRID)
}

const MIN_GRID: Grid = match Grid::new(crate::world::MIN_SIZE) {
    Some(g) => g,
    None => unreachable_grid(),
};

#[expect(
    clippy::panic,
    reason = "a const evaluation failure, never a runtime path"
)]
const fn unreachable_grid() -> Grid {
    panic!("MIN_SIZE is a valid side")
}

/// The state slots of `team` in use under `cfg`.
pub(crate) fn slots(cfg: &RtsConfig, team: Team) -> core::ops::Range<usize> {
    let base = index_of(team) * SIDE_SLOTS;
    base..base + usize::from(cfg.max_units)
}

pub(crate) fn unit_kind(u: &Unit) -> Option<Kind> {
    if u.alive.get() {
        Kind::from_byte(u.kind)
    } else {
        None
    }
}

/// The cell of a target window index from `from`, in `team`'s frame.
pub(crate) fn window_cell(g: Grid, team: Team, from: u16, index: u16) -> Option<u16> {
    let half = i32::from(WINDOW / 2);
    let (fx, fy) = (
        i32::from(index % WINDOW) - half,
        i32::from(index / WINDOW) - half,
    );
    // The side's frame (rule 9): `in_frame` is the identity or a half turn.
    let (dx, dy) = if Dir::North.in_frame(team) == Dir::North {
        (fx, fy)
    } else {
        (-fx, -fy)
    };
    let (x, y) = g.coords(from);
    let nx = u16::try_from(i32::from(x) + dx).ok()?;
    let ny = u16::try_from(i32::from(y) + dy).ok()?;
    g.index(nx, ny)
}

/// The target window index, in `team`'s frame, of a step `dir` on the map:
/// the `target` of a move order (bots and tests use it).
#[must_use]
pub fn move_index(team: Team, dir: Dir) -> u16 {
    let half = WINDOW / 2;
    // A map direction in the side's frame: the half turn is its own inverse.
    match dir.in_frame(team) {
        Dir::North => (half - 1) * WINDOW + half,
        Dir::South => (half + 1) * WINDOW + half,
        Dir::West => half * WINDOW + half - 1,
        Dir::East => half * WINDOW + half + 1,
    }
}

/// Which unit stands on each cell, [`NONE`] if none.
pub(crate) fn occupancy(s: &RtsState) -> [u16; MAX_CELLS] {
    let mut occ = [NONE; MAX_CELLS];
    for (i, u) in s.units.iter().enumerate() {
        if u.alive.get()
            && let Some(o) = occ.get_mut(usize::from(u.cell))
        {
            *o = u16::try_from(i).unwrap_or(NONE);
        }
    }
    occ
}

pub(crate) fn terrain_at(s: &RtsState, cell: u16) -> Option<Terrain> {
    s.terrain
        .get(usize::from(cell))
        .and_then(|&b| Terrain::from_byte(b))
}

pub(crate) fn unit_at(s: &RtsState, occ: &[u16; MAX_CELLS], cell: u16) -> Option<(usize, Unit)> {
    let i = usize::from(*occ.get(usize::from(cell))?);
    s.units.get(i).map(|u| (i, *u))
}

/// Free slots of `team` for spending: empty slots minus productions in
/// progress or waiting (section 13, item 8a).
fn free_slots(cfg: &RtsConfig, s: &RtsState, team: Team) -> usize {
    let range = slots(cfg, team);
    let units = s.units.get(range).unwrap_or(&[]);
    let empty = units.iter().filter(|u| !u.alive.get()).count();
    let reserved = units
        .iter()
        .filter(|u| u.alive.get() && u.producing != 0)
        .count();
    empty.saturating_sub(reserved)
}

fn product_index(kind: Kind) -> Option<usize> {
    PRODUCTS.iter().position(|&p| p == kind)
}

/// The legality of one unit's three rows, as one block of
/// [`MASK_PER_UNIT`] bytes: `[type 6][target 6 x 81][product 6 x 4]`.
pub(crate) fn unit_mask(
    cfg: &RtsConfig,
    s: &RtsState,
    occ: &[u16; MAX_CELLS],
    team: Team,
    slot: usize,
    block: &mut [bool; MASK_PER_UNIT],
) {
    block.fill(false);
    let types = usize::from(N_TYPES);
    let cells = usize::from(WINDOW_CELLS);
    let products = usize::from(N_PRODUCTS);
    let target_at = |t: usize, i: usize| types + t * cells + i;
    let product_at = |t: usize, p: usize| types + types * cells + t * products + p;
    let set = |block: &mut [bool; MASK_PER_UNIT], i: usize| {
        if let Some(b) = block.get_mut(i) {
            *b = true;
        }
    };
    // The no-op: always legal, with target 0 and product 0.
    set(block, 0);
    set(block, target_at(0, 0));
    set(block, product_at(0, 0));
    let Some(u) = s.units.get(slot).copied() else {
        return;
    };
    let Some(kind) = unit_kind(&u) else {
        return;
    };
    let g = grid(cfg);
    let stats = kind.stats();
    let own = |i: usize| side_of_slot(i) == team;
    let side_ore = s.ore.get(index_of(team)).copied().unwrap_or(0);
    let free = free_slots(cfg, s, team);
    for i in 0..WINDOW_CELLS {
        let Some(cell) = window_cell(g, team, u.cell, i) else {
            continue;
        };
        let at = unit_at(s, occ, cell);
        let adjacent = g.adjacent(u.cell, cell);
        let open = terrain_at(s, cell) == Some(Terrain::Open);
        let i = usize::from(i);
        // Move: one of the four adjacent cells, open terrain.
        if stats.speed > 0 && adjacent && open {
            set(block, target_at(1, i));
        }
        // Attack: in range, no own unit there (rule 2).
        if stats.damage > 0
            && g.chebyshev(u.cell, cell) <= u16::from(stats.range)
            && !at.is_some_and(|(j, _)| own(j))
        {
            set(block, target_at(2, i));
        }
        if kind == Kind::Drone && adjacent {
            // Harvest: an adjacent ore node, empty-handed.
            if u.carry == 0 && terrain_at(s, cell) == Some(Terrain::Ore) {
                set(block, target_at(3, i));
            }
            // Deposit: an adjacent finished own Core, carrying.
            if u.carry == 1
                && at.is_some_and(|(j, v)| {
                    own(j) && unit_kind(&v) == Some(Kind::Core) && v.finished.get()
                })
            {
                set(block, target_at(4, i));
            }
            // Build: a new Foundry on an empty open cell, or continue an
            // unfinished own one.
            let new = open
                && at.is_none()
                && side_ore >= u32::from(Kind::Foundry.stats().cost)
                && free > 0;
            let cont = at.is_some_and(|(j, v)| {
                own(j) && unit_kind(&v) == Some(Kind::Foundry) && !v.finished.get()
            });
            if new || cont {
                set(block, target_at(5, i));
            }
        }
    }
    // Produce: a finished, idle structure; target cell 0 only.
    if kind.is_structure() && u.finished.get() && u.producing == 0 && free > 0 {
        let makes: &[Kind] = match kind {
            Kind::Core => &[Kind::Drone],
            Kind::Foundry => &[Kind::Striker, Kind::Lancer],
            _ => &[],
        };
        for &p in makes {
            if side_ore >= u32::from(p.stats().cost)
                && let Some(pi) = product_index(p)
            {
                set(block, product_at(5, pi));
                set(block, target_at(5, 0));
            }
        }
    }
    // Every other type takes product 0; a Drone's build takes Foundry.
    for t in 1..types {
        let any_target = (0..cells).any(|i| block.get(target_at(t, i)).copied().unwrap_or(false));
        if !any_target {
            continue;
        }
        if t == 5 && kind == Kind::Drone {
            if let Some(pi) = product_index(Kind::Foundry) {
                set(block, product_at(5, pi));
            }
        } else if t != 5 {
            set(block, product_at(t, 0));
        }
        set(block, t);
    }
}

/// Decodes `slot`'s three heads into an order, in map coordinates.
fn decode(cfg: &RtsConfig, s: &RtsState, team: Team, slot: usize, heads: [i32; 3]) -> Order {
    let Some(unit) = s.units.get(slot).copied() else {
        return Order::None;
    };
    if unit_kind(&unit).is_none() {
        return Order::None;
    }
    let gr = grid(cfg);
    let cell = u16::try_from(heads[1])
        .ok()
        .and_then(|i| window_cell(gr, team, unit.cell, i));
    match heads[0] {
        order_type::MOVE => cell
            .and_then(|c| {
                Dir::order(Team::Zero)
                    .into_iter()
                    .find(|&d| gr.step(unit.cell, d) == Some(c))
            })
            .map_or(Order::None, Order::Move),
        order_type::ATTACK => cell.map_or(Order::None, Order::Attack),
        order_type::HARVEST => cell.map_or(Order::None, Order::Harvest),
        order_type::DEPOSIT => cell.map_or(Order::None, Order::Deposit),
        order_type::BUILD => {
            let product = usize::try_from(heads[2])
                .ok()
                .and_then(|p| PRODUCTS.get(p))
                .copied();
            match (unit_kind(&unit), product) {
                (Some(Kind::Drone), Some(Kind::Foundry)) => cell.map_or(Order::None, Order::Build),
                (Some(k), Some(p)) if k.is_structure() => Order::Produce(p),
                _ => Order::None,
            }
        }
        _ => Order::None,
    }
}

/// Places a unit of `kind` in the lowest empty slot of `team`, at `cell`;
/// returns its slot.
fn spawn(
    cfg: &RtsConfig,
    s: &mut RtsState,
    team: Team,
    kind: Kind,
    cell: u16,
    finished: bool,
) -> Option<usize> {
    let range = slots(cfg, team);
    let base = range.start;
    let units = s.units.get_mut(range)?;
    let k = units.iter().position(|u| !u.alive.get())?;
    let unit = units.get_mut(k)?;
    *unit = Unit {
        cell,
        kind: kind as u8,
        hp: kind.stats().hp,
        alive: Bool8::TRUE,
        finished: Bool8::new(finished),
        generation: unit.generation.wrapping_add(1),
        ..Unit::default()
    };
    Some(base + k)
}

/// Places every appearing unit by synchronous rounds (section 13, item
/// 8c): each picks its first free adjacent cell in its producer's frame
/// order; a cell picked twice is refused to all its pickers for this tick.
/// Returns the cell of each appearing producer, `NONE` to wait.
fn place_rounds(s: &RtsState, g: Grid, appearing: &[bool; UNIT_SLOTS]) -> [u16; UNIT_SLOTS] {
    let occ = occupancy(s);
    let mut refused = [[false; 4]; UNIT_SLOTS];
    let mut pick = [NONE; UNIT_SLOTS];
    loop {
        for (p, out) in pick.iter_mut().enumerate() {
            *out = NONE;
            if !appearing.get(p).copied().unwrap_or(false) {
                continue;
            }
            let Some(u) = s.units.get(p) else { continue };
            let team = side_of_slot(p);
            for (k, n) in g.neighbours(u.cell, team).into_iter().enumerate() {
                let Some(n) = n else { continue };
                let free = terrain_at(s, n) == Some(Terrain::Open)
                    && occ.get(usize::from(n)) == Some(&NONE);
                if free
                    && !refused
                        .get(p)
                        .and_then(|r| r.get(k))
                        .copied()
                        .unwrap_or(true)
                {
                    *out = n;
                    break;
                }
            }
        }
        let mut count = [0u8; MAX_CELLS];
        for &c in &pick {
            if c != NONE
                && let Some(n) = count.get_mut(usize::from(c))
            {
                *n = n.saturating_add(1);
            }
        }
        let mut changed = false;
        for (p, &c) in pick.iter().enumerate() {
            if c == NONE || count.get(usize::from(c)).copied().unwrap_or(0) <= 1 {
                continue;
            }
            let Some(u) = s.units.get(p) else { continue };
            let team = side_of_slot(p);
            let k = g
                .neighbours(u.cell, team)
                .iter()
                .position(|&n| n == Some(c));
            if let Some(r) = k.and_then(|k| refused.get_mut(p).and_then(|r| r.get_mut(k))) {
                *r = true;
                changed = true;
            }
        }
        if !changed {
            return pick;
        }
    }
}

/// Whether each side still holds a finished Core.
fn cores_alive(s: &RtsState) -> [bool; 2] {
    let mut alive = [false; 2];
    for (i, u) in s.units.iter().enumerate() {
        if unit_kind(u) == Some(Kind::Core)
            && u.finished.get()
            && u.hp > 0
            && let Some(a) = alive.get_mut(index_of(side_of_slot(i)))
        {
            *a = true;
        }
    }
    alive
}

fn order_at(orders: &[Order; UNIT_SLOTS], i: usize) -> Order {
    orders.get(i).copied().unwrap_or(Order::None)
}

fn makes(producer: Kind, product: Kind) -> bool {
    matches!(
        (producer, product),
        (Kind::Core, Kind::Drone) | (Kind::Foundry, Kind::Striker | Kind::Lancer)
    )
}

/// Spending in the decision's first tick, after moves (section 13, items
/// 7, 8 and 11): new builds and productions, all or nothing per side.
fn spend(
    cfg: &RtsConfig,
    s: &mut RtsState,
    orders: &[Order; UNIT_SLOTS],
    placed: &mut [bool; UNIT_SLOTS],
) {
    let g = grid(cfg);
    let occ = occupancy(s);
    let mut claimed = [0u8; MAX_CELLS];
    let mut build = [false; UNIT_SLOTS];
    let mut produce = [None; UNIT_SLOTS];
    for (i, u) in s.units.iter().enumerate() {
        let Some(kind) = unit_kind(u) else { continue };
        if u.hp == 0 {
            continue;
        }
        match order_at(orders, i) {
            Order::Build(c)
                if kind == Kind::Drone
                    && g.adjacent(u.cell, c)
                    && terrain_at(s, c) == Some(Terrain::Open)
                    && occ.get(usize::from(c)) == Some(&NONE) =>
            {
                if let (Some(b), Some(n)) = (build.get_mut(i), claimed.get_mut(usize::from(c))) {
                    *b = true;
                    *n = n.saturating_add(1);
                }
            }
            Order::Produce(p) if makes(kind, p) && u.finished.get() && u.producing == 0 => {
                if let Some(slot) = produce.get_mut(i) {
                    *slot = Some(p);
                }
            }
            _ => {}
        }
    }
    // Any two new builds on one cell are refused (item 7).
    for (i, b) in build.iter_mut().enumerate() {
        if let Order::Build(c) = order_at(orders, i)
            && *b
            && claimed.get(usize::from(c)).copied().unwrap_or(0) > 1
        {
            *b = false;
        }
    }
    for team in [Team::Zero, Team::One] {
        let range = slots(cfg, team);
        let mut cost = 0u32;
        let mut count = 0usize;
        for i in range.clone() {
            if build.get(i).copied().unwrap_or(false) {
                cost += u32::from(Kind::Foundry.stats().cost);
                count += 1;
            }
            if let Some(Some(p)) = produce.get(i) {
                cost += u32::from(p.stats().cost);
                count += 1;
            }
        }
        let side = index_of(team);
        let ore = s.ore.get(side).copied().unwrap_or(0);
        if count == 0 || cost > ore || count > free_slots(cfg, s, team) {
            continue;
        }
        if let Some(o) = s.ore.get_mut(side) {
            *o = ore - cost;
        }
        for i in range {
            if build.get(i).copied().unwrap_or(false)
                && let Order::Build(c) = order_at(orders, i)
                && let Some(slot) = spawn(cfg, s, team, Kind::Foundry, c, false)
                && let Some(p) = placed.get_mut(slot)
            {
                *p = true;
            }
            if let Some(Some(p)) = produce.get(i).copied()
                && let Some(u) = s.units.get_mut(i)
            {
                u.producing = p as u8;
                u.progress = 0;
            }
        }
    }
}

/// Step 2, in a decision's first tick: attacks, from the state at its
/// start; all damage applies together.
fn attacks(s: &mut RtsState, orders: &[Order; UNIT_SLOTS]) {
    let occ = occupancy(s);
    let mut damage = [0u8; UNIT_SLOTS];
    for (i, u) in s.units.iter().enumerate() {
        let (Some(kind), Order::Attack(cell)) = (unit_kind(u), order_at(orders, i)) else {
            continue;
        };
        if let Some(&j) = occ.get(usize::from(cell))
            && j != NONE
            && side_of_slot(usize::from(j)) != side_of_slot(i)
            && let Some(d) = damage.get_mut(usize::from(j))
        {
            *d = d.saturating_add(kind.stats().damage);
        }
    }
    for (u, d) in s.units.iter_mut().zip(damage) {
        u.hp = u.hp.saturating_sub(d);
    }
}

/// Step 3: moves (rule 1, rule 10, section 13 item 5).
fn moves(g: Grid, s: &mut RtsState, orders: &[Order; UNIT_SLOTS]) {
    let mut claims = Claims {
        holder: occupancy(s),
        ..Claims::default()
    };
    for (i, u) in s.units.iter_mut().enumerate() {
        let (Some(kind), Order::Move(dir)) = (unit_kind(u), order_at(orders, i)) else {
            continue;
        };
        if u.hp == 0 {
            continue;
        }
        let (counter, steps) = advance_counter(u.counter, kind.stats().speed);
        u.counter = counter;
        if !steps {
            continue;
        }
        match g.step(u.cell, dir) {
            Some(next) if s.terrain.get(usize::from(next)) == Some(&(Terrain::Open as u8)) => {
                if let Some(t) = claims.target.get_mut(i) {
                    *t = next;
                }
            }
            // A later step into wall, ore or off the grid is blocked.
            _ => u.counter = 0,
        }
    }
    let moved = claims.resolve();
    for (i, u) in s.units.iter_mut().enumerate() {
        let t = claims.target.get(i).copied().unwrap_or(NONE);
        if t == NONE {
            continue;
        }
        if moved.get(i).copied().unwrap_or(false) {
            u.cell = t;
        } else {
            u.counter = 0;
        }
    }
}

/// Step 4: spending in the first tick, then construction, production and
/// placement (section 13, items 7, 8 and 11).
fn build_and_produce(
    cfg: &RtsConfig,
    g: Grid,
    s: &mut RtsState,
    orders: &[Order; UNIT_SLOTS],
    first: bool,
    placed: &mut [bool; UNIT_SLOTS],
) {
    if first {
        spend(cfg, s, orders, placed);
    }
    let foundry_ticks = u16::from(Kind::Foundry.stats().build_ticks);
    for f in 0..UNIT_SLOTS {
        let Some(fu) = s.units.get(f).copied() else {
            continue;
        };
        if unit_kind(&fu) != Some(Kind::Foundry)
            || fu.finished.get()
            || placed.get(f).copied().unwrap_or(false)
        {
            continue;
        }
        let team = side_of_slot(f);
        let built = s.units.iter().enumerate().any(|(d, du)| {
            side_of_slot(d) == team
                && unit_kind(du) == Some(Kind::Drone)
                && du.hp > 0
                && order_at(orders, d) == Order::Build(fu.cell)
                && g.adjacent(du.cell, fu.cell)
        });
        if built && let Some(u) = s.units.get_mut(f) {
            u.progress += 1;
            if u.progress >= foundry_ticks {
                u.finished = Bool8::TRUE;
                u.progress = 0;
            }
        }
    }
    let mut appearing = [false; UNIT_SLOTS];
    for (u, a) in s.units.iter_mut().zip(appearing.iter_mut()) {
        let Some(p) = Kind::from_byte(u.producing) else {
            continue;
        };
        // A unit at 0 hit points does nothing more in this tick.
        if !u.alive.get() || u.hp == 0 {
            continue;
        }
        let ticks = u16::from(p.stats().build_ticks);
        if u.progress < ticks {
            u.progress += 1;
        }
        *a = u.progress >= ticks;
    }
    let picks = place_rounds(s, g, &appearing);
    for (p, &cell) in picks.iter().enumerate() {
        if cell == NONE {
            continue;
        }
        let Some(kind) = s.units.get(p).and_then(|u| Kind::from_byte(u.producing)) else {
            continue;
        };
        if spawn(cfg, s, side_of_slot(p), kind, cell, true).is_some()
            && let Some(u) = s.units.get_mut(p)
        {
            u.producing = 0;
            u.progress = 0;
        }
    }
}

/// One tick (section 4 and section 13). `first` marks the decision's first
/// tick, the only one with attacks and spending; `placed` marks Foundries
/// placed in it, which their Drone continues from the next tick.
fn tick(
    cfg: &RtsConfig,
    s: &mut RtsState,
    orders: &[Order; UNIT_SLOTS],
    first: bool,
    placed: &mut [bool; UNIT_SLOTS],
) {
    let g = grid(cfg);
    if first {
        attacks(s, orders);
    }
    moves(g, s, orders);
    build_and_produce(cfg, g, s, orders, first, placed);
    // Step 5: deaths and slot release; the generation stays.
    for u in &mut s.units {
        if u.alive.get() && u.hp == 0 {
            *u = Unit {
                generation: u.generation,
                ..Unit::default()
            };
        }
    }
}

/// One decision: `decision_ticks` ticks, then harvest and deposit. Returns
/// the Core holders when a tick ended the game (item 10).
fn decide(cfg: &RtsConfig, s: &mut RtsState, orders: &[Order; UNIT_SLOTS]) -> Option<[bool; 2]> {
    for (i, u) in s.units.iter_mut().enumerate() {
        if !matches!(order_at(orders, i), Order::Move(_)) {
            u.counter = 0;
        }
    }
    let mut placed = [false; UNIT_SLOTS];
    for k in 0..cfg.decision_ticks {
        tick(cfg, s, orders, k == 0, &mut placed);
        placed = [false; UNIT_SLOTS];
        s.ticks = s.ticks.wrapping_add(1);
        let alive = cores_alive(s);
        if !(alive[0] && alive[1]) {
            return Some(alive);
        }
    }
    let g = grid(cfg);
    let occ = occupancy(s);
    for i in 0..UNIT_SLOTS {
        let Some(u) = s.units.get(i).copied() else {
            continue;
        };
        if unit_kind(&u) != Some(Kind::Drone) {
            continue;
        }
        let team = side_of_slot(i);
        match order_at(orders, i) {
            Order::Harvest(c)
                if u.carry == 0
                    && terrain_at(s, c) == Some(Terrain::Ore)
                    && g.adjacent(u.cell, c) =>
            {
                if let Some(v) = s.units.get_mut(i) {
                    v.carry = 1;
                }
            }
            Order::Deposit(c) if u.carry == 1 && g.adjacent(u.cell, c) => {
                let core = unit_at(s, &occ, c).is_some_and(|(j, v)| {
                    side_of_slot(j) == team && unit_kind(&v) == Some(Kind::Core) && v.finished.get()
                });
                if core {
                    if let Some(v) = s.units.get_mut(i) {
                        v.carry = 0;
                    }
                    if let Some(o) = s.ore.get_mut(index_of(team)) {
                        *o = o.saturating_add(1);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// The cell of side `team`'s frame position `(x, y)` on the map.
fn frame_cell(g: Grid, team: Team, x: u16, y: u16) -> Option<u16> {
    let c = g.index(x, y)?;
    Some(if team == Team::One { g.mirror(c) } else { c })
}

impl Environment for Rts {
    type State = RtsState;
    type Config = RtsConfig;
    type Obs = RtsObs;
    type Action = ();
    type Header = RecordHeader<MAX_AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = MAX_AGENTS;
    /// Fog hides the enemy, so the search pool refuses this environment
    /// (ADR-0035).
    const PERFECT_INFORMATION: bool = false;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &RtsConfig) -> Result<Space, SpaceError> {
        RtsObs::space(cfg)
    }

    fn action_space(cfg: &RtsConfig) -> Result<Space, SpaceError> {
        Ok(Space::MultiDiscrete {
            nvec: [
                u32::from(N_TYPES),
                u32::from(WINDOW_CELLS),
                u32::from(N_PRODUCTS),
            ]
            .repeat(usize::from(cfg.max_units)),
        })
    }

    /// The map from the reset stream; then each side's Core, ore and
    /// starting Drones, with no further draw (section 13, item 12).
    fn reset(cfg: &RtsConfig, rng: &mut StepRng, s: &mut RtsState) {
        let g = grid(cfg);
        let map = generate(g, |bound: NonZeroU32| rng.below_u32(bound));
        *s = RtsState {
            terrain: map.terrain,
            ore: [START_ORE; 2],
            ..RtsState::default()
        };
        for (team, &core) in [Team::Zero, Team::One].iter().zip(map.cores.iter()) {
            let placed = spawn(cfg, s, *team, Kind::Core, core, true);
            debug_assert!(
                placed.is_some(),
                "max_units >= 3 leaves a slot for the Core"
            );
            for _ in 0..START_DRONES {
                let occ = occupancy(s);
                let cell = g.neighbours(core, *team).into_iter().flatten().find(|&n| {
                    terrain_at(s, n) == Some(Terrain::Open)
                        && occ.get(usize::from(n)) == Some(&NONE)
                });
                // The clear zone keeps a neighbour of the Core open, and
                // max_units >= 3 keeps a slot: both always succeed.
                let placed = cell.and_then(|c| spawn(cfg, s, *team, Kind::Drone, c, true));
                debug_assert!(placed.is_some(), "a starting Drone has a cell and a slot");
            }
        }
        crate::fog::explore(cfg, s);
    }

    fn step(
        cfg: &RtsConfig,
        s: &mut RtsState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let mut orders = [Order::None; UNIT_SLOTS];
        for a in 0..MAX_AGENTS {
            let Some(agent) = u16::try_from(a)
                .ok()
                .and_then(|i| AgentSlot::new(i, MAX_AGENTS))
            else {
                continue;
            };
            let team = side_of_agent(cfg, a);
            for (u, slot) in slots(cfg, team).enumerate() {
                let head = |k: usize| actions.discrete(agent, u * HEADS_PER_UNIT + k).unwrap_or(0);
                let o = decode(cfg, s, team, slot, [head(0), head(1), head(2)]);
                if let Some(o_slot) = orders.get_mut(slot) {
                    *o_slot = o;
                }
            }
        }
        let end = decide(cfg, s, &orders);
        s.decisions = s.decisions.saturating_add(1);
        // Explored grows at the end of each decision only (ruling 2).
        crate::fog::explore(cfg, s);
        if let Some(alive) = end {
            for a in 0..MAX_AGENTS {
                let Some(agent) = u16::try_from(a)
                    .ok()
                    .and_then(|i| AgentSlot::new(i, MAX_AGENTS))
                else {
                    continue;
                };
                let side = index_of(side_of_agent(cfg, a));
                let mine = alive.get(side).copied().unwrap_or(false);
                let theirs = alive.get(1 - side).copied().unwrap_or(false);
                let reward = match (mine, theirs) {
                    (true, false) => 1.0,
                    (false, true) => -1.0,
                    _ => 0.0,
                };
                out.set_reward(agent, reward);
            }
            out.end_episode(EpisodeEnd::Terminated);
        } else if s.decisions >= cfg.step_limit {
            out.end_episode(EpisodeEnd::Truncated);
        }
    }

    fn observe(
        cfg: &RtsConfig,
        s: &RtsState,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, RtsObs>,
    ) {
        let team = side_of_agent(cfg, agent.index());
        if let Some(mut leaf) = out.leaf_mut(RtsObs::PLANES) {
            write_planes(cfg, s, team, |i, v| {
                let _ = leaf.set(i, v);
            });
        }
        if let Some(mut leaf) = out.leaf_mut(RtsObs::GLOBALS) {
            let [ore, played] = globals(cfg, s, team);
            let _ = leaf.set(0, ore);
            let _ = leaf.set(1, played);
        }
    }
}

/// The global leaf of `team`: own ore clamped at `u16::MAX` (a documented
/// observation-only clamp), and decisions played in units of 1/65535 of
/// the step limit.
fn globals(cfg: &RtsConfig, s: &RtsState, team: Team) -> [u16; 2] {
    let ore = s.ore.get(index_of(team)).copied().unwrap_or(0);
    let ore = u16::try_from(ore).unwrap_or(u16::MAX);
    let limit = u64::from(cfg.step_limit.max(1));
    let played = u64::from(s.decisions.min(cfg.step_limit)) * u64::from(u16::MAX) / limit;
    [ore, u16::try_from(played).unwrap_or(u16::MAX)]
}

/// Writes `team`'s planes through `set(index, byte)`. Enemy units are read
/// only on cells `team` sees now, terrain only where it has explored.
fn write_planes(cfg: &RtsConfig, s: &RtsState, team: Team, mut set: impl FnMut(usize, u8)) {
    let gr = grid(cfg);
    let occ = occupancy(s);
    let seen = crate::fog::visible(cfg, s, team);
    let cells = usize::from(gr.cells());
    for y in 0..gr.size() {
        for x in 0..gr.size() {
            let Some(cell) = frame_cell(gr, team, x, y) else {
                continue;
            };
            let pos = usize::from(y) * usize::from(gr.size()) + usize::from(x);
            let visible = seen.get(usize::from(cell)).copied().unwrap_or(false);
            let explored = crate::fog::explored(s, team, cell);
            let terrain = if explored {
                s.terrain.get(usize::from(cell)).copied().unwrap_or(0)
            } else {
                0
            };
            let mut planes = [0u8; 11];
            planes[0] = terrain;
            planes[1] = u8::from(visible);
            planes[2] = u8::from(explored);
            match unit_at(s, &occ, cell) {
                Some((j, u)) if side_of_slot(j) == team => {
                    planes[3] = u.kind;
                    planes[4] = u.hp;
                    planes[5] = u.carry;
                    planes[6] = u8::try_from(u.progress).unwrap_or(u8::MAX);
                    planes[9] = u8::from(u.finished.get());
                    planes[10] = u.producing;
                }
                Some((_, u)) if visible => {
                    planes[7] = u.kind;
                    planes[8] = u.hp;
                }
                _ => {}
            }
            for (k, &v) in planes.iter().enumerate() {
                set(k * cells + pos, v);
            }
        }
    }
}

impl ActionMask for Rts {
    fn action_mask(cfg: &RtsConfig, s: &RtsState, agent: AgentSlot, out: &mut MaskWriter<'_>) {
        let team = side_of_agent(cfg, agent.index());
        let occ = occupancy(s);
        let mut block = [false; MASK_PER_UNIT];
        for (u, slot) in slots(cfg, team).enumerate() {
            unit_mask(cfg, s, &occ, team, slot, &mut block);
            for (k, &legal) in block.iter().enumerate() {
                out.set(u * MASK_PER_UNIT + k, legal);
            }
        }
    }

    /// Every head 0: the no-op of every unit (item 9).
    fn noop(_: &RtsConfig, _: &RtsState, _: AgentSlot, _: &mut ActionWriter<'_>) {}

    fn head_parents(_: &RtsConfig, head: usize) -> [Option<u16>; 2] {
        let root = head - head % HEADS_PER_UNIT;
        if head == root {
            [None, None]
        } else {
            [u16::try_from(root).ok(), None]
        }
    }

    fn entity_group_heads(_: &RtsConfig) -> Option<NonZeroU16> {
        NonZeroU16::new(3)
    }
}

impl tessera_rl_core::Persist for Rts {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}

fn violation(code: &'static str) -> Result<(), InvariantViolation> {
    Err(InvariantViolation { code })
}

impl Validate for Rts {
    fn validate(cfg: &RtsConfig, s: &RtsState) -> Result<(), InvariantViolation> {
        let gr = grid(cfg);
        for (cell, &byte) in s.terrain.iter().enumerate() {
            let inside = cell < usize::from(gr.cells());
            if Terrain::from_byte(byte).is_none() || (!inside && byte != Terrain::Open as u8) {
                return violation("a terrain byte that is not a terrain, or terrain off the grid");
            }
        }
        if s.decisions > cfg.step_limit {
            return violation("more decisions than the step limit");
        }
        if let Err(code) = crate::fog::check_explored(cfg, s) {
            return violation(code);
        }
        for team in [Team::Zero, Team::One] {
            let units = s.units.get(slots(cfg, team)).unwrap_or(&[]);
            let empty = units.iter().filter(|u| !u.alive.get()).count();
            let producing = units
                .iter()
                .filter(|u| u.alive.get() && u.producing != 0)
                .count();
            if producing > empty {
                return violation("more productions than empty slots to place them");
            }
            let core = units
                .iter()
                .any(|u| u.alive.get() && u.kind == Kind::Core as u8 && u.finished.get());
            if !core {
                return violation("a side without a finished Core in a live episode");
            }
        }
        let mut seen = [false; MAX_CELLS];
        for (i, unit) in s.units.iter().enumerate() {
            let in_use = i % SIDE_SLOTS < usize::from(cfg.max_units);
            if !unit.alive.get() {
                let empty = Unit {
                    generation: unit.generation,
                    ..Unit::default()
                };
                if *unit != empty || (!in_use && unit.generation != 0) {
                    return violation("a dead or unused slot that is not empty");
                }
                continue;
            }
            if !in_use {
                return violation("a unit beyond max_units");
            }
            let Some(kind) = Kind::from_byte(unit.kind) else {
                return violation("a unit of no kind");
            };
            let stats = kind.stats();
            if unit.cell >= gr.cells() || terrain_at(s, unit.cell) != Some(Terrain::Open) {
                return violation("a unit off the grid or on wall or ore");
            }
            match seen.get_mut(usize::from(unit.cell)) {
                Some(seen) if !*seen => *seen = true,
                _ => return violation("two units on one cell"),
            }
            if unit.hp == 0 || unit.hp > stats.hp {
                return violation("hit points out of range");
            }
            if unit.carry > u8::from(kind == Kind::Drone) {
                return violation("ore carried by a unit that cannot carry it");
            }
            if (stats.speed == 0 && unit.counter != 0)
                || (stats.speed > 0 && unit.counter >= stats.speed)
            {
                return violation("a movement counter out of range");
            }
            if !unit.finished.get() && kind != Kind::Foundry {
                return violation("an unfinished unit that is not a Foundry");
            }
            match (unit.finished.get(), Kind::from_byte(unit.producing)) {
                (false, _)
                    if unit.producing != 0 || unit.progress >= u16::from(stats.build_ticks) =>
                {
                    return violation("an unfinished Foundry producing or past its build time");
                }
                (true, None) if unit.producing != 0 || unit.progress != 0 => {
                    return violation("progress without a production");
                }
                (true, Some(p))
                    if !makes(kind, p) || unit.progress > u16::from(p.stats().build_ticks) =>
                {
                    return violation("a production its producer cannot make");
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn validate_record(
        cfg: &RtsConfig,
        header: &HeaderView<'_>,
        s: &RtsState,
    ) -> Result<(), InvariantViolation> {
        if header.phase() == Phase::NeverReset {
            if *s != RtsState::default() {
                return violation("a state other than the initial value before the first reset");
            }
            return Ok(());
        }
        Self::validate(cfg, s)?;
        if !(0..MAX_AGENTS).all(|k| header.agent(k).is_some_and(|v| v.alive)) {
            return violation("a side that is not alive");
        }
        Ok(())
    }
}
