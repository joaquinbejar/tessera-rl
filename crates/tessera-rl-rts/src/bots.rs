// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scripted bots of `rts-design.md` section 9: [`Bot::Rush`],
//! [`Bot::Prospector`] and [`Bot::RandomLegal`].
//!
//! Every bot reads the mask the environment writes for its side and picks
//! only legal orders, so a bot never triggers the illegal-action fallback.
//! Movement walks the flow fields of section 5: four goal classes per side
//! (enemy structures, enemy units, ore, own Core), whose obstacles are
//! terrain and structures only, never mobile units, so a unit's own cell
//! and a corridor shared with other movers stay passable. Choices with a
//! direction or a window order go through the side's frame (rule 9), so a
//! bot playing side One acts as the mirror image of one playing side Zero.
//!
//! Only [`Bot::RandomLegal`] draws, from the `StepRng` the caller passes.
//! Win rates among the bots are measured, never assumed here.
//!
//! The bots read the whole state, enemy positions included. That is fair
//! only without fog: section 5 says a field reads only what its side sees
//! or remembers, so once stage 3 adds fog these bots must build their
//! fields through the side's visibility, or they are cheating opponents.

#![deny(clippy::float_arithmetic)]

use core::num::NonZeroU32;

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic};
use tessera_rl_num::rng::StepRng;

use crate::env::{
    HEADS_PER_UNIT, MASK_PER_UNIT, N_PRODUCTS, N_TYPES, Rts, RtsConfig, RtsState, WINDOW,
    WINDOW_CELLS, grid, index_of, move_index, occupancy, order_type, side_of_agent, side_of_slot,
    slots, terrain_at, unit_kind, unit_mask, window_cell,
};
use crate::flow::Field;
use crate::units::Kind;
use crate::world::{Dir, Grid, MAX_CELLS, Team, Terrain};

/// A scripted policy for one side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bot {
    /// Builds a Foundry early and sends Strikers at the nearest known enemy
    /// structure.
    Rush,
    /// Grows its economy first, then Lancers, and attacks once it has an
    /// army.
    Prospector,
    /// A uniformly drawn legal order for every unit.
    RandomLegal,
}

/// Drones a Rush Core keeps before its Foundry.
pub const RUSH_DRONES: usize = 2;
/// Drones a Prospector Core grows to.
pub const PROSPECTOR_DRONES: usize = 4;
/// Fighters a Prospector gathers before it advances.
pub const PROSPECTOR_ARMY: usize = 4;

/// The `product` index of each kind, as the action space numbers them.
const PRODUCT_DRONE: i32 = 0;
const PRODUCT_STRIKER: i32 = 1;
const PRODUCT_LANCER: i32 = 2;
const PRODUCT_FOUNDRY: i32 = 3;

/// One unit's mask block, read by the order helpers.
struct Block([bool; MASK_PER_UNIT]);

impl Block {
    fn get(&self, i: usize) -> bool {
        self.0.get(i).copied().unwrap_or(false)
    }

    fn target(&self, t: i32, i: u16) -> bool {
        let (types, cells) = (usize::from(N_TYPES), usize::from(WINDOW_CELLS));
        usize::try_from(t).is_ok_and(|t| self.get(types + t * cells + usize::from(i)))
    }

    fn product(&self, t: i32, p: i32) -> bool {
        let base = usize::from(N_TYPES) + usize::from(N_TYPES) * usize::from(WINDOW_CELLS);
        match (usize::try_from(t), usize::try_from(p)) {
            (Ok(t), Ok(p)) => self.get(base + t * usize::from(N_PRODUCTS) + p),
            _ => false,
        }
    }

    /// Whether the full order `(t, i, p)` is legal.
    fn legal(&self, t: i32, i: u16, p: i32) -> bool {
        usize::try_from(t).is_ok_and(|ti| self.get(ti)) && self.target(t, i) && self.product(t, p)
    }
}

/// The flow fields of one side, built once per call.
struct Fields {
    enemy_structures: Field,
    enemy_units: Field,
    ore: Field,
    own_core: Field,
}

impl Fields {
    fn build(cfg: &RtsConfig, s: &RtsState, team: Team) -> Self {
        let g = grid(cfg);
        let occ = occupancy(s);
        let structure = |cell: u16| {
            let i = usize::from(occ.get(usize::from(cell)).copied().unwrap_or(u16::MAX));
            s.units
                .get(i)
                .and_then(unit_kind)
                .is_some_and(Kind::is_structure)
        };
        // Obstacles are terrain and structures, never mobile units.
        let passable = |cell: u16| terrain_at(s, cell) == Some(Terrain::Open) && !structure(cell);
        let enemy = |i: usize| side_of_slot(i) != team;
        let mut fields = Self {
            enemy_structures: Field::default(),
            enemy_units: Field::default(),
            ore: Field::default(),
            own_core: Field::default(),
        };
        let es = cells_of(s, |i, k| enemy(i) && k.is_structure());
        fields
            .enemy_structures
            .build(g, es.into_iter().filter(|&c| c != u16::MAX), passable);
        let eu = cells_of(s, |i, _| enemy(i));
        fields
            .enemy_units
            .build(g, eu.into_iter().filter(|&c| c != u16::MAX), passable);
        let ore = (0..g.cells()).filter(|&c| terrain_at(s, c) == Some(Terrain::Ore));
        fields.ore.build(g, ore, passable);
        let core = cells_of(s, |i, k| !enemy(i) && k == Kind::Core);
        fields
            .own_core
            .build(g, core.into_iter().filter(|&c| c != u16::MAX), passable);
        fields
    }
}

/// The cells of the units `want` selects, [`u16::MAX`] elsewhere.
fn cells_of(s: &RtsState, want: impl Fn(usize, Kind) -> bool) -> [u16; crate::env::UNIT_SLOTS] {
    let mut out = [u16::MAX; crate::env::UNIT_SLOTS];
    for (i, u) in s.units.iter().enumerate() {
        if let Some(k) = unit_kind(u)
            && want(i, k)
            && let Some(o) = out.get_mut(i)
        {
            *o = u.cell;
        }
    }
    out
}

/// The window index, in `team`'s frame, of `to` seen from `from`.
fn window_index(g: Grid, team: Team, from: u16, to: u16) -> Option<u16> {
    (0..WINDOW_CELLS).find(|&i| window_cell(g, team, from, i) == Some(to))
}

/// What a bot decided for one unit.
type Order = (i32, u16, i32);

const NOOP: Order = (order_type::NOOP, 0, 0);

/// A legal move down `field`, if any.
fn step_order(team: Team, dir: Dir, block: &Block) -> Option<Order> {
    let i = move_index(team, dir);
    block
        .legal(order_type::MOVE, i, 0)
        .then_some((order_type::MOVE, i, 0))
}

/// The view a unit walks with: its side, the state, the occupancy and its
/// slot.
struct Walker<'a> {
    g: Grid,
    team: Team,
    s: &'a RtsState,
    occ: &'a [u16; MAX_CELLS],
    slot: usize,
}

impl Walker<'_> {
    /// Whether this unit goes first when an own unit stands in its way: a
    /// carrier before an empty-handed unit, then the lower slot. Without a
    /// rule two units meeting head on would wait on each other for ever
    /// (a swap is refused, rule 1).
    fn first(&self, other: usize) -> bool {
        let carry = |i: usize| self.s.units.get(i).map_or(0, |u| u.carry);
        (carry(self.slot), core::cmp::Reverse(self.slot))
            > (carry(other), core::cmp::Reverse(other))
    }

    /// The own mobile unit on `cell`, other than this one.
    fn own_mobile_at(&self, cell: u16) -> Option<usize> {
        let j = usize::from(*self.occ.get(usize::from(cell))?);
        let u = self.s.units.get(j)?;
        let mobile = unit_kind(u).is_some_and(|k| !k.is_structure());
        (mobile && side_of_slot(j) == self.team && j != self.slot).then_some(j)
    }

    /// A legal move down `field`. When an own mobile unit that goes first
    /// stands on the next cell, this unit steps aside to a free cell no
    /// farther from the goal, in its frame order, or waits.
    fn walk(&self, cell: u16, field: &Field, block: &Block) -> Option<Order> {
        let dir = field.descend(self.g, cell, self.team)?;
        let next = self.g.step(cell, dir)?;
        match self.own_mobile_at(next) {
            Some(j) if !self.first(j) => {
                // Aside to a free cell, no farther from the goal if one
                // exists, else any: the unit that goes first must get by.
                let here = field.distance(cell);
                let aside = |close: bool| {
                    Dir::order(self.team).into_iter().find_map(|d| {
                        let n = self.g.step(cell, d)?;
                        let free = self.occ.get(usize::from(n)) == Some(&u16::MAX);
                        (n != next && free && (!close || field.distance(n) <= here))
                            .then(|| step_order(self.team, d, block))
                            .flatten()
                    })
                };
                aside(true).or_else(|| aside(false))
            }
            _ => step_order(self.team, dir, block),
        }
    }
}

/// The first legal attack, in window order, on a cell holding an enemy.
fn strike(
    g: Grid,
    team: Team,
    s: &RtsState,
    occ: &[u16; MAX_CELLS],
    cell: u16,
    block: &Block,
) -> Option<Order> {
    (0..WINDOW_CELLS).find_map(|i| {
        let target = window_cell(g, team, cell, i)?;
        let j = usize::from(*occ.get(usize::from(target))?);
        let enemy = s.units.get(j).and_then(unit_kind).is_some() && side_of_slot(j) != team;
        (enemy && block.legal(order_type::ATTACK, i, 0)).then_some((order_type::ATTACK, i, 0))
    })
}

/// The first legal order of type `t` with product `p` on a 4-neighbour of
/// `cell`, in the side's frame order.
fn adjacent(g: Grid, team: Team, cell: u16, t: i32, p: i32, block: &Block) -> Option<Order> {
    g.neighbours(cell, team)
        .into_iter()
        .flatten()
        .find_map(|n| {
            let i = window_index(g, team, cell, n)?;
            block.legal(t, i, p).then_some((t, i, p))
        })
}

/// Counts of a side's units by role.
#[derive(Default)]
struct Census {
    drones: usize,
    foundries: usize,
    fighters: usize,
    /// The lowest-slot Drone: the only one that starts a Foundry, so two
    /// Drones never split the side's ore and both get refused (item 11).
    builder: Option<usize>,
}

fn census(cfg: &RtsConfig, s: &RtsState, team: Team) -> Census {
    let mut c = Census::default();
    for slot in slots(cfg, team) {
        let Some(u) = s.units.get(slot) else { continue };
        match unit_kind(u) {
            Some(Kind::Drone) => {
                c.drones += 1;
                c.builder = c.builder.or(Some(slot));
            }
            Some(Kind::Foundry) => c.foundries += 1,
            Some(Kind::Striker | Kind::Lancer) => c.fighters += 1,
            _ => {}
        }
    }
    c
}

/// A Drone's economy: deposit, harvest, or walk to the Core or the ore.
fn economy(w: &Walker<'_>, carry: u8, cell: u16, f: &Fields, block: &Block) -> Order {
    if carry > 0 {
        return adjacent(w.g, w.team, cell, order_type::DEPOSIT, 0, block)
            .or_else(|| w.walk(cell, &f.own_core, block))
            .unwrap_or(NOOP);
    }
    adjacent(w.g, w.team, cell, order_type::HARVEST, 0, block)
        .or_else(|| w.walk(cell, &f.ore, block))
        .unwrap_or(NOOP)
}

impl Bot {
    /// Writes the orders of `agent`'s side; `rng` is drawn only by
    /// [`Bot::RandomLegal`].
    pub fn act(
        self,
        cfg: &RtsConfig,
        s: &RtsState,
        agent: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let team = side_of_agent(cfg, agent.index());
        let g = grid(cfg);
        let occ = occupancy(s);
        let fields = (self != Self::RandomLegal).then(|| Fields::build(cfg, s, team));
        let census = census(cfg, s, team);
        let ore = s.ore.get(index_of(team)).copied().unwrap_or(0);
        let mut block = Block([false; MASK_PER_UNIT]);
        for (u, slot) in slots(cfg, team).enumerate() {
            // Every slot is written, an empty one with a no-op, so a reused
            // buffer never keeps a dead unit's order.
            let order = match s.units.get(slot).copied() {
                Some(unit) => match unit_kind(&unit) {
                    Some(kind) => {
                        unit_mask(cfg, s, &occ, team, slot, &mut block.0);
                        match (self, fields.as_ref()) {
                            (Self::RandomLegal, _) => random(&block, rng),
                            (_, Some(f)) => {
                                let w = Walker {
                                    g,
                                    team,
                                    s,
                                    occ: &occ,
                                    slot,
                                };
                                self.scripted(&w, kind, &unit, &census, ore, f, &block)
                            }
                            (_, None) => NOOP,
                        }
                    }
                    None => NOOP,
                },
                None => NOOP,
            };
            let base = u * HEADS_PER_UNIT;
            let _ = out.set_discrete(base, order.0);
            let _ = out.set_discrete(base + 1, i32::from(order.1));
            let _ = out.set_discrete(base + 2, order.2);
        }
    }

    #[expect(
        clippy::too_many_arguments,
        clippy::many_single_char_names,
        reason = "one unit's view of its side"
    )]
    fn scripted(
        self,
        w: &Walker<'_>,
        kind: Kind,
        unit: &crate::units::Unit,
        census: &Census,
        ore: u32,
        f: &Fields,
        block: &Block,
    ) -> Order {
        let produce = |p: i32| {
            block
                .legal(order_type::BUILD, 0, p)
                .then_some((order_type::BUILD, 0, p))
        };
        let (g, team, cell) = (w.g, w.team, unit.cell);
        let (want_drones, fighter) = match self {
            Self::Rush => (RUSH_DRONES, PRODUCT_STRIKER),
            _ => (PROSPECTOR_DRONES, PRODUCT_LANCER),
        };
        match kind {
            Kind::Core if census.drones < want_drones => produce(PRODUCT_DRONE).unwrap_or(NOOP),
            Kind::Core => NOOP,
            Kind::Foundry => produce(fighter).unwrap_or(NOOP),
            Kind::Drone => {
                // Continue an adjacent unfinished own Foundry first.
                let unfinished =
                    g.neighbours(cell, team)
                        .into_iter()
                        .flatten()
                        .find_map(|n| {
                            let j = usize::from(*w.occ.get(usize::from(n))?);
                            let v = w.s.units.get(j)?;
                            let site = unit_kind(v) == Some(Kind::Foundry)
                                && !v.finished.get()
                                && side_of_slot(j) == team;
                            let i = window_index(g, team, cell, n)?;
                            (site && block.legal(order_type::BUILD, i, PRODUCT_FOUNDRY))
                                .then_some((order_type::BUILD, i, PRODUCT_FOUNDRY))
                        });
                // Start one: the side's builder only, when it has the ore.
                let start = census.foundries == 0
                    && census.drones >= want_drones
                    && census.builder == Some(w.slot)
                    && ore >= u32::from(Kind::Foundry.stats().cost);
                unfinished
                    .or_else(|| {
                        start
                            .then(|| {
                                adjacent(g, team, cell, order_type::BUILD, PRODUCT_FOUNDRY, block)
                            })
                            .flatten()
                    })
                    .unwrap_or_else(|| economy(w, unit.carry, cell, f, block))
            }
            Kind::Striker | Kind::Lancer => {
                let advance = self == Self::Rush || census.fighters >= PROSPECTOR_ARMY;
                strike(g, team, w.s, w.occ, cell, block)
                    .or_else(|| {
                        advance
                            .then(|| w.walk(cell, &f.enemy_structures, block))
                            .flatten()
                    })
                    .or_else(|| {
                        advance
                            .then(|| w.walk(cell, &f.enemy_units, block))
                            .flatten()
                    })
                    .unwrap_or(NOOP)
            }
        }
    }
}

/// A uniformly drawn true index of `row`, or 0 if none.
fn pick(rng: &mut StepRng, row: impl Iterator<Item = bool> + Clone) -> usize {
    let count = row.clone().filter(|&b| b).count();
    let Some(n) = u32::try_from(count).ok().and_then(NonZeroU32::new) else {
        return 0;
    };
    let k = usize::try_from(rng.below_u32(n)).unwrap_or(0);
    row.enumerate()
        .filter(|&(_, b)| b)
        .nth(k)
        .map_or(0, |(i, _)| i)
}

fn random(block: &Block, rng: &mut StepRng) -> Order {
    let types = usize::from(N_TYPES);
    let t = pick(rng, (0..types).map(|t| block.get(t)));
    let t = i32::try_from(t).unwrap_or(0);
    let i = pick(rng, (0..WINDOW_CELLS).map(|i| block.target(t, i)));
    let p = pick(rng, (0..i32::from(N_PRODUCTS)).map(|p| block.product(t, p)));
    (
        t,
        u16::try_from(i).unwrap_or(0),
        i32::try_from(p).unwrap_or(0),
    )
}

/// The reference scripted opponent is [`Bot::Rush`]; which bot the learning
/// gate uses is frozen in its protocol (section 9).
impl Heuristic for Rts {
    fn act(
        cfg: &RtsConfig,
        s: &RtsState,
        agent: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        Bot::Rush.act(cfg, s, agent, rng, out);
    }
}

const _: () = assert!(WINDOW % 2 == 1);
