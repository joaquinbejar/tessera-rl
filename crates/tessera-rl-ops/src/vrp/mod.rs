// SPDX-License-Identifier: MIT OR Apache-2.0

//! Capacitated vehicle routing (`tessera/Vrp-v1`): vehicles leave one
//! depot, serve every customer once and come back, an original
//! implementation of the textbook problem (SPEC 16, ADR-0034).
//! Cross-platform tier, integer state.
//!
//! - **Instance:** at reset, the depot (node 0) and `customers` customers
//!   (nodes `1..=customers`) get integer coordinates in `0..grid`, and each
//!   customer a demand in `1..=max_demand`, drawn uniformly from the
//!   instance's `StepRng`. Steps draw no randomness.
//! - **Vehicles:** `vehicles` agents with the same `capacity`, all at the
//!   depot and full at reset. This version takes one vehicle; turns are
//!   `Sequential` (ADR-0013), so several vehicles keep one acting agent.
//! - **Distance:** the Euclidean distance rounded to the nearest integer,
//!   computed when needed from the coordinates with an exact integer square
//!   root on `u64` (`round(sqrt(v))` is `r + 1` when `v - r * r > r`, else
//!   `r`, with `r = isqrt(v)`; no tie is possible). No matrix is stored, so
//!   the state stays `O(customers)` (review of the design).
//! - **Action:** `Discrete(customers + 1)`: go to customer `c`, or back to
//!   the depot (0), which refills the vehicle.
//! - **Mask** (ADR-0014): unvisited customers whose demand fits the
//!   vehicle's load; the depot unless the vehicle is already there with
//!   customers left (no idle loops). Away from the depot the depot is always
//!   legal, and at the depot a full vehicle fits every customer, so some
//!   action is always legal. The no-op is the lowest legal action.
//! - **Reward:** minus the distance of the vehicle's move.
//! - **End:** terminated once every customer is served and every vehicle is
//!   back at the depot. Each step serves a customer or returns to the depot
//!   from a customer, so an episode takes at most `2 * customers` steps:
//!   it always ends, and no truncation is needed.
//! - **Observation:**
//!   - `nodes`, `[customers + 1, 4]` `i32`: x, y, the demand still to
//!     serve (0 once served, and for the depot), and 1 once served;
//!   - `vehicle`, `[3]` `i32`: the observer's node, load and distance
//!     travelled.

pub mod baseline;

use core::num::NonZeroU32;

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, ConfigError, Environment, EpisodeEnd, InvariantViolation, NoAdaptive,
    ObsLeaf, Observation, RecordHeader, Reserved, SpaceError, StepOut, Tier, TurnStructure,
    Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/Vrp-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;

// Registration under the versioned id (SPEC 10, ADR-0054).
tessera_rl_core::register!(Vrp, id = "tessera/Vrp-v1");

/// The most customers.
pub const MAX_CUSTOMERS: usize = 500;
/// Nodes: the depot and the customers.
pub const MAX_NODES: usize = MAX_CUSTOMERS + 1;
/// Agent slots: the most vehicles.
pub const MAX_VEHICLES: usize = 8;
/// The depot's node.
pub const DEPOT: usize = 0;
/// Features per node in the `nodes` leaf.
pub const NODE_FEATURES: usize = 4;
/// Features in the `vehicle` leaf.
pub const VEHICLE_FEATURES: usize = 3;
/// The largest grid side.
pub const MAX_GRID: u32 = 1_000;
/// An upper bound of every observation value: a tour is at most
/// `2 * MAX_CUSTOMERS` moves of at most `MAX_GRID * sqrt(2)` (1,415).
pub const MAX_VALUE: i32 = 2_000_000;
const _: () = assert!(2 * MAX_CUSTOMERS * 1_415 < MAX_VALUE as usize);
const _: () = assert!(MAX_VALUE < 1 << 24, "rewards stay exact in f32");

/// Vehicle routing configuration. `customers` sets the spaces' shapes.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
#[config(validate = check_config)]
pub struct VrpConfig {
    /// Customers, 2 to 500.
    #[config(default = 20, range = 2..=500)]
    pub customers: u32,
    /// Vehicles; this version takes one.
    #[config(default = 1, range = 1..=1)]
    pub vehicles: u32,
    /// Coordinates are in `0..grid`.
    #[config(default = 100, range = 2..=1000)]
    pub grid: u32,
    /// The largest demand, 1 to 255.
    #[config(default = 9, range = 1..=255)]
    pub max_demand: u32,
    /// Each vehicle's capacity, at least `max_demand`.
    #[config(default = 30, range = 1..=10_000)]
    pub capacity: u32,
}

fn check_config(cfg: &VrpConfig) -> Result<(), ConfigError> {
    if cfg.capacity < cfg.max_demand {
        return Err(ConfigError {
            field: "capacity",
            reason: "must be at least max_demand, so every customer can be served",
        });
    }
    Ok(())
}

impl VrpConfig {
    /// Customers, as a count.
    #[must_use]
    pub fn customers(&self) -> usize {
        usize::try_from(self.customers)
            .unwrap_or(2)
            .clamp(2, MAX_CUSTOMERS)
    }

    /// Vehicles, as a count.
    #[must_use]
    pub fn vehicles(&self) -> usize {
        usize::try_from(self.vehicles)
            .unwrap_or(1)
            .clamp(1, MAX_VEHICLES)
    }
}

/// Vehicle routing state. The `u32` fields come first, then `u16`, then
/// bytes, and `reserved` pads to a multiple of 4: no padding.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct VrpState {
    /// Distance each vehicle has travelled.
    pub travelled: [u32; MAX_VEHICLES],
    /// Node x coordinates.
    pub x: [u16; MAX_NODES],
    /// Node y coordinates.
    pub y: [u16; MAX_NODES],
    /// Each vehicle's node.
    pub at: [u16; MAX_VEHICLES],
    /// Each vehicle's remaining load.
    pub load: [u16; MAX_VEHICLES],
    /// Customers of this instance.
    pub customers: u16,
    /// Customers served so far.
    pub served: u16,
    /// Vehicle capacity.
    pub capacity: u16,
    /// Customer demands (0 for the depot).
    pub demand: [u8; MAX_NODES],
    /// 1 once a customer is served.
    pub visited: [u8; MAX_NODES],
    /// Vehicles of this instance.
    pub vehicles: u8,
    /// The vehicle to move.
    pub to_move: u8,
    /// 1 once every customer is served and every vehicle is home.
    pub done: u8,
    /// Padding to a multiple of 4 bytes.
    pub reserved: Reserved<1>,
}

/// The state's size. A change here changes every snapshot.
pub const STATE_BYTES: usize = 3_080;
const _: () = assert!(core::mem::size_of::<VrpState>() == STATE_BYTES);

impl Default for VrpState {
    fn default() -> Self {
        Self {
            travelled: [0; MAX_VEHICLES],
            x: [0; MAX_NODES],
            y: [0; MAX_NODES],
            at: [0; MAX_VEHICLES],
            load: [0; MAX_VEHICLES],
            customers: 0,
            served: 0,
            capacity: 0,
            demand: [0; MAX_NODES],
            visited: [0; MAX_NODES],
            vehicles: 0,
            to_move: 0,
            done: 0,
            reserved: Reserved::ZERO,
        }
    }
}

/// Vehicle routing observation (see the module documentation).
#[derive(Observation)]
pub struct VrpObs {
    /// Per node: x, y, demand still to serve, served.
    #[space(box(low = 0, high = 2_000_000, shape = |c: &VrpConfig| [c.customers + 1, 4]))]
    pub nodes: ObsLeaf<i32>,
    /// The observer: node, load, distance travelled.
    #[space(box(low = 0, high = 2_000_000, shape = [3]))]
    pub vehicle: ObsLeaf<i32>,
}

/// The vehicle routing environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Vrp;

/// `round(sqrt(v))` exactly: `isqrt(v)`, plus one when `v` is past
/// `r * r + r` (then `v >= (r + 0.5)^2 + 0.75`, so no tie is possible).
#[must_use]
pub const fn round_sqrt(v: u64) -> u64 {
    let r = v.isqrt();
    if v - r * r > r { r + 1 } else { r }
}

fn bound(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).unwrap_or(NonZeroU32::MIN)
}

fn slot(index: usize) -> Option<AgentSlot> {
    AgentSlot::new(u16::try_from(index).ok()?, MAX_VEHICLES)
}

impl VrpState {
    /// Customers, as a count.
    #[must_use]
    pub fn customer_count(&self) -> usize {
        usize::from(self.customers).min(MAX_CUSTOMERS)
    }

    /// The rounded distance between nodes `a` and `b`.
    #[must_use]
    pub fn distance(&self, a: usize, b: usize) -> u32 {
        let coord = |v: &[u16; MAX_NODES], n: usize| i64::from(v.get(n).copied().unwrap_or(0));
        let dx = coord(&self.x, a) - coord(&self.x, b);
        let dy = coord(&self.y, a) - coord(&self.y, b);
        let v = u64::try_from(dx * dx + dy * dy).unwrap_or(0);
        u32::try_from(round_sqrt(v)).unwrap_or(u32::MAX)
    }

    /// Vehicle `v`'s node.
    #[must_use]
    pub fn node_of(&self, v: usize) -> usize {
        usize::from(self.at.get(v).copied().unwrap_or(0))
    }

    /// Whether customer `c` is still to serve.
    #[must_use]
    pub fn open(&self, c: usize) -> bool {
        c != DEPOT && c <= self.customer_count() && self.visited.get(c) == Some(&0)
    }

    /// Whether vehicle `v` may go to node `n`.
    #[must_use]
    pub fn legal(&self, v: usize, n: usize) -> bool {
        let here = self.node_of(v);
        if n == DEPOT {
            return here != DEPOT || usize::from(self.served) == self.customer_count();
        }
        let load = self.load.get(v).copied().unwrap_or(0);
        self.open(n) && u16::from(self.demand.get(n).copied().unwrap_or(u8::MAX)) <= load
    }

    /// The lowest legal action of vehicle `v`.
    #[must_use]
    pub fn noop(&self, v: usize) -> usize {
        (0..=self.customer_count())
            .find(|&n| self.legal(v, n))
            .unwrap_or(DEPOT)
    }

    /// Whether every customer is served and every vehicle is home.
    #[must_use]
    pub fn finished(&self) -> bool {
        usize::from(self.served) == self.customer_count()
            && self
                .at
                .iter()
                .take(usize::from(self.vehicles))
                .all(|&a| usize::from(a) == DEPOT)
    }

    /// Moves vehicle `v` to node `n`; returns the distance.
    fn travel(&mut self, v: usize, n: usize) -> u32 {
        let from = self.node_of(v);
        let d = self.distance(from, n);
        let (Some(at), Some(load), Some(travelled)) = (
            self.at.get_mut(v),
            self.load.get_mut(v),
            self.travelled.get_mut(v),
        ) else {
            return 0;
        };
        *at = u16::try_from(n).unwrap_or(0);
        *travelled += d;
        if n == DEPOT {
            *load = self.capacity;
        } else {
            *load -= u16::from(self.demand.get(n).copied().unwrap_or(0));
            if let Some(seen) = self.visited.get_mut(n) {
                *seen = 1;
            }
            self.served += 1;
        }
        d
    }
}

impl Environment for Vrp {
    type State = VrpState;
    type Config = VrpConfig;
    type Obs = VrpObs;
    type Action = ();
    type Header = RecordHeader<MAX_VEHICLES, NoAdaptive>;

    const MAX_AGENTS: usize = MAX_VEHICLES;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &VrpConfig) -> Result<Space, SpaceError> {
        VrpObs::space(cfg)
    }

    fn action_space(cfg: &VrpConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete {
            n: cfg.customers + 1,
        })
    }

    fn initial_agents(cfg: &VrpConfig) -> usize {
        cfg.vehicles()
    }

    fn acting(_: &VrpConfig, state: &VrpState, agent: AgentSlot) -> bool {
        state.done == 0 && agent.index() == usize::from(state.to_move)
    }

    fn reset(cfg: &VrpConfig, rng: &mut StepRng, state: &mut VrpState) {
        let customers = cfg.customers();
        let vehicles = cfg.vehicles();
        let capacity = u16::try_from(cfg.capacity).unwrap_or(u16::MAX);
        *state = VrpState {
            customers: u16::try_from(customers).unwrap_or(2),
            capacity,
            vehicles: u8::try_from(vehicles).unwrap_or(1),
            ..VrpState::default()
        };
        let grid = bound(cfg.grid.min(MAX_GRID));
        let max_demand = bound(cfg.max_demand);
        for n in 0..=customers {
            let (Some(x), Some(y), Some(d)) = (
                state.x.get_mut(n),
                state.y.get_mut(n),
                state.demand.get_mut(n),
            ) else {
                continue;
            };
            *x = u16::try_from(rng.below_u32(grid)).unwrap_or(0);
            *y = u16::try_from(rng.below_u32(grid)).unwrap_or(0);
            if n != DEPOT {
                *d = u8::try_from(rng.below_u32(max_demand) + 1).unwrap_or(1);
            }
        }
        for load in state.load.iter_mut().take(vehicles) {
            *load = capacity;
        }
    }

    fn step(
        _: &VrpConfig,
        state: &mut VrpState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let v = usize::from(state.to_move);
        let Some(mover) = slot(v) else {
            return;
        };
        // The engine replaced an illegal action (ADR-0028); the kernel
        // stays total and reads anything else as the no-op.
        let n = actions
            .discrete(mover, 0)
            .and_then(|a| usize::try_from(a).ok())
            .filter(|&n| state.legal(v, n))
            .unwrap_or_else(|| state.noop(v));
        let d = state.travel(v, n);
        #[expect(clippy::cast_precision_loss, reason = "below MAX_VALUE, exact in f32")]
        let reward = -(d as f32);
        out.set_reward(mover, reward);
        let vehicles = usize::from(state.vehicles).max(1);
        state.to_move = u8::try_from((v + 1) % vehicles).unwrap_or(0);
        if state.finished() {
            state.done = 1;
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &VrpConfig,
        state: &VrpState,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, VrpObs>,
    ) {
        let int = |v: u32| i32::try_from(v).unwrap_or(MAX_VALUE);
        if let Some(mut leaf) = out.leaf_mut(VrpObs::NODES) {
            for n in 0..=state.customer_count() {
                let served = state.visited.get(n).copied().unwrap_or(0);
                let demand = if served == 0 {
                    state.demand.get(n).copied().unwrap_or(0)
                } else {
                    0
                };
                let row = [
                    i32::from(state.x.get(n).copied().unwrap_or(0)),
                    i32::from(state.y.get(n).copied().unwrap_or(0)),
                    i32::from(demand),
                    i32::from(served),
                ];
                for (f, value) in row.into_iter().enumerate() {
                    leaf.set(n * NODE_FEATURES + f, value);
                }
            }
        }
        if let Some(mut leaf) = out.leaf_mut(VrpObs::VEHICLE) {
            let v = agent.index();
            leaf.set(0, int(u32::try_from(state.node_of(v)).unwrap_or(0)));
            leaf.set(1, i32::from(state.load.get(v).copied().unwrap_or(0)));
            leaf.set(2, int(state.travelled.get(v).copied().unwrap_or(0)));
        }
    }
}

impl ActionMask for Vrp {
    fn action_mask(_: &VrpConfig, state: &VrpState, agent: AgentSlot, out: &mut MaskWriter<'_>) {
        let v = agent.index();
        for n in 0..=state.customer_count() {
            out.set(n, state.legal(v, n));
        }
    }

    fn noop(_: &VrpConfig, state: &VrpState, agent: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, i32::try_from(state.noop(agent.index())).unwrap_or(0));
    }
}

impl Validate for Vrp {
    fn validate(cfg: &VrpConfig, s: &VrpState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        // A record that was never reset holds the default state.
        if *s == VrpState::default() {
            return Ok(());
        }
        let (customers, vehicles) = (cfg.customers(), cfg.vehicles());
        if s.customer_count() != customers
            || usize::from(s.vehicles) != vehicles
            || u32::from(s.capacity) != cfg.capacity
        {
            return invalid("the instance differs from the config");
        }
        let grid = cfg.grid.min(MAX_GRID);
        let mut served = 0usize;
        for n in 0..MAX_NODES {
            let (Some(&x), Some(&y), Some(&d), Some(&seen)) =
                (s.x.get(n), s.y.get(n), s.demand.get(n), s.visited.get(n))
            else {
                return invalid("node arrays out of range");
            };
            if n > customers {
                if x != 0 || y != 0 || d != 0 || seen != 0 {
                    return invalid("data beyond the configured customers");
                }
                continue;
            }
            if u32::from(x) >= grid || u32::from(y) >= grid {
                return invalid("a node outside the grid");
            }
            let customer = n != DEPOT;
            if customer != (d != 0) || u32::from(d) > cfg.max_demand {
                return invalid("a demand outside 1..=max_demand, or one at the depot");
            }
            if seen > 1 || (!customer && seen != 0) {
                return invalid("a served flag that is not 0 or 1, or a served depot");
            }
            served += usize::from(seen);
        }
        if usize::from(s.served) != served {
            return invalid("a served count that does not match the flags");
        }
        for v in 0..MAX_VEHICLES {
            let (Some(&at), Some(&load), Some(&travelled)) =
                (s.at.get(v), s.load.get(v), s.travelled.get(v))
            else {
                return invalid("vehicle arrays out of range");
            };
            if v >= vehicles {
                if at != 0 || load != 0 || travelled != 0 {
                    return invalid("data beyond the configured vehicles");
                }
                continue;
            }
            if usize::from(at) > customers || load > s.capacity {
                return invalid("a vehicle off the nodes or over capacity");
            }
            let at = usize::from(at);
            if at != DEPOT && s.visited.get(at) != Some(&1) {
                return invalid("a vehicle at a customer it did not serve");
            }
            if at == DEPOT && load != s.capacity {
                return invalid("a vehicle at the depot that is not full");
            }
        }
        if usize::from(s.to_move) >= vehicles || (s.done == 1) != s.finished() || s.done > 1 {
            return invalid("a turn or done flag that does not match the state");
        }
        Ok(())
    }
}

impl tessera_rl_core::Persist for Vrp {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
