// SPDX-License-Identifier: MIT OR Apache-2.0

//! An example host environment wheel (#272 step 6): a one-trader order
//! book whose resting orders live on the heap, exported to Python with
//! `export_host_env!` as `example/OrderBook-v1`.
//!
//! It is integer-only: prices and sizes are `i64` ticks, with no floats
//! and no random numbers of its own. So equal seeds and actions give equal
//! outputs, which the CI job checks by hashing two runs.
//!
//! One agent slot, so the same id runs through the Gymnasium adapter (one
//! agent only) and the PettingZoo adapter.

use tessera_rl_core::io::{Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, EpisodeEnd, HostEnvironment, HostError, HostOut, HostSeed, HostSnapshot,
    SnapshotReject, SpaceError,
};
use tessera_rl_macros::EnvConfig;

/// The book's id. `example/` is used only by this example: an environment
/// author chooses a namespace of their own.
pub const ORDERBOOK_ID: &str = "example/OrderBook-v1";

/// Sides: hold, buy, sell.
const SIDES: u32 = 3;
/// Price levels: ticks above the last price.
const LEVELS: u32 = 4;
/// The first sell action: an action is `side * LEVELS + level`.
const SELL: usize = 2 * LEVELS as usize;
/// The bound of every observation value.
const BOUND: i64 = 1_000_000;

/// The book's configuration, set from Python as keyword arguments of
/// `tessera_rl.make`.
#[derive(EnvConfig, Debug, Clone, PartialEq)]
pub struct BookConfig {
    /// Steps per episode; the book terminates the episode at this step.
    #[config(default = 32, range = 1..=100_000)]
    pub episode: u32,
    /// The most orders the book holds; a buy beyond it is not placed.
    #[config(default = 64, range = 1..=1_000_000)]
    pub depth: u32,
}

/// One instance: the resting orders live on the heap, in a `Vec` that grows
/// with every buy.
#[derive(Debug)]
pub struct OrderBook {
    cfg: BookConfig,
    /// The bought units, oldest first, at their purchase prices.
    orders: Vec<i64>,
    last: i64,
    step: u32,
}

impl OrderBook {
    fn slot() -> Option<AgentSlot> {
        AgentSlot::new(0, 1)
    }
}

impl HostEnvironment for OrderBook {
    const ENV_ID: &'static str = ORDERBOOK_ID;
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 1;
    const MASKS: bool = true;
    type Config = BookConfig;

    fn new(cfg: &BookConfig) -> Result<Self, HostError> {
        Ok(Self {
            cfg: cfg.clone(),
            orders: Vec::new(),
            last: 0,
            step: 0,
        })
    }

    /// Units held, their total cost, the last price and the step.
    fn observation_space(_: &BookConfig) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![4],
            dtype: DType::I32,
            bounds: BoxBounds::Int {
                low: vec![-BOUND],
                high: vec![BOUND],
            },
        })
    }

    /// One discrete head of `side * LEVELS + level`: a side (hold, buy,
    /// sell) and a price level. One head, so the PettingZoo adapters put
    /// the mask in the observation and PettingZoo's API tests sample with
    /// it.
    fn action_space(_: &BookConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: SIDES * LEVELS })
    }

    fn reset(&mut self, seed: HostSeed, _: &mut HostOut<'_>) {
        self.orders.clear();
        // Integer-only: the opening price comes from the seed and episode.
        self.last = i64::try_from(seed.seed() % 97).unwrap_or(0)
            + i64::from(seed.episode_index() % 100) * 10;
        self.step = 0;
    }

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut HostOut<'_>) {
        let Some(agent) = Self::slot() else {
            return;
        };
        self.step += 1;
        let action = i64::from(actions.discrete(agent, 0).unwrap_or(0));
        let levels = i64::from(LEVELS);
        let (side, level) = (action / levels, action % levels);
        let price = (self.last + level).min(BOUND / 2);
        let depth = usize::try_from(self.cfg.depth).unwrap_or(usize::MAX);
        // A buy rests at its price; a sell fills the oldest unit, and its
        // reward is the price against what that unit cost.
        let reward = match side {
            1 if self.orders.len() < depth => {
                self.orders.push(price);
                0
            }
            // `remove(0)` and the sum in `observe` are linear in the
            // book's size: fine at the default depth of 64; a book meant
            // for large depths would keep a ring buffer and a running sum.
            2 if !self.orders.is_empty() => price - self.orders.remove(0),
            _ => 0,
        };
        #[expect(
            clippy::cast_precision_loss,
            reason = "small integer rewards, exact in f32"
        )]
        out.set_reward(agent, reward as f32);
        self.last = price;
        if self.step >= self.cfg.episode {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(&self, _: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>) {
        let values = [
            i64::try_from(self.orders.len()).unwrap_or(BOUND),
            self.orders.iter().sum::<i64>(),
            self.last,
            i64::from(self.step),
        ];
        if let Some(leaf) = obs.leaf(0) {
            for (bytes, v) in leaf.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                let v = i32::try_from(v.clamp(-BOUND, BOUND)).unwrap_or(0);
                bytes.copy_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn mask(&self, _: AgentSlot, mask: &mut MaskWriter<'_>) {
        mask.fill(true);
        // Selling needs a unit to sell.
        for action in SELL..SELL + LEVELS as usize {
            mask.set(action, !self.orders.is_empty());
        }
    }
}

tessera_rl_core::register_host!(OrderBook, id = "example/OrderBook-v1");

/// Bytes before the orders: last price (`i64`), step and order count
/// (`u32` each).
const FIXED: usize = 16;

/// Snapshots (ADR-0073): the last price, the step and the resting orders,
/// little-endian. The configuration is not state.
impl HostSnapshot for OrderBook {
    const SNAPSHOT_VERSION: u32 = 1;

    fn max_snapshot_len(cfg: &BookConfig) -> usize {
        FIXED + 8 * usize::try_from(cfg.depth).unwrap_or(0)
    }

    fn save(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.last.to_le_bytes());
        out.extend_from_slice(&self.step.to_le_bytes());
        out.extend_from_slice(&u32::try_from(self.orders.len()).unwrap_or(0).to_le_bytes());
        for order in &self.orders {
            out.extend_from_slice(&order.to_le_bytes());
        }
    }

    fn check_snapshot(cfg: &BookConfig, bytes: &[u8]) -> Result<(), SnapshotReject> {
        let reject = |code, reason| Err(SnapshotReject { code, reason });
        let (Some(step), Some(count)) = (word(bytes, 8), word(bytes, 12)) else {
            return reject(1, "shorter than the fixed fields");
        };
        if step > cfg.episode {
            return reject(2, "a step past the episode length");
        }
        if count > cfg.depth {
            return reject(3, "more orders than the book's depth");
        }
        let want = usize::try_from(count).map_or(usize::MAX, |c| FIXED + 8 * c);
        if bytes.len() != want {
            return reject(4, "the order count does not match the bytes");
        }
        Ok(())
    }

    fn load(&mut self, bytes: &[u8]) {
        self.last = long(bytes, 0).unwrap_or(0);
        self.step = word(bytes, 8).unwrap_or(0);
        let count = word(bytes, 12).map_or(0, |c| usize::try_from(c).unwrap_or(0));
        self.orders.clear();
        self.orders
            .extend((0..count).map(|k| long(bytes, FIXED + 8 * k).unwrap_or(0)));
    }
}

/// The `u32` at `at`.
fn word(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)?
        .try_into()
        .ok()
        .map(u32::from_le_bytes)
}

/// The `i64` at `at`.
fn long(bytes: &[u8], at: usize) -> Option<i64> {
    bytes
        .get(at..at + 8)?
        .try_into()
        .ok()
        .map(i64::from_le_bytes)
}

tessera_rl_pyffi::export_host_env!(
    OrderBook,
    module = c"tessera_example_orderbook._tessera_host",
    snapshots
);
