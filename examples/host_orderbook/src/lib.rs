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
    AgentSlot, EpisodeEnd, HostEnvironment, HostError, HostOut, HostSeed, SpaceError,
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
    orders: Vec<i64>,
    position: i64,
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
            position: 0,
            last: 0,
            step: 0,
        })
    }

    /// Position, resting orders, last price and step.
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
        self.position = 0;
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
        // Buying pays the level; selling a held unit earns it.
        let reward = match side {
            1 if self.orders.len() < depth => {
                self.orders.push(price);
                self.position += 1;
                -level
            }
            2 if self.position > 0 => {
                self.position -= 1;
                self.orders.pop();
                level
            }
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
            self.position,
            i64::try_from(self.orders.len()).unwrap_or(BOUND),
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
        // Selling needs a position.
        for action in SELL..SELL + LEVELS as usize {
            mask.set(action, self.position > 0);
        }
    }
}

tessera_rl_core::register_host!(OrderBook, id = "example/OrderBook-v1");

tessera_rl_pyffi::export_host_env!(
    OrderBook,
    module = c"tessera_example_orderbook._tessera_host"
);
