// SPDX-License-Identifier: MIT OR Apache-2.0

//! A host environment with heap state for the host pool tests (ADR-0062):
//! a two-agent order book, integer-only (prices and sizes are `i64` ticks,
//! no floats, no RNG of its own), so equal seeds and actions give equal
//! outputs and the tests can assert them exactly.

#![allow(dead_code, reason = "each test binary uses part of it")]

use tessera_rl_core::io::{Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, EnvConfig, EpisodeEnd, HostEnvironment, HostError, HostSeed, SpaceError, StepOut,
};

/// The book's agents.
pub const AGENTS: usize = 2;

/// Book configuration.
#[derive(Debug, Clone, Copy)]
pub struct BookConfig {
    /// Steps per episode; the book terminates the episode at this step.
    pub episode: u32,
    /// The step (1-based, in every episode) whose `step` panics; 0 never.
    pub panic_at: u32,
    /// Leave flush-to-zero set when `step` returns, as a host library
    /// might (ADR-0062 item 4).
    pub hostile_fp: bool,
    /// Refuse to start (`HostError::Construction`).
    pub refuse: bool,
    /// The step (1-based) whose rewards are NaN; 0 never.
    pub nan_reward_at: u32,
    /// Write no observation at the start of an episode (step 0): the rows
    /// must then read zero, as the engine hands them (ADR-0058).
    pub blank_at_start: bool,
}

impl Default for BookConfig {
    fn default() -> Self {
        Self {
            episode: 5,
            panic_at: 0,
            hostile_fp: false,
            refuse: false,
            nan_reward_at: 0,
            blank_at_start: false,
        }
    }
}

impl EnvConfig for BookConfig {}

/// One instance: the resting orders live on the heap and grow on every
/// buy, so every step allocates.
#[derive(Debug)]
pub struct Book {
    cfg: BookConfig,
    orders: Vec<i64>,
    position: [i64; AGENTS],
    last: i64,
    step: u32,
}

/// The reward every agent gets from a reset.
pub const RESET_REWARD: f32 = 100.0;

/// Choices of the side head (hold, buy, sell) and the level head.
pub const SIDES: u32 = 3;
pub const LEVELS: u32 = 4;

impl HostEnvironment for Book {
    const ENV_ID: &'static str = "test/Book-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = AGENTS;
    const MASKS: bool = true;
    type Config = BookConfig;

    fn new(cfg: &BookConfig) -> Result<Self, HostError> {
        if cfg.refuse {
            return Err(HostError::Construction {
                reason: "the book refused".to_owned(),
            });
        }
        Ok(Self {
            cfg: *cfg,
            orders: Vec::new(),
            position: [0; AGENTS],
            last: 0,
            step: 0,
        })
    }

    fn observation_space(_: &BookConfig) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![4],
            dtype: DType::I32,
            bounds: BoxBounds::Int {
                low: vec![-1_000_000],
                high: vec![1_000_000],
            },
        })
    }

    fn action_space(_: &BookConfig) -> Result<Space, SpaceError> {
        Ok(Space::MultiDiscrete {
            nvec: vec![SIDES, LEVELS],
        })
    }

    /// Writes a reward of `RESET_REWARD` for every agent: visible after an
    /// explicit or `NextStep` reset, discarded in a `SameStep` one.
    fn reset(&mut self, seed: HostSeed, out: &mut StepOut<'_>) {
        for slot in 0..AGENTS {
            if let Some(agent) = AgentSlot::new(u16::try_from(slot).unwrap_or(0), AGENTS) {
                out.set_reward(agent, RESET_REWARD);
            }
        }
        self.orders.clear();
        self.position = [0; AGENTS];
        // Integer-only: the opening price comes from the seed and episode.
        self.last =
            i64::try_from(seed.seed() % 97).unwrap_or(0) + i64::from(seed.episode_index()) * 1000;
        self.step = 0;
    }

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut StepOut<'_>) {
        self.step += 1;
        assert!(
            self.step != self.cfg.panic_at,
            "the book panicked at step {}",
            self.step
        );
        for slot in 0..AGENTS {
            let Some(agent) = AgentSlot::new(u16::try_from(slot).unwrap_or(0), AGENTS) else {
                continue;
            };
            let side = actions.discrete(agent, 0).unwrap_or(0);
            let level = i64::from(actions.discrete(agent, 1).unwrap_or(0));
            let price = self.last + level;
            let Some(position) = self.position.get_mut(slot) else {
                continue;
            };
            let reward = match side {
                1 => {
                    self.orders.push(price);
                    *position += 1;
                    -level
                }
                2 if *position > 0 => {
                    *position -= 1;
                    level
                }
                _ => 0,
            };
            #[expect(clippy::cast_precision_loss, reason = "small integer rewards")]
            let reward = if self.step == self.cfg.nan_reward_at {
                f32::NAN
            } else {
                reward as f32
            };
            out.set_reward(agent, reward);
            self.last = price;
        }
        if self.cfg.hostile_fp {
            tessera_rl_num::fp::hostile::set(tessera_rl_num::fp::hostile::flush_to_zero);
        }
        if self.step >= self.cfg.episode {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(&self, agent: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>) {
        if self.cfg.blank_at_start && self.step == 0 {
            return;
        }
        let values = [
            self.position.get(agent.index()).copied().unwrap_or(0),
            i64::try_from(self.orders.len()).unwrap_or(i64::MAX),
            self.last,
            i64::from(self.step),
        ];
        if let Some(leaf) = obs.leaf(0) {
            for (bytes, v) in leaf.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                bytes.copy_from_slice(&i32::try_from(v).unwrap_or(i32::MAX).to_le_bytes());
            }
        }
    }

    fn mask(&self, agent: AgentSlot, mask: &mut MaskWriter<'_>) {
        mask.fill(true);
        // Selling needs a position: the side head's choice 2.
        let position = self.position.get(agent.index()).copied().unwrap_or(0);
        mask.set(2, position > 0);
    }
}
