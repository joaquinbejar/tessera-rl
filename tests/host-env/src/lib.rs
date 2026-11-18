// SPDX-License-Identifier: MIT OR Apache-2.0

//! A host environment crate that depends on core only (ADR-0062,
//! ADR-0065 item 3): a counter whose observation is its step and the
//! config's offset, and a second environment registered without a
//! factory.

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, EpisodeEnd, HostEnvironment, HostError, HostOut, HostSeed, SpaceError,
};
use tessera_rl_macros::EnvConfig;

/// The counter's configuration.
#[derive(EnvConfig, Debug, Clone, PartialEq)]
pub struct CounterConfig {
    /// Steps per episode.
    #[config(default = 4, range = 1..=1000)]
    pub episode: u32,
    /// Added to every observation.
    #[config(default = 10)]
    pub offset: i32,
    /// Every reward.
    #[config(default = 0.5)]
    pub reward: f32,
}

/// Two agents counting steps.
#[derive(Debug)]
pub struct Counter {
    cfg: CounterConfig,
    step: u32,
    seed: u64,
}

/// The counter's id.
pub const COUNTER_ID: &str = "test/FactoryCounter-v1";

impl HostEnvironment for Counter {
    const ENV_ID: &'static str = COUNTER_ID;
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 2;
    type Config = CounterConfig;

    fn new(cfg: &CounterConfig) -> Result<Self, HostError> {
        Ok(Self {
            cfg: cfg.clone(),
            step: 0,
            seed: 0,
        })
    }

    fn observation_space(_: &CounterConfig) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![2],
            dtype: DType::I32,
            bounds: BoxBounds::Int {
                low: vec![-1_000_000],
                high: vec![1_000_000],
            },
        })
    }

    fn action_space(_: &CounterConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 3 })
    }

    fn reset(&mut self, seed: HostSeed, _: &mut HostOut<'_>) {
        self.step = 0;
        self.seed = seed.seed();
    }

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut HostOut<'_>) {
        self.step += 1;
        for s in 0..2u16 {
            if let Some(agent) = AgentSlot::new(s, 2) {
                let a = actions.discrete(agent, 0).unwrap_or(0);
                #[expect(clippy::cast_precision_loss, reason = "small actions")]
                let extra = a as f32;
                out.set_reward(agent, self.cfg.reward + extra);
            }
        }
        if self.step >= self.cfg.episode {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(&self, _: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>) {
        let values = [
            i32::try_from(self.step).unwrap_or(0) + self.cfg.offset,
            i32::try_from(self.seed % 1000).unwrap_or(0),
        ];
        if let Some(leaf) = obs.leaf(0) {
            for (bytes, v) in leaf.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                bytes.copy_from_slice(&v.to_le_bytes());
            }
        }
    }
}

tessera_rl_core::register_host!(Counter, id = "test/FactoryCounter-v1");

/// A host environment registered with `register_host!` and no factory.
#[derive(Debug)]
pub struct Unfactored;

/// Its configuration (no fields).
#[derive(EnvConfig, Debug, Clone, PartialEq)]
pub struct UnfactoredConfig {}

impl HostEnvironment for Unfactored {
    const ENV_ID: &'static str = "test/Unfactored-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 1;
    type Config = UnfactoredConfig;

    fn new(_: &UnfactoredConfig) -> Result<Self, HostError> {
        Ok(Self)
    }
    fn observation_space(_: &UnfactoredConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }
    fn action_space(_: &UnfactoredConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }
    fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {}
    fn step(&mut self, _: &Actions<'_, ()>, _: &mut HostOut<'_>) {}
    fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}
}

tessera_rl_core::register_host!(Unfactored, id = "test/Unfactored-v1");
