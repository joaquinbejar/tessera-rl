// SPDX-License-Identifier: MIT OR Apache-2.0

//! A host environment crate that depends on core only (ADR-0062,
//! ADR-0065 item 3): a counter whose observation is its step and the
//! config's offset, and a second environment registered without a
//! factory.

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AbortCode, AgentSlot, EpisodeEnd, HostEnvironment, HostError, HostOut, HostSeed, SpaceError,
    TurnStructure,
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
    /// Test knob: the step (1-based) at which an episode aborts; 0 never.
    #[config(default = 0)]
    pub abort_step: u32,
    /// Test knob: the code of a step's abort; a reset's is this plus 100.
    #[config(default = 7, range = 1..=1_000_000)]
    pub abort_code: u32,
    /// Test knob: each instance's first this many resets abort.
    #[config(default = 0)]
    pub reset_aborts: u32,
    /// Test knob: only the instance with this seed aborts; -1 every one.
    #[config(default = -1)]
    pub abort_seed: i64,
    /// Test knob: the step (1-based) that panics, poisoning the pool; 0
    /// never.
    #[config(default = 0)]
    pub panic_step: u32,
    /// Test knob: milliseconds each step sleeps (a long call, for the
    /// reentrancy tests); 0 none.
    #[config(default = 0, range = 0..=10_000)]
    pub sleep_ms: u32,
    /// Test knob: an agent slot that stays alive but never acts; -1 none.
    #[config(default = -1)]
    pub idle_slot: i64,
}

/// Two agents counting steps.
#[derive(Debug)]
pub struct Counter {
    cfg: CounterConfig,
    step: u32,
    seed: u64,
    resets: u32,
}

/// The counter's id.
pub const COUNTER_ID: &str = "test/FactoryCounter-v1";

impl Counter {
    /// Whether the abort knobs apply to this instance.
    fn targeted(&self) -> bool {
        self.cfg.abort_seed < 0 || u64::try_from(self.cfg.abort_seed) == Ok(self.seed)
    }
}

impl HostEnvironment for Counter {
    const ENV_ID: &'static str = COUNTER_ID;
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 2;
    // Mixed, so `acting` is asked (the idle slot knob); every slot acts by
    // default, as in Simultaneous.
    const TURNS: TurnStructure = TurnStructure::Mixed;
    type Config = CounterConfig;

    fn new(cfg: &CounterConfig) -> Result<Self, HostError> {
        Ok(Self {
            cfg: cfg.clone(),
            step: 0,
            seed: 0,
            resets: 0,
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

    fn reset(&mut self, seed: HostSeed, out: &mut HostOut<'_>) {
        self.step = 0;
        self.seed = seed.seed();
        self.resets = self.resets.saturating_add(1);
        if self.resets <= self.cfg.reset_aborts
            && self.targeted()
            && let Some(code) = AbortCode::new(self.cfg.abort_code.saturating_add(100))
        {
            out.abort(code);
        }
    }

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut HostOut<'_>) {
        self.step += 1;
        if self.cfg.sleep_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(self.cfg.sleep_ms.into()));
        }
        assert!(
            self.step != self.cfg.panic_step,
            "test panic at step {}",
            self.step
        );
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
        if self.step == self.cfg.abort_step
            && self.targeted()
            && let Some(code) = AbortCode::new(self.cfg.abort_code)
        {
            out.abort(code);
        }
    }

    fn acting(&self, agent: AgentSlot) -> bool {
        i64::try_from(agent.index()).ok() != Some(self.cfg.idle_slot)
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
