// SPDX-License-Identifier: MIT OR Apache-2.0

//! A host environment crate that depends on core only (ADR-0062,
//! ADR-0065 item 3): a counter whose observation is its step and the
//! config's offset, a second environment registered without a factory,
//! and a spawning environment for the ADR-0013 respawn test (#71).

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

/// The respawner's configuration.
#[derive(EnvConfig, Debug, Clone, PartialEq)]
pub struct RespawnConfig {
    /// Steps per episode.
    #[config(default = 6, range = 1..=1000)]
    pub episode: u32,
    /// The step (1-based) in which agent 1 dies; 0 never.
    #[config(default = 2)]
    pub die_step: u32,
    /// Steps from the death to the spawn of the next agent in slot 1: at
    /// least 1, since a slot freed by death is reused no earlier than the
    /// next step (ADR-0013).
    #[config(default = 1, range = 1..=1000)]
    pub respawn_delay: u32,
}

/// Two agents; agent 1 dies once per episode and a new agent spawns into
/// its slot later (ADR-0013 respawn, #71). Every alive agent's reward is
/// 1 plus its action; its observation is the step and its own spawn count.
#[derive(Debug)]
pub struct Respawner {
    cfg: RespawnConfig,
    step: u32,
    /// Whether slot 1 holds an agent.
    occupied: bool,
    /// Agents spawned into slot 1 this episode.
    spawns: u32,
}

/// The respawner's id.
pub const RESPAWN_ID: &str = "test/Respawn-v1";

/// The respawner's slot count (its `MAX_AGENTS`).
pub const RESPAWN_MAX_AGENTS: usize = 2;

impl HostEnvironment for Respawner {
    const ENV_ID: &'static str = RESPAWN_ID;
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = RESPAWN_MAX_AGENTS;
    const SPAWNS: bool = true;
    type Config = RespawnConfig;

    fn new(cfg: &RespawnConfig) -> Result<Self, HostError> {
        Ok(Self {
            cfg: cfg.clone(),
            step: 0,
            occupied: true,
            spawns: 0,
        })
    }

    fn observation_space(_: &RespawnConfig) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![2],
            dtype: DType::I32,
            bounds: BoxBounds::Int {
                low: vec![0],
                high: vec![1_000_000],
            },
        })
    }

    fn action_space(_: &RespawnConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {
        self.step = 0;
        self.occupied = true;
        self.spawns = 0;
    }

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut HostOut<'_>) {
        self.step += 1;
        for s in 0..2u16 {
            if (s == 0 || self.occupied)
                && let Some(agent) = AgentSlot::new(s, 2)
            {
                let a = actions.discrete(agent, 0).unwrap_or(0);
                #[expect(clippy::cast_precision_loss, reason = "small actions")]
                out.set_reward(agent, 1.0 + a as f32);
            }
        }
        let Some(slot1) = AgentSlot::new(1, 2) else {
            return;
        };
        if self.cfg.die_step > 0 && self.step == self.cfg.die_step && self.occupied {
            out.agent_died(slot1);
            self.occupied = false;
        } else if self.cfg.die_step > 0
            && !self.occupied
            && self.step == self.cfg.die_step.saturating_add(self.cfg.respawn_delay)
        {
            out.agent_spawned(slot1);
            self.occupied = true;
            self.spawns += 1;
        }
        if self.step >= self.cfg.episode {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(&self, agent: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>) {
        let own = if agent.index() == 1 { self.spawns } else { 0 };
        let values = [
            i32::try_from(self.step).unwrap_or(0),
            i32::try_from(own).unwrap_or(0),
        ];
        if let Some(leaf) = obs.leaf(0) {
            for (bytes, v) in leaf.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                bytes.copy_from_slice(&v.to_le_bytes());
            }
        }
    }
}

tessera_rl_core::register_host!(Respawner, id = "test/Respawn-v1");
