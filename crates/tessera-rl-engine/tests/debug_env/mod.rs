// SPDX-License-Identifier: MIT OR Apache-2.0

//! A one-agent environment that can put a NaN in its state or reward, or
//! allocate in `step`, at a given step: the `debug-checks` tests of
//! ADR-0055. Every instance does the same thing at the same step, so the
//! instance an error reports is the first one checked.

#![allow(dead_code, reason = "each test binary uses part of it")]

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, NoAdaptive, ObsLeaf, Observation, RecordHeader, SpaceError,
    StepOut, Tier,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Probe {
    pub steps: u32,
    pub level: f32,
    /// May hold NaN.
    #[state(allow_nan)]
    pub free: f32,
}

/// How `step` allocates at `alloc_at`.
pub const ALLOC_BOX: u32 = 1;
/// `vec![0; n]`: `alloc_zeroed`.
pub const ALLOC_ZEROED: u32 = 2;
/// An allocation grown once: `alloc` then `realloc` (two counts).
pub const ALLOC_GROW: u32 = 3;

/// Independent test switches.
#[expect(clippy::struct_excessive_bools, reason = "independent test switches")]
#[derive(EnvConfig, Debug, Clone)]
pub struct ProbeConfig {
    #[config(default = 50)]
    pub length: u32,
    /// Put NaN in `level` at this step (0: never).
    #[config(default = 0)]
    pub nan_state_at: u32,
    /// Reward NaN at this step (0: never).
    #[config(default = 0)]
    pub nan_reward_at: u32,
    /// Put NaN in the `allow_nan` field at every step.
    #[config(default = false)]
    pub nan_free: bool,
    /// Put NaN in `level` at reset.
    #[config(default = false)]
    pub nan_on_reset: bool,
    /// Allocate (as `alloc_kind`) in `step` at this step (0: never).
    #[config(default = 0)]
    pub alloc_at: u32,
    /// Allocate at every step.
    #[config(default = false)]
    pub alloc_always: bool,
    #[config(default = 1)]
    pub alloc_kind: u32,
    /// Allocate in `reset` (outside any chunk step).
    #[config(default = false)]
    pub alloc_on_reset: bool,
    /// Panic in `step` at this step (0: never).
    #[config(default = 0)]
    pub panic_at: u32,
    /// Spawn the live agent again at this step, an invalid agent event
    /// (0: never).
    #[config(default = 0)]
    pub spawn_alive_at: u32,
}

/// Every state is valid: `import_all` needs the trait.
impl tessera_rl_core::Validate for ProbeEnv {
    fn validate(_: &ProbeConfig, _: &Probe) -> Result<(), tessera_rl_core::InvariantViolation> {
        Ok(())
    }
}

/// The split-phase `import_all` needs the trait.
impl tessera_rl_core::Persist for ProbeEnv {
    const ENV_ID: &'static str = "tessera/DebugProbe-v1";
    const ENV_VERSION: u32 = 1;
}

#[derive(Observation)]
pub struct ProbeObs {
    #[space(box(low = 0, high = 1_000_000, shape = [1]))]
    pub steps: ObsLeaf<i32>,
}

pub struct ProbeEnv;

fn allocate(kind: u32) {
    match kind {
        ALLOC_ZEROED => {
            let v = vec![0_u8; 32];
            core::hint::black_box(&v);
        }
        ALLOC_GROW => {
            let mut v: Vec<u8> = Vec::with_capacity(1);
            v.push(1);
            v.reserve(4096);
            core::hint::black_box(&v);
        }
        _ => {
            let b = Box::new(7_u64);
            core::hint::black_box(&b);
        }
    }
}

impl Environment for ProbeEnv {
    type State = Probe;
    type Config = ProbeConfig;
    type Obs = ProbeObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &ProbeConfig) -> Result<Space, SpaceError> {
        ProbeObs::space(cfg)
    }

    fn action_space(_: &ProbeConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(cfg: &ProbeConfig, _: &mut StepRng, state: &mut Probe) {
        *state = Probe::default();
        if cfg.nan_on_reset {
            state.level = f32::NAN;
        }
        if cfg.alloc_on_reset {
            allocate(cfg.alloc_kind);
        }
    }

    fn step(
        cfg: &ProbeConfig,
        state: &mut Probe,
        _: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        state.steps += 1;
        if state.steps == cfg.nan_state_at {
            state.level = f32::NAN;
        }
        if cfg.nan_free {
            state.free = f32::NAN;
        }
        if cfg.alloc_always || state.steps == cfg.alloc_at {
            allocate(cfg.alloc_kind);
        }
        if state.steps == cfg.spawn_alive_at {
            out.agent_spawned(AgentSlot::FIRST);
        }
        // Panic injection for the recovery tests.
        assert!(state.steps != cfg.panic_at, "injected step panic");
        let reward = if state.steps == cfg.nan_reward_at {
            f32::NAN
        } else {
            1.0
        };
        out.set_reward(AgentSlot::FIRST, reward);
        if state.steps >= cfg.length {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &ProbeConfig,
        state: &Probe,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, ProbeObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(ProbeObs::STEPS) {
            leaf.set(0, i32::try_from(state.steps).unwrap_or(i32::MAX));
        }
    }
}
