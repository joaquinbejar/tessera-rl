// SPDX-License-Identifier: MIT OR Apache-2.0

//! A single-agent environment with variable step cost, for the schedule
//! bench (ADR-0076). A heavy step spins `heavy_spins` rounds of integer
//! mixing against `LIGHT_SPINS` for a light one. Heaviness is drawn at
//! reset for the whole episode (1 in `heavy_one_in`), or, with `redraw`, on
//! every step: a heavy-tailed per-step cost, where Static assignment waits
//! for the worker whose chunks drew the most heavy steps.

use core::hint::black_box;

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Bool8, Environment, EpisodeEnd, NoAdaptive, ObsLeaf, Observation, RecordHeader,
    Reserved, SpaceError, StepOut, Tier,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

pub const LIGHT_SPINS: u32 = 64;

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct VarState {
    pub acc: u64,
    pub steps: u32,
    pub heavy: Bool8,
    pub reserved: Reserved<3>,
}

#[derive(EnvConfig, Debug, Clone)]
pub struct VarConfig {
    #[config(default = 1000, range = 1..=1_000_000)]
    pub length: u32,
    #[config(default = 16, range = 1..=1_000_000)]
    pub heavy_one_in: u32,
    #[config(default = 4096, range = 1..=10_000_000)]
    pub heavy_spins: u32,
    /// 1: draw heaviness on every step; 0: once per episode, at reset.
    #[config(default = 0, range = 0..=1)]
    pub redraw: u32,
}

#[derive(Observation)]
pub struct VarObs {
    #[space(box(low = 0, high = 1_000_000, shape = [1]))]
    pub steps: ObsLeaf<i32>,
}

pub struct VarCost;

impl Environment for VarCost {
    type State = VarState;
    type Config = VarConfig;
    type Obs = VarObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &VarConfig) -> Result<Space, SpaceError> {
        VarObs::space(cfg)
    }

    fn action_space(_cfg: &VarConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn initial_agents(_cfg: &VarConfig) -> usize {
        1
    }

    fn reset(cfg: &VarConfig, rng: &mut StepRng, state: &mut VarState) {
        state.acc = u64::from(rng.next_u32());
        state.steps = 0;
        state.heavy = Bool8::from(rng.next_u32().is_multiple_of(cfg.heavy_one_in));
    }

    fn step(
        cfg: &VarConfig,
        state: &mut VarState,
        _actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        if cfg.redraw == 1 {
            state.heavy = Bool8::from(rng.next_u32().is_multiple_of(cfg.heavy_one_in));
        }
        let spins = if state.heavy.get() {
            cfg.heavy_spins
        } else {
            LIGHT_SPINS
        };
        let mut acc = state.acc;
        for _ in 0..spins {
            acc = black_box(acc.rotate_left(7) ^ 0x9e37_79b9_7f4a_7c15);
        }
        state.acc = acc;
        state.steps += 1;
        out.set_reward(AgentSlot::FIRST, 0.0);
        if state.steps >= cfg.length {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _cfg: &VarConfig,
        state: &VarState,
        _agent: AgentSlot,
        _rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, VarObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(VarObs::STEPS) {
            leaf.set(0, state.steps.cast_signed());
        }
    }
}
