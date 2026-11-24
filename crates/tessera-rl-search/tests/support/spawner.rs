// SPDX-License-Identifier: MIT OR Apache-2.0

//! A three-slot simultaneous environment whose slot 2 spawns after the
//! first step (ADR-0093 erratum test): slots 0 and 1 start alive; actions
//! 0 or 1 add to a total; the episode ends after 4 steps. Its heuristic
//! plays 0. Turns are `Mixed`: with `alternate` 1, slot 0 acts on even
//! steps only (the single-edge rule of ADR-0093 decision 4); with `spawn`
//! 0, slot 2 never spawns.

use tessera_rl_core::io::{ActionWriter, Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, Heuristic, NoAdaptive, ObsLeaf, Observation, RecordHeader,
    SpaceError, StepOut, Tier, TurnStructure,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Steps {
    pub total: u32,
    pub steps: u32,
}

#[derive(EnvConfig, Debug, Clone)]
pub struct SpawnerConfig {
    /// 1: slot 0 acts on even steps only.
    #[config(default = 0, range = 0..=1)]
    pub alternate: u32,
    /// 1: slot 2 spawns after the first step.
    #[config(default = 1, range = 0..=1)]
    pub spawn: u32,
}

#[derive(Observation)]
pub struct StepsObs {
    #[space(box(low = 0, high = 1000, shape = [1]))]
    total: ObsLeaf<i32>,
}

pub struct Spawner;

impl Environment for Spawner {
    type State = Steps;
    type Config = SpawnerConfig;
    type Obs = StepsObs;
    type Action = ();
    type Header = RecordHeader<3, NoAdaptive>;

    const MAX_AGENTS: usize = 3;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Mixed;

    fn observation_space(cfg: &SpawnerConfig) -> Result<Space, SpaceError> {
        StepsObs::space(cfg)
    }

    fn action_space(_: &SpawnerConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn initial_agents(_: &SpawnerConfig) -> usize {
        2
    }

    fn acting(cfg: &SpawnerConfig, state: &Steps, agent: AgentSlot) -> bool {
        agent.index() != 0 || cfg.alternate == 0 || state.steps.is_multiple_of(2)
    }

    fn reset(_: &SpawnerConfig, _: &mut StepRng, state: &mut Steps) {
        *state = Steps::default();
    }

    fn step(
        cfg: &SpawnerConfig,
        state: &mut Steps,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        for slot in 0..3u16 {
            if let Some(agent) = AgentSlot::new(slot, 3) {
                let a = actions.discrete(agent, 0).unwrap_or(0);
                state.total += u32::try_from(a).unwrap_or(0);
            }
        }
        if state.steps == 0
            && cfg.spawn == 1
            && let Some(third) = AgentSlot::new(2, 3)
        {
            out.agent_spawned(third);
        }
        state.steps += 1;
        if state.steps >= 4 {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &SpawnerConfig,
        state: &Steps,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, StepsObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(StepsObs::TOTAL) {
            leaf.set(0, i32::try_from(state.total).unwrap_or(0));
        }
    }
}

impl Heuristic for Spawner {
    fn act(
        _: &SpawnerConfig,
        _: &Steps,
        _: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        out.set_discrete(0, 0);
    }
}

impl tessera_rl_core::Validate for Spawner {
    fn validate(_: &SpawnerConfig, _: &Steps) -> Result<(), tessera_rl_core::InvariantViolation> {
        Ok(())
    }
}
