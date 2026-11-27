// SPDX-License-Identifier: MIT OR Apache-2.0

//! `SurvivalLanes`: Survival-v1 with its agent-lane step switched on, for
//! tests only (ADR-0100 head 3). Survival itself does not set
//! `Environment::AGENT_LANE` until the stop point passes (erratum 2), so
//! this test-only, unregistered and unexported wrapper delegates every item
//! to `Survival` and sets the hook, so that pools can compare the
//! agent-lane step with the reference step.

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentLaneEnvironment, AgentLaneHook, AgentSlot, Environment, NoAdaptive, RecordHeader,
    SpaceError, StepOut, Tier, TurnStructure, agent_lane_hook,
};
use tessera_rl_num::rng::{ObsRng, StepRng};
use tessera_rl_simd::Resolved;
use tessera_rl_world::survival::{MAX_AGENTS, lanes};
use tessera_rl_world::{Survival, SurvivalConfig, SurvivalObs, SurvivalState};

/// Survival-v1 with `AGENT_LANE` set.
pub struct SurvivalLanes;

impl Environment for SurvivalLanes {
    type State = SurvivalState;
    type Config = SurvivalConfig;
    type Obs = SurvivalObs;
    type Action = ();
    type Header = RecordHeader<MAX_AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = MAX_AGENTS;
    const TIER: Tier = <Survival as Environment>::TIER;
    const TURNS: TurnStructure = <Survival as Environment>::TURNS;
    const PERFECT_INFORMATION: bool = <Survival as Environment>::PERFECT_INFORMATION;
    const AGENT_LANE: Option<AgentLaneHook<Self>> = Some(agent_lane_hook::<Self>());

    fn observation_space(cfg: &SurvivalConfig) -> Result<Space, SpaceError> {
        Survival::observation_space(cfg)
    }

    fn action_space(cfg: &SurvivalConfig) -> Result<Space, SpaceError> {
        Survival::action_space(cfg)
    }

    fn initial_agents(cfg: &SurvivalConfig) -> usize {
        Survival::initial_agents(cfg)
    }

    fn acting(cfg: &SurvivalConfig, state: &SurvivalState, agent: AgentSlot) -> bool {
        Survival::acting(cfg, state, agent)
    }

    fn reset(cfg: &SurvivalConfig, rng: &mut StepRng, state: &mut SurvivalState) {
        Survival::reset(cfg, rng, state);
    }

    fn step(
        cfg: &SurvivalConfig,
        state: &mut SurvivalState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        Survival::step(cfg, state, actions, rng, out);
    }

    fn observe(
        cfg: &SurvivalConfig,
        state: &SurvivalState,
        agent: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, SurvivalObs>,
    ) {
        Survival::observe(cfg, state, agent, rng, out);
    }
}

impl AgentLaneEnvironment for SurvivalLanes {
    fn step_agent_lanes(
        cfg: &SurvivalConfig,
        _: &NoAdaptive,
        state: &mut SurvivalState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
        level: Resolved,
    ) {
        lanes::step(cfg, state, actions, rng, out, level);
    }
}
