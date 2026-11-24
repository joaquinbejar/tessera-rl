// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ScalarCartPole`: CartPole's kernels without a lane step (ADR-0071), the
//! scalar reference of the lane differential and the baseline arm of the
//! 1.2x stop (decision 7 and its erratum). Shared by a test and a bench.

use tessera_rl_classic::cartpole::{CartPole, CartPoleConfig, CartPoleObs, CartPoleState};
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, InvariantViolation, NoAdaptive, Persist, RecordHeader, SpaceError,
    StepOut, Tier, Validate,
};
use tessera_rl_num::rng::{ObsRng, StepRng};

/// CartPole without a lane step: the scalar reference.
pub struct ScalarCartPole;

impl Environment for ScalarCartPole {
    type State = CartPoleState;
    type Config = CartPoleConfig;
    type Obs = CartPoleObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &CartPoleConfig) -> Result<Space, SpaceError> {
        CartPole::observation_space(cfg)
    }

    fn action_space(cfg: &CartPoleConfig) -> Result<Space, SpaceError> {
        CartPole::action_space(cfg)
    }

    fn reset(cfg: &CartPoleConfig, rng: &mut StepRng, state: &mut CartPoleState) {
        CartPole::reset(cfg, rng, state);
    }

    fn step(
        cfg: &CartPoleConfig,
        state: &mut CartPoleState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        CartPole::step(cfg, state, actions, rng, out);
    }

    fn observe(
        cfg: &CartPoleConfig,
        state: &CartPoleState,
        agent: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, CartPoleObs>,
    ) {
        CartPole::observe(cfg, state, agent, rng, out);
    }
}

impl Validate for ScalarCartPole {
    fn validate(cfg: &CartPoleConfig, state: &CartPoleState) -> Result<(), InvariantViolation> {
        CartPole::validate(cfg, state)
    }
}

impl Persist for ScalarCartPole {
    const ENV_ID: &'static str = CartPole::ENV_ID;
    const ENV_VERSION: u32 = CartPole::ENV_VERSION;
}
