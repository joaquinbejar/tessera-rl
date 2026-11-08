// SPDX-License-Identifier: MIT OR Apache-2.0

//! A test environment with one continuous action head (`make py-test-hooks`):
//! the reward is the `f32` action as received, so a test reads back its exact
//! bits. It exists to test the Python action path without a real continuous
//! environment.

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, HeaderView, InvariantViolation, NoAdaptive, Observation,
    RecordHeader, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_num::rng::{ObsRng, StepRng};

use crate::split::common::{CountConfig, CountObs, Counts};

pub(crate) struct ContinuousProbe;

impl Environment for ContinuousProbe {
    type State = Counts;
    type Config = CountConfig;
    type Obs = CountObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        CountObs::space(cfg)
    }

    fn action_space(_: &CountConfig) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![1],
            dtype: DType::F32,
            bounds: BoxBounds::Float {
                low: vec![-2.0],
                high: vec![2.0],
            },
        })
    }

    fn reset(_: &CountConfig, _: &mut StepRng, state: &mut Counts) {
        *state = Counts::default();
    }

    fn step(
        cfg: &CountConfig,
        state: &mut Counts,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let action = actions.continuous(AgentSlot::FIRST, 0).unwrap_or(f32::NAN);
        out.set_reward(AgentSlot::FIRST, action);
        state.steps += 1;
        if state.steps >= cfg.length {
            out.end_episode(EpisodeEnd::Truncated);
        }
    }

    fn observe(
        _: &CountConfig,
        _: &Counts,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, CountObs>,
    ) {
    }
}

impl tessera_rl_core::Persist for ContinuousProbe {
    const ENV_ID: &'static str = "tessera/ContinuousProbe-v1";
    const ENV_VERSION: u32 = 1;
}

impl Validate for ContinuousProbe {
    fn validate(_: &CountConfig, _: &Counts) -> Result<(), InvariantViolation> {
        Ok(())
    }

    fn validate_record(
        _: &CountConfig,
        _: &HeaderView<'_>,
        _: &Counts,
    ) -> Result<(), InvariantViolation> {
        Ok(())
    }
}
