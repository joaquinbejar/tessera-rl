// SPDX-License-Identifier: MIT OR Apache-2.0

//! A do-nothing `Persist` environment for the registry tests: every
//! record field differs from the shipped environments' values.

use tessera_rl_core::encoding::{Canonical, Decoder, Encoder, Sink};
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, DecodeError, EnvConfig, EnvState, Environment, InvariantViolation, NoAdaptive,
    Persist, RecordHeader, SpaceError, StateInit, StepOut, Tier, Validate,
};
use tessera_rl_num::rng::{ObsRng, StepRng};

/// The fixture's id, shared by every `Fixture<V>`.
pub const FIXTURE_ID: &str = "test/Fixture-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Cell {
    value: u32,
}

impl Canonical for Cell {
    const ENCODED_LEN: usize = 4;

    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        self.value.encode(out);
    }

    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        Ok(Self {
            value: Canonical::decode(input)?,
        })
    }
}

impl StateInit for Cell {
    fn state_default() -> Self {
        Self::default()
    }
}

impl EnvState for Cell {}

pub struct FixtureConfig;

impl EnvConfig for FixtureConfig {}

/// A fixture whose `ENV_VERSION` is `V`.
pub struct Fixture<const V: u32>;

impl<const V: u32> Environment for Fixture<V> {
    type State = Cell;
    type Config = FixtureConfig;
    type Obs = ();
    type Action = ();
    type Header = RecordHeader<3, NoAdaptive>;

    const MAX_AGENTS: usize = 3;
    const TIER: Tier = Tier::SamePlatform;
    const PERFECT_INFORMATION: bool = false;

    fn observation_space(_: &FixtureConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn action_space(_: &FixtureConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(_: &FixtureConfig, _: &mut StepRng, _: &mut Cell) {}

    fn step(
        _: &FixtureConfig,
        _: &mut Cell,
        _: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
    }

    fn observe(
        _: &FixtureConfig,
        _: &Cell,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, ()>,
    ) {
    }
}

impl<const V: u32> Validate for Fixture<V> {
    fn validate(_: &FixtureConfig, _: &Cell) -> Result<(), InvariantViolation> {
        Ok(())
    }
}

impl<const V: u32> Persist for Fixture<V> {
    const ENV_ID: &'static str = FIXTURE_ID;
    const ENV_VERSION: u32 = V;
}
