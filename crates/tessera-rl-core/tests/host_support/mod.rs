// SPDX-License-Identifier: MIT OR Apache-2.0

//! Do-nothing host environments for the host registry tests (ADR-0062):
//! every record field differs from the defaults.

#![allow(dead_code, reason = "each test binary uses part of it")]

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, EnvConfig, HostEnvironment, HostError, HostSeed, SpaceError, StepOut, TurnStructure,
};

/// The host fixture's id.
pub const HOST_ID: &str = "test/Host-v1";

pub struct HostConfig;

impl EnvConfig for HostConfig {}

/// A host environment with heap state, whose id is `HOST_ID`.
pub struct HostFixture {
    pub book: Vec<i64>,
}

macro_rules! host_impl {
    ($ty:ty, $id:expr, $build:expr) => {
        impl HostEnvironment for $ty {
            const ENV_ID: &'static str = $id;
            const ENV_VERSION: u32 = 5;
            const MAX_AGENTS: usize = 2;
            const PERFECT_INFORMATION: bool = false;
            const TURNS: TurnStructure = TurnStructure::Sequential;
            type Config = HostConfig;

            fn new(_: &HostConfig) -> Result<Self, HostError> {
                Ok($build)
            }

            fn observation_space(_: &HostConfig) -> Result<Space, SpaceError> {
                Ok(Space::Discrete { n: 2 })
            }

            fn action_space(_: &HostConfig) -> Result<Space, SpaceError> {
                Ok(Space::Discrete { n: 2 })
            }

            fn reset(&mut self, _: HostSeed, _: &mut StepOut<'_>) {}

            fn step(&mut self, _: &Actions<'_, ()>, _: &mut StepOut<'_>) {}

            fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}

            fn acting(&self, _: AgentSlot) -> bool {
                true
            }
        }
    };
}

host_impl!(HostFixture, HOST_ID, HostFixture { book: Vec::new() });

/// A host environment whose id equals the `register!` fixture's id.
pub struct HostSameId;

host_impl!(HostSameId, "test/Fixture-v1", HostSameId);
