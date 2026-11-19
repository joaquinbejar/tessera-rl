// SPDX-License-Identifier: MIT OR Apache-2.0

//! Two factories for one id (ADR-0065 item 3): every `make_host` in this
//! binary reports `Registry(DuplicateId)`, as the registration lookup does.

use core::num::NonZeroU32;

use tessera_rl_core::envs::RegistryError;
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, ConfigSource, HostEnvironment, HostError, HostOut, HostSeed, SpaceError,
};
use tessera_rl_engine::{FactoryError, MakeOptions, PoolOptions, make_host};
use tessera_rl_num::identity::Seed;
use tessera_rl_test_host_env::{COUNTER_ID, UnfactoredConfig};

/// A second environment that claims the counter's id.
struct Impostor;

impl HostEnvironment for Impostor {
    const ENV_ID: &'static str = COUNTER_ID;
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

tessera_rl_engine::register_host_factory!(Impostor);

#[test]
fn test_factory_two_factories_for_one_id_are_a_registry_error() {
    assert_eq!(tessera_rl_test_host_glue::linked(), COUNTER_ID);
    let options = MakeOptions::new(PoolOptions::new(NonZeroU32::new(1).unwrap(), Seed::new(0)));
    let err = make_host(COUNTER_ID, &ConfigSource::empty(), &options)
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        err,
        FactoryError::Registry(RegistryError::DuplicateId { id: COUNTER_ID })
    );
}
