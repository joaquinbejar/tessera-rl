// SPDX-License-Identifier: MIT OR Apache-2.0

// The tessera/ namespace is reserved for first-party environments.
use tessera_rl::core::io::{Actions, ObsWriter};
use tessera_rl::core::space::Space;
use tessera_rl::core::{
    AgentSlot, EnvConfig, HostEnvironment, HostError, HostSeed, SpaceError, StepOut,
};

struct Cfg;

impl EnvConfig for Cfg {}

struct Book {
    orders: Vec<i64>,
}

impl HostEnvironment for Book {
    const ENV_ID: &'static str = "tessera/Book-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 1;
    type Config = Cfg;

    fn new(_: &Cfg) -> Result<Self, HostError> {
        Ok(Self { orders: Vec::new() })
    }

    fn observation_space(_: &Cfg) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn action_space(_: &Cfg) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(&mut self, _: HostSeed, _: &mut StepOut<'_>) {
        self.orders.clear();
    }

    fn step(&mut self, _: &Actions<'_, ()>, _: &mut StepOut<'_>) {
        self.orders.push(1);
    }

    fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}
}

tessera_rl::register_host!(Book, id = "tessera/Book-v1");

fn main() {}
