// SPDX-License-Identifier: MIT OR Apache-2.0

// An id that differs from HostEnvironment::ENV_ID does not compile.
use tessera_rl::core::io::{Actions, ObsWriter};
use tessera_rl::core::space::Space;
use tessera_rl::core::{
    AgentSlot, EnvConfig, HostEnvironment, HostError, HostOut, HostSeed, SpaceError,
};

struct Cfg;

impl EnvConfig for Cfg {}

struct Book {
    orders: Vec<i64>,
}

impl HostEnvironment for Book {
    const ENV_ID: &'static str = "example/Book-v1";
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

    fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {
        self.orders.clear();
    }

    fn step(&mut self, _: &Actions<'_, ()>, _: &mut HostOut<'_>) {
        self.orders.push(1);
    }

    fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}
}

tessera_rl::register_host!(Book, id = "example/Book-v2");

fn main() {}
