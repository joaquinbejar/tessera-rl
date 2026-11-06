// SPDX-License-Identifier: MIT OR Apache-2.0

//! A synthetic same-platform environment's files: no catalog environment has
//! this tier. The build id is a parameter here. See `fuzz/harness/persist.rs`.

#![no_main]

use core::num::NonZeroU32;

use libfuzzer_sys::fuzz_target;
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, InvariantViolation, NoAdaptive, Observation, Persist, RecordHeader,
    SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_engine::{EnvPool, NoMasks, PoolOptions, Reseed};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

#[path = "../harness/persist.rs"]
mod harness;

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct State {
    v: f64,
}

#[derive(EnvConfig, Debug, Clone)]
struct Config {
    #[config(default = 1)]
    size: u32,
}

#[derive(Observation)]
struct Obs {
    #[space(box(low = 0, high = 1, shape = [1]))]
    value: tessera_rl_core::ObsLeaf<i32>,
}

struct Same;

impl Environment for Same {
    type State = State;
    type Config = Config;
    type Obs = Obs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;
    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::SamePlatform;
    fn observation_space(cfg: &Config) -> Result<Space, SpaceError> {
        Obs::space(cfg)
    }
    fn action_space(_: &Config) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }
    fn reset(_: &Config, _: &mut StepRng, _: &mut State) {}
    fn step(_: &Config, _: &mut State, _: &Actions<'_, ()>, _: &mut StepRng, _: &mut StepOut<'_>) {}
    fn observe(_: &Config, _: &State, _: AgentSlot, _: &mut ObsRng, _: &mut ObsWriter<'_, '_, Obs>) {}
}

impl Validate for Same {
    fn validate(_: &Config, _: &State) -> Result<(), InvariantViolation> {
        Ok(())
    }
}

impl Persist for Same {
    const ENV_ID: &'static str = "tessera/Same-v1";
    const ENV_VERSION: u32 = 1;
}

fuzz_target!(|data: &[u8]| {
    let options = PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(3));
    let mut p = EnvPool::<Same, NoMasks>::new(Config::default(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    // No checked-in fixture can hold this synthetic file, so overlay the input on a
    // valid one, as well as running it alone.
    let mut file = p.encode_with(&[0], "fuzz", Some("fuzz-build")).unwrap();
    if let Some((&at, rest)) = data.split_first() {
        let start = usize::from(at) % file.len();
        for (out, byte) in file[start..].iter_mut().zip(rest) {
            *out = *byte;
        }
    }
    harness::run(p, &file, Some("fuzz-build"));
    let mut p = EnvPool::<Same, NoMasks>::new(Config::default(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    harness::run(p, data, Some("fuzz-build"));
});
