// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pool-level fault injection: a worker that dies outside the job's catch
//! poisons the pool (ADR-0028).

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, NoAdaptive, ObsLeaf, Observation, RecordHeader, SpaceError, StepOut,
    Tier,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

use crate::sys::Faults;
use crate::{EngineError, EnvPool, PoolOptions, Reseed, SyncOptions};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Tick {
    steps: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct TickConfig {}

#[derive(Observation)]
struct TickObs {
    #[space(box(low = 0, high = 1_000_000, shape = [1]))]
    steps: ObsLeaf<i32>,
}

struct Ticker;

impl Environment for Ticker {
    type State = Tick;
    type Config = TickConfig;
    type Obs = TickObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &TickConfig) -> Result<Space, SpaceError> {
        TickObs::space(cfg)
    }

    fn action_space(_: &TickConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(_: &TickConfig, _: &mut StepRng, _: &mut Tick) {}

    fn step(
        _: &TickConfig,
        state: &mut Tick,
        _: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
        state.steps += 1;
    }

    fn observe(
        _: &TickConfig,
        _: &Tick,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, TickObs>,
    ) {
    }
}

#[test]
fn test_pool_worker_death_mid_epoch_poisons_then_reports() {
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(4).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(32).unwrap(), Seed::new(1))
    };
    let faults = Faults {
        die_in_epoch: Some(1),
        ..Faults::default()
    };
    let mut p = EnvPool::<Ticker>::new(TickConfig {}, options)
        .unwrap()
        .with_workers_faults(SyncOptions::new(NonZeroUsize::new(2).unwrap()), faults)
        .unwrap();
    // The epoch where worker 1 dies completes, but its chunks may be
    // untouched: the run reports a panic and poisons the pool.
    assert_eq!(
        p.reset_all(Reseed::Keep).map(|_| ()),
        Err(EngineError::Panicked)
    );
    assert!(p.is_poisoned());
    // The next run reports the death instead of waiting for it.
    assert_eq!(
        p.reset_all(Reseed::Keep).map(|_| ()),
        Err(EngineError::WorkerDied)
    );
}
