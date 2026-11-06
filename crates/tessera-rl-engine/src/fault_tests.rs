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

/// A worker that dies in a Pipelined half job: `recv` reports
/// `WorkerDied`, the pool stays poisoned, and every later send or writer
/// reports the death permanently (ADR-0047 section 7).
#[cfg(not(loom))]
#[test]
fn test_pipelined_worker_death_reports_worker_died_permanently() {
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
        .pipelined_faults(
            crate::PipelinedOptions::new(NonZeroUsize::new(2).unwrap()),
            faults,
        )
        .unwrap();
    let [a, b] = p.reset_all(Reseed::Keep).unwrap();
    let actions = [0_i32; 16];
    let input = crate::ActionInput {
        discrete: &actions,
        continuous: &[],
    };
    p.send(a, input).map_err(|e| e.error).unwrap();
    // The death is visible with no `recv` in between.
    let visible = (0..10_000_000).any(|_| {
        std::thread::yield_now();
        p.is_poisoned()
    });
    assert!(visible, "a dead worker poisons the wrapper");
    let err = p.send(b, input).unwrap_err();
    assert_eq!(err.error, EngineError::WorkerDied);
    drop(err);
    assert_eq!(p.recv().map(|_| ()), Err(EngineError::WorkerDied));
    assert!(p.is_poisoned());
    assert_eq!(
        p.reset_all(Reseed::Keep).map(|_| ()),
        Err(EngineError::WorkerDied)
    );
    assert_eq!(p.idle_token(0).map(|_| ()), Err(EngineError::Poisoned));
}

/// Receives, and sends again while nothing is left, until the injected death
/// shows: the surviving worker can finish every id before the other worker
/// reaches its death point, and `recv` then rightly says `NothingInFlight`.
#[cfg(not(loom))]
fn recv_until_death(p: &mut crate::AsyncPool<Ticker, crate::NoMasks>, ids: &[u32]) -> EngineError {
    let actions = vec![0_i32; ids.len()];
    let input = crate::ActionInput {
        discrete: &actions,
        continuous: &[],
    };
    // The death can already have happened, so this send can report it too.
    if let Err(e) = p.send(ids, input) {
        return e;
    }
    for _ in 0..100_000 {
        match p.recv() {
            Ok(t) => drop(t),
            Err(EngineError::NothingInFlight) => {
                std::thread::yield_now();
                if let Err(e) = p.send(ids, input) {
                    return e;
                }
            }
            Err(e) => return e,
        }
    }
    // The injected death never showed: the callers' assertions fail on this.
    EngineError::NothingInFlight
}

/// A worker that dies in an Async drain job: `recv` reports `WorkerDied`
/// (the surviving worker drains the ring, so nothing hangs), and every later
/// send or writer reports the death permanently (ADR-0047 section 7).
#[cfg(not(loom))]
#[test]
fn test_async_worker_death_reports_worker_died_permanently() {
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
        .async_faults(
            crate::AsyncOptions::new(NonZeroUsize::new(2).unwrap(), NonZeroUsize::new(4).unwrap()),
            faults,
        )
        .unwrap();
    drop(p.reset_all(Reseed::Keep).unwrap());
    let ids: Vec<u32> = (0..8).collect();
    let seen = recv_until_death(&mut p, &ids);
    assert!(
        matches!(seen, EngineError::WorkerDied | EngineError::Poisoned),
        "{seen:?}"
    );
    assert!(p.is_poisoned());
    assert!(matches!(
        p.send(
            &[20],
            crate::ActionInput {
                discrete: &[0],
                continuous: &[]
            }
        ),
        Err(EngineError::Poisoned | EngineError::WorkerDied)
    ));
    assert_eq!(
        p.reset_all(Reseed::Keep).map(|_| ()),
        Err(EngineError::WorkerDied)
    );
}

/// The forced schedule of that test's CI failure: the dying worker waits until
/// the other one has delivered every id, so `recv` has nothing to report yet.
#[cfg(not(loom))]
#[test]
fn test_async_worker_death_after_all_ids_are_delivered_reports_nothing_in_flight_first() {
    static GATE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(4).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(32).unwrap(), Seed::new(1))
    };
    let faults = Faults {
        die_in_epoch: Some(1),
        death_gate: Some(&GATE),
        ..Faults::default()
    };
    let mut p = EnvPool::<Ticker>::new(TickConfig {}, options)
        .unwrap()
        .async_faults(
            crate::AsyncOptions::new(NonZeroUsize::new(2).unwrap(), NonZeroUsize::new(4).unwrap()),
            faults,
        )
        .unwrap();
    drop(p.reset_all(Reseed::Keep).unwrap());
    let ids: Vec<u32> = (0..8).collect();
    let actions = [0_i32; 8];
    p.send(
        &ids,
        crate::ActionInput {
            discrete: &actions,
            continuous: &[],
        },
    )
    .unwrap();
    // `try_recv`: a blocking `recv` would wait for the held worker's job.
    let mut batches = 0;
    while batches < 2 {
        if let Some(t) = p.try_recv().unwrap() {
            drop(t);
            batches += 1;
        }
        std::thread::yield_now();
    }
    assert_eq!(p.recv().map(|_| ()), Err(EngineError::NothingInFlight));
    let mut records = vec![p.snapshot(0).unwrap(); 32];
    assert!(!p.is_poisoned());
    GATE.store(true, core::sync::atomic::Ordering::Release);
    // The death is visible with no `recv` in between.
    let visible = (0..10_000_000).any(|_| {
        std::thread::yield_now();
        p.is_poisoned()
    });
    assert!(visible, "a dead worker poisons the wrapper");
    assert_eq!(p.snapshot_all(&mut records), Err(EngineError::Poisoned));
    assert_eq!(
        p.send(
            &ids,
            crate::ActionInput {
                discrete: &actions,
                continuous: &[],
            },
        ),
        Err(EngineError::WorkerDied)
    );
    assert_eq!(p.reset(&[0], Reseed::Keep), Err(EngineError::WorkerDied));
    assert_eq!(
        p.reset_all(Reseed::Keep).map(|_| ()),
        Err(EngineError::WorkerDied)
    );
}
