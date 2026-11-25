// SPDX-License-Identifier: MIT OR Apache-2.0

//! The privileged state buffer (ADR-0079 items 7 and 10): one row per
//! instance equal to `write_state`, zero while pending, written by every
//! call that writes the observations (reset, step, restore, fork, import),
//! in Inline and Sync mode; an output of the shared storage with its write
//! generation; refused after the first reset, twice, and by the
//! split-phase modes.

mod common;

use core::num::{NonZeroU32, NonZeroUsize};

use common::{CountConfig, Counter, Counts};
use tessera_rl_core::io::ObsWriter;
use tessera_rl_core::space::Space;
use tessera_rl_core::{ObsLeaf, Observation, PrivilegedState, SpaceError};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AutoReset, DecodeLimits, EngineError, EnvPool, Masked, OutputField,
    PipelinedOptions, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_macros::Observation;
use tessera_rl_num::identity::Seed;

const N: u32 = 6;
/// Bytes of one row: three little-endian `i32`s.
const ROW: usize = 12;

/// Both counters and the step count: what no single agent observes.
#[derive(Observation)]
struct CountState {
    #[space(box(low = -1_000_000, high = 1_000_000, shape = [3]))]
    all: ObsLeaf<i32>,
}

impl PrivilegedState for Counter {
    fn state_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        CountState::space(cfg)
    }

    fn write_state(_: &CountConfig, state: &Counts, out: &mut ObsWriter<'_, '_, ()>) {
        if let Some(mut leaf) = out.leaf_mut(CountState::ALL) {
            leaf.set(0, state.value[0]);
            leaf.set(1, state.value[1]);
            leaf.set(2, i32::try_from(state.steps).unwrap_or(i32::MAX));
        }
    }
}

fn options(auto_reset: AutoReset) -> PoolOptions {
    PoolOptions {
        auto_reset,
        chunk_size: NonZeroU32::new(2).unwrap_or(NonZeroU32::MIN),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap_or(NonZeroU32::MIN), Seed::new(9))
    }
}

fn cfg(length: u32) -> CountConfig {
    CountConfig {
        length,
        ..CountConfig::default()
    }
}

fn pool(
    length: u32,
    auto_reset: AutoReset,
    workers: Option<usize>,
) -> Result<EnvPool<Counter, Masked>, EngineError> {
    let p = EnvPool::with_masks(cfg(length), options(auto_reset))?.with_privileged_state()?;
    match workers.and_then(NonZeroUsize::new) {
        Some(w) => p.with_workers(SyncOptions::new(w)),
        None => Ok(p),
    }
}

/// A pool without the opt-in.
fn plain() -> Result<EnvPool<Counter, Masked>, EngineError> {
    EnvPool::with_masks(cfg(3), options(AutoReset::SameStep))
}

/// `write_state` of `state` into a fresh zeroed row.
fn expected(state: &Counts) -> Vec<u8> {
    let mut row = vec![0u8; ROW];
    {
        let mut leaves: [&mut [u8]; 1] = [&mut row];
        Counter::write_state(&cfg(1), state, &mut ObsWriter::new(&mut leaves));
    }
    row
}

/// The privileged row of instance `i`.
fn row(p: &EnvPool<Counter, Masked>, i: usize) -> Vec<u8> {
    p.outputs()
        .ok()
        .and_then(|o| o.privileged_state(0))
        .and_then(|rows| rows.get(i * ROW..(i + 1) * ROW))
        .map(<[u8]>::to_vec)
        .unwrap_or_default()
}

const ONES: [i32; 2 * N as usize] = [1; 2 * N as usize];

fn ones() -> ActionInput<'static> {
    ActionInput {
        discrete: &ONES,
        continuous: &[],
    }
}

/// Every row equals `write_state` of its instance's state.
fn assert_rows_match(p: &EnvPool<Counter, Masked>) {
    for i in 0..N {
        let state = p.state(i).unwrap_or_default();
        let k = usize::try_from(i).unwrap_or(usize::MAX);
        assert_eq!(row(p, k), expected(&state), "instance {i}");
    }
}

#[test]
fn test_privileged_state_rows_equal_write_state_inline_and_sync() {
    for workers in [None, Some(3)] {
        let mut p = pool(4, AutoReset::SameStep, workers).unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        assert_rows_match(&p);
        for _ in 0..9 {
            p.step(ones()).unwrap();
            assert_rows_match(&p);
        }
        assert!(row(&p, 0).iter().any(|&b| b != 0), "rows hold the state");
    }
}

#[test]
fn test_privileged_state_pending_instances_rows_are_zero() {
    let mut p = pool(2, AutoReset::Disabled, None).unwrap();
    // Before the first reset every instance is pending.
    assert!((0..6).all(|i| row(&p, i) == vec![0; ROW]));
    p.reset_all(Reseed::Keep).unwrap();
    p.step(ones()).unwrap();
    p.step(ones()).unwrap();
    // Every episode ended: pending, zero rows.
    assert!((0..6).all(|i| row(&p, i) == vec![0; ROW]));
    p.reset(&[2], Reseed::Keep).unwrap();
    assert_eq!(row(&p, 2), expected(&p.state(2).unwrap()));
    assert_eq!(row(&p, 3), vec![0; ROW]);
}

#[test]
fn test_privileged_state_region_in_shared_storage_follows_the_generation() {
    let mut p = pool(5, AutoReset::SameStep, None).unwrap();
    let storage = p.outputs().unwrap().storage();
    let region = storage.region(OutputField::PrivilegedState(0)).unwrap();
    assert_eq!(region.len, ROW * 6);
    assert_eq!(storage.region(OutputField::PrivilegedState(1)), None);
    p.reset_all(Reseed::Keep).unwrap();
    let g = storage.generation();
    let before = row(&p, 1);
    // A refused call writes nothing: same generation, same rows.
    let short = [1; 3];
    let refused = p.step(ActionInput {
        discrete: &short,
        continuous: &[],
    });
    assert!(refused.is_err());
    assert_eq!(storage.generation(), g);
    assert_eq!(row(&p, 1), before);
    p.step(ones()).unwrap();
    assert!(storage.generation() > g, "a step starts a new generation");
    // Read in the borrow window of the last call.
    assert_eq!(row(&p, 1), expected(&p.state(1).unwrap()));
}

#[test]
fn test_privileged_state_rows_persist_across_restore_fork_and_import() {
    let mut p = pool(9, AutoReset::SameStep, None).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    for _ in 0..3 {
        p.step(ones()).unwrap();
    }
    let snap = p.snapshot(0).unwrap();
    let bytes = p.encode(&[0]).unwrap();
    p.step(ones()).unwrap();
    p.restore(4, &snap).unwrap();
    assert_eq!(row(&p, 4), expected(snap.state()));
    assert_rows_match(&p);
    p.fork(1, &[2, 3]).unwrap();
    assert_eq!(row(&p, 2), row(&p, 1));
    assert_eq!(row(&p, 3), row(&p, 1));
    assert_rows_match(&p);
    let records = p.decode(&bytes, DecodeLimits::default()).unwrap();
    p.import(&[5], &records).unwrap();
    assert_eq!(row(&p, 5), expected(snap.state()));
    assert_rows_match(&p);
}

#[test]
fn test_privileged_state_without_opt_in_has_no_region() {
    let p = plain().unwrap();
    assert!(!p.has_privileged_state());
    assert!(p.privileged_state_space().is_none());
    let out = p.outputs().unwrap();
    assert_eq!(out.privileged_state(0), None);
    assert_eq!(out.storage().region(OutputField::PrivilegedState(0)), None);
    let with = pool(3, AutoReset::SameStep, None).unwrap();
    assert!(with.has_privileged_state());
    // The derived space is a `Dict` of its leaves, as an observation's.
    assert!(
        matches!(with.privileged_state_space(), Some(Space::Dict(leaves)) if leaves.len() == 1)
    );
    // The other regions keep their sizes.
    let other = with.outputs().unwrap();
    let len = |o: &tessera_rl_engine::Outputs| o.observation(0).map(<[u8]>::len);
    assert_eq!(len(other), len(out));
    assert_eq!(other.masks().len(), out.masks().len());
}

#[test]
fn test_privileged_state_after_reset_or_twice_is_refused() {
    let mut p = plain().unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    assert!(matches!(
        p.with_privileged_state(),
        Err(EngineError::PrivilegedState { .. })
    ));
    let twice = pool(3, AutoReset::SameStep, None)
        .unwrap()
        .with_privileged_state();
    assert!(matches!(twice, Err(EngineError::PrivilegedState { .. })));
}

#[test]
fn test_privileged_state_split_phase_modes_refuse_and_hand_the_pool_back() {
    let w = NonZeroUsize::new(2).unwrap();
    let refused = pool(3, AutoReset::SameStep, None)
        .unwrap()
        .pipelined(PipelinedOptions::new(w));
    let e = refused.err().unwrap();
    assert!(matches!(e.error, EngineError::PrivilegedState { .. }));
    assert!(e.pool.is_some());
    let refused = pool(3, AutoReset::SameStep, None)
        .unwrap()
        .async_mode(AsyncOptions::new(w, NonZeroUsize::new(2).unwrap()));
    let e = refused.err().unwrap();
    assert!(matches!(e.error, EngineError::PrivilegedState { .. }));
    assert!(e.pool.is_some());
}
