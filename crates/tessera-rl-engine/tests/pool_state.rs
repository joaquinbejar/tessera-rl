// SPDX-License-Identifier: MIT OR Apache-2.0

//! Whole-pool snapshots with their context and `restore_pool` (ADR-0075
//! item 7, ADR-0078 item 6, #465): records, frame stacks, branch counter
//! and normalization statistics in every mode, checked before any write.
//! Archive contexts are covered in the classic `archive_resets` tests.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers unwrap and index: a failure is the assertion"
)]

mod common;

use core::num::{NonZeroU32, NonZeroUsize};

use common::{CountConfig, Counter};
use tessera_rl_core::StateHash;
use tessera_rl_core::space::Space;
use tessera_rl_engine::{
    ActionInput, AsyncOptions, EngineError, EnvPool, Masked, PipelinedOptions, PoolContext,
    PoolOptions, PoolState, Reseed, TransformError, Transforms,
};
use tessera_rl_num::identity::Seed;

type Pool = EnvPool<Counter, Masked>;

fn leaf() -> String {
    let space = Pool::with_masks(cfg(), options(1, 1))
        .unwrap()
        .observation_space()
        .unwrap();
    match space {
        Space::Dict(entries) => entries[0].0.clone(),
        _ => String::new(),
    }
}

fn cfg() -> CountConfig {
    CountConfig {
        length: 1000,
        ignore_rng: true,
        ..CountConfig::default()
    }
}

fn options(n: u32, chunk: u32) -> PoolOptions {
    PoolOptions {
        chunk_size: NonZeroU32::new(chunk).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(7))
    }
}

/// A `Normalize` and a `FrameStack` on the counter's leaf.
fn stacked() -> Transforms {
    Transforms::new()
        .normalize(leaf(), 5.0)
        .frame_stack(leaf(), 2)
}

fn built(n: u32, chunk: u32, t: Transforms) -> Pool {
    let mut p = Pool::with_masks(cfg(), options(n, chunk))
        .unwrap()
        .with_transforms(t)
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn step(p: &mut Pool, a: i32) {
    let n = p.outputs().unwrap().rewards().len();
    let actions = vec![a; n];
    p.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap();
}

/// What a pool snapshot holds, comparable.
fn print(state: &PoolState<Counter>) -> (Vec<StateHash>, Vec<u8>, PoolContext) {
    (
        state.records().iter().map(tessera_rl_engine::Record::hash).collect(),
        state.transforms().to_vec(),
        state.context().clone(),
    )
}

fn observation(p: &Pool) -> Vec<u8> {
    p.outputs().unwrap().observation(0).unwrap().to_vec()
}

#[test]
fn test_pool_state_sync_round_trip_continues_exactly() {
    let mut p = built(4, 2, stacked());
    step(&mut p, 1);
    p.epoch_barrier().unwrap();
    step(&mut p, 0);
    p.fork_branched(0, &[1], None).unwrap();
    let state = p.snapshot_pool().unwrap();
    assert_ne!(state.transforms(), []);
    step(&mut p, 1);
    let want = observation(&p);
    // Move every part of the pool on: records, stacks, statistics, freeze
    // and the branch counter.
    step(&mut p, 0);
    p.epoch_barrier().unwrap();
    p.freeze_normalization(true);
    p.fork_branched(2, &[3], None).unwrap();
    assert_ne!(p.pool_context().unwrap(), *state.context());
    p.restore_pool(&state, None).unwrap();
    assert_eq!(print(&p.snapshot_pool().unwrap()), print(&state));
    assert!(!p.normalization_frozen(), "the freeze is restored too");
    step(&mut p, 1);
    assert_eq!(observation(&p), want);
}

#[test]
fn test_pool_state_refusals_mutate_nothing() {
    let mut p = built(4, 2, stacked());
    step(&mut p, 1);
    p.epoch_barrier().unwrap();
    let before = print(&p.snapshot_pool().unwrap());
    let obs = observation(&p);

    // Another transform identity (ADR-0078 item 2.3).
    let other = built(
        4,
        2,
        Transforms::new()
            .normalize(leaf(), 4.0)
            .frame_stack(leaf(), 2),
    );
    assert!(matches!(
        p.restore_pool(&other.snapshot_pool().unwrap(), None),
        Err(EngineError::Transform(
            TransformError::IdentityMismatch { .. }
        ))
    ));
    // Another instance count.
    let small = built(2, 2, stacked());
    assert!(matches!(
        p.restore_pool(&small.snapshot_pool().unwrap(), None),
        Err(EngineError::RecordCount { .. })
    ));
    // No transforms at all.
    let mut plain = Pool::with_masks(cfg(), options(4, 2)).unwrap();
    plain.reset_all(Reseed::Keep).unwrap();
    assert!(
        p.restore_pool(&plain.snapshot_pool().unwrap(), None)
            .is_err()
    );

    assert_eq!(print(&p.snapshot_pool().unwrap()), before);
    assert_eq!(observation(&p), obs);
}

#[test]
fn test_pool_state_pipelined_round_trip() {
    let mut p = built(8, 2, stacked())
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
        .map_err(|e| e.error)
        .unwrap();
    drop(p.reset_all(Reseed::Keep).unwrap());
    let (state, cycles) = p.snapshot_pool().unwrap();
    drop(p.reset_all(Reseed::Base(Seed::new(99))).unwrap());
    p.epoch_barrier().unwrap();
    p.freeze_normalization(true).unwrap();
    assert_ne!(print(&p.snapshot_pool().unwrap().0), print(&state));
    p.restore_pool(&state, &cycles, None).unwrap();
    let (again, again_cycles) = p.snapshot_pool().unwrap();
    assert_eq!(print(&again), print(&state));
    assert_eq!(again_cycles, cycles);
}

#[test]
fn test_pool_state_async_round_trip() {
    let mut frozen = built(6, 2, stacked());
    frozen.freeze_normalization(true);
    let mut a = frozen
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(3).unwrap(),
        ))
        .unwrap();
    drop(a.reset_all(Reseed::Keep).unwrap());
    let state = a.snapshot_pool().unwrap();
    drop(a.reset_all(Reseed::Base(Seed::new(99))).unwrap());
    assert_ne!(print(&a.snapshot_pool().unwrap()), print(&state));
    a.restore_pool(&state, None).unwrap();
    assert_eq!(print(&a.snapshot_pool().unwrap()), print(&state));
}
