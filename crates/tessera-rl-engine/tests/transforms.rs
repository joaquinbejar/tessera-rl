// SPDX-License-Identifier: MIT OR Apache-2.0

//! Observation transforms on pools (ADR-0078 and its erratum 1): frame
//! stacks, one-hot rows and running normalization, through every path that
//! observes, the continuation operations, the pool context and the modes.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers unwrap and index: a failure is the assertion"
)]

mod common;

use core::num::{NonZeroU32, NonZeroUsize};

use common::{CountConfig, Counter};
use tessera_rl_core::space::{DType, Space};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AutoReset, EngineError, EnvPool, Masked, PoolOptions, Reseed,
    Schedule, SyncOptions, TransformError, Transforms,
};
use tessera_rl_num::identity::Seed;

type Pool = EnvPool<Counter, Masked>;

/// The name of the counter's one observation leaf.
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

fn built(n: u32, chunk: u32, t: Transforms) -> Pool {
    Pool::with_masks(cfg(), options(n, chunk))
        .unwrap()
        .with_transforms(t)
        .unwrap()
}

fn reset(mut p: Pool) -> Pool {
    p.reset_all(Reseed::Keep).unwrap();
    p
}

/// Steps with action `a` for every agent of every instance.
fn step(p: &mut Pool, a: i32) {
    let n = p.outputs().unwrap().rewards().len();
    let actions = vec![a; n];
    p.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap();
}

/// Leaf 0's row of agent `slot` of instance `i`, as `i32`s.
fn row_i32(p: &Pool, i: usize, slot: usize) -> Vec<i32> {
    let out = p.outputs().unwrap();
    let all = out.observation(0).unwrap();
    let per = all.len() / out.rewards().len();
    let k = i * 2 + slot;
    all[k * per..(k + 1) * per]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_ne_bytes(*b))
        .collect()
}

fn row_f32(p: &Pool, i: usize, slot: usize) -> Vec<f32> {
    row_i32(p, i, slot)
        .into_iter()
        .map(|v| f32::from_bits(v.cast_unsigned()))
        .collect()
}

fn leaf_space(space: &Space) -> &Space {
    match space {
        Space::Dict(entries) => &entries[0].1,
        other => other,
    }
}

#[test]
fn test_transforms_frame_stack_exports_k_frames_oldest_first() {
    let mut p = reset(built(2, 1, Transforms::new().frame_stack(leaf(), 3)));
    let space = p.observation_space().unwrap();
    assert!(
        matches!(leaf_space(&space), Space::Box { shape, dtype: DType::I32, .. } if *shape == [3, 2])
    );
    let v = row_i32(&p, 0, 0)[0];
    // Reset fills the stack with the first observation.
    assert_eq!(row_i32(&p, 0, 0), [v, 0, v, 0, v, 0]);
    step(&mut p, 1);
    assert_eq!(row_i32(&p, 0, 0), [v, 0, v, 0, v + 1, 0]);
    step(&mut p, 1);
    step(&mut p, 1);
    // The ring wrapped: the oldest frame is gone.
    assert_eq!(row_i32(&p, 0, 0), [v + 1, 0, v + 2, 0, v + 3, 0]);
    assert_eq!(p.transform_state_bytes(), 2 * (8 + 3 * 8));
}

#[test]
fn test_transforms_one_hot_rows_and_out_of_range_values() {
    let p = reset(built(1, 1, Transforms::new().one_hot(leaf(), 4)));
    let space = p.observation_space().unwrap();
    assert!(matches!(leaf_space(&space), Space::MultiBinary { shape } if *shape == [2, 4]));
    let out = p.outputs().unwrap();
    let row = &out.observation(0).unwrap()[..8];
    let v = Pool::with_masks(cfg(), options(1, 1)).map(reset).unwrap();
    let value = row_i32(&v, 0, 0)[0];
    let expect_first: Vec<u8> = (0..4).map(|c| u8::from(value == c)).collect();
    assert_eq!(&row[..4], &expect_first[..]);
    // The second element is 0: class 0.
    assert_eq!(&row[4..], &[1, 0, 0, 0]);
}

#[test]
fn test_transforms_normalize_identity_before_barrier_then_statistics() {
    let mut p = reset(built(4, 2, Transforms::new().normalize(leaf(), 1_000.0)));
    let raw = reset(Pool::with_masks(cfg(), options(4, 2)).unwrap());
    let before = row_f32(&p, 0, 0);
    let value = row_i32(&raw, 0, 0)[0];
    #[expect(clippy::cast_precision_loss, reason = "small test values")]
    let expected = value as f32;
    assert!(
        (before[0] - expected).abs() <= expected.abs() * 1e-6,
        "{before:?} vs {value}"
    );
    p.epoch_barrier().unwrap();
    step(&mut p, 0);
    // After the barrier the mean of the reset observations is subtracted.
    let after = row_f32(&p, 0, 0);
    assert_ne!(after[0].to_bits(), before[0].to_bits(), "{after:?}");
    // A constant element: mean 0, variance 0, so 0 / sqrt(1e-8) is 0.
    assert_eq!(after[1], 0.0);
}

#[test]
fn test_transforms_identical_across_workers_and_schedules() {
    let t = Transforms::new()
        .normalize(leaf(), 5.0)
        .frame_stack(leaf(), 2);
    let run = |workers: usize, schedule: Schedule| {
        let mut o = options(13, 2);
        o.schedule = schedule;
        let mut p = Pool::with_masks(cfg(), o)
            .unwrap()
            .with_transforms(t.clone())
            .unwrap();
        if workers > 0 {
            p = p
                .with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
                .unwrap();
        }
        p.reset_all(Reseed::Keep).unwrap();
        let mut seen = Vec::new();
        for s in 0..6 {
            step(&mut p, s % 2);
            if s % 2 == 1 {
                p.epoch_barrier().unwrap();
            }
            seen.push(p.outputs().unwrap().observation(0).unwrap().to_vec());
        }
        (seen, p.pool_context().unwrap())
    };
    let inline = run(0, Schedule::Static);
    for (w, s) in [
        (1, Schedule::Static),
        (2, Schedule::Static),
        (7, Schedule::Static),
        (3, Schedule::Dynamic),
    ] {
        assert_eq!(run(w, s), inline, "{w} workers, {s:?}");
    }
}

#[test]
fn test_transforms_identity_depends_on_chunk_size_only_with_normalize() {
    let stack =
        |chunk| built(4, chunk, Transforms::new().frame_stack(leaf(), 2)).transform_identity();
    let norm =
        |chunk| built(4, chunk, Transforms::new().normalize(leaf(), 5.0)).transform_identity();
    assert_eq!(stack(1), stack(2));
    assert_ne!(norm(1), norm(2));
    assert_eq!(built(4, 1, Transforms::new()).transform_identity(), 0);
}

#[test]
fn test_transforms_fork_then_step_equals_an_unforked_step() {
    let mut p = reset(built(2, 1, Transforms::new().frame_stack(leaf(), 3)));
    step(&mut p, 1);
    step(&mut p, 1);
    p.fork(0, &[1]).unwrap();
    // Keep mode: the fork reads the stack, no newest frame twice.
    assert_eq!(row_i32(&p, 1, 0), row_i32(&p, 0, 0));
    let stack = row_i32(&p, 0, 0);
    assert_ne!(stack[2..4], stack[4..6], "the two newest frames differ");
    step(&mut p, 1);
    assert_eq!(row_i32(&p, 1, 0), row_i32(&p, 0, 0));
    assert_eq!(row_i32(&p, 1, 1), row_i32(&p, 0, 1));
}

#[test]
fn test_transforms_restore_with_transforms_continues_and_plain_restore_refills() {
    let mut p = reset(built(1, 1, Transforms::new().frame_stack(leaf(), 3)));
    step(&mut p, 1);
    step(&mut p, 0);
    let record = p.snapshot(0).unwrap();
    let mut bytes = vec![0; p.transform_state_bytes()];
    p.snapshot_transforms(0, &mut bytes).unwrap();
    let kept = row_i32(&p, 0, 0);
    step(&mut p, 1);
    let next = row_i32(&p, 0, 0);
    p.restore_with_transforms(0, &record, &bytes).unwrap();
    assert_eq!(row_i32(&p, 0, 0), kept);
    step(&mut p, 1);
    assert_eq!(row_i32(&p, 0, 0), next, "an exact continuation");
    // A plain restore refills the stack from the restored observation.
    p.restore(0, &record).unwrap();
    let r = row_i32(&p, 0, 0);
    assert_eq!(r[..2], r[2..4]);
    assert_eq!(r[2..4], r[4..6]);
    assert_eq!(r[4..6], kept[4..6]);
    // A wrong length is refused before anything is written.
    assert!(matches!(
        p.restore_with_transforms(0, &record, &bytes[1..]),
        Err(EngineError::Transform(TransformError::StateLength { .. }))
    ));
}

#[test]
fn test_transforms_pool_snapshot_with_stacks_continues_exactly() {
    let mut p = reset(built(3, 2, Transforms::new().frame_stack(leaf(), 2)));
    step(&mut p, 1);
    let records: Vec<_> = (0..3).map(|i| p.snapshot(i).unwrap()).collect();
    let mut bytes = vec![0; 3 * p.transform_state_bytes()];
    p.snapshot_transforms_all(&mut bytes).unwrap();
    step(&mut p, 0);
    let want = p.outputs().unwrap().observation(0).unwrap().to_vec();
    p.restore_all_with_transforms(&records, &bytes).unwrap();
    step(&mut p, 0);
    assert_eq!(p.outputs().unwrap().observation(0).unwrap(), &want[..]);
}

#[test]
fn test_transforms_restore_then_barrier_leaves_statistics_unchanged() {
    let mut p = reset(built(2, 1, Transforms::new().normalize(leaf(), 5.0)));
    step(&mut p, 1);
    p.epoch_barrier().unwrap();
    let before = p.pool_context().unwrap();
    let record = p.snapshot(0).unwrap();
    // Re-observations never accumulate (rev #346 choice 3).
    p.restore(0, &record).unwrap();
    p.fork(0, &[1]).unwrap();
    p.epoch_barrier().unwrap();
    assert_eq!(p.pool_context().unwrap(), before);
}

#[test]
fn test_transforms_freeze_stops_accumulation() {
    let mut p = reset(built(2, 1, Transforms::new().normalize(leaf(), 5.0)));
    p.epoch_barrier().unwrap();
    let before = p.pool_context().unwrap();
    p.freeze_normalization(true);
    step(&mut p, 1);
    step(&mut p, 0);
    p.epoch_barrier().unwrap();
    let mut after = p.pool_context().unwrap();
    assert!(after.transforms.frozen);
    after.transforms.frozen = false;
    assert_eq!(after.transforms, before.transforms);
}

#[test]
fn test_transforms_pool_context_restores_statistics_and_refuses_another_identity() {
    let mut a = reset(built(2, 1, Transforms::new().normalize(leaf(), 5.0)));
    step(&mut a, 1);
    a.epoch_barrier().unwrap();
    a.freeze_normalization(true);
    let context = a.pool_context().unwrap();
    assert!(context.transforms.frozen);
    let mut b = reset(built(2, 1, Transforms::new().normalize(leaf(), 5.0)));
    b.restore_pool_context(context.clone(), None).unwrap();
    assert_eq!(b.pool_context().unwrap(), context);
    assert!(
        b.normalization_frozen(),
        "the freeze travels with the context"
    );
    let mut other = reset(built(2, 1, Transforms::new().normalize(leaf(), 4.0)));
    assert!(matches!(
        other.restore_pool_context(context.clone(), None),
        Err(EngineError::Transform(
            TransformError::IdentityMismatch { .. }
        ))
    ));
    let mut plain = reset(Pool::with_masks(cfg(), options(2, 1)).unwrap());
    assert!(plain.restore_pool_context(context, None).is_err());
}

#[test]
fn test_transforms_same_step_terminal_row_keeps_the_ended_episode() {
    let mut o = options(1, 1);
    o.auto_reset = AutoReset::SameStep;
    let c = CountConfig { length: 2, ..cfg() };
    let mut p = Pool::with_masks(c, o)
        .unwrap()
        .with_transforms(Transforms::new().frame_stack(leaf(), 2))
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let v = row_i32(&p, 0, 0)[0];
    step(&mut p, 1);
    step(&mut p, 1);
    let out = p.outputs().unwrap();
    let term: Vec<i32> = out.terminal_observation(0).unwrap()[..16]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_ne_bytes(*b))
        .collect();
    assert_eq!(term, [v + 1, 0, v + 2, 0]);
    // The new episode's stack is refilled.
    let fresh = row_i32(&p, 0, 0);
    assert_eq!(fresh[..2], fresh[2..4]);
}

#[test]
fn test_transforms_dead_agent_rows_are_zero_and_respawn_refills() {
    let c = CountConfig {
        kill_at: 3,
        respawn: true,
        ..cfg()
    };
    let mut p = Pool::with_masks(c, options(1, 1))
        .unwrap()
        .with_transforms(Transforms::new().frame_stack(leaf(), 3))
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let mut died = false;
    for _ in 0..12 {
        step(&mut p, 1);
        let alive = p.outputs().unwrap().alive()[1].get();
        if !alive {
            died = true;
            assert!(row_i32(&p, 0, 1).iter().all(|&x| x == 0));
            // The dead agent's frames are gone from the arena too.
            let mut bytes = vec![0; p.transform_state_bytes()];
            p.snapshot_transforms(0, &mut bytes).unwrap();
            let half = bytes.len() / 2;
            assert!(
                bytes[half..].iter().all(|&b| b == 0),
                "agent 1's arena slice"
            );
            assert!(
                bytes[..half].iter().any(|&b| b != 0),
                "agent 0 keeps its frames"
            );
        } else if died {
            // The respawned agent sees only its own frames.
            let r = row_i32(&p, 0, 1);
            assert_eq!(r[..2], r[2..4]);
            return;
        }
    }
    panic!("agent 1 did not die and respawn");
}

#[test]
fn test_transforms_refusals() {
    let t = |x: Transforms| {
        Pool::with_masks(cfg(), options(2, 1))
            .unwrap()
            .with_transforms(x)
            .err()
    };
    assert!(matches!(
        t(Transforms::new().frame_stack("nope", 2)),
        Some(EngineError::Transform(TransformError::UnknownLeaf { .. }))
    ));
    assert!(matches!(
        t(Transforms::new().frame_stack(leaf(), 1)),
        Some(EngineError::Transform(TransformError::StackDepth { k: 1 }))
    ));
    assert!(matches!(
        t(Transforms::new().frame_stack(leaf(), 65)),
        Some(EngineError::Transform(TransformError::StackDepth { k: 65 }))
    ));
    assert!(matches!(
        t(Transforms::new().normalize(leaf(), 0.0)),
        Some(EngineError::Transform(TransformError::Clip { .. }))
    ));
    assert!(matches!(
        t(Transforms::new()
            .normalize(leaf(), 1.0)
            .normalize(leaf(), 1.0)),
        Some(EngineError::Transform(
            TransformError::NormalizeTwice { .. }
        ))
    ));
    assert!(matches!(
        t(Transforms::new()
            .frame_stack(leaf(), 2)
            .normalize(leaf(), 1.0)),
        Some(EngineError::Transform(
            TransformError::NormalizeAfterStack { .. }
        ))
    ));
    assert!(matches!(
        t(Transforms::new().one_hot(leaf(), 0)),
        Some(EngineError::Transform(TransformError::Unsupported { .. }))
    ));
    // After a reset, or twice.
    let p = reset(Pool::with_masks(cfg(), options(2, 1)).unwrap());
    assert!(matches!(
        p.with_transforms(Transforms::new().frame_stack(leaf(), 2)),
        Err(EngineError::Transform(TransformError::AfterReset))
    ));
    let p = built(2, 1, Transforms::new().frame_stack(leaf(), 2));
    assert!(matches!(
        p.with_transforms(Transforms::new().frame_stack(leaf(), 2)),
        Err(EngineError::Transform(TransformError::AfterReset))
    ));
}

#[test]
fn test_transforms_encode_is_refused() {
    let p = reset(built(1, 1, Transforms::new().frame_stack(leaf(), 2)));
    assert!(matches!(
        p.encode(&[0]),
        Err(tessera_rl_engine::PersistError::Transforms)
    ));
}

#[test]
fn test_transforms_async_refuses_unfrozen_normalize_and_runs_frozen() {
    let o = || AsyncOptions::new(NonZeroUsize::new(2).unwrap(), NonZeroUsize::new(2).unwrap());
    let refused = built(4, 1, Transforms::new().normalize(leaf(), 5.0)).async_mode(o());
    assert!(matches!(
        refused.err().map(|e| e.error),
        Some(EngineError::Transform(TransformError::NormalizeInAsync))
    ));
    let mut frozen = built(4, 1, Transforms::new().normalize(leaf(), 5.0));
    frozen.freeze_normalization(true);
    let mut a = frozen.async_mode(o()).unwrap();
    a.reset_all(Reseed::Keep).unwrap();
    let stacked = built(4, 1, Transforms::new().frame_stack(leaf(), 2)).async_mode(o());
    assert!(stacked.is_ok());
}

#[test]
fn test_transforms_pipelined_matches_sync_with_stacks_and_statistics() {
    use tessera_rl_engine::PipelinedOptions;
    let t = Transforms::new()
        .normalize(leaf(), 5.0)
        .frame_stack(leaf(), 2);
    let mut reference = reset(built(8, 2, t.clone()));
    let mut p = built(8, 2, t)
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
        .map_err(|e| e.error)
        .unwrap();
    let halves = p.halves();
    let [mut ta, mut tb] = p.reset_all(Reseed::Keep).unwrap();
    for s in 0..6 {
        let a = vec![s % 2; 16];
        reference
            .step(ActionInput {
                discrete: &a,
                continuous: &[],
            })
            .unwrap();
        for (token, half) in [(ta, &halves[0]), (tb, &halves[1])] {
            let rows =
                usize::try_from(half.start).unwrap() * 2..usize::try_from(half.end).unwrap() * 2;
            p.send(
                token,
                ActionInput {
                    discrete: &a[rows],
                    continuous: &[],
                },
            )
            .map_err(|e| e.error)
            .unwrap();
        }
        ta = p.recv().unwrap();
        tb = p.recv().unwrap();
    }
    drop((ta, tb));
    let mut pool = p.into_pool().map_err(|e| e.kind).unwrap();
    assert_eq!(
        pool.outputs().unwrap().observation(0).unwrap(),
        reference.outputs().unwrap().observation(0).unwrap()
    );
    pool.epoch_barrier().unwrap();
    reference.epoch_barrier().unwrap();
    assert_eq!(
        pool.pool_context().unwrap(),
        reference.pool_context().unwrap()
    );
}

/// A frozen Normalize in Async mode with runs of claimed ids that split
/// chunks across workers (REVIEW-headA-r1): no accumulator view is handed
/// out, and the outputs equal a frozen Sync pool's. 9 instances in chunks
/// of 4 and batches of 3: every batch is reachable. Run under Miri in CI.
#[test]
fn test_transforms_async_frozen_normalize_with_split_chunks_matches_sync() {
    let t = Transforms::new()
        .normalize(leaf(), 5.0)
        .frame_stack(leaf(), 2);
    let mut reference = built(9, 4, t.clone());
    reference.freeze_normalization(true);
    let mut reference = reset(reference);
    let mut frozen = built(9, 4, t);
    frozen.freeze_normalization(true);
    let mut a = frozen
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(3).unwrap(),
        ))
        .unwrap();
    drop(a.reset_all(Reseed::Keep).unwrap());
    let ids: Vec<u32> = (0..9).collect();
    for s in 0..3 {
        let actions = vec![s % 2; 18];
        reference
            .step(ActionInput {
                discrete: &actions,
                continuous: &[],
            })
            .unwrap();
        a.send(
            &ids,
            ActionInput {
                discrete: &actions,
                continuous: &[],
            },
        )
        .unwrap();
        let mut got = 0;
        while got < ids.len() {
            let token = a.recv().unwrap();
            got += token.ids().count();
        }
    }
    let Ok(pool) = a.into_pool() else {
        panic!("the Async pool is quiescent");
    };
    assert_eq!(
        pool.outputs().unwrap().observation(0).unwrap(),
        reference.outputs().unwrap().observation(0).unwrap()
    );
    assert_eq!(
        pool.pool_context().unwrap(),
        reference.pool_context().unwrap()
    );
}
