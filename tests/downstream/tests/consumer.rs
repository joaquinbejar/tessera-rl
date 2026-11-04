// SPDX-License-Identifier: MIT OR Apache-2.0

//! The API a downstream adapter uses, exercised from outside the workspace's
//! inner crates: `Environment`, `EnvPool` (inline and with workers),
//! snapshot, restore, fork and branched fork, with discrete actions and one
//! agent. Every comparison is bit-exact.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl::engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, NoMasks, PoolOptions, Record,
    Reseed, SyncOptions,
};
use tessera_rl::num::identity::Seed;
use tessera_rl_downstream_test::{Quote, QuoteConfig};

type Pool = EnvPool<Quote, NoMasks>;

const N: u32 = 8;
const HEADS: usize = 2;

fn options(seed: u64) -> PoolOptions {
    PoolOptions {
        auto_reset: AutoReset::SameStep,
        chunk_size: NonZeroU32::new(2).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(seed))
    }
}

fn pool(seed: u64, workers: Option<usize>) -> Pool {
    let mut p = EnvPool::new(QuoteConfig::default(), options(seed)).unwrap();
    if let Some(w) = workers {
        p = p
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap();
    }
    p.reset_all(Reseed::Keep).unwrap();
    p
}

/// A fixed, instance-dependent action schedule: bid and ask offsets.
fn actions(t: usize) -> Vec<i32> {
    (0..N as usize)
        .flat_map(|i| {
            let bid = i32::try_from((t + i) % 3).unwrap();
            let ask = i32::try_from((t * 7 + i * 5) % 3).unwrap();
            [bid, ask]
        })
        .collect()
}

/// Everything compared after one step, floats as bits.
#[derive(Debug, PartialEq, Eq)]
struct Frame {
    obs: Vec<u8>,
    rewards: Vec<u32>,
    terminated: Vec<bool>,
    hashes: Vec<u64>,
}

fn step(p: &mut Pool, t: usize) -> Frame {
    let a = actions(t);
    assert_eq!(a.len(), N as usize * HEADS);
    let out = p
        .step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
    let frame = Frame {
        obs: out.observation(0).unwrap().to_vec(),
        rewards: out.rewards().iter().map(|r| r.to_bits()).collect(),
        terminated: out.episode_terminated().iter().map(|b| b.get()).collect(),
        hashes: Vec::new(),
    };
    Frame {
        hashes: (0..N).map(|i| p.state_hash(i).unwrap().get()).collect(),
        ..frame
    }
}

fn run(p: &mut Pool, from: usize, steps: usize) -> Vec<Frame> {
    (from..from + steps).map(|t| step(p, t)).collect()
}

#[test]
fn test_downstream_same_seed_replays_and_another_seed_differs() {
    let a = run(&mut pool(7, None), 0, 100);
    assert_eq!(a, run(&mut pool(7, None), 0, 100));
    assert_ne!(a, run(&mut pool(8, None), 0, 100));
    // The 32-step episodes auto-reset within the run.
    assert!(a.iter().any(|f| f.terminated.iter().any(|&t| t)));
}

#[test]
fn test_downstream_workers_match_inline_bit_exact() {
    let inline = run(&mut pool(7, None), 0, 100);
    for workers in [1, 2, 4] {
        assert_eq!(
            run(&mut pool(7, Some(workers)), 0, 100),
            inline,
            "{workers} workers"
        );
    }
}

#[test]
fn test_downstream_restore_continues_exactly() {
    let mut p = pool(7, None);
    let _ = run(&mut p, 0, 10);
    let mut records = vec![p.snapshot(0).unwrap(); N as usize];
    p.snapshot_all(&mut records).unwrap();
    let context = p.pool_context().unwrap();
    let first = run(&mut p, 10, 60);
    p.restore_all(&records).unwrap();
    p.restore_pool_context(context).unwrap();
    assert_eq!(run(&mut p, 10, 60), first);
}

#[test]
fn test_downstream_restore_into_another_pool_continues_exactly() {
    let mut p = pool(7, None);
    let _ = run(&mut p, 0, 10);
    let records: Vec<Record<Quote>> = (0..N).map(|i| p.snapshot(i).unwrap()).collect();
    let context = p.pool_context().unwrap();
    let first = run(&mut p, 10, 60);
    // A fresh pool with another seed: the records carry every instance's
    // identity, so the seed of the pool does not matter.
    let mut q = pool(99, Some(2));
    q.restore_all(&records).unwrap();
    q.restore_pool_context(context).unwrap();
    assert_eq!(run(&mut q, 10, 60), first);
}

#[test]
fn test_downstream_fork_copies_the_future_and_a_branched_fork_diverges() {
    let mut p = pool(7, None);
    let _ = run(&mut p, 0, 5);
    // Instances 1..4 become copies of 0, 5..8 branched copies of 0.
    p.fork(0, &[1, 2, 3]).unwrap();
    p.fork_branched(0, &[4, 5, 6, 7], None).unwrap();
    for i in 1..4 {
        assert_eq!(p.state_hash(i).unwrap(), p.state_hash(0).unwrap());
    }
    // Same actions for every instance from here on: the plain copies have
    // exactly the source's future, the branched ones their own.
    let same = |p: &mut Pool| {
        let a = vec![1i32; N as usize * HEADS];
        p.step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
        (0..N).map(|i| p.state_hash(i).unwrap()).collect::<Vec<_>>()
    };
    let mut diverged = false;
    for _ in 0..20 {
        let h = same(&mut p);
        assert!(h[1..4].iter().all(|x| *x == h[0]));
        diverged |= h[4..].iter().any(|x| *x != h[0]);
    }
    assert!(diverged, "a branched fork draws its own randomness");
}

#[test]
fn test_downstream_reward_is_the_exact_change_in_marked_value() {
    let mut p = pool(7, None);
    for t in 0..31 {
        let before: Vec<i64> = (0..N)
            .map(|i| p.snapshot(i).unwrap().state().marked_value())
            .collect();
        let frame = step(&mut p, t);
        for (i, before) in before.iter().enumerate() {
            let after = p
                .snapshot(u32::try_from(i).unwrap())
                .unwrap()
                .state()
                .marked_value();
            let reward = f32::from_bits(frame.rewards[i]);
            #[expect(clippy::cast_precision_loss, reason = "small exact integers")]
            let expected = (after - before) as f32;
            assert_eq!(
                reward.to_bits(),
                expected.to_bits(),
                "instance {i} at step {t}"
            );
        }
    }
}

#[test]
fn test_downstream_import_validates_the_inventory_bound_including_i32_min() {
    let mut p = pool(7, None);
    let _ = run(&mut p, 0, 3);
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let before = p.state_hash(1).unwrap();
    for inventory in [11, -11, i32::MIN] {
        let mut state = *good.state();
        state.inventory = inventory;
        let bad = DecodedRecord::from_parts(*good.header(), state).unwrap();
        assert!(
            matches!(
                p.import(&[1], &[bad]),
                Err(EngineError::Invariant { index: 0, .. })
            ),
            "inventory {inventory} is refused"
        );
        assert_eq!(p.state_hash(1).unwrap(), before, "nothing is written");
    }
    p.import(&[1], &[good]).unwrap();
    assert_eq!(p.state_hash(1).unwrap(), p.state_hash(0).unwrap());
}
