// SPDX-License-Identifier: MIT OR Apache-2.0

//! Snapshot, restore and fork (ADR-0010, ADR-0007), including the
//! normative continuation test: same slot, another slot, another pool.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers unwrap and index: a failure is the assertion"
)]

mod common;

use core::num::NonZeroU32;

use common::{CountConfig, Counter};
use tessera_rl_core::Bool8;
use tessera_rl_core::encoding::{decode_from_slice, encode_to_vec};
use tessera_rl_engine::{
    ActionInput, EngineError, EnvPool, Masked, PoolContext, PoolOptions, Record, Reseed,
};
use tessera_rl_num::identity::Seed;

type Pool = EnvPool<Counter, Masked>;

fn pool(n: u32, seed: u64) -> Pool {
    let cfg = CountConfig {
        length: 50,
        ..CountConfig::default()
    };
    let mut p = EnvPool::with_masks(
        cfg,
        PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

/// Everything the continuation test compares for instance `i`.
#[derive(Debug, PartialEq)]
struct Frame {
    obs: Vec<u8>,
    rewards: Vec<f32>,
    flags: Vec<Bool8>,
    masks: Vec<Bool8>,
    generations: Vec<u32>,
    hash: u64,
}

fn frame(p: &Pool, i: usize) -> Frame {
    let out = p.outputs();
    let rows = |len: usize| i * len..(i + 1) * len;
    let mask_len = out.masks().len() / out.rewards().len();
    let mut flags = out.agent_terminated()[rows(2)].to_vec();
    flags.extend_from_slice(&out.agent_truncated()[rows(2)]);
    flags.extend_from_slice(&out.alive()[rows(2)]);
    flags.push(out.episode_terminated()[i]);
    flags.push(out.episode_truncated()[i]);
    Frame {
        obs: out.observation(0).unwrap()[rows(16)].to_vec(),
        rewards: out.rewards()[rows(2)].to_vec(),
        flags,
        masks: out.masks()[rows(2 * mask_len)].to_vec(),
        generations: out.generations()[rows(2)].to_vec(),
        hash: p.state_hash(u32::try_from(i).unwrap()).unwrap().get(),
    }
}

/// Runs `steps` steps where instance `i` takes `script[t]` (other
/// instances act 0) and returns instance `i`'s frames.
fn run(p: &mut Pool, i: usize, script: &[[i32; 2]]) -> Vec<Frame> {
    let n = usize::try_from(p.options().num_envs.get()).unwrap();
    script
        .iter()
        .map(|act| {
            let mut actions = vec![0; n * 2];
            actions[i * 2..i * 2 + 2].copy_from_slice(act);
            p.step(ActionInput {
                discrete: &actions,
                continuous: &[],
            })
            .unwrap();
            frame(p, i)
        })
        .collect()
}

/// Legal for both agents (action 2 is illegal for agent 0), so the script
/// also runs under `debug-checks`.
const SCRIPT: [[i32; 2]; 6] = [[1, 0], [0, 2], [1, 1], [1, 2], [0, 0], [0, 2]];

#[test]
fn test_continuation_same_slot_other_slot_other_pool() {
    let mut a = pool(3, 1);
    run(&mut a, 0, &SCRIPT[..2]);
    let snap = a.snapshot(0).unwrap();
    let expected = run(&mut a, 0, &SCRIPT);

    a.restore(0, &snap).unwrap();
    assert_eq!(run(&mut a, 0, &SCRIPT), expected, "same slot");

    a.restore(2, &snap).unwrap();
    assert_eq!(run(&mut a, 2, &SCRIPT), expected, "other slot");

    let mut b = pool(4, 999);
    b.restore(1, &snap).unwrap();
    assert_eq!(run(&mut b, 1, &SCRIPT), expected, "other pool, other seed");
}

#[test]
fn test_restore_recomputes_observations_and_masks() {
    let mut a = pool(2, 3);
    run(&mut a, 0, &SCRIPT[..1]);
    let before = frame(&a, 0);
    let snap = a.snapshot(0).unwrap();
    a.restore(1, &snap).unwrap();
    let copied = frame(&a, 1);
    assert_eq!(
        (copied.obs, copied.masks, copied.hash),
        (before.obs, before.masks, before.hash)
    );
    // Flags of the copy are cleared: it has not stepped yet.
    assert_eq!(a.outputs().rewards()[2..4], [0.0, 0.0]);
}

#[test]
fn test_fork_gives_identical_futures() {
    let mut a = pool(3, 5);
    run(&mut a, 0, &SCRIPT[..2]);
    a.fork(0, &[1, 2]).unwrap();
    let act = [1, 2, 1, 2, 1, 2];
    for _ in 0..4 {
        a.step(ActionInput {
            discrete: &act,
            continuous: &[],
        })
        .unwrap();
        let h: Vec<u64> = (0..3).map(|i| a.state_hash(i).unwrap().get()).collect();
        assert!(h.iter().all(|&x| x == h[0]));
    }
}

#[test]
fn test_fork_branched_diverges_deterministically() {
    let branch = |seed| {
        let mut p = pool(3, seed);
        p.fork_branched(0, &[1, 2], None).unwrap();
        let act = [1; 6];
        let mut hashes = Vec::new();
        for _ in 0..10 {
            p.step(ActionInput {
                discrete: &act,
                continuous: &[],
            })
            .unwrap();
            hashes.push(
                (0..3)
                    .map(|i| p.state_hash(i).unwrap().get())
                    .collect::<Vec<_>>(),
            );
        }
        (hashes, p.pool_context().unwrap())
    };
    let (a, ctx) = branch(7);
    let (b, _) = branch(7);
    assert_eq!(a, b, "deterministic");
    assert_eq!(ctx, PoolContext { branch_counter: 1 });
    let last = a.last().unwrap();
    assert!(last[0] != last[1] && last[1] != last[2], "branches diverge");
}

#[test]
fn test_fork_branched_labels_and_counter_errors() {
    let mut p = pool(3, 1);
    let before = p.state_hash(1).unwrap();
    assert_eq!(
        p.fork_branched(0, &[1, 2], Some(&[4, 4])),
        Err(EngineError::DuplicateLabel { label: 4 })
    );
    assert_eq!(
        p.fork_branched(0, &[1, 2], Some(&[4])),
        Err(EngineError::LabelCount {
            labels: 1,
            destinations: 2
        })
    );
    p.restore_pool_context(PoolContext {
        branch_counter: u64::MAX,
    })
    .unwrap();
    assert_eq!(
        p.fork_branched(0, &[1], None),
        Err(EngineError::BranchCounterExhausted)
    );
    assert_eq!(p.state_hash(1).unwrap(), before, "nothing mutated");
    assert_eq!(
        p.fork(0, &[1, 1]),
        Err(EngineError::DuplicateInstance { id: 1 })
    );
}

#[test]
fn test_poisoned_pool_accepts_only_full_restore() {
    let p = pool(2, 1);
    let mut records = [p.snapshot(0).unwrap(); 2];
    p.snapshot_all(&mut records).unwrap();
    let cfg = CountConfig {
        panic_at: 1,
        ..CountConfig::default()
    };
    let mut bad = EnvPool::<Counter, Masked>::with_masks(
        cfg,
        PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(1)),
    )
    .unwrap();
    bad.reset_all(Reseed::Keep).unwrap();
    assert!(
        bad.step(ActionInput {
            discrete: &[0; 4],
            continuous: &[]
        })
        .is_err()
    );
    assert_eq!(bad.restore(0, &records[0]), Err(EngineError::Poisoned));
    assert_eq!(bad.fork(0, &[1]), Err(EngineError::Poisoned));
    assert_eq!(bad.fork_branched(0, &[1], None), Err(EngineError::Poisoned));
    assert_eq!(bad.snapshot(0).map(|_| ()), Err(EngineError::Poisoned));
    assert_eq!(bad.state_hash(0).map(|_| ()), Err(EngineError::Poisoned));
    assert_eq!(
        bad.snapshot_all(&mut records.clone()),
        Err(EngineError::Poisoned)
    );
    assert_eq!(bad.pool_context(), Err(EngineError::Poisoned));
    assert_eq!(
        bad.restore_pool_context(PoolContext { branch_counter: 0 }),
        Err(EngineError::Poisoned)
    );
    assert_eq!(
        bad.restore_all(&records[..1]),
        Err(EngineError::RecordCount {
            records: 1,
            instances: 2
        })
    );
    bad.restore_all(&records).unwrap();
    assert!(!bad.is_poisoned());
    assert_eq!(bad.state_hash(0).unwrap(), p.state_hash(0).unwrap());
}

#[test]
fn test_record_encoding_round_trips_and_matches_state_hash() {
    let mut p = pool(1, 2);
    run(&mut p, 0, &SCRIPT[..3]);
    let record = p.snapshot(0).unwrap();
    let bytes = encode_to_vec(&record);
    let decoded: Record<Counter> = decode_from_slice(&bytes).unwrap();
    assert_eq!(decoded.hash(), p.state_hash(0).unwrap());
    assert_eq!(Record::<Counter>::record_size(), 40 + 2 * 16 + 16);
}
