// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replay files (ADR-0072 section 10) for Leduc (sequential, hidden information): record then
//! replay with zero divergences, Inline and Sync, with byte-identical files.

#![expect(clippy::unwrap_used, reason = "tests: a failure is the assertion")]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_engine::replay::{Recorder, Replay, ReplayWriterOptions};
use tessera_rl_engine::{
    ActionInput, DecodeLimits, EnvPool, Masked, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_games::{Leduc, LeducConfig};
use tessera_rl_num::identity::Seed;

const N: u32 = 3;
const AGENTS: usize = 2;
const CHOICES: usize = 3;

fn pool(workers: usize) -> EnvPool<Leduc, Masked> {
    let options = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(17));
    let p = EnvPool::with_masks(LeducConfig::default(), options).unwrap();
    if workers == 0 {
        p
    } else {
        p.with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
            .unwrap()
    }
}

/// Every agent of every instance, cycling through the actions; illegal ones
/// are replaced by the no-op under the default policy.
fn actions(t: usize) -> Vec<i32> {
    (0..N as usize * AGENTS)
        .map(|k| i32::try_from((t + 3 * k) % CHOICES).unwrap())
        .collect()
}

fn record(workers: usize) -> (Vec<u8>, Vec<u64>) {
    let mut p = pool(workers);
    let options = ReplayWriterOptions {
        keyframe_every: 16,
        checkpoint_every: 4,
        outputs_hash: false,
    };
    let mut rec = Recorder::new(&mut p, Vec::new(), options).unwrap();
    rec.reset_all(Reseed::Base(Seed::new(5))).unwrap();
    for t in 0..40 {
        rec.step(ActionInput {
            discrete: &actions(t),
            continuous: &[],
        })
        .unwrap();
        if t == 20 {
            rec.fork_branched(0, &[1, 2], None).unwrap();
            rec.reset(&[1], Reseed::List(&[Seed::new(8)])).unwrap();
        }
    }
    let bytes = rec.finish().unwrap();
    let hashes = (0..N).map(|i| p.state_hash(i).unwrap().get()).collect();
    (bytes, hashes)
}

#[test]
fn test_replay_leduc_round_trip_inline_and_sync_has_no_divergence() {
    let (inline, hashes) = record(0);
    let (sync, sync_hashes) = record(2);
    assert_eq!(inline, sync);
    assert_eq!(hashes, sync_hashes);
    for workers in [0, 2] {
        let mut p = pool(workers);
        let summary = Replay::open(&inline, DecodeLimits::default())
            .unwrap()
            .run(&mut p)
            .unwrap();
        assert_eq!(summary.calls, 43);
        assert!(summary.checkpoints >= 10);
        let replayed: Vec<u64> = (0..N).map(|i| p.state_hash(i).unwrap().get()).collect();
        assert_eq!(replayed, hashes);
    }
}
