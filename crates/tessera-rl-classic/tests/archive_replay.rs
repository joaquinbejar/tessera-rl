// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replay of archive resets (ADR-0075 item 7, the `ResetSource` and
//! `ResetFrom` events of replay format version 1): a recording of a pool
//! that resets from an archive replays with no divergence given the same
//! archive, and is refused without it or with another one.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_classic::{CartPole, CartPoleConfig};
use tessera_rl_engine::replay::{Recorder, Replay, ReplayError, ReplayWriterOptions};
use tessera_rl_engine::{
    ActionInput, Archive, AutoReset, DecodeLimits, EngineError, EnvPool, PoolOptions, Record,
    Reseed, ResetSource, SyncOptions,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 4;

fn pool(seed: u64, workers: usize) -> EnvPool<CartPole> {
    let mut o = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(seed));
    o.auto_reset = AutoReset::SameStep;
    let p = EnvPool::new(CartPoleConfig::default(), o).unwrap();
    if workers == 0 {
        p
    } else {
        p.with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
            .unwrap()
    }
}

fn records(count: u64) -> Vec<Record<CartPole>> {
    (0..count)
        .map(|k| {
            let mut p = pool(100 + k, 0);
            p.reset_all(Reseed::Keep).unwrap();
            let a = [1i32; N as usize];
            for _ in 0..=k {
                p.step(ActionInput {
                    discrete: &a,
                    continuous: &[],
                })
                .unwrap();
            }
            p.snapshot(0).unwrap()
        })
        .collect()
}

fn archive(count: u64) -> Archive<CartPole> {
    Archive::new(&CartPoleConfig::default(), &records(count)).unwrap()
}

/// Records 120 steps of a pool that starts with an archive source, changes
/// its weights at step 40, and calls `reset_from` at step 80.
fn record(arc: &Archive<CartPole>, workers: usize) -> (Vec<u8>, Vec<u64>) {
    let mut p = pool(7, workers);
    p.reset_all(Reseed::Keep).unwrap();
    p.set_reset_source(ResetSource::Archive {
        archive: arc.clone(),
        weights: Some(vec![1, 0, 2]),
    })
    .unwrap();
    let options = ReplayWriterOptions {
        keyframe_every: 32,
        checkpoint_every: 8,
        outputs_hash: false,
    };
    let mut rec = Recorder::new(&mut p, Vec::new(), options).unwrap();
    for t in 0..120u32 {
        let a: Vec<i32> = (0..N).map(|i| i32::try_from((t + i) % 2).unwrap()).collect();
        rec.step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
        if t == 40 {
            rec.set_reset_source(ResetSource::Archive {
                archive: arc.clone(),
                weights: Some(vec![3, 1, 0]),
            })
            .unwrap();
            // A refused source logs nothing.
            assert!(
                rec.set_reset_source(ResetSource::Archive {
                    archive: arc.clone(),
                    weights: Some(vec![0, 0, 0]),
                })
                .is_err()
            );
        }
        if t == 80 {
            rec.reset_from(&[2, 0], &[1, 2]).unwrap();
        }
    }
    let bytes = rec.finish().unwrap();
    let hashes = (0..N).map(|i| p.state_hash(i).unwrap().get()).collect();
    (bytes, hashes)
}

#[test]
fn test_archive_replay_round_trip_has_no_divergence() {
    let arc = archive(3);
    let (inline, hashes) = record(&arc, 0);
    let (sync, sync_hashes) = record(&arc, 2);
    assert_eq!(inline, sync, "Inline and Sync write the same file");
    assert_eq!(hashes, sync_hashes);
    for workers in [0, 2] {
        let mut p = pool(7, workers);
        let summary = Replay::open(&inline, DecodeLimits::default())
            .unwrap()
            .run_with_archive(&mut p, Some(&arc))
            .unwrap();
        // 120 steps, one set_reset_source, one reset_from.
        assert_eq!(summary.calls, 122);
        assert!(summary.checkpoints >= 15);
        let replayed: Vec<u64> = (0..N).map(|i| p.state_hash(i).unwrap().get()).collect();
        assert_eq!(replayed, hashes);
    }
}

#[test]
fn test_archive_replay_needs_the_recorded_archive() {
    let arc = archive(3);
    let (bytes, _) = record(&arc, 0);
    let file = Replay::open(&bytes, DecodeLimits::default()).unwrap();
    let mut p = pool(7, 0);
    assert!(matches!(
        file.run(&mut p),
        Err(ReplayError::Engine(EngineError::ArchiveMismatch { found: 0, .. }))
    ));
    let other = archive(4);
    let mut q = pool(7, 0);
    assert!(matches!(
        file.run_with_archive(&mut q, Some(&other)),
        Err(ReplayError::Engine(EngineError::ArchiveMismatch { .. }))
    ));
}

#[test]
fn test_archive_replay_header_names_the_source_at_recording_start() {
    let arc = archive(3);
    let (bytes, _) = record(&arc, 0);
    let file = Replay::open(&bytes, DecodeLimits::default()).unwrap();
    let header = file.header().archive.unwrap();
    assert_eq!(header.digest, arc.digest());
    assert_eq!(header.weights().collect::<Vec<_>>(), [1, 0, 2]);
}

#[test]
fn test_archive_replay_fresh_pool_writes_no_archive() {
    let mut p = pool(7, 0);
    p.reset_all(Reseed::Keep).unwrap();
    let mut rec = Recorder::new(&mut p, Vec::new(), ReplayWriterOptions::default()).unwrap();
    rec.step(ActionInput {
        discrete: &[0; N as usize],
        continuous: &[],
    })
    .unwrap();
    let bytes = rec.finish().unwrap();
    let file = Replay::open(&bytes, DecodeLimits::default()).unwrap();
    assert!(file.header().archive.is_none());
    let mut q = pool(7, 0);
    file.run(&mut q).unwrap();
}

#[test]
fn test_archive_replay_seek_restores_the_archive_source() {
    // Seek to a periodic keyframe of a file with an archive source (#378
    // cursor, ADR-0075): with the archive, the pool's reset source becomes
    // the keyframe's; the rest of the file then replays with no divergence.
    let arc = archive(3);
    let (bytes, hashes) = record(&arc, 0);
    let file = Replay::open(&bytes, DecodeLimits::default()).unwrap();
    let keyframes = file.keyframes().unwrap();
    let periodic = *keyframes.get(1).unwrap();
    let mut p = pool(7, 0);
    let mut cur = file.cursor_with_archive(&p, Some(&arc)).unwrap();
    file.seek(&mut p, &mut cur, periodic).unwrap();
    assert_eq!(
        p.pool_context().unwrap().reset_source,
        tessera_rl_engine::ResetSourceContext::Archive {
            digest: arc.digest(),
            weights: vec![1, 0, 2],
        }
    );
    while file.advance(&mut p, &mut cur, |_| false).unwrap().is_some() {}
    file.finish(cur).unwrap();
    let replayed: Vec<u64> = (0..N).map(|i| p.state_hash(i).unwrap().get()).collect();
    assert_eq!(replayed, hashes);
    // Without the archive: refused before any write.
    let mut q = pool(7, 0);
    q.reset_all(Reseed::Keep).unwrap();
    let before: Vec<u64> = (0..N).map(|i| q.state_hash(i).unwrap().get()).collect();
    let mut bare = file.cursor(&q).unwrap();
    assert!(matches!(
        file.seek(&mut q, &mut bare, periodic),
        Err(ReplayError::Engine(EngineError::ArchiveMismatch { found: 0, .. }))
    ));
    let after: Vec<u64> = (0..N).map(|i| q.state_hash(i).unwrap().get()).collect();
    assert_eq!(after, before);
}
