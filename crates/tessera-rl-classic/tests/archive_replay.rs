// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replay of archive resets (ADR-0075 item 7, the `ResetSource` and
//! `ResetFrom` events of replay format version 1): a recording of a pool
//! that resets from an archive replays with no divergence given the same
//! archive, and is refused without it or with another one.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion; forged-file offsets"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_classic::{CartPole, CartPoleConfig};
use tessera_rl_engine::replay::{
    DivergenceKind, Recorder, Replay, ReplayError, ReplayWriterOptions,
};
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
        let a: Vec<i32> = (0..N)
            .map(|i| i32::try_from((t + i) % 2).unwrap())
            .collect();
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
        Err(ReplayError::Engine(EngineError::ArchiveMismatch {
            found: 0,
            ..
        }))
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
fn test_archive_replay_async_recorder_refuses_an_archive_source() {
    use tessera_rl_engine::AsyncOptions;
    use tessera_rl_engine::replay::AsyncRecorder;
    let mut p = pool(7, 0);
    p.set_reset_source(ResetSource::Archive {
        archive: archive(2),
        weights: None,
    })
    .unwrap();
    let mut a = p
        .async_mode(AsyncOptions::new(NonZeroUsize::MIN, NonZeroUsize::MIN))
        .unwrap();
    assert!(matches!(
        AsyncRecorder::new(&mut a, Vec::new(), ReplayWriterOptions::default()),
        Err(ReplayError::Unsupported { .. })
    ));
}

/// The first index of `needle` in `hay`.
fn find(hay: &[u8], needle: &[u8]) -> usize {
    hay.windows(needle.len()).position(|w| w == needle).unwrap()
}

/// Rewrites the trailer's checksum over the event bytes (from the end of
/// the header to the `End` frame), as the writer computes it.
fn reseal_events(bytes: &mut [u8], header_end: usize) {
    let n = bytes.len();
    let events_end = n - 16 - 5;
    let sum = tessera_rl_core::encoding::checksum(&bytes[header_end..events_end]);
    bytes[n - 8..].copy_from_slice(&sum.to_le_bytes());
}

/// The header's archive block (`digest`, `count:u32`, weights) and the
/// header checksum after it and `chunk_size`: returns (weights start,
/// checksum position).
fn header_archive(bytes: &[u8], digest: u64, weights: usize) -> (usize, usize) {
    let at = find(bytes, &digest.to_le_bytes());
    let w = at + 8 + 4;
    (w, w + 4 * weights + 4)
}

#[test]
fn test_archive_replay_forged_header_source_is_malformed() {
    // A header whose archive block differs from the first keyframe's pool
    // context (ADR-0072 r4 via #411: the header names the source at
    // recording start).
    let arc = archive(3);
    let (mut bytes, _) = record(&arc, 0);
    let (w, sum_at) = header_archive(&bytes, arc.digest(), 3);
    // Weights (1, 0, 2) become (1, 0, 5), still a positive total.
    bytes[w + 8..w + 12].copy_from_slice(&5u32.to_le_bytes());
    let sum = tessera_rl_core::encoding::checksum(&bytes[..sum_at]);
    bytes[sum_at..sum_at + 8].copy_from_slice(&sum.to_le_bytes());
    let file = Replay::open(&bytes, DecodeLimits::default()).unwrap();
    assert_eq!(
        file.header().archive.unwrap().weights().collect::<Vec<_>>(),
        [1, 0, 5]
    );
    let mut p = pool(7, 0);
    assert!(matches!(
        file.run_with_archive(&mut p, Some(&arc)),
        Err(ReplayError::Malformed {
            event_index: 0,
            field: "first keyframe reset source"
        })
    ));
}

#[test]
fn test_archive_replay_forged_reset_source_event_names_another_archive() {
    // A `ResetSource` event whose digest is not the archive given: refused
    // with `ArchiveMismatch`, even though the first keyframe's matches.
    let arc = archive(3);
    let other = archive(4);
    let (mut bytes, _) = record(&arc, 0);
    let (_, sum_at) = header_archive(&bytes, arc.digest(), 3);
    let header_end = sum_at + 8;
    // Tag 10, body length 1 + 8 + 4 + 3 * 4 = 25, then `1` (Archive) and
    // the digest: the first `ResetSource` event (the refused call logged
    // nothing).
    let mut pattern = vec![10u8];
    pattern.extend_from_slice(&25u32.to_le_bytes());
    pattern.push(1);
    pattern.extend_from_slice(&arc.digest().to_le_bytes());
    let at = header_end + find(&bytes[header_end..], &pattern);
    bytes[at + 6..at + 14].copy_from_slice(&other.digest().to_le_bytes());
    reseal_events(&mut bytes, header_end);
    let file = Replay::open(&bytes, DecodeLimits::default()).unwrap();
    let mut p = pool(7, 0);
    let err = file.run_with_archive(&mut p, Some(&arc)).unwrap_err();
    assert!(
        matches!(
            err,
            ReplayError::Engine(EngineError::ArchiveMismatch { expected, found })
                if expected == other.digest() && found == arc.digest()
        ),
        "{err:?}"
    );
}

#[test]
fn test_archive_replay_forged_periodic_keyframe_source_diverges() {
    // A periodic keyframe (call 32, before the weights change at step 40)
    // whose reset source differs from the replayed pool's.
    let arc = archive(3);
    let (mut bytes, _) = record(&arc, 0);
    let (_, sum_at) = header_archive(&bytes, arc.digest(), 3);
    let header_end = sum_at + 8;
    let digest = arc.digest().to_le_bytes();
    // The digest's first occurrence after the header is the first
    // keyframe's; the second, the periodic keyframe's.
    let first = header_end + find(&bytes[header_end..], &digest);
    let second = first + 8 + find(&bytes[first + 8..], &digest);
    // Then `count:u32` and the weights (1, 0, 2): (1, 0, 3).
    let w = second + 8 + 4;
    assert_eq!(&bytes[w..w + 12], &[1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]);
    bytes[w + 8..w + 12].copy_from_slice(&3u32.to_le_bytes());
    reseal_events(&mut bytes, header_end);
    let file = Replay::open(&bytes, DecodeLimits::default()).unwrap();
    let mut p = pool(7, 0);
    let err = file.run_with_archive(&mut p, Some(&arc)).unwrap_err();
    assert!(
        matches!(
            err,
            ReplayError::Diverged {
                kind: DivergenceKind::ResetSource,
                ..
            }
        ),
        "{err:?}"
    );
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
    // The cursor itself is refused: no seek can write.
    assert!(matches!(
        file.cursor(&q),
        Err(ReplayError::Engine(EngineError::ArchiveMismatch {
            found: 0,
            ..
        }))
    ));
    let after: Vec<u64> = (0..N).map(|i| q.state_hash(i).unwrap().get()).collect();
    assert_eq!(after, before);
}

/// Records 60 steps of a pool that starts with `start` and switches to
/// `then` at step 20.
fn record_switch(start: Option<&Archive<CartPole>>, then: &Archive<CartPole>) -> Vec<u8> {
    let mut p = pool(7, 0);
    p.reset_all(Reseed::Keep).unwrap();
    if let Some(a) = start {
        p.set_reset_source(ResetSource::Archive {
            archive: a.clone(),
            weights: None,
        })
        .unwrap();
    }
    let mut rec = Recorder::new(&mut p, Vec::new(), ReplayWriterOptions::default()).unwrap();
    for t in 0..60u32 {
        let a: Vec<i32> = (0..N).map(|i| i32::try_from((t + i) % 2).unwrap()).collect();
        rec.step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
        if t == 20 {
            rec.set_reset_source(ResetSource::Archive {
                archive: then.clone(),
                weights: None,
            })
            .unwrap();
        }
    }
    rec.finish().unwrap()
}

/// Replays `file` into a reset pool and returns the error, asserting the
/// pool's records and context are unchanged.
fn untouched(file: &Replay<'_>, archive: Option<&Archive<CartPole>>) -> ReplayError {
    let mut p = pool(7, 0);
    p.reset_all(Reseed::Base(Seed::new(99))).unwrap();
    let before: Vec<u64> = (0..N).map(|i| p.state_hash(i).unwrap().get()).collect();
    let context = p.pool_context().unwrap();
    let err = file.run_with_archive(&mut p, archive).unwrap_err();
    let after: Vec<u64> = (0..N).map(|i| p.state_hash(i).unwrap().get()).collect();
    assert_eq!(after, before, "the pool was written before the refusal");
    assert_eq!(p.pool_context().unwrap(), context);
    err
}

#[test]
fn test_archive_replay_fresh_then_archive_without_it_refused_before_stepping() {
    // A file that starts with fresh resets and switches to an archive at a
    // `ResetSource` event: without that archive, refused before any write
    // (ADR-0075 item 7, REVIEW-implB-r1).
    let arc = archive(3);
    let bytes = record_switch(None, &arc);
    let file = Replay::open(&bytes, DecodeLimits::default()).unwrap();
    assert!(file.header().archive.is_none());
    assert!(matches!(
        untouched(&file, None),
        ReplayError::Engine(EngineError::ArchiveMismatch { found: 0, .. })
    ));
    // With it, the file replays.
    let mut p = pool(7, 0);
    file.run_with_archive(&mut p, Some(&arc)).unwrap();
}

#[test]
fn test_archive_replay_two_archives_in_one_file_refused_before_stepping() {
    let a = archive(3);
    let b = archive(4);
    let bytes = record_switch(Some(&a), &b);
    let file = Replay::open(&bytes, DecodeLimits::default()).unwrap();
    for given in [Some(&a), Some(&b), None] {
        let err = untouched(&file, given);
        assert!(
            matches!(err, ReplayError::Unsupported { feature } if feature.contains("one archive per file")),
            "{err:?}"
        );
    }
}
