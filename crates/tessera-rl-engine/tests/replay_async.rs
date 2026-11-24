// SPDX-License-Identifier: MIT OR Apache-2.0

//! Async replay files (ADR-0072 section 3, `Recv`): a round trip with
//! interleaved subset sends and receives on 4 instances and 2 workers,
//! checkpoints only at quiescent points, a tampered `Recv` id, and async
//! files refused by the other replays and vice versa.
//!
//! Batch membership depends on worker timing (ADR-0006), so the recorded
//! batches differ between runs; replay (with `batch_size` 1) waits for each
//! recorded batch, so every recording replays without divergence.

#![cfg(feature = "replay")]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests: a failure is the assertion"
)]

mod common;

use core::num::{NonZeroU32, NonZeroUsize};

use common::{CountConfig, Counter};
use tessera_rl_core::encoding::checksum;
use tessera_rl_engine::replay::{
    AsyncRecorder, DivergenceKind, Event, EventTag, Ids, Recorder, Replay, ReplayError,
    ReplayWriterOptions,
};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AsyncPool, DecodeLimits, EnvPool, Masked, PersistError, PoolOptions,
    Reseed,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 4;

type Asy = AsyncPool<Counter, Masked>;

fn cfg() -> CountConfig {
    CountConfig {
        length: 5,
        ..CountConfig::default()
    }
}

fn sync_pool() -> EnvPool<Counter, Masked> {
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(1).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(31))
    };
    EnvPool::with_masks(cfg(), options).unwrap()
}

fn asy(batch: usize) -> Asy {
    sync_pool()
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(batch).unwrap(),
        ))
        .unwrap()
}

/// Two actions (one per agent) for each id.
fn actions(ids: &[u32], t: u32) -> Vec<i32> {
    ids.iter()
        .flat_map(|&id| {
            let a = i32::try_from((id + t) % 2).unwrap();
            [a, 1 - a]
        })
        .collect()
}

type Rec<'p> = AsyncRecorder<'p, Counter, Masked, Vec<u8>>;

fn send(rec: &mut Rec<'_>, ids: &[u32], t: u32) {
    rec.send(
        ids,
        ActionInput {
            discrete: &actions(ids, t),
            continuous: &[],
        },
    )
    .unwrap();
}

fn recv(rec: &mut Rec<'_>) -> Vec<u32> {
    let token = rec.recv().unwrap();
    token.ids().collect()
}

fn hashes(p: &mut Asy) -> Vec<u64> {
    (0..N)
        .map(|i| p.snapshot(i).unwrap().hash().get())
        .collect()
}

/// Records interleaved subset sends and receives (batches of 2), a partial
/// reset and a partial restore; returns the file and the final hashes.
fn record(options: ReplayWriterOptions) -> (Vec<u8>, Vec<u64>) {
    let mut p = asy(2);
    let mut rec = AsyncRecorder::new(&mut p, Vec::new(), options).unwrap();
    drop(rec.reset_all(Reseed::Base(Seed::new(3))).unwrap());
    // Round 0: only ids 0 and 1 are in flight.
    send(&mut rec, &[0, 1], 0);
    let first = recv(&mut rec);
    assert_eq!(first.len(), 2);
    for t in 1..6 {
        send(&mut rec, &[0, 2], t);
        send(&mut rec, &[3, 1], t);
        let mut got = recv(&mut rec);
        // Resend what came back while the other batch may still run.
        let again: Vec<u32> = got.clone();
        send(&mut rec, &again, t + 10);
        got.extend(recv(&mut rec));
        got.extend(recv(&mut rec));
        assert_eq!(got.len(), 6);
    }
    rec.reset(&[1, 2], Reseed::List(&[Seed::new(8), Seed::new(9)]))
        .unwrap();
    assert_eq!(recv(&mut rec).len(), 2);
    let saved = rec_snapshot(&mut rec, 0);
    rec.restore(&[3], &[saved]).unwrap();
    send(&mut rec, &[0], 20);
    assert_eq!(recv(&mut rec).len(), 2);
    let bytes = rec.finish().unwrap();
    (bytes, hashes(&mut p))
}

fn rec_snapshot(rec: &mut Rec<'_>, id: u32) -> tessera_rl_engine::Record<Counter> {
    rec.snapshot(id).unwrap()
}

fn options(checkpoint_every: u64, outputs_hash: bool) -> ReplayWriterOptions {
    ReplayWriterOptions {
        keyframe_every: 7,
        checkpoint_every,
        outputs_hash,
    }
}

const LIMITS: DecodeLimits = DecodeLimits {
    max_instances: 1 << 24,
    max_decoded_bytes: 1 << 30,
    max_host_record_bytes: 1 << 26,
};

#[test]
fn test_replay_async_round_trip_with_subset_sends_has_no_divergence() {
    for _ in 0..3 {
        let (bytes, recorded) = record(options(3, true));
        let replay = Replay::open(&bytes, LIMITS).unwrap();
        assert!(replay.header().async_pool);
        let mut p = asy(1);
        let summary = replay.run_async(&mut p).unwrap();
        assert!(summary.checkpoints >= 2);
        assert_eq!(hashes(&mut p), recorded);
        let recvs = replay
            .events()
            .filter(|e| matches!(e, Ok((_, Event::Recv { .. }))))
            .count();
        assert_eq!(recvs, 1 + 5 * 3 + 2);
    }
}

#[test]
fn test_replay_async_checkpoints_and_keyframes_only_when_quiescent() {
    let (bytes, _) = record(options(1, false));
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let mut outstanding: usize = 0;
    let mut checkpoints = 0;
    for item in replay.events() {
        let (index, event) = item.unwrap();
        match event {
            Event::Step {
                ids: Ids::List(ids),
                ..
            }
            | Event::Reset {
                ids: Ids::List(ids),
                ..
            }
            | Event::StateWrite {
                ids: Ids::List(ids),
                ..
            } => outstanding += ids.len(),
            Event::Reset { ids: Ids::All, .. } => outstanding = 0,
            Event::Recv { ids, .. } => outstanding -= ids.len(),
            Event::Checkpoint { .. } => {
                assert_eq!(outstanding, 0, "checkpoint at event {index} not quiescent");
                checkpoints += 1;
            }
            Event::Keyframe { ids: Ids::All, .. } => {
                assert_eq!(outstanding, 0, "keyframe at event {index} not quiescent");
            }
            _ => {}
        }
    }
    assert!(checkpoints >= 3);
}

/// Event offsets: past the header (first-party, no identity).
fn header_len(b: &[u8]) -> usize {
    let u16_at = |at: usize| usize::from(u16::from_le_bytes([b[at], b[at + 1]]));
    let mut at = 48;
    for _ in 0..2 {
        at += 2 + u16_at(at);
    }
    at += 2;
    at += 2 + u16_at(at);
    let config = u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize;
    // Flags and host fields (4), the version 1 extension, the checksum (8).
    past_extension(b, at + 4 + config + 12 + 4) + 8
}

/// Offset just past the version 1 extension that starts at `q`: the policy
/// map, the curriculum flag, the archive block and the chunk size.
fn past_extension(b: &[u8], mut q: usize) -> usize {
    let slots = usize::from(u16::from_le_bytes([b[q], b[q + 1]]));
    q += 2 + slots + 1;
    if b[q] == 1 {
        let n = u32::from_le_bytes(b[q + 9..q + 13].try_into().unwrap()) as usize;
        q += 1 + 8 + 4 + 4 * n;
    } else {
        q += 1;
    }
    q + 4
}

fn frames(b: &[u8]) -> Vec<(usize, u8)> {
    let mut at = header_len(b);
    let mut out = Vec::new();
    while at < b.len() {
        let len = u32::from_le_bytes(b[at + 1..at + 5].try_into().unwrap()) as usize;
        out.push((at, b[at]));
        at += 5 + len;
    }
    out
}

fn reseal(b: &mut [u8]) {
    let fr = frames(b);
    let (end, _) = *fr.last().unwrap();
    let start = header_len(b);
    let sum = checksum(&b[start..end]);
    b[end + 13..end + 21].copy_from_slice(&sum.to_le_bytes());
}

#[test]
fn test_replay_async_tampered_recv_id_is_a_recv_mismatch() {
    let (mut bytes, _) = record(options(1024, false));
    // The first Recv holds ids 0 and 1, in some order: replace its second id
    // with 3, which was not sent yet.
    let (at, _) = frames(&bytes)
        .into_iter()
        .find(|f| f.1 == EventTag::Recv as u8)
        .unwrap();
    // Frame (5), ids marker (1), count (4), first id (4).
    bytes[at + 5 + 1 + 4 + 4..at + 5 + 1 + 4 + 8].copy_from_slice(&3u32.to_le_bytes());
    reseal(&mut bytes);
    let err = Replay::open(&bytes, LIMITS)
        .unwrap()
        .run_async(&mut asy(1))
        .unwrap_err();
    assert!(
        matches!(
            err,
            ReplayError::Diverged {
                kind: DivergenceKind::RecvMismatch,
                instance: Some(3),
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn test_replay_async_file_and_sync_file_refuse_the_other_replay() {
    let (bytes, _) = record(options(4, false));
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    assert!(matches!(
        replay.run(&mut sync_pool()),
        Err(ReplayError::Persist(PersistError::Incompatible { .. }))
    ));
    // A replay pool with batches of 2 cannot wait for recorded batches.
    assert!(matches!(
        replay.run_async(&mut asy(2)),
        Err(ReplayError::Persist(PersistError::Incompatible { .. }))
    ));
    let mut p = sync_pool();
    let mut rec = Recorder::new(&mut p, Vec::new(), ReplayWriterOptions::default()).unwrap();
    rec.reset_all(Reseed::Keep).unwrap();
    let sync = rec.finish().unwrap();
    assert!(matches!(
        Replay::open(&sync, LIMITS).unwrap().run_async(&mut asy(1)),
        Err(ReplayError::Persist(PersistError::Incompatible { .. }))
    ));
}

#[test]
fn test_replay_async_recv_tag_in_a_sync_file_is_refused() {
    let mut p = sync_pool();
    let mut rec = Recorder::new(&mut p, Vec::new(), ReplayWriterOptions::default()).unwrap();
    rec.reset_all(Reseed::Keep).unwrap();
    let sync = rec.finish().unwrap();
    let (end, _) = *frames(&sync).last().unwrap();
    let mut forged = sync[..end].to_vec();
    forged.extend_from_slice(&[EventTag::Recv as u8, 0, 0, 0, 0]);
    forged.extend_from_slice(&sync[end..]);
    // Fix the end event's count and checksum.
    let count = u64::try_from(frames(&forged).len() - 1).unwrap();
    let (end, _) = *frames(&forged).last().unwrap();
    forged[end + 5..end + 13].copy_from_slice(&count.to_le_bytes());
    reseal(&mut forged);
    assert!(matches!(
        Replay::open(&forged, LIMITS),
        Err(ReplayError::ReservedTag { tag: 8 })
    ));
}

#[test]
fn test_replay_async_truncated_file_reports_unclaimed_ids() {
    let (bytes, _) = record(options(1024, false));
    // Cut the file just before its first Recv: ids 0 and 1 were sent and no
    // event claims them.
    let (cut, _) = frames(&bytes)
        .into_iter()
        .find(|f| f.1 == EventTag::Recv as u8)
        .unwrap();
    let start = header_len(&bytes);
    let events = &bytes[start..cut];
    let mut file = bytes[..cut].to_vec();
    let count = u64::try_from(frames(&bytes).iter().take_while(|f| f.0 < cut).count()).unwrap();
    file.push(0);
    file.extend_from_slice(&16u32.to_le_bytes());
    file.extend_from_slice(&count.to_le_bytes());
    file.extend_from_slice(&checksum(events).to_le_bytes());
    let summary = Replay::open(&file, LIMITS)
        .unwrap()
        .run_async(&mut asy(1))
        .unwrap();
    assert_eq!(summary.unclaimed, vec![0, 1]);
    // A whole file leaves nothing unclaimed.
    let summary = Replay::open(&bytes, LIMITS)
        .unwrap()
        .run_async(&mut asy(1))
        .unwrap();
    assert_eq!(summary.unclaimed, Vec::<u32>::new());
}
