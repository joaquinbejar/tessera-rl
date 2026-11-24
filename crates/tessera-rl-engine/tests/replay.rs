// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replay files (ADR-0072): record then replay with zero divergences in
//! Inline and Sync modes, byte-identical files across modes, every reseed
//! variant, state writes with keyframes, poisoning, the divergence report,
//! and typed refusals of damaged, reserved, newer and foreign files.

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
    DivergenceKind, Event, EventTag, Ids, RecordedReseed, Recorder, Replay, ReplayError,
    ReplayWriterOptions,
};
use tessera_rl_engine::{
    ActionInput, DecodeLimits, EnvPool, Masked, PersistError, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;

type Pool = EnvPool<Counter, Masked>;

const N: u32 = 4;

fn config() -> CountConfig {
    CountConfig {
        length: 6,
        ..CountConfig::default()
    }
}

fn pool_with(cfg: CountConfig, workers: usize) -> Pool {
    let options = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(11));
    let p = EnvPool::with_masks(cfg, options).unwrap();
    if workers == 0 {
        p
    } else {
        p.with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
            .unwrap()
    }
}

fn options() -> ReplayWriterOptions {
    ReplayWriterOptions {
        keyframe_every: 5,
        checkpoint_every: 3,
        outputs_hash: false,
    }
}

/// Two agents per instance, cycling through the choices. Counter masks
/// action 2 for agent 0 (even `k`), so agent 0 takes 1 there: with
/// `debug-checks` (on under workspace feature unification) an illegal action
/// fails the call instead of taking the no-op.
fn actions(t: usize) -> Vec<i32> {
    (0..N as usize * 2)
        .map(|k| {
            let c = (t + k) % 3;
            let c = if k % 2 == 0 && c == 2 { 1 } else { c };
            i32::try_from(c).unwrap()
        })
        .collect()
}

fn step(rec: &mut Recorder<'_, Counter, Masked, Vec<u8>>, t: usize) {
    rec.step(ActionInput {
        discrete: &actions(t),
        continuous: &[],
    })
    .unwrap();
}

/// The recorded script: every reseed variant, subset resets, a restore, a
/// fork and a branched fork between steps.
fn script(rec: &mut Recorder<'_, Counter, Masked, Vec<u8>>) {
    rec.reset_all(Reseed::Base(Seed::new(7))).unwrap();
    for t in 0..4 {
        step(rec, t);
    }
    rec.reset(&[1, 3], Reseed::List(&[Seed::new(21), Seed::new(22)]))
        .unwrap();
    rec.reset(&[2, 0], Reseed::Each(&[None, Some(Seed::new(9))]))
        .unwrap();
    rec.reset(&[3], Reseed::Keep).unwrap();
    step(rec, 4);
    rec.fork_branched(0, &[1, 2], None).unwrap();
    step(rec, 5);
    let record = rec.pool().snapshot(1).unwrap();
    rec.restore(3, &record).unwrap();
    rec.fork(2, &[0]).unwrap();
    for t in 6..14 {
        step(rec, t);
    }
}

fn record_with(cfg: CountConfig, workers: usize) -> (Vec<u8>, Pool) {
    let mut p = pool_with(cfg, workers);
    let mut rec = Recorder::new(&mut p, Vec::new(), options()).unwrap();
    script(&mut rec);
    let bytes = rec.finish().unwrap();
    (bytes, p)
}

fn record(workers: usize) -> (Vec<u8>, Pool) {
    record_with(config(), workers)
}

fn hashes(p: &Pool) -> Vec<u64> {
    (0..N).map(|i| p.state_hash(i).unwrap().get()).collect()
}

/// Byte offset of the first event: past the shared ADR-0011 header and the
/// replay extension.
fn header_len(b: &[u8]) -> usize {
    let u16_at = |at: usize| usize::from(u16::from_le_bytes([b[at], b[at + 1]]));
    let mut at = 48;
    for _ in 0..2 {
        at += 2 + u16_at(at);
    }
    at += 2; // asset_count
    at += 2 + u16_at(at); // descriptor
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

/// The (offset, tag, body length) of every frame, the end event included.
fn frames(b: &[u8]) -> Vec<(usize, u8, usize)> {
    let mut at = header_len(b);
    let mut out = Vec::new();
    while at < b.len() {
        let len = u32::from_le_bytes(b[at + 1..at + 5].try_into().unwrap()) as usize;
        out.push((at, b[at], len));
        at += 5 + len;
    }
    out
}

/// Rewrites the end event's count and checksum for the events before it.
fn reseal(b: &mut [u8]) {
    let fr = frames(b);
    let (end_at, _, _) = *fr.last().unwrap();
    let start = header_len(b);
    let count = u64::try_from(fr.len() - 1).unwrap();
    let sum = checksum(&b[start..end_at]);
    b[end_at + 5..end_at + 13].copy_from_slice(&count.to_le_bytes());
    b[end_at + 13..end_at + 21].copy_from_slice(&sum.to_le_bytes());
}

/// `b` with an event inserted before the end event, resealed.
fn with_event(b: &[u8], tag: u8, body: &[u8]) -> Vec<u8> {
    let (end_at, _, _) = *frames(b).last().unwrap();
    let mut out = b[..end_at].to_vec();
    out.push(tag);
    out.extend_from_slice(&u32::try_from(body.len()).unwrap().to_le_bytes());
    out.extend_from_slice(body);
    out.extend_from_slice(&b[end_at..]);
    reseal(&mut out);
    out
}

const LIMITS: DecodeLimits = DecodeLimits {
    max_instances: 1 << 24,
    max_decoded_bytes: 1 << 30,
    max_host_record_bytes: 1 << 26,
};

#[test]
fn test_replay_inline_round_trip_has_no_divergence() {
    let (bytes, recorded) = record(0);
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let mut p = pool_with(config(), 0);
    let summary = replay.run(&mut p).unwrap();
    assert_eq!(summary.calls, 21);
    assert_eq!(summary.checkpoints, 7);
    assert_eq!(summary.keyframes, 8);
    assert!(!summary.poisoned);
    assert_eq!(summary.events, replay.event_count());
    assert_eq!(hashes(&p), hashes(&recorded));
}

#[test]
fn test_replay_sync_round_trip_has_no_divergence_and_identical_bytes() {
    let (inline, _) = record(0);
    let (sync, recorded) = record(2);
    assert_eq!(inline, sync);
    let mut p = pool_with(config(), 2);
    Replay::open(&sync, LIMITS).unwrap().run(&mut p).unwrap();
    assert_eq!(hashes(&p), hashes(&recorded));
    let mut q = pool_with(config(), 0);
    Replay::open(&sync, LIMITS).unwrap().run(&mut q).unwrap();
    assert_eq!(hashes(&q), hashes(&recorded));
}

#[test]
fn test_replay_events_hold_every_reseed_variant_and_state_write() {
    let (bytes, _) = record(0);
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let events: Vec<Event<'_>> = replay.events().map(|e| e.unwrap().1).collect();
    assert!(matches!(
        events[0],
        Event::Keyframe {
            ids: Ids::All,
            branch_counter: 0,
            ..
        }
    ));
    let reseeds: Vec<&RecordedReseed> = events
        .iter()
        .filter_map(|e| match e {
            Event::Reset { reseed, .. } => Some(reseed),
            _ => None,
        })
        .collect();
    assert_eq!(
        reseeds,
        [
            &RecordedReseed::Base(Seed::new(7)),
            &RecordedReseed::List(vec![Seed::new(21), Seed::new(22)]),
            &RecordedReseed::Each(vec![None, Some(Seed::new(9))]),
            &RecordedReseed::Keep,
        ]
    );
    // Every state write is followed by a keyframe of the written ids.
    let mut writes = 0;
    for pair in events.windows(2) {
        if let Event::StateWrite {
            ids,
            branch_counter,
            ..
        } = &pair[0]
        {
            writes += 1;
            assert!(
                matches!(&pair[1], Event::Keyframe { ids: k, branch_counter: c, .. } if k == ids && c == branch_counter)
            );
        }
    }
    assert_eq!(writes, 3);
    assert!(events.iter().any(|e| matches!(
        e,
        Event::StateWrite {
            branch_counter: 1,
            ..
        }
    )));
    // The last event is a checkpoint.
    assert!(matches!(events.last(), Some(Event::Checkpoint { .. })));
}

#[test]
fn test_replay_shares_the_snapshot_header_under_its_own_magic() {
    let (bytes, recorded) = record(0);
    let snapshot = recorded.encode(&(0..N).collect::<Vec<_>>()).unwrap();
    let mut at = 48;
    for _ in 0..2 {
        at += 2 + usize::from(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
    }
    at += 2;
    at += 2 + usize::from(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
    assert_eq!(bytes[..8], *b"TESRPLY\x1a");
    assert_eq!(snapshot[..8], *b"TESSERA\x1a");
    assert_eq!(bytes[8..at], snapshot[8..at]);
}

#[test]
fn test_replay_checkpoint_mismatch_names_the_last_matching_checkpoint() {
    let (mut bytes, _) = record(0);
    let checkpoints: Vec<usize> = frames(&bytes)
        .iter()
        .filter(|f| f.1 == EventTag::Checkpoint as u8)
        .map(|f| f.0)
        .collect();
    // Flip a hash byte of instance 2 in the second checkpoint.
    let at = checkpoints[1] + 5 + 1 + 2 * 8;
    bytes[at] ^= 1;
    reseal(&mut bytes);
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let index_of = |offset: usize| {
        u64::try_from(frames(&bytes).iter().position(|f| f.0 == offset).unwrap()).unwrap()
    };
    let err = replay.run(&mut pool_with(config(), 0)).unwrap_err();
    let (second, first) = (index_of(checkpoints[1]), index_of(checkpoints[0]));
    assert!(
        matches!(
            err,
            ReplayError::Diverged {
                event_index,
                kind: DivergenceKind::Checkpoint,
                instance: Some(2),
                last_matching_checkpoint: Some(last),
            } if event_index == second && last == first
        ),
        "{err:?}"
    );
}

#[test]
fn test_replay_changed_config_is_incompatible() {
    // A pool whose environment ignores its RNG is not the recorded one: the
    // config check refuses it before any event.
    let (bytes, _) = record(0);
    let other = CountConfig {
        ignore_rng: true,
        ..config()
    };
    let err = Replay::open(&bytes, LIMITS)
        .unwrap()
        .run(&mut pool_with(other, 0))
        .unwrap_err();
    assert!(matches!(
        err,
        ReplayError::Persist(PersistError::Incompatible {
            field: "config_hash"
        })
    ));
}

#[test]
fn test_replay_poisoning_call_replays_to_the_same_poison() {
    let cfg = CountConfig {
        panic_at: 3,
        ..config()
    };
    let mut p = pool_with(cfg.clone(), 0);
    let mut rec = Recorder::new(&mut p, Vec::new(), options()).unwrap();
    rec.reset_all(Reseed::Keep).unwrap();
    step(&mut rec, 0);
    step(&mut rec, 1);
    let err = rec.step(ActionInput {
        discrete: &actions(2),
        continuous: &[],
    });
    assert!(matches!(err, Err(ReplayError::Engine(_))));
    assert!(matches!(
        rec.step(ActionInput {
            discrete: &actions(3),
            continuous: &[],
        }),
        Err(ReplayError::RecorderClosed)
    ));
    let bytes = rec.finish().unwrap();
    let fr = frames(&bytes);
    assert_eq!(fr[fr.len() - 2].1, EventTag::Poisoned as u8);
    let mut q = pool_with(cfg, 0);
    let summary = Replay::open(&bytes, LIMITS).unwrap().run(&mut q).unwrap();
    assert!(summary.poisoned);
    assert!(q.is_poisoned());
    // The same file into a pool that does not panic: the poison is missing.
    let err = Replay::open(&bytes, LIMITS)
        .unwrap()
        .run(&mut pool_with(config(), 0));
    assert!(err.is_err());
}

#[test]
fn test_replay_refused_calls_are_not_logged() {
    let mut p = pool_with(config(), 0);
    let mut rec = Recorder::new(&mut p, Vec::new(), ReplayWriterOptions::default()).unwrap();
    rec.reset_all(Reseed::Keep).unwrap();
    assert!(rec.reset(&[9], Reseed::Keep).is_err());
    assert!(
        rec.step(ActionInput {
            discrete: &[0; 3],
            continuous: &[],
        })
        .is_err()
    );
    let bytes = rec.finish().unwrap();
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    // Keyframe, reset, final checkpoint.
    assert_eq!(replay.event_count(), 3);
    replay.run(&mut pool_with(config(), 0)).unwrap();
}

#[test]
fn test_replay_every_flip_and_truncation_is_a_typed_error() {
    let (bytes, _) = record(0);
    for at in 0..bytes.len() {
        let mut b = bytes.clone();
        b[at] ^= 0x10;
        assert!(Replay::open(&b, LIMITS).is_err(), "flip at {at} accepted");
    }
    for len in 0..bytes.len() {
        assert!(
            Replay::open(&bytes[..len], LIMITS).is_err(),
            "prefix {len} accepted"
        );
    }
}

#[test]
fn test_replay_flips_name_the_region() {
    let (bytes, _) = record(0);
    let mut b = bytes.clone();
    b[20] ^= 1; // penalty bits, inside the header
    assert!(matches!(
        Replay::open(&b, LIMITS),
        Err(ReplayError::HeaderChecksum)
    ));
    let fr = frames(&bytes);
    let mut b = bytes.clone();
    b[fr[1].0 + 5] ^= 1; // the first byte of the second event's body
    assert!(matches!(
        Replay::open(&b, LIMITS),
        Err(ReplayError::EventChecksum)
    ));
    let mut b = bytes.clone();
    let end = fr.last().unwrap().0;
    b[end + 5] ^= 1; // the end event's count
    assert!(matches!(
        Replay::open(&b, LIMITS),
        Err(ReplayError::EventCount { .. })
    ));
    let mut b = bytes.clone();
    b.push(0);
    assert!(matches!(
        Replay::open(&b, LIMITS),
        Err(ReplayError::TrailingBytes)
    ));
}

#[test]
fn test_replay_unknown_and_reserved_tags_are_refused() {
    let (bytes, _) = record(0);
    assert!(Replay::open(&with_event(&bytes, EventTag::Poisoned as u8, &[]), LIMITS).is_ok());
    assert!(matches!(
        Replay::open(&with_event(&bytes, 42, &[]), LIMITS),
        Err(ReplayError::UnknownTag { tag: 42 })
    ));
    for tag in [3, 7, 8] {
        assert!(matches!(
            Replay::open(&with_event(&bytes, tag, &[]), LIMITS),
            Err(ReplayError::ReservedTag { tag: t }) if t == tag
        ));
    }
}

#[test]
fn test_replay_newer_version_and_foreign_magic_are_refused() {
    let (bytes, recorded) = record(0);
    let mut b = bytes.clone();
    b[8..12].copy_from_slice(&2u32.to_le_bytes());
    assert!(matches!(
        Replay::open(&b, LIMITS),
        Err(ReplayError::UnsupportedVersion { found: 2 })
    ));
    let snapshot = recorded.encode(&[0]).unwrap();
    assert!(matches!(
        Replay::open(&snapshot, LIMITS),
        Err(ReplayError::BadMagic)
    ));
    // And a snapshot reader refuses a replay file.
    assert!(matches!(
        recorded.decode(&bytes, LIMITS),
        Err(PersistError::BadMagic)
    ));
}

#[test]
fn test_replay_limits_refuse_before_allocating() {
    let (bytes, _) = record(0);
    let few = DecodeLimits {
        max_instances: N - 1,
        ..LIMITS
    };
    assert!(matches!(
        Replay::open(&bytes, few),
        Err(ReplayError::Persist(PersistError::TooManyInstances { .. }))
    ));
    let small = DecodeLimits {
        max_decoded_bytes: u64::try_from(bytes.len()).unwrap() - 1,
        ..LIMITS
    };
    assert!(matches!(
        Replay::open(&bytes, small),
        Err(ReplayError::TooLarge { .. })
    ));
}

#[test]
fn test_replay_pool_of_another_size_is_incompatible() {
    let (bytes, _) = record(0);
    let options = PoolOptions::new(NonZeroU32::new(N + 1).unwrap(), Seed::new(11));
    let mut p = EnvPool::<Counter, Masked>::with_masks(config(), options).unwrap();
    assert!(matches!(
        Replay::open(&bytes, LIMITS).unwrap().run(&mut p),
        Err(ReplayError::Persist(PersistError::Incompatible {
            field: "instance_count"
        }))
    ));
}

/// The pinned format (ADR-0072 section 6): XXH3-64 and length of a fixed
/// recording with the two version-dependent parts cut out, the library
/// version string and the header checksum over it. Any change to the
/// header, an event tag or an event body changes them; update the constants
/// only together with a format version bump.
const FIXTURE_LEN: usize = 1580;
const FIXTURE_XXH3: u64 = 17_966_924_819_680_012_336;

#[test]
fn test_replay_encoded_fixture_is_pinned() {
    let mut p = pool_with(config(), 0);
    let options = ReplayWriterOptions {
        keyframe_every: 3,
        checkpoint_every: 2,
        outputs_hash: false,
    };
    let mut rec = Recorder::new(&mut p, Vec::new(), options).unwrap();
    rec.reset_all(Reseed::Base(Seed::new(3))).unwrap();
    step(&mut rec, 0);
    rec.reset(&[1], Reseed::Each(&[Some(Seed::new(4))]))
        .unwrap();
    rec.fork_branched(0, &[2], Some(&[7])).unwrap();
    step(&mut rec, 1);
    let bytes = rec.finish().unwrap();
    // Cut the library version (its u16 length stays) and the header checksum.
    let lib_at = 48 + 2 + usize::from(u16::from_le_bytes([bytes[48], bytes[49]]));
    let lib_len = usize::from(u16::from_le_bytes([bytes[lib_at], bytes[lib_at + 1]]));
    let header_end = header_len(&bytes);
    let mut pinned = bytes[..lib_at + 2].to_vec();
    pinned.extend_from_slice(&bytes[lib_at + 2 + lib_len..header_end - 8]);
    pinned.extend_from_slice(&bytes[header_end..]);
    assert_eq!(
        (pinned.len(), checksum(&pinned)),
        (FIXTURE_LEN, FIXTURE_XXH3),
        "replay format changed: pinned length and XXH3-64"
    );
}

/// `b` with the header's tier byte set to `tier` (offset 13: magic, version,
/// payload encoding) and the header checksum rewritten.
fn with_tier(b: &[u8], tier: u8) -> Vec<u8> {
    let mut out = b.to_vec();
    out[13] = tier;
    let end = header_len(&out) - 8;
    let sum = checksum(&out[..end]);
    out[end..end + 8].copy_from_slice(&sum.to_le_bytes());
    out
}

#[test]
fn test_replay_first_party_file_refuses_the_host_tier() {
    let (bytes, _) = record(0);
    assert_eq!(bytes[13], 0);
    assert!(matches!(
        Replay::open(&with_tier(&bytes, 2), LIMITS),
        Err(ReplayError::Persist(PersistError::InvalidField {
            field: "tier"
        }))
    ));
}

#[test]
fn test_replay_detached_recorder_writes_the_same_file() {
    let (expected, _) = record(0);
    let mut p = pool_with(config(), 0);
    let mut detached = Recorder::new(&mut p, Vec::new(), options())
        .unwrap()
        .detach();
    {
        let mut rec = detached.attach(&mut p);
        rec.reset_all(Reseed::Base(Seed::new(7))).unwrap();
        detached = rec.detach();
    }
    // The rest of the script, reattached for each call.
    let mut rec = detached.attach(&mut p);
    for t in 0..4 {
        step(&mut rec, t);
    }
    let detached = rec.detach();
    let mut rec = detached.attach(&mut p);
    rec.reset(&[1, 3], Reseed::List(&[Seed::new(21), Seed::new(22)]))
        .unwrap();
    rec.reset(&[2, 0], Reseed::Each(&[None, Some(Seed::new(9))]))
        .unwrap();
    rec.reset(&[3], Reseed::Keep).unwrap();
    step(&mut rec, 4);
    rec.fork_branched(0, &[1, 2], None).unwrap();
    step(&mut rec, 5);
    let record = rec.pool().snapshot(1).unwrap();
    rec.restore(3, &record).unwrap();
    rec.fork(2, &[0]).unwrap();
    let detached = rec.detach();
    let mut rec = detached.attach(&mut p);
    for t in 6..14 {
        step(&mut rec, t);
    }
    assert_eq!(rec.finish().unwrap(), expected);
}

/// #377: the pool rebuilt from the header (config decoded from its
/// canonical bytes, options from its identity fields) replays without
/// divergence, in Inline and Sync mode.
#[test]
fn test_replay_run_rebuilt_from_header_has_no_divergence() {
    let (bytes, recorded) = record(0);
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    for workers in [0, 2] {
        let (p, summary) = replay
            .run_rebuilt(|cfg, options| {
                let p = EnvPool::<Counter, Masked>::with_masks(cfg, options)?;
                if workers == 0 {
                    Ok(p)
                } else {
                    p.with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
                }
            })
            .unwrap();
        assert_eq!(summary.events, replay.event_count());
        assert_eq!(hashes(&p), hashes(&recorded));
    }
}

/// #377: the options come from the header, and a pool built with another
/// config than the header's is refused as a caller-built one is.
#[test]
fn test_replay_build_pool_options_from_header_and_config_checked() {
    let (bytes, _) = record(0);
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let options = replay.pool_options().unwrap();
    assert_eq!(options.num_envs.get(), N);
    let mut other = config();
    other.length += 1;
    let Err(err) =
        replay.build_pool(|_, options| EnvPool::<Counter, Masked>::with_masks(other, options))
    else {
        panic!("a pool with another config is refused");
    };
    assert!(
        matches!(err, ReplayError::Persist(PersistError::Incompatible { .. })),
        "{err:?}"
    );
}

/// A recorded step with an illegal (masked-out) action, which the default
/// policy replaces by the no-op (ADR-0028): replay re-submits the same
/// illegal action and reproduces the state. Under `debug-checks` the call
/// fails instead, so the test is off there.
#[cfg(not(feature = "debug-checks"))]
#[test]
fn test_replay_illegal_action_replaced_by_noop_replays_without_divergence() {
    let mut p = pool_with(config(), 0);
    let mut rec = Recorder::new(&mut p, Vec::new(), ReplayWriterOptions::default()).unwrap();
    rec.reset_all(Reseed::Keep).unwrap();
    // Counter masks action 2 for agent 0; the mask row is [N, 2 agents, 3].
    let masks = rec.pool().outputs().unwrap().masks();
    assert!(
        !masks[2].get(),
        "action 2 must be illegal for instance 0, agent 0"
    );
    assert!(masks[1].get());
    let mut actions = vec![1; N as usize * 2];
    actions[0] = 2;
    rec.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap();
    step(&mut rec, 1);
    let bytes = rec.finish().unwrap();
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let submitted = replay.events().find_map(|e| match e {
        Ok((_, Event::Step { discrete, .. })) => Some(discrete),
        _ => None,
    });
    assert_eq!(submitted.unwrap()[0], 2);
    let mut q = pool_with(config(), 0);
    replay.run(&mut q).unwrap();
    assert_eq!(hashes(&q), hashes(&p));
}
