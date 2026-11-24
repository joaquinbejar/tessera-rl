// SPDX-License-Identifier: MIT OR Apache-2.0

//! The version 1 freeze (#411): the fields ADR-0075, ADR-0077, ADR-0079 and
//! ADR-0081 folded into replay version 1 (asset entries are ADR-0077's,
//! tested with #364). A default file round-trips; each
//! field's invalid values are refused at open; valid non-default values open
//! and are refused by replay until their features land.

#![cfg(feature = "replay")]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests: a failure is the assertion"
)]

mod common;

use core::num::NonZeroU32;

use common::{CountConfig, Counter};
use tessera_rl_core::encoding::checksum;
use tessera_rl_engine::replay::{
    DivergenceKind, Event, EventTag, Recorder, Replay, ReplayError, ReplayWriterOptions,
    ResetSourceRecord,
};
use tessera_rl_engine::{
    ActionInput, DecodeLimits, EngineError, EnvPool, Masked, PersistError, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 3;

type Pool = EnvPool<Counter, Masked>;

fn pool_chunk(chunk: u32) -> Pool {
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(chunk).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(2))
    };
    EnvPool::with_masks(CountConfig::default(), options).unwrap()
}

fn pool() -> Pool {
    pool_chunk(64)
}

fn recording() -> Vec<u8> {
    let mut p = pool();
    let mut rec = Recorder::new(&mut p, Vec::new(), ReplayWriterOptions::default()).unwrap();
    rec.reset_all(Reseed::Keep).unwrap();
    for t in 0..4 {
        let a: Vec<i32> = (0..N * 2)
            .map(|k| i32::try_from((k + t) % 2).unwrap())
            .collect();
        rec.step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
    }
    rec.finish().unwrap()
}

const LIMITS: DecodeLimits = DecodeLimits {
    max_instances: 1 << 24,
    max_decoded_bytes: 1 << 30,
    max_host_record_bytes: 1 << 26,
};

/// A first-party file split into its parts.
struct Parts {
    /// Header bytes up to the asset entries (`asset_count` included).
    head: Vec<u8>,
    /// Asset entries.
    assets: Vec<u8>,
    /// From the descriptor to the host fields included.
    middle: Vec<u8>,
    /// The version 1 extension.
    ext: Vec<u8>,
    /// Events, without the end event.
    events: Vec<(u8, Vec<u8>)>,
}

fn u16_at(b: &[u8], at: usize) -> usize {
    usize::from(u16::from_le_bytes([b[at], b[at + 1]]))
}

fn u32_at(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize
}

fn split(b: &[u8]) -> Parts {
    let mut at = 48;
    for _ in 0..2 {
        at += 2 + u16_at(b, at);
    }
    let count = u16_at(b, at);
    at += 2;
    let head_end = at;
    for _ in 0..count {
        at += 1 + usize::from(b[at]) + 16;
    }
    let assets_end = at;
    at += 2 + u16_at(b, at); // descriptor
    at += 4 + u32_at(b, at); // config
    at += 12 + 4; // action layout, flags and host fields (first-party)
    let ext_start = at;
    at += 2 + u16_at(b, at) + 2; // policy map, curriculum and league flags
    at += if b[at] == 1 {
        13 + 4 * u32_at(b, at + 9)
    } else {
        1
    };
    at += 4;
    let ext_end = at;
    at += 8; // header checksum
    let mut events = Vec::new();
    loop {
        let tag = b[at];
        let len = u32_at(b, at + 1);
        if tag == 0 {
            break;
        }
        events.push((tag, b[at + 5..at + 5 + len].to_vec()));
        at += 5 + len;
    }
    Parts {
        head: b[..head_end].to_vec(),
        assets: b[head_end..assets_end].to_vec(),
        middle: b[assets_end..ext_start].to_vec(),
        ext: b[ext_start..ext_end].to_vec(),
        events,
    }
}

fn join(p: &Parts, asset_count: u16) -> Vec<u8> {
    let mut out = p.head.clone();
    let n = out.len();
    out[n - 2..].copy_from_slice(&asset_count.to_le_bytes());
    out.extend_from_slice(&p.assets);
    out.extend_from_slice(&p.middle);
    out.extend_from_slice(&p.ext);
    let sum = checksum(&out);
    out.extend_from_slice(&sum.to_le_bytes());
    let start = out.len();
    for (tag, body) in &p.events {
        out.push(*tag);
        out.extend_from_slice(&u32::try_from(body.len()).unwrap().to_le_bytes());
        out.extend_from_slice(body);
    }
    let sum = checksum(&out[start..]);
    out.push(0);
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&u64::try_from(p.events.len()).unwrap().to_le_bytes());
    out.extend_from_slice(&sum.to_le_bytes());
    out
}

/// An extension: policy map groups, curriculum byte, no league, archive
/// block, chunk.
fn ext(groups: &[u8], curriculum: u8, archive: &[u8], chunk: u32) -> Vec<u8> {
    ext_league(groups, curriculum, 0, archive, chunk)
}

/// An extension with a league byte (ADR-0088 item 8).
fn ext_league(groups: &[u8], curriculum: u8, league: u8, archive: &[u8], chunk: u32) -> Vec<u8> {
    let mut out = u16::try_from(groups.len()).unwrap().to_le_bytes().to_vec();
    out.extend_from_slice(groups);
    out.push(curriculum);
    out.push(league);
    out.extend_from_slice(archive);
    out.extend_from_slice(&chunk.to_le_bytes());
    out
}

fn archive_block(kind: u8, weights: &[u32]) -> Vec<u8> {
    let mut out = vec![kind];
    if kind == 1 {
        out.extend_from_slice(&7u64.to_le_bytes());
        out.extend_from_slice(&u32::try_from(weights.len()).unwrap().to_le_bytes());
        for w in weights {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    out
}

fn with_ext(e: Vec<u8>) -> Vec<u8> {
    let mut p = split(&recording());
    p.ext = e;
    join(&p, 0)
}

fn with_event(curriculum: u8, tag: u8, body: Vec<u8>) -> Vec<u8> {
    let mut p = split(&recording());
    p.ext = ext(&[], curriculum, &[0], 64);
    p.events.push((tag, body));
    join(&p, 0)
}

fn invalid_field(bytes: &[u8]) -> &'static str {
    match Replay::open(bytes, LIMITS) {
        Err(ReplayError::Persist(PersistError::InvalidField { field })) => field,
        other => panic_free(&format!("{other:?}")),
    }
}

/// A test failure without `panic!` (denied workspace-wide).
fn panic_free(what: &str) -> &'static str {
    assert!(what.is_empty(), "expected InvalidField, got {what}");
    ""
}

fn first_event_error(bytes: &[u8]) -> ReplayError {
    let replay = Replay::open(bytes, LIMITS).unwrap();
    replay.events().find_map(Result::err).unwrap()
}

fn run_error(bytes: &[u8]) -> ReplayError {
    Replay::open(bytes, LIMITS)
        .unwrap()
        .run(&mut pool())
        .unwrap_err()
}

#[test]
fn test_replay_v1_default_file_round_trips_with_default_fields() {
    let bytes = recording();
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let h = replay.header();
    assert_eq!(h.policy_map, []);
    assert!(!h.has_curriculum);
    assert!(!h.has_league);
    assert!(h.archive.is_none());
    assert_eq!(h.chunk_size, 64);
    assert!(matches!(
        replay.events().next(),
        Some(Ok((
            0,
            Event::Keyframe {
                reset_source: ResetSourceRecord::Fresh,
                ..
            }
        )))
    ));
    replay.run(&mut pool()).unwrap();
    // The split and join helpers reproduce the file exactly.
    assert_eq!(join(&split(&bytes), 0), bytes);
}

#[test]
fn test_replay_v1_policy_map_invalid_or_mismatched_refused() {
    // Counter has 2 agent slots.
    assert_eq!(
        invalid_field(&with_ext(ext(&[0, 5], 0, &[0], 64))),
        "policy map group"
    );
    assert_eq!(
        invalid_field(&with_ext(ext(&[0], 0, &[0], 64))),
        "policy map"
    );
    // A valid non-default map opens; replay into a default-map pool is
    // refused as a map mismatch (ADR-0079 item 6, #347 stage B2).
    let mapped = with_ext(ext(&[0, 2], 0, &[0], 64));
    assert!(matches!(
        run_error(&mapped),
        ReplayError::Engine(EngineError::PolicyMapMismatch { expected: 0, .. })
    ));
    // An explicit all-learner map is the default.
    Replay::open(&with_ext(ext(&[0, 0], 0, &[0], 64)), LIMITS)
        .unwrap()
        .run(&mut pool())
        .unwrap();
}

#[test]
fn test_replay_v1_curriculum_and_archive_refusals() {
    assert_eq!(
        invalid_field(&with_ext(ext(&[], 2, &[0], 64))),
        "has_curriculum"
    );
    assert_eq!(
        invalid_field(&with_ext(ext(&[], 0, &[2], 64))),
        "archive digest_present"
    );
    assert_eq!(
        invalid_field(&with_ext(ext(&[], 0, &archive_block(1, &[0, 0]), 64))),
        "archive weights"
    );
    assert_eq!(
        invalid_field(&with_ext(ext(&[], 0, &archive_block(1, &[]), 64))),
        "archive weights"
    );
    let curriculum = with_ext(ext(&[], 1, &[0], 64));
    assert!(matches!(
        run_error(&curriculum),
        ReplayError::Unsupported { .. }
    ));
    let archive = with_ext(ext(&[], 0, &archive_block(1, &[1, 0, 3]), 64));
    let replay = Replay::open(&archive, LIMITS).unwrap();
    let weights: Vec<u32> = replay.header().archive.unwrap().weights().collect();
    assert_eq!(weights, [1, 0, 3]);
    // ADR-0075 (#363): the header names an archive and none is given:
    // refused before anything is replayed. (With the archive, a fresh first
    // keyframe is malformed: classic's archive_replay forged-header test.)
    assert!(matches!(
        run_error(&archive),
        ReplayError::Engine(EngineError::ArchiveMismatch { found: 0, .. })
    ));
}

#[test]
fn test_replay_v1_chunk_size_is_checked_only_when_recorded() {
    let bytes = recording();
    assert!(matches!(
        Replay::open(&bytes, LIMITS)
            .unwrap()
            .run(&mut pool_chunk(2)),
        Err(ReplayError::Persist(PersistError::Incompatible {
            field: "chunk_size"
        }))
    ));
    let unrecorded = with_ext(ext(&[], 0, &[0], 0));
    Replay::open(&unrecorded, LIMITS)
        .unwrap()
        .run(&mut pool_chunk(2))
        .unwrap();
}

fn adaptive_body(entries: &[(u32, &[u8])]) -> Vec<u8> {
    let mut out = u32::try_from(entries.len()).unwrap().to_le_bytes().to_vec();
    for (id, values) in entries {
        out.extend_from_slice(&id.to_le_bytes());
        out.extend_from_slice(values);
    }
    out
}

#[test]
fn test_replay_v1_adaptive_event_refusals() {
    let good = adaptive_body(&[(0, &[1, 2]), (2, &[3, 4])]);
    // Without a curriculum, tag 7 is refused at open.
    assert!(matches!(
        Replay::open(
            &with_event(0, EventTag::Adaptive as u8, good.clone()),
            LIMITS
        ),
        Err(ReplayError::ReservedTag { tag: 7 })
    ));
    // With one, a sound body parses, and replay refuses it for now.
    let file = with_event(1, EventTag::Adaptive as u8, good);
    let replay = Replay::open(&file, LIMITS).unwrap();
    assert!(replay.events().any(|e| matches!(
        e,
        Ok((_, Event::Adaptive { value_len: 2, ref entries })) if entries.len() == 2
    )));
    for bad in [
        adaptive_body(&[(1, &[1, 2]), (1, &[3, 4])]),
        adaptive_body(&[(2, &[1, 2]), (0, &[3, 4])]),
        adaptive_body(&[(N, &[1, 2])]),
        adaptive_body(&[]),
        // Entries of an id with no adaptive values (L = 0).
        adaptive_body(&[(0, &[]), (1, &[])]),
        {
            let mut b = adaptive_body(&[(0, &[1, 2]), (1, &[3, 4])]);
            b.pop();
            b
        },
    ] {
        let file = with_event(1, EventTag::Adaptive as u8, bad);
        assert!(matches!(
            first_event_error(&file),
            ReplayError::Malformed { .. }
        ));
    }
}

#[test]
fn test_replay_v1_reset_source_and_reset_from_refusals() {
    // ADR-0075 (#363): a fresh `ResetSource` replays as a call.
    let fresh = with_event(0, EventTag::ResetSource as u8, vec![0]);
    Replay::open(&fresh, LIMITS)
        .unwrap()
        .run(&mut pool())
        .unwrap();
    let mut from = 2u32.to_le_bytes().to_vec();
    for w in [0u32, 2, 5, 9] {
        from.extend_from_slice(&w.to_le_bytes());
    }
    let file = with_event(0, EventTag::ResetFrom as u8, from);
    let replay = Replay::open(&file, LIMITS).unwrap();
    assert!(replay.events().any(|e| matches!(
        e,
        Ok((_, Event::ResetFrom { ref ids, ref entries })) if ids == &[0, 2] && entries == &[5, 9]
    )));
    // A `reset_from` on a pool without an archive fails on replay.
    assert!(matches!(
        run_error(&file),
        ReplayError::Diverged {
            kind: DivergenceKind::CallFailed,
            ..
        }
    ));
    let words = |w: &[u32]| -> Vec<u8> { w.iter().flat_map(|x| x.to_le_bytes()).collect() };
    for (tag, bad) in [
        (EventTag::ResetSource, vec![2]),
        (EventTag::ResetSource, archive_block(1, &[0])),
        (EventTag::ResetSource, archive_block(1, &[])),
        (EventTag::ResetFrom, words(&[0])),
        (EventTag::ResetFrom, words(&[2, 1, 1, 0, 0])),
        (EventTag::ResetFrom, words(&[1, N, 0])),
        (EventTag::ResetFrom, words(&[1, 0, 1 << 24])),
        (EventTag::ResetFrom, words(&[2, 0, 1, 0])),
    ] {
        let file = with_event(0, tag as u8, bad);
        assert!(
            matches!(first_event_error(&file), ReplayError::Malformed { .. }),
            "{tag:?}"
        );
    }
}

#[test]
fn test_replay_v1_keyframe_archive_source_differing_from_the_header_is_malformed() {
    let mut p = split(&recording());
    // The first keyframe: branch counter (8), then the reset source byte.
    let body = &mut p.events[0].1;
    let mut source = archive_block(1, &[1]);
    source.remove(0);
    body[8] = 1;
    for (k, b) in source.into_iter().enumerate() {
        body.insert(9 + k, b);
    }
    let file = join(&p, 0);
    // ADR-0075 (#363): a keyframe names an archive and none is given:
    // refused before anything is replayed.
    assert!(matches!(
        run_error(&file),
        ReplayError::Engine(EngineError::ArchiveMismatch { found: 0, .. })
    ));
}

#[test]
fn test_replay_seek_keyframe_archive_source_without_the_archive_refused_pool_untouched() {
    // A keyframe whose reset source is an archive (ADR-0075, #363): keyframes()
    // lists it, and seek restores that source only with the archive it names.
    // A cursor without the archive is refused with `ArchiveMismatch` before
    // importing anything (REVIEW-4c-design-r1 point 2: restore the full
    // context or refuse).
    let mut p = split(&recording());
    let body = &mut p.events[0].1;
    let mut source = archive_block(1, &[1]);
    source.remove(0);
    body[8] = 1;
    for (k, b) in source.into_iter().enumerate() {
        body.insert(9 + k, b);
    }
    let file = join(&p, 0);
    let replay = Replay::open(&file, LIMITS).unwrap();
    assert!(replay.keyframes().unwrap().contains(&0));
    let mut pool = pool();
    pool.reset_all(Reseed::Base(Seed::new(11))).unwrap();
    let before: Vec<u64> = (0..N).map(|i| pool.state_hash(i).unwrap().get()).collect();
    let context = pool.pool_context().unwrap();
    // The cursor itself is refused (the file names an archive, none is
    // given), so no seek can write.
    assert!(matches!(
        replay.cursor(&pool),
        Err(ReplayError::Engine(EngineError::ArchiveMismatch {
            found: 0,
            ..
        }))
    ));
    let after: Vec<u64> = (0..N).map(|i| pool.state_hash(i).unwrap().get()).collect();
    assert_eq!(after, before);
    assert_eq!(pool.pool_context().unwrap(), context);
}

fn with_league_event(league: u8, tag: u8, body: Vec<u8>) -> Vec<u8> {
    let mut p = split(&recording());
    p.ext = ext_league(&[], 0, league, &[0], 64);
    p.events.push((tag, body));
    join(&p, 0)
}

#[test]
fn test_replay_v1_league_fields_are_reserved() {
    // ADR-0088 item 8: the flag is 0 or 1.
    assert_eq!(
        invalid_field(&with_ext(ext_league(&[], 0, 2, &[0], 64))),
        "has_league"
    );
    // A league opens, and replay refuses it until the league lands.
    let league = with_ext(ext_league(&[], 0, 1, &[0], 64));
    let replay = Replay::open(&league, LIMITS).unwrap();
    assert!(replay.header().has_league);
    assert!(matches!(
        replay.run(&mut pool()),
        Err(ReplayError::Unsupported {
            feature: "leagues (ADR-0088)"
        })
    ));
    // Group code 4, opponent, opens, and replay refuses it as a policy map.
    let opponent = with_ext(ext(&[0, 4], 0, &[0], 64));
    assert!(matches!(
        run_error(&opponent),
        ReplayError::Unsupported { .. }
    ));
}

#[test]
fn test_replay_v1_league_event_needs_a_league() {
    let body = vec![1, 2, 3];
    // Without a league, tag 12 is refused at open.
    assert!(matches!(
        Replay::open(
            &with_league_event(0, EventTag::League as u8, body.clone()),
            LIMITS
        ),
        Err(ReplayError::ReservedTag { tag: 12 })
    ));
    // With one, it opens with its body as opaque bytes.
    let file = with_league_event(1, EventTag::League as u8, body);
    let replay = Replay::open(&file, LIMITS).unwrap();
    assert!(
        replay
            .events()
            .any(|e| matches!(e, Ok((_, Event::League { body: &[1, 2, 3] }))))
    );
    assert_eq!(EventTag::from_u8(12), Some(EventTag::League));
    assert_eq!(EventTag::from_u8(13), None);
}
