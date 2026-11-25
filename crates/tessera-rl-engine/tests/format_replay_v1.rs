// SPDX-License-Identifier: MIT OR Apache-2.0

//! The replay file, version 1, against its committed fixture
//! `tests/fixtures/formats/replay-v1.bin` (ADR-0072): a recording of the
//! Counter environment with resets, a fork and steps. It opens with the
//! recorded header fields, replays on a fresh pool with no divergence and to
//! the recording pool's states, today's writer writes the same bytes apart
//! from the library version and the header checksum over it, and a newer or
//! another-major version is refused with the typed error naming it. A
//! changed writer is resolved by a new version that keeps this file and its
//! read-back test, or by a fix; the fixture is regenerated only by
//! `test_format_replay_v1_write_fixture` with `TESSERA_BLESS_FORMATS=1`.

#![cfg(feature = "replay")]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests: a failure is the assertion"
)]

mod common;

use core::num::NonZeroU32;

use common::{CountConfig, Counter};
use tessera_rl_engine::replay::{
    FORMAT_VERSION, REPLAY_MAGIC, Recorder, Replay, ReplayError, ReplayWriterOptions,
};
use tessera_rl_engine::{ActionInput, DecodeLimits, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

type Pool = EnvPool<Counter, Masked>;

const FIXTURE: &[u8] = include_bytes!("fixtures/formats/replay-v1.bin");
const N: u32 = 4;
const LIMITS: DecodeLimits = DecodeLimits {
    max_instances: 1 << 24,
    max_decoded_bytes: 1 << 30,
    max_host_record_bytes: 1 << 26,
};

fn pool() -> Pool {
    let cfg = CountConfig {
        length: 6,
        ..CountConfig::default()
    };
    let options = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(11));
    EnvPool::with_masks(cfg, options).unwrap()
}

/// Two agents per instance; agent 0 never takes the action Counter masks.
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

/// The fixture's recording: a seeded reset, a step, a reseeded reset of
/// one instance, a branched fork, a step. Returns the file and the pool.
fn record() -> (Vec<u8>, Pool) {
    let mut p = pool();
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
    (bytes, p)
}

/// The string at `at`, a u16 length then its bytes, and the offset past it.
fn string_at(b: &[u8], at: usize) -> (&str, usize) {
    let len = usize::from(u16::from_le_bytes([b[at], b[at + 1]]));
    (
        core::str::from_utf8(&b[at + 2..at + 2 + len]).unwrap(),
        at + 2 + len,
    )
}

/// Byte offset of the first event: past the shared ADR-0011 header and the
/// replay extension (as `tests/replay.rs` reads it).
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

/// Offset just past the version 1 extension that starts at `q`.
fn past_extension(b: &[u8], mut q: usize) -> usize {
    let slots = usize::from(u16::from_le_bytes([b[q], b[q + 1]]));
    q += 2 + slots + 2;
    if b[q] == 1 {
        let n = u32::from_le_bytes(b[q + 9..q + 13].try_into().unwrap()) as usize;
        q += 1 + 8 + 4 + 4 * n;
    } else {
        q += 1;
    }
    q + 4
}

/// `b` without its library version string (its length field kept) and
/// without the header checksum: the parts that change with the release.
fn without_release_parts(b: &[u8]) -> Vec<u8> {
    let (_, lib_at) = string_at(b, 48);
    let (_, past_lib) = string_at(b, lib_at);
    let header_end = header_len(b);
    let mut out = b[..lib_at + 2].to_vec();
    out.extend_from_slice(&b[past_lib..header_end - 8]);
    out.extend_from_slice(&b[header_end..]);
    out
}

#[test]
#[ignore = "writes the fixture; run with TESSERA_BLESS_FORMATS=1"]
fn test_format_replay_v1_write_fixture() {
    if std::env::var("TESSERA_BLESS_FORMATS").as_deref() != Ok("1") {
        return;
    }
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/formats/replay-v1.bin"
    );
    std::fs::write(path, record().0).unwrap();
}

#[test]
fn test_format_replay_v1_fixture_reads_back() {
    // The header fields, read literally.
    assert_eq!(&FIXTURE[..8], &REPLAY_MAGIC);
    assert_eq!(u32::from_le_bytes(FIXTURE[8..12].try_into().unwrap()), 1);
    assert_eq!(FIXTURE[13], 0, "tier: first party");
    assert_eq!(string_at(FIXTURE, 48).0, "tessera/Counter-v1");
    // The events, replayed on a fresh pool: no divergence, the recorded
    // counts, and the recording pool's final states.
    let replay = Replay::open(FIXTURE, LIMITS).unwrap();
    let mut p = pool();
    let summary = replay.run(&mut p).unwrap();
    assert_eq!(
        (
            summary.events,
            summary.calls,
            summary.checkpoints,
            summary.keyframes
        ),
        (11, 5, 3, 3),
        "{summary:?}"
    );
    assert!(
        !summary.poisoned && summary.unclaimed.is_empty(),
        "{summary:?}"
    );
    let (_, recorded) = record();
    for i in 0..N {
        assert_eq!(
            p.state_hash(i).unwrap(),
            recorded.state_hash(i).unwrap(),
            "instance {i}"
        );
    }
}

#[test]
fn test_format_replay_v1_writer_writes_the_fixture() {
    assert_eq!(FORMAT_VERSION, 1);
    let (bytes, _) = record();
    assert_eq!(
        without_release_parts(&bytes),
        without_release_parts(FIXTURE)
    );
}

#[test]
fn test_format_replay_newer_and_other_major_versions_are_refused() {
    for found in [2u32, 1 << 16] {
        let mut b = FIXTURE.to_vec();
        b[8..12].copy_from_slice(&found.to_le_bytes());
        assert!(matches!(
            Replay::open(&b, LIMITS),
            Err(ReplayError::UnsupportedVersion { found: f }) if f == found
        ));
    }
    let message = ReplayError::UnsupportedVersion { found: 2 }.to_string();
    assert!(
        message.contains('2') && message.contains("version"),
        "{message}"
    );
}
