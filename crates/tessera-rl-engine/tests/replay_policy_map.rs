// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replay of a pool with a policy map (ADR-0079 items 6 and 10, erratum B2,
//! #347 stage B2): the sync and async recorders write the map into the
//! header; a recording with a heuristic slot replays into a pool built with
//! the same map and reproduces its state hashes; a pool with another map
//! (the default one included) is refused with `PolicyMapMismatch` before any
//! event, the pool unchanged; an external-only map records and replays.

#![cfg(all(feature = "replay", not(loom)))]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests: a failure is the assertion"
)]

mod common;

use core::num::{NonZeroU32, NonZeroUsize};

use common::{CountConfig, Counter, Counts};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic, PolicyGroup};
use tessera_rl_engine::replay::{
    AsyncRecorder, Recorder, Replay, ReplayError, ReplayWriterOptions,
};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AsyncPool, DecodeLimits, EngineError, EnvPool, Masked, PolicyMap,
    PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::StepRng;

const N: u32 = 8;
const NU: usize = 8;
const STEPS: usize = 12;

const LIMITS: DecodeLimits = DecodeLimits {
    max_instances: 1 << 24,
    max_decoded_bytes: 1 << 30,
    max_host_record_bytes: 1 << 26,
};

/// The heuristic of `tests/heuristic_split.rs` (without its out-of-range
/// switches): it draws from its RNG, so a replay that read the submitted
/// rows instead of recomputing them would diverge.
impl Heuristic for Counter {
    fn act(
        _cfg: &CountConfig,
        _state: &Counts,
        agent: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let draw = rng.below_u32(NonZeroU32::new(2).unwrap_or(NonZeroU32::MIN));
        let action = i32::try_from(draw).unwrap_or(0) + i32::try_from(agent.index()).unwrap_or(0);
        out.set_discrete(0, action);
    }
}

fn cfg() -> CountConfig {
    CountConfig {
        length: 7,
        ..CountConfig::default()
    }
}

fn plain() -> EnvPool<Counter, Masked> {
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(2).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(31))
    };
    EnvPool::with_masks(cfg(), options).unwrap()
}

fn heuristic() -> EnvPool<Counter, Masked> {
    plain().with_heuristic_slots(&[1]).unwrap()
}

fn external() -> EnvPool<Counter, Masked> {
    plain()
        .with_policy_map(PolicyMap::new(2).assign(1, PolicyGroup::External).unwrap())
        .unwrap()
}

fn asy(pool: EnvPool<Counter, Masked>, batch: usize) -> AsyncPool<Counter, Masked> {
    pool.async_mode(AsyncOptions::new(
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(batch).unwrap(),
    ))
    .unwrap()
}

/// The action rows of `ids` at step `t`: a learner action in slot 0 and a
/// garbage row (ignored for a heuristic slot, legal for an external one)
/// in slot 1.
fn rows(ids: &[u32], t: usize) -> Vec<i32> {
    ids.iter()
        .flat_map(|&i| {
            let i = usize::try_from(i).unwrap();
            [i32::from((i + t).is_multiple_of(2)), 1]
        })
        .collect()
}

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

fn options() -> ReplayWriterOptions {
    ReplayWriterOptions {
        keyframe_every: 5,
        checkpoint_every: 3,
        outputs_hash: false,
    }
}

fn hashes(p: &EnvPool<Counter, Masked>) -> Vec<u64> {
    (0..N).map(|i| p.state_hash(i).unwrap().get()).collect()
}

fn async_hashes(p: &mut AsyncPool<Counter, Masked>) -> Vec<u64> {
    (0..N)
        .map(|i| p.snapshot(i).unwrap().hash().get())
        .collect()
}

/// A sync recording of `pool`; returns the file and the final hashes.
fn record_sync(mut pool: EnvPool<Counter, Masked>) -> (Vec<u8>, Vec<u64>) {
    let all: Vec<u32> = (0..N).collect();
    let mut rec = Recorder::new(&mut pool, Vec::new(), options()).unwrap();
    rec.reset_all(Reseed::Base(Seed::new(3))).unwrap();
    for t in 0..STEPS {
        rec.step(input(&rows(&all, t))).unwrap();
    }
    let bytes = rec.finish().unwrap();
    (bytes, hashes(&pool))
}

/// An async recording of `pool` with shuffled subset sends; returns the
/// file and the final hashes.
fn record_async(pool: EnvPool<Counter, Masked>) -> (Vec<u8>, Vec<u64>) {
    let mut p = asy(pool, 2);
    let mut rec = AsyncRecorder::new(&mut p, Vec::new(), options()).unwrap();
    drop(rec.reset_all(Reseed::Base(Seed::new(3))).unwrap());
    let mut ids: Vec<u32> = (0..N).collect();
    let mut state = 0x9e37_79b9_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        usize::try_from(state >> 33).unwrap()
    };
    for t in 0..STEPS {
        for k in (1..NU).rev() {
            ids.swap(k, next() % (k + 1));
        }
        let mut sent = 0;
        while sent < NU {
            let take = (1 + next() % 3).min(NU - sent);
            let subset = &ids[sent..sent + take];
            rec.send(subset, input(&rows(subset, t))).unwrap();
            sent += take;
        }
        let mut back = 0;
        while back < NU {
            back += rec.recv().unwrap().ids().count();
        }
    }
    let bytes = rec.finish().unwrap();
    (bytes, async_hashes(&mut p))
}

fn mismatch(err: &ReplayError) -> Option<(u64, u64)> {
    match err {
        ReplayError::Engine(EngineError::PolicyMapMismatch { expected, found }) => {
            Some((*expected, *found))
        }
        _ => None,
    }
}

#[test]
fn test_replay_policy_map_header_round_trips_the_codes() {
    let (bytes, _) = record_sync(heuristic());
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let header = replay.header();
    assert_eq!(header.policy_map, [0, 2]);
    let map = heuristic().policy_map().clone();
    assert_eq!(header.policy_map().unwrap(), Some(map.clone()));
    assert_eq!(header.policy_map_digest().unwrap(), map.digest());
    // The default map is a count of 0.
    let (bytes, _) = record_sync(plain());
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    assert_eq!(replay.header().policy_map, &[] as &[u8]);
    assert_eq!(replay.header().policy_map().unwrap(), None);
    assert_eq!(replay.header().policy_map_digest().unwrap(), 0);
}

#[test]
fn test_replay_policy_map_sync_heuristic_recording_reproduces_hashes() {
    let (bytes, recorded) = record_sync(heuristic());
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let mut pool = heuristic();
    let summary = replay.run(&mut pool).unwrap();
    assert!(summary.verified());
    assert!(summary.keyframes >= 1);
    assert_eq!(hashes(&pool), recorded);
    // The heuristic slot changed the trajectory: a default-map recording
    // with the same submitted rows ends elsewhere.
    let (_, plain_hashes) = record_sync(plain());
    assert_ne!(plain_hashes, recorded);
}

#[test]
fn test_replay_policy_map_async_shuffled_recording_reproduces_hashes() {
    let (bytes, recorded) = record_async(heuristic());
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    assert_eq!(replay.header().policy_map, [0, 2]);
    let mut pool = asy(heuristic(), 1);
    let summary = replay.run_async(&mut pool).unwrap();
    assert!(summary.verified());
    assert_eq!(async_hashes(&mut pool), recorded);
}

#[test]
fn test_replay_policy_map_other_map_refused_before_any_event() {
    let (bytes, _) = record_sync(heuristic());
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let recorded = heuristic().policy_map().digest();
    for mut pool in [plain(), external()] {
        pool.reset_all(Reseed::Base(Seed::new(77))).unwrap();
        let before = hashes(&pool);
        let err = replay.run(&mut pool).unwrap_err();
        assert_eq!(
            mismatch(&err),
            Some((pool.policy_map().digest(), recorded)),
            "{err:?}"
        );
        assert_eq!(hashes(&pool), before);
        assert!(!pool.is_poisoned());
        // The cursor entry point refuses it the same way.
        let err = replay.cursor(&pool).unwrap_err();
        assert!(mismatch(&err).is_some(), "{err:?}");
    }
    // A default-map file into a pool with a map is refused too.
    let (bytes, _) = record_sync(plain());
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let mut pool = heuristic();
    pool.reset_all(Reseed::Base(Seed::new(77))).unwrap();
    let before = hashes(&pool);
    let err = replay.run(&mut pool).unwrap_err();
    assert_eq!(mismatch(&err), Some((recorded, 0)), "{err:?}");
    assert_eq!(hashes(&pool), before);
}

#[test]
fn test_replay_policy_map_async_other_map_refused_before_any_event() {
    let (bytes, _) = record_async(heuristic());
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let mut pool = asy(plain(), 1);
    drop(pool.reset_all(Reseed::Base(Seed::new(77))).unwrap());
    let before = async_hashes(&mut pool);
    let err = replay.run_async(&mut pool).unwrap_err();
    assert!(mismatch(&err).is_some(), "{err:?}");
    assert_eq!(async_hashes(&mut pool), before);
}

#[test]
fn test_replay_policy_map_external_only_map_records_and_replays() {
    let (bytes, recorded) = record_sync(external());
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    assert_eq!(replay.header().policy_map, [0, 1]);
    let mut pool = external();
    assert!(replay.run(&mut pool).unwrap().verified());
    assert_eq!(hashes(&pool), recorded);
    // An external slot is not a learner: a default-map pool is refused.
    assert!(mismatch(&replay.run(&mut plain()).unwrap_err()).is_some());

    let (bytes, recorded) = record_async(external());
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let mut pool = asy(external(), 1);
    assert!(replay.run_async(&mut pool).unwrap().verified());
    assert_eq!(async_hashes(&mut pool), recorded);
}

#[test]
fn test_replay_policy_map_cursor_seek_to_a_keyframe_reproduces_hashes() {
    // A seek applies the keyframe with the header map's digest as its pool
    // context, so a heuristic pool accepts it and replays to the same end.
    let (bytes, recorded) = record_sync(heuristic());
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let keyframes = replay.keyframes().unwrap();
    let target = *keyframes.last().unwrap();
    let mut pool = heuristic();
    let mut cur = replay.cursor(&pool).unwrap();
    replay.seek(&mut pool, &mut cur, target).unwrap();
    assert_eq!(
        replay.advance(&mut pool, &mut cur, |_| false).unwrap(),
        None
    );
    drop(replay.finish(cur).unwrap());
    assert_eq!(hashes(&pool), recorded);
}

/// `b` with the library version string and the header checksum cut out
/// (both change with a release), as `replay.rs`'s pinned fixture does:
/// the length and XXH3-64 of the whole file, and the XXH3-64 of the header
/// part alone.
fn pinned(b: &[u8]) -> ((usize, u64), u64) {
    let u16_at = |at: usize| usize::from(u16::from_le_bytes([b[at], b[at + 1]]));
    let lib_at = 48 + 2 + u16_at(48);
    let lib_len = u16_at(lib_at);
    let mut at = lib_at + 2 + lib_len;
    at += 2; // asset_count
    at += 2 + u16_at(at); // descriptor
    let config = u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize;
    // The extension: flags and host fields (4) after the config and the
    // three counts (12), then the policy map, the two flags, the reset
    // source (Fresh here) and the chunk size.
    let mut q = at + 4 + config + 12 + 4;
    q += 2 + u16_at(q) + 2 + 1 + 4;
    let header_end = q + 8;
    let mut out = b[..lib_at + 2].to_vec();
    out.extend_from_slice(&b[lib_at + 2 + lib_len..header_end - 8]);
    let header = tessera_rl_core::encoding::checksum(&out);
    out.extend_from_slice(&b[header_end..]);
    (
        (out.len(), tessera_rl_core::encoding::checksum(&out)),
        header,
    )
}

/// #384 head D (ADR-0088 erratum 3, rev's condition): a file without a
/// league is byte-identical to one written before league replay landed.
/// The pins are the bytes main c98e1890 writes for these recordings (a
/// sync and an async recording with heuristic slots, codes 0 and 2), with
/// the version string and header checksum cut out: the whole sync file,
/// and the async file's header and length (its events follow the order
/// `recv` returns ids in, which depends on worker timing, so only those
/// are reproducible).
#[test]
fn test_replay_policy_map_files_without_a_league_are_byte_identical() {
    let (sync, _) = record_sync(heuristic());
    let (asy, _) = record_async(heuristic());
    let (sync_file, _) = pinned(&sync);
    let ((async_len, _), async_header) = pinned(&asy);
    assert_eq!(
        (sync_file, async_len, async_header),
        (PINNED_SYNC, PINNED_ASYNC_LEN, PINNED_ASYNC_HEADER),
        "a file without a league changed"
    );
}

const PINNED_SYNC: (usize, u64) = (4044, 5_428_336_282_661_761_961);
const PINNED_ASYNC_LEN: usize = 14344;
const PINNED_ASYNC_HEADER: u64 = 10_175_052_044_108_022_029;
