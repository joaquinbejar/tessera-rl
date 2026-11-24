// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host snapshots (ADR-0073): the continuation test of SPEC 9.1 on a host
//! with heap state and its own RNG, check-then-apply, the identity digest,
//! poisoning, pending instances (ADR-0068 Follow-up), the host header rules
//! r2h (ADR-0068 item 5) and the host file.

#![cfg(feature = "host")]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers unwrap and slice: a failure is the assertion"
)]

mod host_script;
mod host_walk;

use core::num::{NonZeroU32, NonZeroUsize};
use std::sync::atomic::Ordering;

use host_script::{ABORT, NOTHING, SLOTS, Script, ScriptConfig, Shared, TERMINATE};
use host_walk::{Walk, WalkConfig};
use tessera_rl_core::encoding::{decode_from_slice, encode_to_vec};
use tessera_rl_core::{NoAdaptive, RecordHeader, RecordLayout};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodeLimits, EngineError, HOST_MAGIC, HostPool, HostRecord, MAGIC,
    Outputs, PersistError, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::{RngIdentity, Seed};

type WalkPool = HostPool<Walk, 1>;
type ScriptPool = HostPool<Script<0, false>, SLOTS>;

const LENGTH: u32 = 64;

fn options(n: u32, seed: u64) -> PoolOptions {
    let mut o = PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed));
    o.max_episode_steps = NonZeroU32::new(LENGTH);
    o
}

fn walk_pool(n: u32, seed: u64, workers: usize) -> WalkPool {
    let pool = WalkPool::new(WalkConfig { length: LENGTH }, options(n, seed)).unwrap();
    if workers == 0 {
        pool
    } else {
        pool.with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
            .unwrap()
    }
}

fn script_pool(shared: &std::sync::Arc<Shared>, n: u32, mode: AutoReset) -> ScriptPool {
    let mut o = options(n, 7);
    o.auto_reset = mode;
    ScriptPool::new(ScriptConfig::new(shared), o).unwrap()
}

/// Everything a call shows for instance `i`: observation bytes, rewards,
/// flags and pending state.
fn view(out: &Outputs, i: usize, agents: usize) -> Vec<u8> {
    let rows = i * agents..(i + 1) * agents;
    let mut v = Vec::new();
    let obs = out.observation(0).unwrap();
    let per = obs.len() / out.pending_reset().len();
    v.extend_from_slice(&obs[i * per..(i + 1) * per]);
    for r in &out.rewards()[rows.clone()] {
        v.extend_from_slice(&r.to_bits().to_le_bytes());
    }
    for f in [out.agent_terminated(), out.agent_truncated()] {
        v.extend(f[rows.clone()].iter().map(|b| u8::from(b.get())));
    }
    v.push(u8::from(out.pending_reset()[i].get()));
    v
}

fn walk_actions(t: u32, n: usize) -> Vec<i32> {
    (0..n)
        .map(|i| i32::try_from((t as usize * 7 + i * 3) % 3).unwrap())
        .collect()
}

/// Runs `k` steps of instance `i`'s view from step `t0`, with its hash
/// after each.
fn run(pool: &mut WalkPool, inst: usize, start: u32, count: u32) -> Vec<(Vec<u8>, u64)> {
    let n = pool.options().num_envs.get() as usize;
    (start..start + count)
        .map(|step| {
            let actions = walk_actions(step, n);
            let out = pool
                .step(ActionInput {
                    discrete: &actions,
                    continuous: &[],
                })
                .unwrap();
            let row = view(out, inst, 1);
            let hash = pool.state_hash(u32::try_from(inst).unwrap()).unwrap().get();
            (row, hash)
        })
        .collect()
}

#[test]
fn test_host_snapshot_continuation_same_other_slot_and_pool_matches() {
    for workers in [0, 2] {
        let mut a = walk_pool(4, 11, workers);
        a.reset_all(Reseed::Keep).unwrap();
        let _ = run(&mut a, 1, 0, 5);
        let record = a.snapshot(1).unwrap();
        let expected = run(&mut a, 1, 5, 9);

        // The same slot.
        a.restore(1, &record).unwrap();
        assert_eq!(run(&mut a, 1, 5, 9), expected, "same slot");

        // Another slot of another pool with another seed: actions follow
        // the slot, so drive slot 3 with slot 1's actions.
        let mut b = walk_pool(4, 999, workers);
        b.reset_all(Reseed::Keep).unwrap();
        b.restore(1, &record).unwrap();
        assert_eq!(run(&mut b, 1, 5, 9), expected, "another pool");
        assert_eq!(b.state_hash(1).unwrap(), a.state_hash(1).unwrap());
    }
}

#[test]
fn test_host_snapshot_restore_other_slot_observes_the_record() {
    let mut a = walk_pool(4, 11, 0);
    a.reset_all(Reseed::Keep).unwrap();
    let _ = run(&mut a, 1, 0, 5);
    let record = a.snapshot(1).unwrap();
    a.restore(3, &record).unwrap();
    let out = a.outputs().unwrap();
    assert_eq!(view(out, 3, 1)[..12], view(out, 1, 1)[..12]);
    assert_eq!(a.state_hash(3).unwrap(), a.state_hash(1).unwrap());
}

#[test]
fn test_host_snapshot_save_after_restore_same_bytes() {
    let mut a = walk_pool(2, 3, 0);
    a.reset_all(Reseed::Keep).unwrap();
    let _ = run(&mut a, 0, 0, 6);
    let record = a.snapshot(0).unwrap();
    a.restore(1, &record).unwrap();
    let again = a.snapshot(1).unwrap();
    assert_eq!(again.state_bytes(), record.state_bytes());
    assert_eq!(again.hash(), record.hash());
}

#[test]
fn test_host_snapshot_encode_decode_import_continues() {
    let mut a = walk_pool(3, 5, 0);
    a.reset_all(Reseed::Keep).unwrap();
    let _ = run(&mut a, 2, 0, 4);
    let file = a.encode(&[2, 0]).unwrap();
    assert_eq!(file[..8], HOST_MAGIC);
    let expected = run(&mut a, 2, 4, 6);

    let mut b = walk_pool(3, 77, 0);
    b.reset_all(Reseed::Keep).unwrap();
    let records = b.decode(&file, DecodeLimits::default()).unwrap();
    assert_eq!(records.len(), 2);
    b.import(&[2, 0], &records).unwrap();
    assert_eq!(run(&mut b, 2, 4, 6), expected);
}

#[test]
fn test_host_snapshot_fork_destinations_continue_as_source() {
    let mut a = walk_pool(4, 9, 0);
    a.reset_all(Reseed::Keep).unwrap();
    let _ = run(&mut a, 0, 0, 3);
    a.fork(0, &[2, 3]).unwrap();
    let h0 = a.state_hash(0).unwrap();
    assert_eq!(a.state_hash(2).unwrap(), h0);
    assert_eq!(a.state_hash(3).unwrap(), h0);
}

#[test]
fn test_host_snapshot_rejected_second_record_mutates_nothing() {
    let mut a = walk_pool(3, 1, 0);
    a.reset_all(Reseed::Keep).unwrap();
    let _ = run(&mut a, 0, 0, 3);
    let good = a.snapshot(0).unwrap();
    let mut bytes = good.to_bytes().unwrap();
    // Truncate the path: the count no longer matches the bytes.
    let n = bytes.len();
    bytes.truncate(n - 4);
    let len_at = 8 + HostRecord::<1>::header_len() + 4;
    let len = u32::from_le_bytes(bytes[len_at..len_at + 4].try_into().unwrap()) - 4;
    bytes[len_at..len_at + 4].copy_from_slice(&len.to_le_bytes());
    let bad = HostRecord::<1>::from_bytes(&bytes, 1 << 20).unwrap();
    let before: Vec<_> = (0..3).map(|i| a.state_hash(i).unwrap()).collect();
    let err = a
        .import(&[0, 1, 2], &[good.clone(), bad, good])
        .unwrap_err();
    assert!(
        matches!(
            err,
            EngineError::SnapshotRejected {
                index: 1,
                code: host_walk::COUNT,
                ..
            }
        ),
        "{err:?}"
    );
    let after: Vec<_> = (0..3).map(|i| a.state_hash(i).unwrap()).collect();
    assert_eq!(before, after);
    assert!(!a.is_poisoned());
}

/// Replaces the version field of an encoded record.
fn with_version(record: &HostRecord<1>, version: u32) -> HostRecord<1> {
    let mut bytes = record.to_bytes().unwrap();
    let at = 8 + HostRecord::<1>::header_len();
    bytes[at..at + 4].copy_from_slice(&version.to_le_bytes());
    HostRecord::from_bytes(&bytes, 1 << 20).unwrap()
}

#[test]
fn test_host_snapshot_other_version_or_identity_refused() {
    let mut a = walk_pool(2, 1, 0);
    a.reset_all(Reseed::Keep).unwrap();
    let record = a.snapshot(0).unwrap();
    assert!(matches!(
        a.restore(1, &with_version(&record, 2)),
        Err(EngineError::InvalidRecord { index: 0, reason }) if reason == "another SNAPSHOT_VERSION"
    ));
    let mut other = WalkPool::new(WalkConfig { length: LENGTH + 1 }, options(2, 1)).unwrap();
    other.reset_all(Reseed::Keep).unwrap();
    assert!(matches!(
        other.restore(0, &record),
        Err(EngineError::InvalidRecord { reason, .. }) if reason.starts_with("the identity digest")
    ));
}

#[test]
fn test_host_snapshot_identity_mismatch_refused_before_check_snapshot() {
    let shared = Shared::new();
    let mut a = script_pool(&shared, 2, AutoReset::NextStep);
    a.reset_all(Reseed::Keep).unwrap();
    let record = a.snapshot(0).unwrap();
    let mut cfg = ScriptConfig::new(&shared);
    cfg.initial = 2;
    let mut b = ScriptPool::new(cfg, options(2, 7)).unwrap();
    b.reset_all(Reseed::Keep).unwrap();
    let checks = shared.checks.load(Ordering::Relaxed);
    assert!(matches!(
        b.restore(0, &record),
        Err(EngineError::InvalidRecord { reason, .. }) if reason.starts_with("the identity digest")
    ));
    assert_eq!(shared.checks.load(Ordering::Relaxed), checks);
}

#[test]
fn test_host_snapshot_bytes_over_max_refused() {
    let shared = Shared::new();
    let mut a = script_pool(&shared, 2, AutoReset::NextStep);
    a.reset_all(Reseed::Keep).unwrap();
    let record = a.snapshot(0).unwrap();
    let mut bytes = record.to_bytes().unwrap();
    let at = 8 + HostRecord::<SLOTS>::header_len() + 4;
    bytes[at..at + 4].copy_from_slice(&9u32.to_le_bytes());
    bytes.push(0);
    let long = HostRecord::<SLOTS>::from_bytes(&bytes, 1 << 20).unwrap();
    assert!(matches!(
        a.restore(1, &long),
        Err(EngineError::InvalidRecord { reason, .. }) if reason == "state bytes over max_snapshot_len"
    ));
    assert!(HostRecord::<SLOTS>::from_bytes(&bytes, 8).is_err());
}

#[test]
fn test_host_snapshot_save_over_bound_poisons_in_snapshot_and_fork() {
    for fork in [false, true] {
        let shared = Shared::new();
        let mut a = script_pool(&shared, 2, AutoReset::NextStep);
        a.reset_all(Reseed::Keep).unwrap();
        shared.save_extra.store(true, Ordering::Relaxed);
        let err = if fork {
            a.fork(0, &[1]).unwrap_err()
        } else {
            a.snapshot(0).unwrap_err()
        };
        assert!(
            matches!(
                err,
                EngineError::SnapshotTooLong {
                    instance: 0,
                    len: 9,
                    max: 8
                }
            ),
            "{err:?}"
        );
        assert!(a.is_poisoned());
    }
}

#[test]
fn test_host_snapshot_panics_in_hooks_poison() {
    for hook in 0..3 {
        let shared = Shared::new();
        let mut a = script_pool(&shared, 2, AutoReset::NextStep);
        a.reset_all(Reseed::Keep).unwrap();
        let record = a.snapshot(0).unwrap();
        let flag = [&shared.panic_save, &shared.panic_check, &shared.panic_load][hook];
        flag.store(true, Ordering::Relaxed);
        let err = if hook == 0 {
            a.snapshot(0).map(|_| ()).unwrap_err()
        } else {
            a.restore(1, &record).unwrap_err()
        };
        assert!(matches!(err, EngineError::Panicked), "{hook}: {err:?}");
        assert!(a.is_poisoned());
    }
}

/// ADR-0068 item 5: a poisoned host pool refuses every snapshot operation,
/// the recovery paths of `EnvPool` included.
#[test]
fn test_host_snapshot_poisoned_pool_refuses_everything() {
    let shared = Shared::new();
    let mut a = script_pool(&shared, 2, AutoReset::NextStep);
    a.reset_all(Reseed::Keep).unwrap();
    let record = a.snapshot(0).unwrap();
    shared.panic_save.store(true, Ordering::Relaxed);
    assert!(a.snapshot(0).is_err());
    shared.panic_save.store(false, Ordering::Relaxed);
    assert!(a.is_poisoned());
    let loads = shared.loads.load(Ordering::Relaxed);
    let poisoned = |r: Result<(), EngineError>| matches!(r, Err(EngineError::Poisoned));
    assert!(poisoned(a.snapshot(0).map(|_| ())));
    assert!(poisoned(
        a.snapshot_all(&mut [HostRecord::new(), HostRecord::new()])
    ));
    assert!(poisoned(a.state_hash(0).map(|_| ())));
    assert!(poisoned(a.restore(0, &record)));
    assert!(poisoned(a.import(&[0], core::slice::from_ref(&record))));
    assert!(poisoned(a.restore_all(&[record.clone(), record.clone()])));
    assert!(poisoned(a.fork(0, &[1])));
    assert!(matches!(
        a.encode(&[0]),
        Err(PersistError::Engine(EngineError::Poisoned))
    ));
    assert_eq!(shared.loads.load(Ordering::Relaxed), loads);
}

fn step_all(pool: &mut ScriptPool, action: i32) -> Result<(), EngineError> {
    let n = pool.options().num_envs.get() as usize;
    let a = vec![action; n * SLOTS];
    pool.step(ActionInput {
        discrete: &a,
        continuous: &[],
    })
    .map(|_| ())
}

/// Calls of `observe`, `mask` and `acting` summed over instances.
fn observed(shared: &Shared) -> u64 {
    (0..host_script::MAX_INSTANCES)
        .map(|i| {
            Shared::get(&shared.observes, i)
                + Shared::get(&shared.masks, i)
                + Shared::get(&shared.actings, i)
        })
        .sum()
}

/// ADR-0068 item 5 and Follow-up: every aborted host header (an aborted
/// explicit reset: Live; an aborted same-call auto-reset: Scratch; an
/// aborted next-step auto-reset: Live) round-trips through the reader in
/// each auto-reset mode, and its restore observes nothing.
#[test]
fn test_host_snapshot_aborted_headers_round_trip_unobserved() {
    for mode in [
        AutoReset::NextStep,
        AutoReset::SameStep,
        AutoReset::Disabled,
    ] {
        for arm in 0..2 {
            let shared = Shared::new();
            let mut a = script_pool(&shared, 2, mode);
            if arm == 0 {
                // Live: an explicit reset aborts.
                shared.abort_resets(0, 1);
                a.reset_all(Reseed::Keep).unwrap();
            } else {
                // The episode ends; the auto-reset (same call in SameStep,
                // next call in NextStep) aborts. Disabled never resets.
                a.reset_all(Reseed::Keep).unwrap();
                shared.abort_resets(0, 1);
                step_all(&mut a, TERMINATE).unwrap();
                if mode == AutoReset::NextStep {
                    step_all(&mut a, NOTHING).unwrap();
                }
            }
            let record = a.snapshot(0).unwrap();
            assert!(record.pending_reset(), "{mode:?} arm {arm}");
            let mut b = script_pool(&shared, 2, mode);
            b.reset_all(Reseed::Keep).unwrap();
            let before = observed(&shared);
            b.restore(1, &record).unwrap();
            assert_eq!(observed(&shared), before, "{mode:?} arm {arm}: observed");
            let out = b.outputs().unwrap();
            assert!(out.pending_reset()[1].get());
            let obs = out.observation(0).unwrap();
            let per = obs.len() / 2;
            assert!(obs[per..].iter().all(|&x| x == 0));
            assert_eq!(b.state_hash(1).unwrap(), record.hash());
        }
    }
}

/// An aborted step leaves the instance pending too.
#[test]
fn test_host_snapshot_aborted_step_round_trips() {
    for mode in [AutoReset::NextStep, AutoReset::Disabled] {
        let shared = Shared::new();
        let mut a = script_pool(&shared, 1, mode);
        a.reset_all(Reseed::Keep).unwrap();
        step_all(&mut a, ABORT).unwrap();
        let record = a.snapshot(0).unwrap();
        assert!(record.pending_reset());
        let mut b = script_pool(&shared, 1, mode);
        b.restore(0, &record).unwrap();
        assert_eq!(b.state_hash(0).unwrap(), record.hash());
    }
}

type Header = RecordHeader<SLOTS, NoAdaptive>;

/// Rewrites the header of an encoded record.
fn with_header(record: &HostRecord<SLOTS>, edit: impl FnOnce(&mut Header)) -> HostRecord<SLOTS> {
    let mut bytes = record.to_bytes().unwrap();
    let range = 8..8 + HostRecord::<SLOTS>::header_len();
    let mut header: Header = decode_from_slice(&bytes[range.clone()]).unwrap();
    edit(&mut header);
    bytes[range].copy_from_slice(&encode_to_vec(&header));
    HostRecord::from_bytes(&bytes, 1 << 20).unwrap()
}

/// ADR-0068 item 3: r2h, not r2. An aborted `same_step` header (pending,
/// not fresh, no steps) is accepted; the same header with a step counted
/// is refused.
#[test]
fn test_host_snapshot_r2h_accepts_aborted_refuses_steps_counted() {
    let shared = Shared::new();
    let mut a = script_pool(&shared, 2, AutoReset::SameStep);
    a.reset_all(Reseed::Keep).unwrap();
    shared.abort_resets(0, 1);
    step_all(&mut a, TERMINATE).unwrap();
    let aborted = a.snapshot(0).unwrap();
    let engine = |r: &HostRecord<SLOTS>| {
        let bytes = r.to_bytes().unwrap();
        let h: Header =
            decode_from_slice(&bytes[8..8 + HostRecord::<SLOTS>::header_len()]).unwrap();
        *h.engine()
    };
    let e = engine(&aborted);
    assert!(e.pending_reset.get() && !e.fresh.get() && e.episode_steps == 0);
    a.restore(1, &aborted).unwrap();

    let counted = with_header(&aborted, |h| {
        let id = h.engine().identity;
        let e = h.engine_mut();
        e.episode_steps = 1;
        e.identity = RngIdentity::from_parts(
            id.seed(),
            id.branch_salt(),
            id.logical_id(),
            id.episode_index(),
            1,
        );
    });
    assert!(matches!(
        a.restore(1, &counted),
        Err(EngineError::InvalidRecord { reason, .. })
            if reason == "a pending reset in same_step mode with steps counted"
    ));
}

#[test]
fn test_host_snapshot_host_file_and_env_file_refuse_each_other() {
    let mut a = walk_pool(2, 1, 0);
    a.reset_all(Reseed::Keep).unwrap();
    let mut file = a.encode(&[0]).unwrap();
    file[..8].copy_from_slice(&MAGIC);
    assert!(matches!(
        a.decode(&file, DecodeLimits::default()),
        Err(PersistError::BadMagic)
    ));
}

#[test]
fn test_host_snapshot_file_other_library_version_imports() {
    let mut a = walk_pool(2, 1, 0);
    a.reset_all(Reseed::Keep).unwrap();
    let file = a.encode_with(&[0], "9.9.9").unwrap();
    let records = a.decode(&file, DecodeLimits::default()).unwrap();
    a.import(&[1], &records).unwrap();
}

#[test]
fn test_host_snapshot_file_record_over_limit_refused() {
    let mut a = walk_pool(2, 1, 0);
    a.reset_all(Reseed::Keep).unwrap();
    let _ = run(&mut a, 0, 0, 3);
    let file = a.encode(&[0]).unwrap();
    let limits = DecodeLimits {
        max_host_record_bytes: 8,
        ..DecodeLimits::default()
    };
    assert!(matches!(
        a.decode(&file, limits),
        Err(PersistError::FieldTooLong { .. })
    ));
    let mut damaged = file.clone();
    let k = damaged.len() - 12;
    damaged[k] ^= 1;
    assert!(matches!(
        a.decode(&damaged, DecodeLimits::default()),
        Err(PersistError::Checksum)
    ));
}

/// ADR-0073 item 4: through `DynHostPool`, a pool built without
/// `with_dyn_snapshots` has none (`NoSnapshots`); with it, encoded records
/// round-trip.
#[test]
fn test_host_snapshot_dyn_pool_needs_with_dyn_snapshots() {
    use tessera_rl_engine::DynHostPool;

    let mut plain: Box<dyn DynHostPool> = Box::new(walk_pool(2, 1, 0));
    assert!(!plain.snapshots() && !plain.layout().snapshots);
    assert!(matches!(
        plain.snapshot_records(&[0]),
        Err(EngineError::NoSnapshots)
    ));
    assert!(matches!(plain.fork(0, &[1]), Err(EngineError::NoSnapshots)));

    let mut pool: Box<dyn DynHostPool> = Box::new(walk_pool(2, 1, 0).with_dyn_snapshots());
    assert!(pool.snapshots() && pool.layout().snapshots);
    pool.reset_all(Reseed::Keep).unwrap();
    let records = pool.snapshot_records(&[0]).unwrap();
    let hash = pool.state_hash(0).unwrap();
    pool.restore_records(&[1], &[&records[0]]).unwrap();
    assert_eq!(pool.state_hash(1).unwrap(), hash);
    let file = pool.encode(&[0, 1]).unwrap();
    pool.import_file(&[1, 0], &file, DecodeLimits::default())
        .unwrap();
    assert!(matches!(
        pool.restore_records(&[0], &[&records[0][..4]]),
        Err(EngineError::InvalidRecord { index: 0, .. })
    ));
}

/// Review of #344: `max_snapshot_len` over `MAX_HOST_SNAPSHOT_BYTES` is
/// refused before anything is reserved, without poisoning.
#[test]
fn test_host_snapshot_max_len_over_cap_refused() {
    let mut pool = WalkPool::new(WalkConfig { length: u32::MAX }, options(1, 0)).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    assert!(matches!(
        pool.snapshot(0),
        Err(EngineError::SnapshotBound { cap, .. }) if cap == tessera_rl_engine::MAX_HOST_SNAPSHOT_BYTES
    ));
    assert!(!pool.is_poisoned());
}

/// Review of #344: `max_snapshot_len` is author code, run once under the
/// panic boundary: a panic poisons.
#[test]
fn test_host_snapshot_max_len_panic_poisons() {
    let shared = Shared::new();
    let mut a = script_pool(&shared, 2, AutoReset::NextStep);
    a.reset_all(Reseed::Keep).unwrap();
    shared.panic_max_len.store(true, Ordering::Relaxed);
    assert!(matches!(a.snapshot(0), Err(EngineError::Panicked)));
    assert!(a.is_poisoned());
}

/// The host file layout, version 1 (ADR-0073 item 5), pinned field by
/// field: magic, format version, the ADR-0011 header fields with tier 2,
/// the host descriptor, then one record (identity, header, version,
/// length, bytes) and the checksum.
#[test]
fn test_host_snapshot_file_layout_v1_is_pinned() {
    let mut a = walk_pool(1, 5, 0);
    a.reset_all(Reseed::Keep).unwrap();
    let file = a.encode_with(&[0], "9.9.9").unwrap();
    let identity = a.snapshot_identity().unwrap();
    let record = a.snapshot(0).unwrap().to_bytes().unwrap();
    let mut want = Vec::new();
    want.extend_from_slice(b"TESHOST\x1a");
    want.extend_from_slice(&1u32.to_le_bytes()); // format version
    want.extend_from_slice(&[0, 2, 0, 0, 0, 0, 0, 0]); // raw, tier 2, mask, next_step, illegal, reserved
    want.extend_from_slice(&0u32.to_le_bytes()); // penalty bits
    want.extend_from_slice(&LENGTH.to_le_bytes()); // max_episode_steps
    want.extend_from_slice(&1u32.to_le_bytes()); // ENV_VERSION
    want.extend_from_slice(&file[32..40]); // config hash, checked below
    want.extend_from_slice(&1u32.to_le_bytes()); // count
    let header_len = u32::try_from(HostRecord::<1>::header_len()).unwrap();
    want.extend_from_slice(&header_len.to_le_bytes()); // record_len
    for part in [&b"test/Walk-v1"[..], b"9.9.9"] {
        want.extend_from_slice(&u16::try_from(part.len()).unwrap().to_le_bytes());
        want.extend_from_slice(part);
    }
    want.extend_from_slice(&0u16.to_le_bytes()); // assets
    // Descriptor: version 1, SNAPSHOT_VERSION 1, agents 1, initial 1,
    // SPAWNS and MASKS false, three reserved bytes.
    want.extend_from_slice(&14u16.to_le_bytes());
    want.extend_from_slice(&[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]);
    want.extend_from_slice(&record);
    assert_eq!(&file[..file.len() - 8], &want[..]);
    assert_eq!(record[..8], identity.to_le_bytes());
    let hash =
        tessera_rl_core::config::config_hash("test/Walk-v1", 1, &WalkConfig { length: LENGTH })
            .unwrap();
    assert_eq!(file[32..40], hash.to_le_bytes());
    let sum = tessera_rl_core::encoding::checksum(&file[..file.len() - 8]);
    assert_eq!(file[file.len() - 8..], sum.to_le_bytes());
}
