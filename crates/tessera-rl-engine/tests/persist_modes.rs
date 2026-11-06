// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persisted snapshots in the split-phase modes (ADR-0011, ADR-0047): a file moves
//! between modes, its refusals, recovery and the tokens around it.

#![cfg(not(loom))]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers unwrap and slice: a failure is the assertion"
)]

mod common;

use core::num::{NonZeroU32, NonZeroUsize};

use common::{CountConfig, Counter};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AsyncPool, DecodeLimits, DecodedRecord, EngineError, EnvPool,
    Masked, PersistError, PipelinedOptions, PipelinedPool, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 32;
const LIMITS: DecodeLimits = DecodeLimits {
    max_instances: 1 << 24,
    max_decoded_bytes: 1 << 30,
};

type Pool = EnvPool<Counter, Masked>;
type Piped = PipelinedPool<Counter, Masked>;
type Asy = AsyncPool<Counter, Masked>;

fn options() -> PoolOptions {
    PoolOptions {
        chunk_size: NonZeroU32::new(4).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(5))
    }
}

fn cfg() -> CountConfig {
    CountConfig {
        length: 50,
        ..CountConfig::default()
    }
}

fn sync_with(cfg: CountConfig) -> Pool {
    EnvPool::with_masks(cfg, options()).unwrap()
}

fn piped(cfg: CountConfig) -> Piped {
    sync_with(cfg)
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
        .unwrap()
}

fn asy(cfg: CountConfig) -> Asy {
    sync_with(cfg)
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(N as usize).unwrap(),
        ))
        .unwrap()
}

fn ids() -> Vec<u32> {
    (0..N).collect()
}

/// The actions of step `t`, `[rows, 2]`, the same per instance in any mode.
fn actions(t: i32, ids: &[u32]) -> Vec<i32> {
    ids.iter()
        .flat_map(|&i| {
            let v = (i.cast_signed() + t) % 2;
            [v, v]
        })
        .collect()
}

fn sync_steps(p: &mut Pool, from: i32, n: i32) {
    for t in from..from + n {
        p.step(ActionInput {
            discrete: &actions(t, &ids()),
            continuous: &[],
        })
        .unwrap();
    }
}

fn piped_steps(p: &mut Piped, from: i32, n: i32) {
    let halves = p.halves();
    for t in from..from + n {
        for (h, range) in halves.iter().enumerate() {
            let token = p.idle_token(u32::try_from(h).unwrap()).unwrap();
            let ids: Vec<u32> = range.clone().collect();
            let a = actions(t, &ids);
            p.send(
                token,
                ActionInput {
                    discrete: &a,
                    continuous: &[],
                },
            )
            .map_err(|e| e.error)
            .unwrap();
        }
        for _ in 0..2 {
            drop(p.recv().unwrap());
        }
    }
}

fn asy_steps(p: &mut Asy, from: i32, n: i32) {
    for t in from..from + n {
        let all = ids();
        p.send(
            &all,
            ActionInput {
                discrete: &actions(t, &all),
                continuous: &[],
            },
        )
        .unwrap();
        let mut got = 0;
        while got < all.len() {
            let token = p.recv().unwrap();
            got += token.ids().count();
        }
    }
}

/// A Sync reference: the file after `played` steps, and after `played + more`.
fn reference(played: i32, more: i32) -> (Vec<u8>, Vec<u8>) {
    let mut p = sync_with(cfg());
    p.reset_all(Reseed::Keep).unwrap();
    sync_steps(&mut p, 0, played);
    let start = p.encode(&ids()).unwrap();
    sync_steps(&mut p, played, more);
    (start, p.encode(&ids()).unwrap())
}

#[test]
fn test_persist_a_sync_file_continues_identically_in_pipelined() {
    let (start, end) = reference(3, 4);
    let mut p = piped(cfg());
    drop(p.reset_all(Reseed::Keep).unwrap());
    let records = p.decode(&start, LIMITS).unwrap();
    p.check_records(&records).unwrap();
    p.import_all(&records).unwrap();
    assert_eq!(p.encode(&ids()).unwrap(), start);
    piped_steps(&mut p, 3, 4);
    assert_eq!(p.encode(&ids()).unwrap(), end);
}

#[test]
fn test_persist_a_sync_file_continues_identically_in_async() {
    let (start, end) = reference(3, 4);
    let mut p = asy(cfg());
    drop(p.reset_all(Reseed::Keep).unwrap());
    let records = p.decode(&start, LIMITS).unwrap();
    p.check_records(&records).unwrap();
    p.import_all(&records).unwrap();
    assert_eq!(p.encode(&ids()).unwrap(), start);
    asy_steps(&mut p, 3, 4);
    assert_eq!(p.encode(&ids()).unwrap(), end);
}

#[test]
fn test_persist_split_files_continue_identically_in_sync() {
    let (start, end) = reference(0, 5);
    let mut p = piped(cfg());
    drop(p.reset_all(Reseed::Keep).unwrap());
    let mut a = asy(cfg());
    drop(a.reset_all(Reseed::Keep).unwrap());
    let from_piped = p.encode(&ids()).unwrap();
    let from_async = a.encode(&ids()).unwrap();
    assert_eq!(from_piped, start);
    assert_eq!(from_async, start);
    let mut q = sync_with(cfg());
    q.reset_all(Reseed::Keep).unwrap();
    let records = q.decode(&from_piped, LIMITS).unwrap();
    q.import_all(&records).unwrap();
    sync_steps(&mut q, 0, 5);
    assert_eq!(q.encode(&ids()).unwrap(), end);
}

/// Pins today only: the cycle state moves no trajectory while no pool-level
/// adaptive subsystem exists (see `PipelinedPool::import_all`).
#[test]
fn test_persist_pipelined_import_at_unequal_cycle_positions_gives_equal_results() {
    let (start, end) = reference(2, 3);
    let mut fresh = piped(cfg());
    drop(fresh.reset_all(Reseed::Keep).unwrap());
    let mut advanced = piped(cfg());
    drop(advanced.reset_all(Reseed::Keep).unwrap());
    // One half a step ahead of the other when the file arrives.
    piped_steps(&mut advanced, 0, 2);
    let halves = advanced.halves();
    let token = advanced.idle_token(0).unwrap();
    let first: Vec<u32> = halves[0].clone().collect();
    let a = actions(9, &first);
    advanced
        .send(
            token,
            ActionInput {
                discrete: &a,
                continuous: &[],
            },
        )
        .map_err(|e| e.error)
        .unwrap();
    drop(advanced.recv().unwrap());
    for p in [&mut fresh, &mut advanced] {
        let records = p.decode(&start, LIMITS).unwrap();
        p.import_all(&records).unwrap();
        piped_steps(p, 2, 3);
        assert_eq!(p.encode(&ids()).unwrap(), end);
    }
}

#[test]
fn test_persist_pipelined_refuses_with_work_in_flight_and_decodes_regardless() {
    let (start, _) = reference(0, 0);
    let mut p = piped(cfg());
    drop(p.reset_all(Reseed::Keep).unwrap());
    let token = p.idle_token(0).unwrap();
    let first: Vec<u32> = p.halves()[0].clone().collect();
    let a = actions(0, &first);
    p.send(
        token,
        ActionInput {
            discrete: &a,
            continuous: &[],
        },
    )
    .map_err(|e| e.error)
    .unwrap();
    assert!(matches!(
        p.encode(&[0]),
        Err(PersistError::Engine(EngineError::NotQuiescent))
    ));
    let records = p.decode(&start, LIMITS).unwrap();
    p.check_records(&records).unwrap();
    assert_eq!(p.import_all(&records), Err(EngineError::NotQuiescent));
    drop(p.recv().unwrap());
    assert!(p.encode(&[0]).is_ok());
}

#[test]
fn test_persist_async_refuses_unreceived_ids_and_decodes_regardless() {
    let (start, _) = reference(0, 0);
    let mut p = asy(cfg());
    drop(p.reset_all(Reseed::Keep).unwrap());
    let all = ids();
    p.send(
        &all,
        ActionInput {
            discrete: &actions(0, &all),
            continuous: &[],
        },
    )
    .unwrap();
    assert!(matches!(
        p.encode(&[0]),
        Err(PersistError::Engine(EngineError::NotQuiescent))
    ));
    let records = p.decode(&start, LIMITS).unwrap();
    p.check_records(&records).unwrap();
    let mut got = 0;
    while got < all.len() {
        got += p.recv().unwrap().ids().count();
    }
    assert!(p.encode(&[0]).is_ok());
}

#[test]
fn test_persist_a_live_token_allows_reading_and_blocks_only_the_import() {
    let (start, _) = reference(0, 0);
    let mut p = piped(cfg());
    let tokens = p.reset_all(Reseed::Keep).unwrap();
    let before = tokens[0].rewards().to_vec();
    assert!(p.encode(&[0]).is_ok());
    let records = p.decode(&start, LIMITS).unwrap();
    p.check_records(&records).unwrap();
    assert!(matches!(
        p.import_all(&records),
        Err(EngineError::HalfBorrowed { .. })
    ));
    assert_eq!(tokens[0].rewards(), before);
    drop(tokens);
    p.import_all(&records).unwrap();

    let mut a = asy(cfg());
    let token = a.reset_all(Reseed::Keep).unwrap();
    assert!(a.encode(&[0]).is_ok());
    assert!(matches!(
        a.import_all(&records),
        Err(EngineError::Borrowed { .. })
    ));
    drop(token);
    a.import_all(&records).unwrap();
}

/// A record `validate` refuses: more steps than the episode length.
fn refused(records: &[DecodedRecord<Counter>]) -> Vec<DecodedRecord<Counter>> {
    let mut out = records.to_vec();
    let last = out.len() - 1;
    let mut state = *out[last].state();
    state.steps = 10_000;
    out[last] = DecodedRecord::from_parts(*out[last].header(), state).unwrap();
    out
}

#[test]
fn test_persist_pipelined_poison_refuses_encode_and_a_good_file_recovers() {
    let panicking = CountConfig {
        panic_at: 1,
        length: 50,
        ..CountConfig::default()
    };
    let start = {
        let mut q = sync_with(panicking.clone());
        q.reset_all(Reseed::Keep).unwrap();
        q.encode(&ids()).unwrap()
    };
    let mut p = piped(panicking);
    drop(p.reset_all(Reseed::Keep).unwrap());
    for (h, range) in p.halves().iter().enumerate() {
        let token = p.idle_token(u32::try_from(h).unwrap()).unwrap();
        let ids: Vec<u32> = range.clone().collect();
        let a = actions(0, &ids);
        p.send(
            token,
            ActionInput {
                discrete: &a,
                continuous: &[],
            },
        )
        .map_err(|e| e.error)
        .unwrap();
    }
    assert_eq!(p.recv().map(|_| ()), Err(EngineError::Panicked));
    assert!(p.is_poisoned());
    // A half may still run: decode and check_records form no `&EnvPool`.
    let records = p.decode(&start, LIMITS).unwrap();
    p.check_records(&records).unwrap();
    for empty in [&[][..], &[0][..]] {
        assert!(matches!(
            p.encode(empty),
            Err(PersistError::Engine(EngineError::Poisoned))
        ));
    }
    let bad = refused(&records);
    assert!(matches!(
        p.check_records(&bad),
        Err(EngineError::Invariant { .. })
    ));
    assert!(p.import_all(&bad).is_err());
    assert!(p.is_poisoned(), "a failed check keeps the poison");
    p.import_all(&records).unwrap();
    assert!(!p.is_poisoned());
    assert_eq!(p.encode(&ids()).unwrap(), start);
}

#[test]
fn test_persist_async_poison_refuses_encode_and_a_good_file_recovers() {
    let panicking = CountConfig {
        panic_at: 1,
        length: 50,
        ..CountConfig::default()
    };
    let start = {
        let mut q = sync_with(panicking.clone());
        q.reset_all(Reseed::Keep).unwrap();
        q.encode(&ids()).unwrap()
    };
    let mut p = asy(panicking);
    drop(p.reset_all(Reseed::Keep).unwrap());
    let all = ids();
    p.send(
        &all,
        ActionInput {
            discrete: &actions(0, &all),
            continuous: &[],
        },
    )
    .unwrap();
    while !p.is_poisoned() {
        if let Err(e) = p.recv() {
            assert!(matches!(e, EngineError::Panicked | EngineError::Poisoned));
        }
    }
    let records = p.decode(&start, LIMITS).unwrap();
    p.check_records(&records).unwrap();
    for empty in [&[][..], &[0][..]] {
        assert!(matches!(
            p.encode(empty),
            Err(PersistError::Engine(EngineError::Poisoned))
        ));
    }
    let bad = refused(&records);
    assert!(matches!(
        p.check_records(&bad),
        Err(EngineError::Invariant { .. })
    ));
    assert!(p.import_all(&bad).is_err());
    assert!(p.is_poisoned(), "a failed check keeps the poison");
    p.import_all(&records).unwrap();
    assert!(!p.is_poisoned());
    assert_eq!(p.encode(&ids()).unwrap(), start);
}

#[test]
fn test_persist_split_check_records_counts_records() {
    let (start, _) = reference(0, 0);
    let p = piped(cfg());
    let a = asy(cfg());
    let records = p.decode(&start, LIMITS).unwrap();
    for r in [
        p.check_records(&records[..1]),
        a.check_records(&records[..1]),
    ] {
        assert!(matches!(r, Err(EngineError::RecordCount { .. })));
    }
}
