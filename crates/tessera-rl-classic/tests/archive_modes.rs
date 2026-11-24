// SPDX-License-Identifier: MIT OR Apache-2.0

//! Archive resets in every execution mode (ADR-0075 items 2, 5 and 10):
//! Inline, Sync, Pipelined and Async give bit-identical states on the same
//! calls; the reset source is set only at quiescence; `reset_from` in Async
//! mode equals Sync's.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_classic::{CartPole, CartPoleConfig};
use tessera_rl_engine::{
    ActionInput, Archive, AsyncOptions, AutoReset, EngineError, EnvPool, PipelinedOptions,
    PoolOptions, Record, Reseed, ResetSource, SyncOptions,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 4;
const STEPS: u32 = 150;

fn options() -> PoolOptions {
    PoolOptions {
        auto_reset: AutoReset::SameStep,
        chunk_size: NonZeroU32::MIN,
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(21))
    }
}

fn pool() -> EnvPool<CartPole> {
    EnvPool::new(CartPoleConfig::default(), options()).unwrap()
}

fn archive() -> Archive<CartPole> {
    let mut src = pool();
    src.reset_all(Reseed::Keep).unwrap();
    let a = [1i32; N as usize];
    let mut out = Vec::new();
    for _ in 0..3 {
        src.step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
        out.push(src.snapshot(0).unwrap());
    }
    Archive::new(&CartPoleConfig::default(), &out).unwrap()
}

fn source(arc: &Archive<CartPole>) -> ResetSource<CartPole> {
    ResetSource::Archive {
        archive: arc.clone(),
        weights: Some(vec![2, 1, 3]),
    }
}

/// Instance `i`'s action at step `t`: pushes one way most of the time, so
/// episodes end and archive auto-resets run.
fn action(t: u32, i: u32) -> i32 {
    i32::from(!(t / 7 + i).is_multiple_of(3))
}

fn hashes(records: &[Record<CartPole>]) -> Vec<u64> {
    records.iter().map(|r| r.hash().get()).collect()
}

/// Inline (`workers == 0`) or Sync.
fn run_sync(arc: &Archive<CartPole>, workers: usize) -> Vec<u64> {
    let mut p = pool();
    if workers > 0 {
        p = p
            .with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
            .unwrap();
    }
    p.set_reset_source(source(arc)).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    for t in 0..STEPS {
        let a: Vec<i32> = (0..N).map(|i| action(t, i)).collect();
        p.step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
    }
    hashes(&(0..N).map(|i| p.snapshot(i).unwrap()).collect::<Vec<_>>())
}

fn run_pipelined(arc: &Archive<CartPole>) -> Vec<u64> {
    let mut p = pool()
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
        .unwrap();
    p.set_reset_source(source(arc)).unwrap();
    let halves = p.halves();
    let [mut ta, mut tb] = p.reset_all(Reseed::Keep).unwrap();
    for t in 0..STEPS {
        for token in [ta, tb] {
            let h = usize::try_from(token.half()).unwrap();
            let a: Vec<i32> = halves[h].clone().map(|i| action(t, i)).collect();
            p.send(
                token,
                ActionInput {
                    discrete: &a,
                    continuous: &[],
                },
            )
            .unwrap();
        }
        ta = p.recv().unwrap();
        tb = p.recv().unwrap();
    }
    drop((ta, tb));
    let mut out = vec![pool().snapshot(0).unwrap(); N as usize];
    p.snapshot_all(&mut out).unwrap();
    hashes(&out)
}

fn run_async(arc: &Archive<CartPole>) -> Vec<u64> {
    let mut p = pool()
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(N as usize).unwrap(),
        ))
        .unwrap();
    p.set_reset_source(source(arc)).unwrap();
    drop(p.reset_all(Reseed::Keep).unwrap());
    let ids: Vec<u32> = (0..N).collect();
    for t in 0..STEPS {
        let a: Vec<i32> = ids.iter().map(|&i| action(t, i)).collect();
        p.send(
            &ids,
            ActionInput {
                discrete: &a,
                continuous: &[],
            },
        )
        .unwrap();
        let mut got = 0;
        while got < ids.len() {
            got += p.recv().unwrap().len();
        }
    }
    hashes(&ids.iter().map(|&i| p.snapshot(i).unwrap()).collect::<Vec<_>>())
}

#[test]
fn test_archive_modes_inline_sync_pipelined_async_are_bit_identical() {
    let arc = archive();
    let inline = run_sync(&arc, 0);
    assert_eq!(run_sync(&arc, 2), inline, "Sync");
    assert_eq!(run_pipelined(&arc), inline, "Pipelined");
    assert_eq!(run_async(&arc), inline, "Async");
}

#[test]
fn test_archive_modes_reset_source_waits_for_quiescence() {
    let arc = archive();
    // Pipelined: a half in flight, then a live token.
    let mut p = pool()
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
        .unwrap();
    let halves = p.halves();
    let [ta, tb] = p.reset_all(Reseed::Keep).unwrap();
    let a: Vec<i32> = halves[0].clone().map(|_| 0).collect();
    p.send(
        ta,
        ActionInput {
            discrete: &a,
            continuous: &[],
        },
    )
    .unwrap();
    assert!(matches!(
        p.set_reset_source(source(&arc)),
        Err(EngineError::NotQuiescent)
    ));
    let ta = p.recv().unwrap();
    assert!(matches!(
        p.set_reset_source(source(&arc)),
        Err(EngineError::HalfBorrowed { .. })
    ));
    drop((ta, tb));
    p.set_reset_source(source(&arc)).unwrap();

    // Async: an id in flight, then a live token.
    let mut q = pool()
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(N as usize).unwrap(),
        ))
        .unwrap();
    drop(q.reset_all(Reseed::Keep).unwrap());
    let ids: Vec<u32> = (0..N).collect();
    q.send(
        &ids,
        ActionInput {
            discrete: &[0; N as usize],
            continuous: &[],
        },
    )
    .unwrap();
    assert!(q.set_reset_source(source(&arc)).is_err(), "ids in flight");
    let token = q.recv().unwrap();
    assert!(matches!(
        q.set_reset_source(source(&arc)),
        Err(EngineError::Borrowed { .. })
    ));
    drop(token);
    q.set_reset_source(source(&arc)).unwrap();
}

#[test]
fn test_archive_modes_async_reset_from_equals_sync() {
    let arc = archive();
    let mut s = pool();
    s.set_reset_source(source(&arc)).unwrap();
    s.reset_all(Reseed::Keep).unwrap();
    s.reset_from(&[3, 1], &[2, 0]).unwrap();
    let want = hashes(&(0..N).map(|i| s.snapshot(i).unwrap()).collect::<Vec<_>>());

    let mut a = pool()
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        ))
        .unwrap();
    a.set_reset_source(source(&arc)).unwrap();
    drop(a.reset_all(Reseed::Keep).unwrap());
    a.reset_from(&[3, 1], &[2, 0]).unwrap();
    let token = a.recv().unwrap();
    let mut delivered: Vec<u32> = token.ids().collect();
    delivered.sort_unstable();
    assert_eq!(delivered, [1, 3]);
    drop(token);
    let got = hashes(&(0..N).map(|i| a.snapshot(i).unwrap()).collect::<Vec<_>>());
    assert_eq!(got, want);
    // Out of range: refused, nothing delivered.
    assert!(matches!(
        a.reset_from(&[0], &[3]),
        Err(EngineError::Archive { .. })
    ));
    // An id still in flight: refused like `reset` (ADR-0047 section 3).
    a.send(
        &[0, 2],
        ActionInput {
            discrete: &[0, 0],
            continuous: &[],
        },
    )
    .unwrap();
    assert!(matches!(
        a.reset_from(&[2], &[0]),
        Err(EngineError::InFlight { .. })
    ));
}
