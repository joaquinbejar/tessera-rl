// SPDX-License-Identifier: MIT OR Apache-2.0

//! Live watch (ADR-0083 item 7, issue #378 stage 4d): each mode's batch
//! boundary serves only idle instances, the served record is exactly
//! `snapshot(id)` at that point, a poisoned pool reports `Poisoned`, a
//! detach while a request is pending is never served, and with no
//! boundary (training stopped) `take` returns `None` without blocking.

#![cfg(not(loom))]
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod common;

use core::num::{NonZeroU32, NonZeroUsize};

use common::{CountConfig, Counter};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AutoReset, EngineError, EnvPool, Masked, PipelinedOptions,
    PoolOptions, Record, Reseed, SyncOptions, WatchError, WatchHandle,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 16;
const CHUNK: u32 = 4;

type Pool = EnvPool<Counter, Masked>;

fn options() -> PoolOptions {
    PoolOptions {
        auto_reset: AutoReset::NextStep,
        chunk_size: NonZeroU32::new(CHUNK).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(5))
    }
}

fn cfg() -> CountConfig {
    CountConfig {
        length: 50,
        ..CountConfig::default()
    }
}

fn pool_with(cfg: CountConfig) -> Pool {
    EnvPool::with_masks(cfg, options()).unwrap()
}

/// Legal actions for `ids.len()` instances (two agents, action 1).
fn actions(rows: usize) -> Vec<i32> {
    vec![1; rows * 2]
}

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

fn record_len() -> usize {
    Record::<Counter>::encoded_len()
}

/// Takes once; the test fails if nothing is ready.
fn take_ready(handle: &mut WatchHandle) -> (u64, Vec<u8>) {
    let mut out = vec![0; handle.record_len()];
    let g = handle.take(&mut out).unwrap().expect("a served record");
    (g, out)
}

fn assert_nothing_ready(handle: &mut WatchHandle) {
    let mut out = vec![0; handle.record_len()];
    assert_eq!(handle.take(&mut out), Ok(None));
}

fn sync_served_record_equals_snapshot(mut pool: Pool) {
    let mut handle = pool.attach_watch().unwrap();
    assert_eq!(handle.record_len(), record_len());
    pool.reset_all(Reseed::Keep).unwrap();
    handle.request(3).unwrap();
    // No boundary yet since the request.
    assert_nothing_ready(&mut handle);
    let a = actions(usize::try_from(N).unwrap());
    pool.step(input(&a)).unwrap();
    let (g, bytes) = take_ready(&mut handle);
    // reset_all, then step: the second boundary since attach.
    assert_eq!(g, 2);
    assert_eq!(bytes, pool.snapshot(3).unwrap().to_bytes());
    // Taken: the slot is idle again, and the next step serves nothing.
    pool.step(input(&a)).unwrap();
    assert_nothing_ready(&mut handle);
}

#[test]
fn test_watch_inline_served_record_equals_snapshot() {
    sync_served_record_equals_snapshot(pool_with(cfg()));
}

#[test]
fn test_watch_sync_served_record_equals_snapshot() {
    sync_served_record_equals_snapshot(
        pool_with(cfg())
            .with_workers(SyncOptions::new(NonZeroUsize::new(4).unwrap()))
            .unwrap(),
    );
}

/// Every writer is a boundary: reset, restore and fork serve too.
#[test]
fn test_watch_inline_writers_are_boundaries_served() {
    let mut pool = pool_with(cfg());
    let mut handle = pool.attach_watch().unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let record = pool.snapshot(1).unwrap();
    handle.request(2).unwrap();
    pool.fork(1, &[2]).unwrap();
    let (_, bytes) = take_ready(&mut handle);
    assert_eq!(bytes, pool.snapshot(2).unwrap().to_bytes());
    handle.request(4).unwrap();
    pool.restore(4, &record).unwrap();
    let (_, bytes) = take_ready(&mut handle);
    assert_eq!(bytes, record.to_bytes());
    handle.request(5).unwrap();
    pool.reset(&[5], Reseed::Keep).unwrap();
    let (_, bytes) = take_ready(&mut handle);
    assert_eq!(bytes, pool.snapshot(5).unwrap().to_bytes());
}

/// Pipelined: a request for the other half stays pending until that
/// half's `recv`; the served record equals the snapshot after it.
#[test]
fn test_watch_pipelined_serves_only_the_received_half() {
    let mut p = pool_with(cfg())
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
        .unwrap();
    let mut handle = p.attach_watch().unwrap();
    let [a, b] = p.reset_all(Reseed::Keep).unwrap();
    let [ha, hb] = p.halves();
    let target = hb.start;
    handle.request(target).unwrap();
    let rows_a = usize::try_from(ha.end - ha.start).unwrap();
    let rows_b = usize::try_from(hb.end - hb.start).unwrap();
    p.send(a, input(&actions(rows_a)))
        .map_err(|e| e.error)
        .unwrap();
    p.send(b, input(&actions(rows_b)))
        .map_err(|e| e.error)
        .unwrap();
    let first = p.recv().unwrap();
    assert_eq!(first.ids(), ha);
    // Half a's boundary: the target is in half b, still in flight or not
    // yet received, so nothing is served.
    assert_nothing_ready(&mut handle);
    let second = p.recv().unwrap();
    assert_eq!(second.ids(), hb);
    let (_, bytes) = take_ready(&mut handle);
    drop((first, second));
    let template = pool_with(cfg()).snapshot(0).unwrap();
    let mut records = vec![template; usize::try_from(N).unwrap()];
    p.snapshot_all(&mut records).unwrap();
    let k = usize::try_from(target).unwrap();
    assert_eq!(bytes, records[k].to_bytes());
}

/// Async: only the returned batch's ids are served.
#[test]
fn test_watch_async_serves_only_the_batch() {
    let mut p = pool_with(cfg())
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(1).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        ))
        .unwrap();
    let mut handle = p.attach_watch().unwrap();
    drop(p.reset_all(Reseed::Keep).unwrap());
    handle.request(5).unwrap();
    p.send(&[0, 1], input(&actions(2))).unwrap();
    let batch = p.recv().unwrap();
    assert!(!batch.contains(5));
    assert_nothing_ready(&mut handle);
    drop(batch);
    p.send(&[5, 2], input(&actions(2))).unwrap();
    let batch = p.recv().unwrap();
    assert!(batch.contains(5));
    drop(batch);
    let (_, bytes) = take_ready(&mut handle);
    assert_eq!(bytes, p.snapshot(5).unwrap().to_bytes());
}

/// A poisoned pool serves nothing and `take` reports `Poisoned`.
#[test]
fn test_watch_poisoned_pool_take_is_poisoned() {
    let mut pool = pool_with(CountConfig {
        panic_at: 2,
        ..cfg()
    });
    let mut handle = pool.attach_watch().unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let a = actions(usize::try_from(N).unwrap());
    handle.request(0).unwrap();
    pool.step(input(&a)).unwrap();
    let _ = take_ready(&mut handle);
    handle.request(0).unwrap();
    assert_eq!(pool.step(input(&a)).map(|_| ()), Err(EngineError::Panicked));
    assert!(pool.is_poisoned());
    let mut out = vec![0; handle.record_len()];
    assert_eq!(handle.take(&mut out), Err(WatchError::Poisoned));
}

/// Detach while a request is pending: never served, then `Detached`.
#[test]
fn test_watch_detach_while_pending_is_never_served() {
    let mut pool = pool_with(cfg());
    let mut handle = pool.attach_watch().unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    handle.request(0).unwrap();
    pool.detach_watch();
    let a = actions(usize::try_from(N).unwrap());
    pool.step(input(&a)).unwrap();
    let mut out = vec![0; handle.record_len()];
    assert_eq!(handle.take(&mut out), Err(WatchError::Detached));
    assert_eq!(handle.request(0), Err(WatchError::Detached));
    assert!(handle.is_detached());
}

/// Dropping the pool detaches too.
#[test]
fn test_watch_dropped_pool_is_detached() {
    let mut pool = pool_with(cfg());
    let mut handle = pool.attach_watch().unwrap();
    drop(pool);
    assert_eq!(handle.request(0), Err(WatchError::Detached));
}

/// Training stopped: no boundary comes, and `take` returns `None` at once.
#[test]
fn test_watch_stopped_training_take_is_none_without_blocking() {
    let mut pool = pool_with(cfg());
    let mut handle = pool.attach_watch().unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    handle.request(0).unwrap();
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        assert_nothing_ready(&mut handle);
    }
    assert!(start.elapsed() < core::time::Duration::from_secs(5));
    // The request stays pending for the next boundary.
    assert_eq!(handle.request(0), Err(WatchError::Busy));
}

/// One handle per pool; a dropped handle's slot is replaced.
#[test]
fn test_watch_second_attach_refused_until_handle_dropped() {
    let mut pool = pool_with(cfg());
    let handle = pool.attach_watch().unwrap();
    assert_eq!(
        pool.attach_watch().map(|_| ()),
        Err(EngineError::WatchAttached)
    );
    drop(handle);
    let _again = pool.attach_watch().unwrap();
    assert!(pool.has_watch());
}

/// The handle moves to another thread (it is `Send`) and takes there while
/// this thread steps.
#[test]
fn test_watch_handle_on_another_thread_receives_records() {
    let mut pool = pool_with(cfg());
    let mut handle = pool.attach_watch().unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let viewer = {
        let stop = std::sync::Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut out = vec![0; handle.record_len()];
            let mut got = 0_u32;
            while !stop.load(std::sync::atomic::Ordering::Acquire) {
                let _ = handle.request(1);
                if handle.take(&mut out).unwrap().is_some() {
                    got += 1;
                }
                std::thread::yield_now();
            }
            got
        })
    };
    let a = actions(usize::try_from(N).unwrap());
    for _ in 0..2_000 {
        pool.step(input(&a)).unwrap();
    }
    stop.store(true, std::sync::atomic::Ordering::Release);
    let _served = viewer.join().unwrap();
}
