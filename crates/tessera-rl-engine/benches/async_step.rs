// SPDX-License-Identifier: MIT OR Apache-2.0

//! Async-mode cost (ADR-0047): every instance sent once and received in
//! batches of 64, against one Sync step of the whole pool with the same
//! workers, by wait backend. No claims beyond the numbers this prints.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::{NonZeroU32, NonZeroUsize};
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AsyncPool, EnvPool, PoolOptions, Reseed, SyncOptions, WaitBackend,
};
use tessera_rl_num::identity::Seed;

#[path = "../tests/common/mod.rs"]
mod common;

use common::{CountConfig, Counter};

const N: u32 = 256;
const BATCH: usize = 64;

fn base() -> EnvPool<Counter> {
    sized(N)
}

fn sized(n: u32) -> EnvPool<Counter> {
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(32).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(1))
    };
    let cfg = CountConfig {
        length: 1000,
        ..CountConfig::default()
    };
    EnvPool::new(cfg, options).unwrap()
}

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_round_256");
    let workers = NonZeroUsize::new(4).unwrap();
    let n = usize::try_from(N).unwrap();
    let all = vec![1i32; n * 2];
    let ids: Vec<u32> = (0..N).collect();
    for (backend, label) in [(WaitBackend::Park, "park"), (WaitBackend::Native, "native")] {
        let sync = SyncOptions {
            spin: 2_000,
            wait: backend,
            ..SyncOptions::new(workers)
        };
        let mut pool: AsyncPool<Counter, _> = base()
            .async_mode(AsyncOptions {
                sync: sync.clone(),
                batch_size: NonZeroUsize::new(BATCH).unwrap(),
            })
            .unwrap();
        drop(pool.reset_all(Reseed::Keep).unwrap());
        group.bench_function(format!("{label}_spin2000"), |bench| {
            bench.iter(|| {
                if pool.send(&ids, input(&all)).is_err() {
                    drop(pool.reset_all(Reseed::Keep).unwrap());
                    return;
                }
                for _ in 0..n / BATCH {
                    let t = pool.recv().unwrap();
                    black_box(t.len());
                }
            });
        });
        let mut sync_pool = base().with_workers(sync).unwrap();
        sync_pool.reset_all(Reseed::Keep).unwrap();
        group.bench_function(format!("sync_{label}_spin2000"), |bench| {
            bench.iter(|| {
                if sync_pool.step(input(&all)).is_err() {
                    sync_pool.reset_all(Reseed::Keep).unwrap();
                }
                black_box(sync_pool.outputs().unwrap().rewards());
            });
        });
    }
    group.finish();
}

/// The live watch's fast path on `recv` (ADR-0083 item 7, issue #378 4d
/// design section 5): one Async round of 8192 instances in batches of 2048
/// (4 workers) with no watch (`watch_none`) and with a slot attached and
/// nothing requested (`watch_idle`), in one group. The test environment
/// stands in for CartPole, which the engine's benches cannot depend on (a
/// cycle).
fn bench_watch(c: &mut Criterion) {
    const WATCH_N: u32 = 8192;
    const WATCH_BATCH: usize = 2048;
    let mut group = c.benchmark_group("watch_async_8192");
    let n = usize::try_from(WATCH_N).unwrap();
    let all = vec![1i32; n * 2];
    let ids: Vec<u32> = (0..WATCH_N).collect();
    for watched in [false, true] {
        let mut pool: AsyncPool<Counter, _> = sized(WATCH_N)
            .async_mode(AsyncOptions::new(
                NonZeroUsize::new(4).unwrap(),
                NonZeroUsize::new(WATCH_BATCH).unwrap(),
            ))
            .unwrap();
        let handle = watched.then(|| pool.attach_watch().unwrap());
        drop(pool.reset_all(Reseed::Keep).unwrap());
        let name = if watched { "watch_idle" } else { "watch_none" };
        group.bench_function(name, |bench| {
            bench.iter(|| {
                if pool.send(&ids, input(&all)).is_err() {
                    drop(pool.reset_all(Reseed::Keep).unwrap());
                    return;
                }
                for _ in 0..n / WATCH_BATCH {
                    let t = pool.recv().unwrap();
                    black_box(t.len());
                }
            });
        });
        drop(handle);
    }
    group.finish();
}

criterion_group!(benches, bench, bench_watch);
criterion_main!(benches);
