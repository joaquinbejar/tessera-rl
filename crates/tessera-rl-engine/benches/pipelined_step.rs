// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pipelined-mode cycle cost (ADR-0047): one send and one recv of each
//! half, against one Sync step of the whole pool with the same workers, by
//! wait backend. No claims beyond the numbers this prints.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::{NonZeroU32, NonZeroUsize};
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_engine::{
    ActionInput, EnvPool, PipelinedOptions, PipelinedPool, PoolOptions, Reseed, SyncOptions,
    WaitBackend,
};
use tessera_rl_num::identity::Seed;

#[path = "../tests/common/mod.rs"]
mod common;

use common::{CountConfig, Counter};

const N: u32 = 256;

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
    let mut group = c.benchmark_group("pipelined_cycle_256");
    let workers = NonZeroUsize::new(4).unwrap();
    for (backend, label) in [(WaitBackend::Park, "park"), (WaitBackend::Native, "native")] {
        let sync = SyncOptions {
            spin: 2_000,
            wait: backend,
            ..SyncOptions::new(workers)
        };
        let mut piped: PipelinedPool<Counter, _> = base()
            .pipelined(PipelinedOptions { sync: sync.clone() })
            .unwrap();
        let [ha, hb] = piped.halves();
        let len = |h: core::ops::Range<u32>| usize::try_from(h.end - h.start).unwrap() * 2;
        let (act_a, act_b) = (vec![1i32; len(ha)], vec![1i32; len(hb)]);
        let [a, b] = piped.reset_all(Reseed::Keep).unwrap();
        let mut tokens = Some((a, b));
        group.bench_function(format!("{label}_spin2000"), |bench| {
            bench.iter(|| {
                let Some((a, b)) = tokens.take() else {
                    return;
                };
                let ok =
                    piped.send(a, input(&act_a)).is_ok() && piped.send(b, input(&act_b)).is_ok();
                if ok {
                    let a = piped.recv().unwrap();
                    let b = piped.recv().unwrap();
                    black_box(a.rewards());
                    tokens = Some((a, b));
                } else {
                    // An episode ended (Disabled is not used here, but stay
                    // safe): drain and reset.
                    while piped.try_recv().ok().flatten().is_some() {}
                    let [a, b] = piped.reset_all(Reseed::Keep).unwrap();
                    tokens = Some((a, b));
                }
            });
        });
        drop(tokens.take());
        let all = vec![1i32; usize::try_from(N).unwrap() * 2];
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
/// design section 5): one Pipelined cycle of 8192 instances (4 workers)
/// with no watch (`watch_none`) and with a slot attached and nothing
/// requested (`watch_idle`), in one group. The test environment stands in
/// for CartPole, which the engine's benches cannot depend on (a cycle).
fn bench_watch(c: &mut Criterion) {
    const WATCH_N: u32 = 8192;
    let mut group = c.benchmark_group("watch_pipelined_8192");
    let workers = NonZeroUsize::new(4).unwrap();
    for watched in [false, true] {
        let mut piped: PipelinedPool<Counter, _> = sized(WATCH_N)
            .pipelined(PipelinedOptions::new(workers))
            .unwrap();
        let handle = watched.then(|| piped.attach_watch().unwrap());
        let [ha, hb] = piped.halves();
        let len = |h: core::ops::Range<u32>| usize::try_from(h.end - h.start).unwrap() * 2;
        let (act_a, act_b) = (vec![1i32; len(ha)], vec![1i32; len(hb)]);
        let [a, b] = piped.reset_all(Reseed::Keep).unwrap();
        let mut tokens = Some((a, b));
        let name = if watched { "watch_idle" } else { "watch_none" };
        group.bench_function(name, |bench| {
            bench.iter(|| {
                let Some((a, b)) = tokens.take() else {
                    return;
                };
                let ok =
                    piped.send(a, input(&act_a)).is_ok() && piped.send(b, input(&act_b)).is_ok();
                if ok {
                    let a = piped.recv().unwrap();
                    let b = piped.recv().unwrap();
                    black_box(a.rewards());
                    tokens = Some((a, b));
                } else {
                    while piped.try_recv().ok().flatten().is_some() {}
                    let [a, b] = piped.reset_all(Reseed::Keep).unwrap();
                    tokens = Some((a, b));
                }
            });
        });
        drop(tokens.take());
        drop(handle);
    }
    group.finish();
}

criterion_group!(benches, bench, bench_watch);
criterion_main!(benches);
