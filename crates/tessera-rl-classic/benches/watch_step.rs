// SPDX-License-Identifier: MIT OR Apache-2.0

//! The live watch's fast path on CartPole (ADR-0083 item 7; #378 4d design
//! section 5, the acceptance benchmark). Each group runs the same pool with
//! no watch (`watch_none`, the baseline) and with a slot attached and nothing
//! requested (`watch_idle`), back to back. Acceptance: `watch_idle`'s median
//! within 1.0% of `watch_none`'s and inside the baseline's run-to-run spread
//! over three interleaved runs.

#![expect(
    clippy::unwrap_used,
    reason = "bench setup unwraps: a failure aborts the bench"
)]

use core::num::{NonZeroU32, NonZeroUsize};
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_classic::{CartPole, CartPoleConfig};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AutoReset, EnvPool, NoMasks, PipelinedOptions, PoolOptions, Reseed,
    SyncOptions,
};
use tessera_rl_num::identity::Seed;

fn pool(n: u32) -> EnvPool<CartPole, NoMasks> {
    let options = PoolOptions {
        auto_reset: AutoReset::SameStep,
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(1))
    };
    EnvPool::new(CartPoleConfig::default(), options).unwrap()
}

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

fn name(watched: bool) -> &'static str {
    if watched { "watch_idle" } else { "watch_none" }
}

fn workers() -> NonZeroUsize {
    NonZeroUsize::new(4).unwrap()
}

fn bench_sync(c: &mut Criterion) {
    for n in [1024_u32, 8192] {
        let mut group = c.benchmark_group(format!("watch_cartpole_sync_{n}"));
        let actions = vec![1i32; usize::try_from(n).unwrap()];
        for watched in [false, true] {
            let mut p = pool(n).with_workers(SyncOptions::new(workers())).unwrap();
            let handle = watched.then(|| p.attach_watch().unwrap());
            p.reset_all(Reseed::Keep).unwrap();
            group.bench_function(name(watched), |b| {
                b.iter(|| black_box(p.step(input(&actions)).unwrap().rewards().len()));
            });
            drop(handle);
        }
        group.finish();
    }
}

fn bench_pipelined(c: &mut Criterion) {
    const N: u32 = 8192;
    let mut group = c.benchmark_group("watch_cartpole_pipelined_8192");
    for watched in [false, true] {
        let mut piped = pool(N).pipelined(PipelinedOptions::new(workers())).unwrap();
        let handle = watched.then(|| piped.attach_watch().unwrap());
        let [ha, hb] = piped.halves();
        let len = |h: core::ops::Range<u32>| usize::try_from(h.end - h.start).unwrap();
        let (act_a, act_b) = (vec![1i32; len(ha)], vec![1i32; len(hb)]);
        let [a, b] = piped.reset_all(Reseed::Keep).unwrap();
        let mut tokens = Some((a, b));
        group.bench_function(name(watched), |bench| {
            bench.iter(|| {
                let (a, b) = tokens.take().unwrap();
                piped.send(a, input(&act_a)).unwrap();
                piped.send(b, input(&act_b)).unwrap();
                let a = piped.recv().unwrap();
                let b = piped.recv().unwrap();
                black_box(a.rewards().len());
                tokens = Some((a, b));
            });
        });
        drop(tokens.take());
        drop(handle);
    }
    group.finish();
}

fn bench_async(c: &mut Criterion) {
    const N: u32 = 8192;
    const BATCH: usize = 2048;
    let mut group = c.benchmark_group("watch_cartpole_async_8192");
    let n = usize::try_from(N).unwrap();
    let all = vec![1i32; n];
    let ids: Vec<u32> = (0..N).collect();
    for watched in [false, true] {
        let mut p = pool(N)
            .async_mode(AsyncOptions::new(
                workers(),
                NonZeroUsize::new(BATCH).unwrap(),
            ))
            .unwrap();
        let handle = watched.then(|| p.attach_watch().unwrap());
        drop(p.reset_all(Reseed::Keep).unwrap());
        group.bench_function(name(watched), |bench| {
            bench.iter(|| {
                p.send(&ids, input(&all)).unwrap();
                for _ in 0..n / BATCH {
                    black_box(p.recv().unwrap().len());
                }
            });
        });
        drop(handle);
    }
    group.finish();
}

criterion_group!(benches, bench_sync, bench_pipelined, bench_async);
criterion_main!(benches);
