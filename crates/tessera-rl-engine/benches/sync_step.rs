// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sync-mode step cost by wait backend (ADR-0005, ADR-0025): Inline as the
//! baseline, then park and the native address wait, with and without the
//! spin budget (spin 0 makes every batch block, which isolates the
//! backend).

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::{NonZeroU32, NonZeroUsize};
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_engine::{ActionInput, EnvPool, PoolOptions, Reseed, SyncOptions, WaitBackend};
use tessera_rl_num::identity::Seed;

#[path = "../tests/common/mod.rs"]
mod common;

use common::{CountConfig, Counter};

const N: u32 = 256;

fn pool(sync: Option<SyncOptions>) -> EnvPool<Counter> {
    sized_pool(N, sync)
}

fn sized_pool(n: u32, sync: Option<SyncOptions>) -> EnvPool<Counter> {
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(32).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(1))
    };
    let cfg = CountConfig {
        length: 1000,
        ..CountConfig::default()
    };
    let p = EnvPool::new(cfg, options).unwrap();
    let mut p = match sync {
        Some(s) => p.with_workers(s).unwrap(),
        None => p,
    };
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn bench(c: &mut Criterion) {
    let actions = vec![1i32; usize::try_from(N).unwrap() * 2];
    let mut group = c.benchmark_group("sync_step_256");
    let mut run = |name: &str, mut p: EnvPool<Counter>| {
        group.bench_function(name, |b| {
            b.iter(|| {
                // Long episodes; reset when one ends so every step steps.
                if p.step(ActionInput {
                    discrete: &actions,
                    continuous: &[],
                })
                .is_err()
                {
                    p.reset_all(Reseed::Keep).unwrap();
                }
                black_box(p.outputs().unwrap().rewards());
            });
        });
    };
    run("inline", pool(None));
    let workers = NonZeroUsize::new(4).unwrap();
    for (backend, label) in [(WaitBackend::Park, "park"), (WaitBackend::Native, "native")] {
        for (spin, spin_label) in [(0, "spin0"), (2_000, "spin2000")] {
            let sync = SyncOptions {
                spin,
                wait: backend,
                ..SyncOptions::new(workers)
            };
            run(&format!("{label}_{spin_label}"), pool(Some(sync)));
        }
    }
    group.finish();
}

/// The live watch's fast path (ADR-0083 item 7, issue #378 4d design
/// section 5): the same Sync pool (4 workers) with no watch (`watch_none`,
/// the baseline: one pointer test per call) and with a slot attached and
/// nothing requested (`watch_idle`: one `Relaxed` load per call), in one
/// group so they run back to back. The test environment stands in for
/// CartPole, which the engine's benches cannot depend on (a cycle).
fn bench_watch(c: &mut Criterion) {
    let workers = NonZeroUsize::new(4).unwrap();
    for n in [1024_u32, 8192] {
        let mut group = c.benchmark_group(format!("watch_sync_{n}"));
        let actions = vec![1i32; usize::try_from(n).unwrap() * 2];
        for watched in [false, true] {
            let mut p = sized_pool(n, Some(SyncOptions::new(workers)));
            let handle = watched.then(|| p.attach_watch().unwrap());
            let name = if watched { "watch_idle" } else { "watch_none" };
            group.bench_function(name, |b| {
                b.iter(|| {
                    if p.step(ActionInput {
                        discrete: &actions,
                        continuous: &[],
                    })
                    .is_err()
                    {
                        p.reset_all(Reseed::Keep).unwrap();
                    }
                    black_box(p.outputs().unwrap().rewards());
                });
            });
            drop(handle);
        }
        group.finish();
    }
}

criterion_group!(benches, bench, bench_watch);
criterion_main!(benches);
