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
#[expect(dead_code, reason = "shared with the tests")]
mod common;

use common::{CountConfig, Counter};

const N: u32 = 256;

fn pool(sync: Option<SyncOptions>) -> EnvPool<Counter> {
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(32).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1))
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
                black_box(p.outputs().rewards());
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

criterion_group!(benches, bench);
criterion_main!(benches);
