// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sync-mode step cost by chunk schedule (ADR-0076): Static against Dynamic
//! on 4 and 8 workers.
//!
//! - `counter`: uniform cost, where Dynamic must not regress beyond noise;
//!   at chunk size 1 it also measures contention on the claim cursors.
//! - `varcost`: heaviness drawn per episode (1 instance in 16, 64 times the
//!   work), spread over many chunks: Static is close to balanced already.
//! - `varskew`: heaviness redrawn every step (1 in 64, 1024 times the work),
//!   few chunks per worker: a heavy-tailed per-step cost, the case Dynamic
//!   is for.

use core::num::{NonZeroU32, NonZeroUsize};
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_engine::{ActionInput, EnvPool, PoolOptions, Reseed, Schedule, SyncOptions};
use tessera_rl_num::identity::Seed;

#[path = "../tests/common/mod.rs"]
mod common;

#[path = "support/varcost.rs"]
mod varcost;

use common::{CountConfig, Counter};
use varcost::{VarConfig, VarCost};

macro_rules! run_case {
    ($group:expr, $name:expr, $env:ty, $cfg:expr, $n:expr, $chunk:expr, $workers:expr, $schedule:expr) => {{
        let options = PoolOptions {
            chunk_size: NonZeroU32::new($chunk).unwrap(),
            schedule: $schedule,
            ..PoolOptions::new(NonZeroU32::new($n).unwrap(), Seed::new(1))
        };
        let mut p = EnvPool::<$env>::new($cfg, options)
            .unwrap()
            .with_workers(SyncOptions::new(NonZeroUsize::new($workers).unwrap()))
            .unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        let actions = vec![1i32; p.outputs().unwrap().rewards().len()];
        $group.bench_function($name, |b| {
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
    }};
}

fn label(s: Schedule) -> &'static str {
    match s {
        Schedule::Static => "static",
        Schedule::Dynamic => "dynamic",
    }
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("schedule_step");
    let counter = || CountConfig {
        length: 1000,
        ..CountConfig::default()
    };
    for workers in [4usize, 8] {
        for chunk in [64u32, 1] {
            for s in [Schedule::Static, Schedule::Dynamic] {
                let name = format!("counter_n4096/chunk{chunk}/w{workers}/{}", label(s));
                run_case!(group, name, Counter, counter(), 4096, chunk, workers, s);
            }
        }
        for s in [Schedule::Static, Schedule::Dynamic] {
            let name = format!("varcost_n1024/chunk8/w{workers}/{}", label(s));
            run_case!(
                group,
                name,
                VarCost,
                VarConfig::default(),
                1024,
                8,
                workers,
                s
            );
        }
        let skew = || VarConfig {
            heavy_one_in: 64,
            heavy_spins: 64 * 1024,
            redraw: 1,
            ..VarConfig::default()
        };
        for s in [Schedule::Static, Schedule::Dynamic] {
            let name = format!("varskew_n256/chunk4/w{workers}/{}", label(s));
            run_case!(group, name, VarCost, skew(), 256, 4, workers, s);
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
