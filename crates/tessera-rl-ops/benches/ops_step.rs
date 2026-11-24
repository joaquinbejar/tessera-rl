// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inline step and reset cost of JobShop-v1 (the default 6x6 and the
//! largest 64x32 shape) and Vrp-v1 (20 and 500 customers, masks 501 wide):
//! full pool steps with auto-reset and masks.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;
use tessera_rl_ops::{JobShop, JobShopConfig, Vrp, VrpConfig};

const N: u32 = 1024;

fn bench(c: &mut Criterion) {
    let options = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1));
    let mut group = c.benchmark_group("ops_pool_1024");
    for (name, jobs, machines) in [("jobshop_6x6", 6, 6), ("jobshop_64x32", 64, 32)] {
        let cfg = JobShopConfig {
            jobs,
            machines,
            max_duration: 99,
        };
        // Scattered jobs; a finished one is replaced by the no-op.
        let actions: Vec<i32> = (0..N).map(|i| ((i * 7) % jobs).cast_signed()).collect();
        let mut pool = EnvPool::<JobShop, Masked>::with_masks(cfg, options).unwrap();
        pool.reset_all(Reseed::Keep).unwrap();
        group.bench_function(format!("{name}_step"), |b| {
            b.iter(|| {
                let out = pool
                    .step(ActionInput {
                        discrete: &actions,
                        continuous: &[],
                    })
                    .unwrap();
                black_box(out.rewards());
            });
        });
        group.bench_function(format!("{name}_reset_all"), |b| {
            b.iter(|| black_box(pool.reset_all(Reseed::Keep).unwrap().rewards().len()));
        });
    }
    for (name, customers) in [("vrp_20", 20u32), ("vrp_500", 500)] {
        let cfg = VrpConfig {
            customers,
            grid: 1000,
            capacity: 100,
            ..VrpConfig::default()
        };
        // Scattered nodes; a served or unfitting one is replaced by the no-op.
        let actions: Vec<i32> = (0..N * 8)
            .map(|i| ((i * 37) % (customers + 1)).cast_signed())
            .collect();
        let mut pool = EnvPool::<Vrp, Masked>::with_masks(cfg, options).unwrap();
        pool.reset_all(Reseed::Keep).unwrap();
        group.bench_function(format!("{name}_step"), |b| {
            b.iter(|| {
                let out = pool
                    .step(ActionInput {
                        discrete: &actions,
                        continuous: &[],
                    })
                    .unwrap();
                black_box(out.rewards());
            });
        });
        group.bench_function(format!("{name}_reset_all"), |b| {
            b.iter(|| black_box(pool.reset_all(Reseed::Keep).unwrap().rewards().len()));
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
