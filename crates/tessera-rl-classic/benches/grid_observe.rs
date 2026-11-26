// SPDX-License-Identifier: MIT OR Apache-2.0

//! The observe cost of the grid worlds (#751): `restore_all` at N = 8192 in
//! chunks of 8, which writes every record and re-observes every instance,
//! on `Grid` and `GridLanes` (default `GridConfig`, 768-byte observations)
//! and on `CartPole` as the control (16-byte observations). The record
//! write is about 45 instructions per instance (#751 prof-r1), so the rest
//! is the observe. Throughput is declared as `size_of::<Record<E>>()`
//! times N records, the convention `tools/bench/soa_bound.py` reads. A
//! report, not a gate.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use tessera_rl_classic::cartpole::{CartPole, CartPoleConfig};
use tessera_rl_classic::{Grid, GridConfig, GridLanes};
use tessera_rl_core::Environment;
use tessera_rl_engine::{ActionInput, EnvPool, PoolOptions, Record, Reseed};
use tessera_rl_num::identity::Seed;

const N: u32 = 8192;

fn record_bytes<E: Environment>() -> u64 {
    u64::try_from(size_of::<Record<E>>()).unwrap()
}

// `restore_all` of `$env`'s own records, taken after a reset and one step
// with action 1, so every instance holds live state.
macro_rules! restore_case {
    ($group:expr, $label:literal, $env:ty, $cfg:expr) => {{
        let group: &mut BenchmarkGroup<'_, WallTime> = $group;
        let mut p = EnvPool::<$env>::new(
            $cfg,
            PoolOptions {
                chunk_size: NonZeroU32::new(8).unwrap(),
                ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1))
            },
        )
        .unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        let actions = vec![1; p.outputs().unwrap().rewards().len()];
        p.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        })
        .unwrap();
        let mut records = vec![p.snapshot(0).unwrap(); usize::try_from(N).unwrap()];
        p.snapshot_all(&mut records).unwrap();

        group.throughput(Throughput::Bytes(record_bytes::<$env>() * u64::from(N)));
        group.bench_function(concat!($label, "/restore"), |b| {
            b.iter(|| p.restore_all(black_box(&records)).unwrap());
        });
    }};
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_observe_8192");
    restore_case!(&mut group, "aos_cartpole", CartPole, CartPoleConfig {});
    restore_case!(&mut group, "aos_grid", Grid, GridConfig::default());
    restore_case!(
        &mut group,
        "soa_grid_lanes",
        GridLanes,
        GridConfig::default()
    );
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
