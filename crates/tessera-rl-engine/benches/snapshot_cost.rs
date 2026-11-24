// SPDX-License-Identifier: MIT OR Apache-2.0

//! Wall-clock cost of the snapshot operations against the SPEC 9.2 targets
//! (#362): `snapshot`, `restore`, `fork` and `fork_branched` at most 1 us per
//! KB per instance, `state_hash` about 5 GB/s, on the reference x86 machine.
//!
//! Every case declares its throughput in record bytes (header plus state),
//! so Criterion reports bytes per second: 1 us per KB is about 0.95 GiB/s.
//! Two environments: the test counter (a small record, where the fixed cost
//! per call dominates) and `Bulk` (a 4 KiB state, where the copy does).
//! `snapshot_instructions` counts instructions of the same calls; this bench
//! measures time and is a report, not a gate.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use tessera_rl_core::Environment;
use tessera_rl_engine::{ActionInput, EnvPool, PoolOptions, Record, Reseed};
use tessera_rl_num::identity::Seed;

#[path = "../tests/common/mod.rs"]
mod common;

#[path = "support/bulk.rs"]
mod bulk;

use bulk::{Bulk, BulkConfig};
use common::{CountConfig, Counter};

/// Instances per pool: enough to leave the per-call overhead behind, small
/// enough that the Bulk pool (about 4 MiB of state) fits a shared cache level
/// on the reference machine only partly, as a training pool would.
const N: u32 = 1024;

fn record_bytes<E: Environment>() -> u64 {
    u64::try_from(size_of::<Record<E>>()).unwrap()
}

macro_rules! cases {
    ($group:expr, $label:literal, $env:ty, $cfg:expr) => {{
        let group: &mut BenchmarkGroup<'_, WallTime> = $group;
        let bytes = record_bytes::<$env>();
        let n = u64::from(N);
        let dst: Vec<u32> = (1..N).collect();

        // A pool reset and stepped once, so every record holds live state.
        let mut p = EnvPool::<$env>::new(
            $cfg,
            PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1)),
        )
        .unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        let actions = vec![0; p.outputs().unwrap().rewards().len()];
        p.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        })
        .unwrap();
        let mut records = vec![p.snapshot(0).unwrap(); usize::try_from(N).unwrap()];

        group.throughput(Throughput::Bytes(bytes * n));
        group.bench_function(concat!($label, "/snapshot_all"), |b| {
            b.iter(|| p.snapshot_all(black_box(&mut records)).unwrap());
        });
        group.bench_function(concat!($label, "/restore"), |b| {
            b.iter(|| {
                for (id, record) in (0..N).zip(&records) {
                    p.restore(id, black_box(record)).unwrap();
                }
            });
        });

        group.throughput(Throughput::Bytes(bytes * (n - 1)));
        group.bench_function(concat!($label, "/fork"), |b| {
            b.iter(|| p.fork(0, black_box(&dst)).unwrap());
        });
        group.bench_function(concat!($label, "/fork_branched"), |b| {
            b.iter(|| p.fork_branched(0, black_box(&dst), None).unwrap());
        });

        group.throughput(Throughput::Bytes(bytes));
        group.bench_function(concat!($label, "/snapshot_one"), |b| {
            b.iter(|| black_box(p.snapshot(black_box(7)).unwrap()));
        });
        let record = p.snapshot(7).unwrap();
        group.bench_function(concat!($label, "/record_hash"), |b| {
            b.iter(|| black_box(black_box(&record).hash()));
        });
        group.bench_function(concat!($label, "/state_hash"), |b| {
            b.iter(|| black_box(p.state_hash(black_box(7)).unwrap()));
        });
    }};
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot_cost_1024");
    cases!(
        &mut group,
        "counter",
        Counter,
        CountConfig {
            length: 1000,
            ..CountConfig::default()
        }
    );
    cases!(&mut group, "bulk4k", Bulk, BulkConfig::default());
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
