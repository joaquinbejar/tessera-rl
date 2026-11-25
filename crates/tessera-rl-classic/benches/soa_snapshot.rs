// SPDX-License-Identifier: MIT OR Apache-2.0

//! The gather cost of a lane-wise pool's snapshots (ADR-0090 decision 7):
//! `snapshot_all`, `snapshot_one`, `fork`, `fork_branched` and `state_hash`
//! on `SoaCartPole` (state in columns: a snapshot gathers four fields, a
//! fork scatters them) against `CartPole` (state in rows), same state, same
//! records, at N = 8192 in chunks of 8. A report, not a gate: no claim is
//! made from these numbers.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use tessera_rl_classic::cartpole::{CartPole, CartPoleConfig};
use tessera_rl_core::Environment;
use tessera_rl_engine::{ActionInput, EnvPool, PoolOptions, Record, Reseed};
use tessera_rl_num::identity::Seed;

#[path = "support/soa_cartpole.rs"]
mod soa_cartpole;
use soa_cartpole::SoaCartPole;

const N: u32 = 8192;

fn record_bytes<E: Environment>() -> u64 {
    u64::try_from(size_of::<Record<E>>()).unwrap()
}

macro_rules! cases {
    ($group:expr, $label:literal, $env:ty) => {{
        let group: &mut BenchmarkGroup<'_, WallTime> = $group;
        let bytes = record_bytes::<$env>();
        let n = u64::from(N);
        let dst: Vec<u32> = (1..N).collect();

        // A pool reset and stepped once, so every record holds live state.
        let mut p = EnvPool::<$env>::new(
            CartPoleConfig {},
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

        group.throughput(Throughput::Bytes(bytes * n));
        group.bench_function(concat!($label, "/snapshot_all"), |b| {
            b.iter(|| p.snapshot_all(black_box(&mut records)).unwrap());
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
        group.bench_function(concat!($label, "/state_hash"), |b| {
            b.iter(|| black_box(p.state_hash(black_box(7)).unwrap()));
        });
    }};
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("soa_snapshot_8192");
    cases!(&mut group, "aos_cartpole", CartPole);
    cases!(&mut group, "soa_cartpole", SoaCartPole);
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
