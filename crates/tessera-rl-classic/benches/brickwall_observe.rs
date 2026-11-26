// SPDX-License-Identifier: MIT OR Apache-2.0

//! The pixel observe cost of `tessera/BrickWall-v1` (#755): `restore_all` at
//! N = 4096 in chunks of 8, which writes every record and re-observes every
//! instance, with the pixel observation (84 x 84 bytes, the wall drawn as runs
//! of adjacent bricks) and with the state observation as the control. The
//! records are taken after a reset and one step, so every wall is full: the
//! case with the most bricks to draw. Throughput is declared as
//! `size_of::<Record<BrickWall>>()` times N records, as `grid_observe` does.
//! A report, not a gate.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use tessera_rl_classic::brickwall::{OBS_PIXELS, OBS_STATE};
use tessera_rl_classic::{BrickWall, BrickWallConfig};
use tessera_rl_engine::{ActionInput, EnvPool, PoolOptions, Record, Reseed};
use tessera_rl_num::identity::Seed;

const N: u32 = 4096;

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("brickwall_observe_4096");
    for (label, obs) in [("pixels", OBS_PIXELS), ("state", OBS_STATE)] {
        let mut p = EnvPool::<BrickWall>::new(
            BrickWallConfig {
                obs,
                ..BrickWallConfig::default()
            },
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
        group.throughput(Throughput::Bytes(
            u64::try_from(size_of::<Record<BrickWall>>()).unwrap() * u64::from(N),
        ));
        group.bench_function(format!("{label}/restore"), |b| {
            b.iter(|| p.restore_all(black_box(&records)).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
