// SPDX-License-Identifier: MIT OR Apache-2.0

//! The narrow loads and the byte gather of `I32Lanes` per SIMD level
//! (ADR-0100 decision 4): 16384 lanes per iteration. Reported only; ADR-0100
//! decision 6's stop point is what decides (SPEC 5.5 asks for a bench per
//! kernel).

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_num::math::lane_int::I32Lanes;
use tessera_rl_simd::{LaneKernel, Lanes, Resolved};

const CHUNKS: usize = 16_384 / 8;

#[derive(Clone, Copy)]
enum Op {
    FromI16,
    FromU8,
    GatherU8,
}

struct Run<'a> {
    op: Op,
    s16: &'a [[i16; 8]],
    u8s: &'a [[u8; 8]],
    idx: &'a [[i32; 8]],
    map: &'a [u8],
    out: &'a mut [[i32; 8]],
}

impl LaneKernel for Run<'_> {
    type Output = ();

    #[expect(clippy::inline_always, reason = "the LaneKernel contract")]
    #[inline(always)]
    fn run<S: Lanes>(self) {
        for (k, out) in self.out.iter_mut().enumerate() {
            *out = match self.op {
                Op::FromI16 => self
                    .s16
                    .get(k)
                    .map_or([0; 8], |v| S::I32::from_i16(*v).to_array()),
                Op::FromU8 => self
                    .u8s
                    .get(k)
                    .map_or([0; 8], |v| S::I32::from_u8(*v).to_array()),
                Op::GatherU8 => self
                    .idx
                    .get(k)
                    .map_or([0; 8], |i| S::I32::gather_u8(self.map, *i).to_array()),
            };
        }
    }
}

fn bench(c: &mut Criterion) {
    let s16: Vec<[i16; 8]> = (0..CHUNKS)
        .map(|k| core::array::from_fn(|i| i16::try_from((k * 8 + i) % 30_000).unwrap_or(0)))
        .collect();
    let u8s: Vec<[u8; 8]> = (0..CHUNKS)
        .map(|k| core::array::from_fn(|i| u8::try_from((k * 8 + i) % 256).unwrap_or(0)))
        .collect();
    let map = vec![7u8; 96 * 96];
    let idx: Vec<[i32; 8]> = (0..CHUNKS)
        .map(|k| core::array::from_fn(|i| i32::try_from((k * 8 + i) * 37 % (96 * 96)).unwrap_or(0)))
        .collect();
    let mut out = vec![[0; 8]; CHUNKS];
    for (name, op) in [
        ("from_i16", Op::FromI16),
        ("from_u8", Op::FromU8),
        ("gather_u8", Op::GatherU8),
    ] {
        let mut group = c.benchmark_group(name);
        for level in Resolved::available() {
            group.bench_function(format!("{:?}", level.level()), |bench| {
                bench.iter(|| {
                    level.dispatch_lanes(Run {
                        op,
                        s16: black_box(&s16),
                        u8s: black_box(&u8s),
                        idx: black_box(&idx),
                        map: black_box(&map),
                        out: &mut out,
                    });
                    black_box(&out);
                });
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
