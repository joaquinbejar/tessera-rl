// SPDX-License-Identifier: MIT OR Apache-2.0

//! `I32Lanes` and `Fx32Lanes` per SIMD level (#345 stage 2 part 1): the
//! `i32` wrapping multiply and the `Fx32` saturating multiply and add, over
//! 64 KiB of operands (16384 lanes) per iteration. No claim is made from
//! these numbers; SPEC 5.5 asks for a bench per kernel.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_num::math::lane_int::{Fx32Lanes, I32Lanes};
use tessera_rl_simd::{LaneKernel, Lanes, Resolved};

const CHUNKS: usize = 16_384 / 8;

#[derive(Clone, Copy)]
enum Op {
    WrappingMul,
    SaturatingMul,
    SaturatingAdd,
}

struct Run<'a> {
    op: Op,
    a: &'a [[i32; 8]],
    b: &'a [[i32; 8]],
    out: &'a mut [[i32; 8]],
}

impl LaneKernel for Run<'_> {
    type Output = ();

    #[expect(clippy::inline_always, reason = "the LaneKernel contract")]
    #[inline(always)]
    fn run<S: Lanes>(self) {
        for ((a, b), out) in self.a.iter().zip(self.b).zip(self.out.iter_mut()) {
            let (a, b) = (S::I32::from_array(*a), S::I32::from_array(*b));
            *out = match self.op {
                Op::WrappingMul => a.wrapping_mul(b).to_array(),
                Op::SaturatingMul => S::Fx32::from_raw(a)
                    .saturating_mul(S::Fx32::from_raw(b))
                    .to_raw()
                    .to_array(),
                Op::SaturatingAdd => S::Fx32::from_raw(a)
                    .saturating_add(S::Fx32::from_raw(b))
                    .to_raw()
                    .to_array(),
            };
        }
    }
}

/// A fixed word stream for the operands (splitmix64).
fn words(seed: u64) -> Vec<[i32; 8]> {
    let mut s = seed;
    (0..CHUNKS)
        .map(|_| {
            core::array::from_fn(|_| {
                s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                #[expect(clippy::cast_possible_truncation, reason = "the low word")]
                let w = (z ^ (z >> 31)) as i32;
                w >> 8
            })
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let (a, b) = (words(1), words(2));
    let mut out = vec![[0; 8]; CHUNKS];
    for (name, op) in [
        ("i32_wrapping_mul", Op::WrappingMul),
        ("fx32_saturating_mul", Op::SaturatingMul),
        ("fx32_saturating_add", Op::SaturatingAdd),
    ] {
        let mut group = c.benchmark_group(name);
        for level in Resolved::available() {
            group.bench_function(format!("{:?}", level.level()), |bench| {
                bench.iter(|| {
                    level.dispatch_lanes(Run {
                        op,
                        a: black_box(&a),
                        b: black_box(&b),
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
