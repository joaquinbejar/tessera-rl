// SPDX-License-Identifier: MIT OR Apache-2.0

//! Push and pop cost of the Async id rings (ADR-0047), single thread:
//! a submit and claim, and a completion push and pop, per id.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_engine::rings::{CompletionRing, IdBufferPool, SubmitRing};

const N: usize = 1024;
const N_IDS: u32 = 1024;

fn rings(c: &mut Criterion) {
    let mut submit = SubmitRing::new(N).unwrap();
    c.bench_function("id_rings/submit_claim_1024", |b| {
        b.iter(|| {
            for id in 0..N_IDS {
                submit.push(black_box(id)).unwrap();
            }
            for _ in 0..N {
                black_box(submit.claim().unwrap());
            }
        });
    });
    let mut done = CompletionRing::new(N).unwrap();
    c.bench_function("id_rings/complete_pop_1024", |b| {
        b.iter(|| {
            for id in 0..N_IDS {
                done.push(black_box(id)).unwrap();
            }
            for _ in 0..N {
                black_box(done.pop().unwrap());
            }
        });
    });
    let pool = IdBufferPool::new(N, 64).unwrap();
    c.bench_function("id_rings/buffer_acquire_release", |b| {
        b.iter(|| {
            let h = pool.acquire(black_box(false)).unwrap();
            black_box(pool.release(h));
        });
    });
}

criterion_group!(benches, rings);
criterion_main!(benches);
