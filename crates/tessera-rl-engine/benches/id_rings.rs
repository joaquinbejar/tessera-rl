// SPDX-License-Identifier: MIT OR Apache-2.0

//! Push and pop cost of the Async id rings (ADR-0047): single thread, a
//! submit and claim and a completion push and pop per id; then contended,
//! one producer and four claimers on the submission ring and four producers
//! and one consumer on the completion ring, which shows whether the head
//! and tail positions share a cache line badly (ADR-0047 carried
//! requirement). Each contended iteration spawns its threads, so the
//! numbers include that cost; compare them with each other only.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use std::hint::black_box;
use std::sync::atomic::{AtomicU32, Ordering};

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
            black_box(pool.release(h).unwrap());
        });
    });
}

const CONTENDED: u32 = 1 << 14;
const THREADS: usize = 4;

fn contended(c: &mut Criterion) {
    c.bench_function("id_rings/contended_submit_1p_4c_16384", |b| {
        let submit = SubmitRing::new(N).unwrap();
        b.iter(|| {
            let claimed = AtomicU32::new(0);
            std::thread::scope(|s| {
                for _ in 0..THREADS {
                    let (submit, claimed) = (&submit, &claimed);
                    s.spawn(move || {
                        while claimed.load(Ordering::Relaxed) < CONTENDED {
                            if submit.claim().unwrap().is_some() {
                                claimed.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    });
                }
                // The single producer, as in the mode (through `&`).
                for id in 0..CONTENDED {
                    while submit.push_shared(id % N_IDS).is_err() {
                        std::hint::spin_loop();
                    }
                }
            });
        });
    });
    c.bench_function("id_rings/contended_complete_4p_1c_16384", |b| {
        let done = CompletionRing::new(N).unwrap();
        let per = CONTENDED / u32::try_from(THREADS).unwrap();
        b.iter(|| {
            let shared = &done;
            let popped = std::thread::scope(|s| {
                for _ in 0..THREADS {
                    s.spawn(move || {
                        for id in 0..per {
                            while shared.push(id % N_IDS).is_err() {
                                std::hint::spin_loop();
                            }
                        }
                    });
                }
                // The single consumer pops while the producers push, as in
                // the mode (through `&`).
                let mut popped = 0;
                while popped < CONTENDED {
                    if shared.pop_shared().unwrap().is_some() {
                        popped += 1;
                    }
                }
                popped
            });
            black_box(popped);
        });
    });
}

criterion_group!(benches, rings, contended);
criterion_main!(benches);
