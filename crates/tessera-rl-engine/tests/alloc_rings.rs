// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Async id rings and token id buffers never allocate after
//! construction (ADR-0047, SPEC 5.5): one id stays claimed (a stalled
//! instance) while every other id is submitted, claimed, completed and
//! resubmitted 100,000 times, and batch buffers are acquired and released
//! out of order. One test per binary, since the counting allocator counts
//! on every thread.

#[expect(dead_code, reason = "the shared test environments are not used here")]
mod common;

#[expect(dead_code, reason = "this test uses only the counting allocator")]
#[path = "common/alloc_diag.rs"]
mod alloc_diag;

use tessera_rl_engine::rings::{CompletionRing, IdBufferPool, SubmitRing};

#[test]
fn test_alloc_rings_stalled_id_and_buffer_cycling_do_not_allocate() {
    let n = 8_usize;
    let mut submit = SubmitRing::new(n).unwrap();
    let mut done = CompletionRing::new(n).unwrap();
    let pool = IdBufferPool::new(n, 2).unwrap();
    let ids: Vec<u32> = (0..u32::try_from(n).unwrap()).collect();
    for &id in &ids {
        submit.push(id).unwrap();
    }
    let stalled = submit.claim().unwrap().unwrap();
    let generations = [0_u64, 0];
    alloc_diag::start();
    for round in 0..100_000_u64 {
        for _ in 1..n {
            let id = submit.claim().unwrap().unwrap();
            done.push(id).unwrap();
        }
        for _ in 1..n {
            let id = done.pop().unwrap().unwrap();
            submit.push(id).unwrap();
        }
        // Three batch tokens live, released out of order.
        let a = pool.acquire(false).unwrap();
        let b = pool.acquire(false).unwrap();
        let c = pool.acquire(false).unwrap();
        pool.fill(&b, &[1, 2], &generations).unwrap();
        let _ = std::hint::black_box(pool.entries(&b).count());
        let (first, second) = if round % 2 == 0 { (b, a) } else { (a, b) };
        let _ = pool.release(first);
        let _ = pool.release(c);
        let _ = pool.release(second);
    }
    done.push(stalled).unwrap();
    let last = done.pop().unwrap();
    let count = alloc_diag::stop();
    assert_eq!(last, Some(stalled));
    assert_eq!(
        count,
        0,
        "allocations in the measured window:\n{}",
        alloc_diag::report()
    );
}
