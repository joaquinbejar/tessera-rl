// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Async id rings and token id buffers never allocate after
//! construction (ADR-0047, SPEC 5.5): one id stays claimed (a stalled
//! instance) while every other id is submitted, claimed, completed and
//! resubmitted 100,000 times, and batch buffers are acquired and released
//! out of order. One test per binary, since the counting allocator counts
//! on every thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Counts allocations on every thread while enabled. The shared counting
/// allocator of `common/alloc_diag.rs` needs the test environments of
/// `common`, which this binary does not use, so it has its own.
struct Counting;

static COUNTING: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicU64 = AtomicU64::new(0);

#[expect(unsafe_code, reason = "test-only counting allocator")]
// SAFETY: every call forwards to `System` with the same arguments; the
// counter is a side effect only.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller's contract for `alloc` is passed through.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract for `dealloc` is passed through.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller's contract for `realloc` is passed through.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

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
    COUNTING.store(true, Ordering::SeqCst);
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
        let _ = std::hint::black_box(pool.entries(&b).unwrap().count());
        let (first, second) = if round % 2 == 0 { (b, a) } else { (a, b) };
        assert!(pool.release(first).is_ok());
        assert!(pool.release(c).is_ok());
        assert!(pool.release(second).is_ok());
    }
    done.push(stalled).unwrap();
    let last = done.pop().unwrap();
    COUNTING.store(false, Ordering::SeqCst);
    let count = COUNT.load(Ordering::SeqCst);
    assert_eq!(last, Some(stalled));
    assert_eq!(count, 0, "allocations in the measured window");
}
