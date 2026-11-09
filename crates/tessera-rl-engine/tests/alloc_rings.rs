// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Async id rings and token id buffers never allocate after
//! construction (ADR-0047, SPEC 5.5): one id stays claimed (a stalled
//! instance) while every other id is submitted, claimed, completed and
//! resubmitted 100,000 times, and batch buffers are acquired and released
//! out of order. One test per binary, since the counting allocator counts
//! on every thread.
//!
//! Each counted allocation also records its thread and size, so a failure
//! names its source: this test once saw 2 allocations on aarch64 musl, cause
//! not proven (run 42613007854). The count stays on every thread on purpose
//! until a repeat says where they come from.

use core::cell::Cell;
use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};

const EVENTS: usize = 16;
const NAME: usize = 48;

/// One recorded allocation, all atomics so that writing needs no lock.
struct Event {
    size: AtomicUsize,
    name: [AtomicU8; NAME],
    name_len: AtomicUsize,
}

#[expect(
    clippy::declare_interior_mutable_const,
    reason = "the initializer of the static table, one copy per slot"
)]
const EMPTY: Event = Event {
    size: AtomicUsize::new(0),
    name: [const { AtomicU8::new(0) }; NAME],
    name_len: AtomicUsize::new(0),
};

static TABLE: [Event; EVENTS] = [EMPTY; EVENTS];
static NEXT: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Set while this thread records: nested allocations are not counted.
    static RECORDING: Cell<bool> = const { Cell::new(false) };
}

/// Counts one allocation of `size` bytes and records it.
fn count(size: usize) {
    if !COUNTING.load(Ordering::Acquire) || RECORDING.with(|r| r.replace(true)) {
        return;
    }
    COUNT.fetch_add(1, Ordering::Relaxed);
    let slot = NEXT.fetch_add(1, Ordering::Relaxed);
    if let Some(e) = TABLE.get(slot) {
        e.size.store(size, Ordering::Relaxed);
        // `thread::current` is safe here: no counted window spans a thread's
        // exit of the kind that would make it panic, and an allocation under
        // the guard is not counted.
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("<unnamed>").as_bytes();
        let n = name.len().min(NAME);
        for (dst, &b) in e.name.iter().zip(name.iter().take(n)) {
            dst.store(b, Ordering::Relaxed);
        }
        e.name_len.store(n, Ordering::Relaxed);
    }
    RECORDING.with(|r| r.set(false));
}

fn start() {
    COUNT.store(0, Ordering::Relaxed);
    NEXT.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Release);
}

fn stop() -> u64 {
    COUNTING.store(false, Ordering::Release);
    COUNT.load(Ordering::Relaxed)
}

/// `(thread, size)` of the recorded events of the last window.
fn recorded() -> Vec<(String, usize)> {
    let n = NEXT.load(Ordering::Relaxed).min(EVENTS);
    TABLE
        .iter()
        .take(n)
        .map(|e| {
            let len = e.name_len.load(Ordering::Relaxed).min(NAME);
            let bytes: Vec<u8> = e
                .name
                .iter()
                .take(len)
                .map(|b| b.load(Ordering::Relaxed))
                .collect();
            (
                String::from_utf8_lossy(&bytes).into_owned(),
                e.size.load(Ordering::Relaxed),
            )
        })
        .collect()
}

fn report(total: u64) -> String {
    let mut out = String::new();
    let events = recorded();
    for (thread, size) in &events {
        let _ = writeln!(out, "{size} bytes on thread {thread:?}");
    }
    let shown = u64::try_from(events.len()).unwrap_or(u64::MAX);
    if total > shown {
        let _ = writeln!(out, "... and {} more, not recorded", total - shown);
    }
    out
}

/// Counts on every thread while enabled, and records each allocation.
struct Counting;

#[expect(unsafe_code, reason = "test-only counting allocator")]
// SAFETY: every call forwards to `System` with the same arguments; the
// counters are atomics and a const-initialized thread local, which never
// allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        // SAFETY: the caller's contract for `alloc` is passed through.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract for `dealloc` is passed through.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        // SAFETY: the caller's contract for `realloc` is passed through.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The recording works, on another thread than the test's: one binary, one
/// test, so it runs before the measured window, which clears it.
fn check_the_recording_names_another_thread() {
    let probe = std::thread::Builder::new()
        .name("alloc-probe".to_owned())
        .spawn(|| {
            while !COUNTING.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            drop(std::hint::black_box(Vec::<u8>::with_capacity(1234)));
        })
        .unwrap();
    start();
    probe.join().unwrap();
    let total = stop();
    assert!(
        recorded().iter().any(|(t, s)| t == "alloc-probe" && *s == 1234),
        "the recording missed a known allocation:\n{}",
        report(total)
    );
}

use tessera_rl_engine::rings::{CompletionRing, IdBufferPool, SubmitRing};

#[test]
fn test_alloc_rings_stalled_id_and_buffer_cycling_do_not_allocate() {
    check_the_recording_names_another_thread();
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
    start();
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
    let count = stop();
    assert_eq!(last, Some(stalled));
    assert_eq!(
        count,
        0,
        "allocations in the measured window:\n{}",
        report(count)
    );
}
