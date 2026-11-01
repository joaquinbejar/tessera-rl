// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sync mode never allocates after construction, on any thread (SPEC 5.5).
//!
//! A counting global allocator for this test binary only; test code, not
//! one of the audited `unsafe` modules. It counts on every thread while
//! enabled, so this binary holds a single test.
//!
//! On a failure the test prints, for each counted allocation (up to
//! [`MAX_TRACES`]), its size, its thread and a backtrace. The capture runs
//! under a per-thread guard, so its own allocations are not counted, and
//! the trace storage is allocated before counting starts. The assertion
//! itself is unchanged: zero allocations on every thread.

use core::num::{NonZeroU32, NonZeroUsize};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

mod common;

use common::{CountConfig, Counter};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

/// Counted allocations described on a failure.
const MAX_TRACES: usize = 16;
static TRACES: Mutex<Vec<String>> = Mutex::new(Vec::new());

thread_local! {
    /// Set while this thread captures a trace: its allocations are the
    /// diagnostic's, not the code under test's.
    static CAPTURING: Cell<bool> = const { Cell::new(false) };
}

/// Counts one allocation of `size` bytes, unless it is the diagnostic's own,
/// and records where it came from.
fn count(size: usize) {
    if !COUNTING.load(Ordering::Relaxed) || CAPTURING.with(|c| c.replace(true)) {
        return;
    }
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut traces) = TRACES.lock()
        && traces.len() < MAX_TRACES
    {
        let thread = std::thread::current();
        traces.push(format!(
            "{size} bytes on thread {:?} ({:?})\n{}",
            thread.name(),
            thread.id(),
            std::backtrace::Backtrace::force_capture()
        ));
    }
    CAPTURING.with(|c| c.set(false));
}

struct Counting;

#[expect(unsafe_code, reason = "test-only counting allocator")]
// SAFETY: every method forwards to `System` with the same arguments; the
// counters are atomics, which never allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        // SAFETY: forwarded unchanged; the caller upholds `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn test_sync_reset_and_step_do_not_allocate_on_any_thread() {
    let options = PoolOptions {
        auto_reset: AutoReset::SameStep,
        chunk_size: NonZeroU32::new(8).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(1))
    };
    let cfg = CountConfig {
        length: 4,
        ..CountConfig::default()
    };
    let mut pool = EnvPool::<Counter, Masked>::with_masks(cfg, options)
        .unwrap()
        .with_workers(SyncOptions::new(NonZeroUsize::new(4).unwrap()))
        .unwrap();
    let actions = vec![1i32; 128];
    // Warm up: the first epoch may initialize thread-local runtime state.
    pool.reset_all(Reseed::Keep).unwrap();
    pool.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap();
    TRACES.lock().unwrap().reserve(MAX_TRACES);
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    for _ in 0..20 {
        let _ = std::hint::black_box(pool.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        }));
    }
    let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
    COUNTING.store(false, Ordering::Relaxed);
    let n = ALLOCATIONS.load(Ordering::Relaxed);
    let traces = TRACES.lock().unwrap().join("\n----\n");
    assert_eq!(n, 0, "allocations in the measured window:\n{traces}");
}
