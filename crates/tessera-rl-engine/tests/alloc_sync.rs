// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sync mode never allocates after construction, on any thread (SPEC 5.5).
//!
//! A counting global allocator for this test binary only; test code, not
//! one of the audited `unsafe` modules. It counts on every thread while
//! enabled, so this binary holds a single test.

use core::num::{NonZeroU32, NonZeroUsize};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

mod common;

use common::{CountConfig, Counter};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

struct Counting;

#[expect(unsafe_code, reason = "test-only counting allocator")]
// SAFETY: every method forwards to `System` with the same arguments; the
// counters are atomics, which never allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            diag(layout.size());
        }
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarded unchanged; the caller upholds `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

thread_local! { static IN: core::cell::Cell<bool> = const { core::cell::Cell::new(false) }; }
static TRACES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
fn diag(size: usize) {
    if IN.with(|f| f.replace(true)) { return; }
    let bt = std::backtrace::Backtrace::force_capture();
    let s = format!("size {size} thread {:?}\n{bt}", std::thread::current().name());
    TRACES.lock().unwrap().push(s);
    IN.with(|f| f.set(false));
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
    for t in TRACES.lock().unwrap().iter() { eprintln!("=== {t}"); }
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
}
