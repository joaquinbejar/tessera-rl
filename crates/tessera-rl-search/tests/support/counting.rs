// SPDX-License-Identifier: MIT OR Apache-2.0

//! A counting global allocator and the measured `SearchGroup` build shared
//! by the memory tests. Each test binary that includes this module holds
//! exactly one measuring test, so no other test's thread runs inside a
//! window. It is test code, not one of the audited `unsafe` modules; its
//! `unsafe` only forwards to the system allocator.
//!
//! Inside a measurement window the allocator keeps per-thread totals: the
//! number compared with the bound is the measuring thread's own net bytes.
//! The libtest harness thread (`main`) is excluded and its net bytes are
//! printed; any other thread (an engine worker, a thread the measured code
//! spawned) that calls the allocator in the window fails the test with its
//! name and net bytes.

#![expect(
    clippy::indexing_slicing,
    reason = "the copied name length is clamped to both buffers"
)]
#![expect(
    clippy::print_stderr,
    reason = "test diagnostics: the harness thread's excluded bytes are reported"
)]

use core::cell::Cell;
use core::num::NonZeroU32;
use core::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;

use tessera_rl_engine::{AutoReset, EnvPool, Masked, PoolOptions, Record, Reseed};
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{
    DEFAULT_GROUP_BUDGET, Eviction, GumbelConfig, Mcts, PuctConfig, RootSelection, SearchGroup,
    SearchOptions, SearchPool,
};

/// A measurement window is open.
static WINDOW: AtomicBool = AtomicBool::new(false);
/// Net bytes of the harness thread (`main`) in the window.
static MAIN_BYTES: AtomicI64 = AtomicI64::new(0);
/// Allocator calls and net bytes of every other foreign thread.
static FOREIGN_CALLS: AtomicU64 = AtomicU64::new(0);
static FOREIGN_BYTES: AtomicI64 = AtomicI64::new(0);
/// The first foreign thread's name, copied (truncated) without allocating.
static FOREIGN_NAME: Mutex<([u8; 48], usize)> = Mutex::new(([0; 48], 0));

thread_local! {
    /// This thread opened the current window.
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    /// The measuring thread's net bytes in the window.
    static MINE: Cell<i64> = const { Cell::new(0) };
    /// The allocator is classifying a call on this thread: a nested call
    /// (`thread::current` may allocate once for a thread std did not
    /// spawn) is counted without classifying it again.
    static BUSY: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

fn delta(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// Attributes one allocator call of `bytes` net to its thread.
fn count(bytes: i64) {
    if !WINDOW.load(Ordering::SeqCst) {
        return;
    }
    if MEASURING.try_with(Cell::get).unwrap_or(false) {
        let _ = MINE.try_with(|m| m.set(m.get() + bytes));
        return;
    }
    if BUSY.try_with(|b| b.replace(true)).unwrap_or(true) {
        FOREIGN_CALLS.fetch_add(1, Ordering::SeqCst);
        FOREIGN_BYTES.fetch_add(bytes, Ordering::SeqCst);
        return;
    }
    let current = std::thread::current();
    let name = current.name().unwrap_or("<unnamed>");
    if name == "main" {
        MAIN_BYTES.fetch_add(bytes, Ordering::SeqCst);
    } else {
        FOREIGN_BYTES.fetch_add(bytes, Ordering::SeqCst);
        if FOREIGN_CALLS.fetch_add(1, Ordering::SeqCst) == 0
            && let Ok(mut slot) = FOREIGN_NAME.try_lock()
        {
            let n = name.len().min(slot.0.len());
            slot.0[..n].copy_from_slice(&name.as_bytes()[..n]);
            slot.1 = n;
        }
    }
    drop(current);
    let _ = BUSY.try_with(|b| b.set(false));
}

#[expect(unsafe_code, reason = "test-only counting allocator")]
// SAFETY: every method forwards to `System` with the same arguments; the
// counting only touches atomics, const-initialized thread-locals and a
// `try_lock`, and a nested allocation it may cause is not classified again.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(delta(layout.size()));
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count(-delta(layout.size()));
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(delta(layout.size()));
        // SAFETY: forwarded unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(delta(new_size) - delta(layout.size()));
        // SAFETY: forwarded unchanged; the caller upholds `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Runs `f` in a measurement window on this thread and returns its result with the measuring thread's net bytes.
///
/// # Panics
///
/// If a thread other than this one and `main` called the allocator in the
/// window.
pub fn window<R>(f: impl FnOnce() -> R) -> (R, i64) {
    MAIN_BYTES.store(0, Ordering::SeqCst);
    FOREIGN_CALLS.store(0, Ordering::SeqCst);
    FOREIGN_BYTES.store(0, Ordering::SeqCst);
    MINE.with(|m| m.set(0));
    MEASURING.with(|m| m.set(true));
    WINDOW.store(true, Ordering::SeqCst);
    let r = f();
    WINDOW.store(false, Ordering::SeqCst);
    MEASURING.with(|m| m.set(false));
    let mine = MINE.with(Cell::get);
    let main = MAIN_BYTES.load(Ordering::SeqCst);
    if main != 0 {
        eprintln!("harness thread main: {main:+} bytes in the window (excluded)");
    }
    let calls = FOREIGN_CALLS.load(Ordering::SeqCst);
    let (buf, n) = *FOREIGN_NAME.lock().unwrap();
    assert!(
        calls == 0,
        "measurement disturbed by thread {:?}, {:+} bytes in {calls} allocator calls",
        String::from_utf8_lossy(&buf[..n]),
        FOREIGN_BYTES.load(Ordering::SeqCst),
    );
    (r, mine)
}

pub fn record() -> Record<Connect4> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(21))
    };
    let mut p = EnvPool::<Connect4, Masked>::with_masks(Connect4Config {}, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p.snapshot(0).unwrap()
}

/// Builds a group of `trees` Connect4 trees from `root` (built before the
/// window) and returns the bytes the measuring thread retains and the
/// group's bound.
pub fn measured(
    root: &Record<Connect4>,
    trees: usize,
    capacity: u32,
    batch: u32,
    gumbel: bool,
) -> (i64, u64) {
    let ((bound, g), live) = window(|| {
        let mut cfg = PuctConfig::default();
        cfg.simulations = 32;
        cfg.batch_size = batch;
        if gumbel {
            cfg.root = RootSelection::Gumbel(GumbelConfig::default());
        }
        let options = || {
            let mut o = SearchOptions::new(
                NonZeroU32::new(capacity).unwrap(),
                NonZeroU32::new(batch).unwrap(),
            );
            o.eviction = Eviction::Oldest;
            o
        };
        let slots = (0..trees)
            .map(|_| {
                let pool = SearchPool::new(Connect4Config {}, options(), root, EnvPool::with_masks)
                    .unwrap();
                Mcts::new(pool, cfg).unwrap()
            })
            .collect();
        let shapes = vec![(options(), cfg); trees];
        let predicted =
            SearchGroup::<Connect4, Masked>::bound_bytes(&Connect4Config {}, &shapes).unwrap();
        let g = SearchGroup::new(&Connect4Config {}, slots, DEFAULT_GROUP_BUDGET).unwrap();
        assert_eq!(
            g.memory_bytes(),
            predicted,
            "checked before and after construction alike"
        );
        // Dropped after the window closes, so its bytes count as retained.
        (g.memory_bytes(), g)
    });
    drop(g);
    (live, bound)
}
