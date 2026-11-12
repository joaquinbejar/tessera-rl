// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-thread attribution of allocator calls inside a measurement window
//! (issue #43), shared by the engine's counting allocators: the plain one
//! of `support/counting.rs` and the recording one of
//! `common/alloc_diag.rs`. Neither the window state nor the classification
//! allocates: atomics, const-initialized thread-locals without `Drop`, and
//! `thread::current`, which does not allocate on a thread std spawned or
//! on `main`.
//!
//! The rule, inside a window:
//!
//! - COUNTED: the measuring thread (the one that opened the window,
//!   recognized by a thread-local flag, so by identity and never by name:
//!   it is counted also when libtest runs the test on `main`) and the
//!   engine's gang workers, whose names start with [`WORKER_PREFIX`]
//!   (`sys/gang.rs`).
//! - EXCLUDED: the libtest harness thread `main`; its calls and net bytes
//!   are printed when the window closes, never subtracted from a count.
//! - FAILS: any other thread, named or not. Closing the window panics with
//!   "measurement disturbed by thread <name>, +N bytes ...".
//!
//! A binary that opens a window holds only its one measuring test, so no
//! other test's libtest thread can run inside it.

#![allow(dead_code, reason = "each test binary uses part of it")]
#![expect(
    clippy::print_stderr,
    reason = "test diagnostics: the harness thread's excluded bytes are reported"
)]

use core::cell::Cell;
use core::sync::atomic::{AtomicBool, AtomicI64, AtomicU8, AtomicU64, AtomicUsize, Ordering};

/// The name prefix of the engine's gang workers (`sys/gang.rs`).
pub const WORKER_PREFIX: &str = "tessera-worker-";

/// Bytes of the first foreign thread's name kept for the failure message.
const NAME: usize = 48;

/// A measurement window is open.
static WINDOW: AtomicBool = AtomicBool::new(false);
/// Allocation calls (`alloc`, `alloc_zeroed`, `realloc`) and net bytes of
/// the counted threads.
static COUNTED_CALLS: AtomicU64 = AtomicU64::new(0);
static COUNTED_BYTES: AtomicI64 = AtomicI64::new(0);
/// The part of the counted allocation calls made by gang workers.
static WORKER_CALLS: AtomicU64 = AtomicU64::new(0);
/// Allocator calls and net bytes of the harness thread `main` (excluded).
static MAIN_CALLS: AtomicU64 = AtomicU64::new(0);
static MAIN_BYTES: AtomicI64 = AtomicI64::new(0);
/// Allocator calls and net bytes of every other thread.
static FOREIGN_CALLS: AtomicU64 = AtomicU64::new(0);
static FOREIGN_BYTES: AtomicI64 = AtomicI64::new(0);
/// The first foreign thread's name, copied (truncated) without a lock.
static FOREIGN_NAMED: AtomicBool = AtomicBool::new(false);
static FOREIGN_NAME: [AtomicU8; NAME] = [const { AtomicU8::new(0) }; NAME];
static FOREIGN_NAME_LEN: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// This thread opened the current window.
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    /// The classification of a call is running on this thread: a nested
    /// call cannot be attributed and fails the window.
    static BUSY: Cell<bool> = const { Cell::new(false) };
}

/// The class of the thread making an allocator call in a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// The thread that opened the window.
    Measuring,
    /// A gang worker of the engine.
    Worker,
    /// The libtest harness thread, excluded.
    Main,
    /// Any other thread: the window fails.
    Foreign,
}

/// What a window measured on the counted threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tally {
    /// Allocation calls of the measuring thread and the workers.
    pub calls: u64,
    /// Their net bytes (frees subtract).
    pub bytes: i64,
    /// The part of `calls` made by gang workers.
    pub worker_calls: u64,
    /// Net bytes of the harness thread `main`, excluded from the above.
    pub main_bytes: i64,
}

/// Converts a size to a signed byte delta.
#[must_use]
pub fn delta(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

fn name_foreign(name: &[u8]) {
    if FOREIGN_NAMED.swap(true, Ordering::SeqCst) {
        return;
    }
    let n = name.len().min(NAME);
    for (dst, &b) in FOREIGN_NAME.iter().zip(name.iter().take(n)) {
        dst.store(b, Ordering::SeqCst);
    }
    FOREIGN_NAME_LEN.store(n, Ordering::SeqCst);
}

fn classify() -> Class {
    if MEASURING.try_with(Cell::get).unwrap_or(false) {
        return Class::Measuring;
    }
    if BUSY.try_with(|b| b.replace(true)).unwrap_or(true) {
        name_foreign(b"<unattributed: nested call or exiting thread>");
        return Class::Foreign;
    }
    let current = std::thread::current();
    let class = match current.name() {
        Some("main") => Class::Main,
        Some(n) if n.starts_with(WORKER_PREFIX) => Class::Worker,
        other => {
            name_foreign(other.unwrap_or("<unnamed>").as_bytes());
            Class::Foreign
        }
    };
    drop(current);
    let _ = BUSY.try_with(|b| b.set(false));
    class
}

/// Attributes one allocator call of `bytes` net to its thread; `allocation`
/// is true for `alloc`, `alloc_zeroed` and `realloc`. Returns the class, or
/// `None` outside a window.
pub fn note(bytes: i64, allocation: bool) -> Option<Class> {
    if !WINDOW.load(Ordering::SeqCst) {
        return None;
    }
    let class = classify();
    match class {
        Class::Measuring | Class::Worker => {
            if allocation {
                COUNTED_CALLS.fetch_add(1, Ordering::SeqCst);
                if class == Class::Worker {
                    WORKER_CALLS.fetch_add(1, Ordering::SeqCst);
                }
            }
            COUNTED_BYTES.fetch_add(bytes, Ordering::SeqCst);
        }
        Class::Main => {
            MAIN_CALLS.fetch_add(1, Ordering::SeqCst);
            MAIN_BYTES.fetch_add(bytes, Ordering::SeqCst);
        }
        Class::Foreign => {
            FOREIGN_CALLS.fetch_add(1, Ordering::SeqCst);
            FOREIGN_BYTES.fetch_add(bytes, Ordering::SeqCst);
        }
    }
    Some(class)
}

/// Opens a window on this thread, the measuring one.
pub fn begin() {
    for c in [&COUNTED_CALLS, &WORKER_CALLS, &MAIN_CALLS, &FOREIGN_CALLS] {
        c.store(0, Ordering::SeqCst);
    }
    for b in [&COUNTED_BYTES, &MAIN_BYTES, &FOREIGN_BYTES] {
        b.store(0, Ordering::SeqCst);
    }
    FOREIGN_NAME_LEN.store(0, Ordering::SeqCst);
    FOREIGN_NAMED.store(false, Ordering::SeqCst);
    MEASURING.with(|m| m.set(true));
    WINDOW.store(true, Ordering::SeqCst);
}

fn close() {
    WINDOW.store(false, Ordering::SeqCst);
    MEASURING.with(|m| m.set(false));
}

/// Closes the window opened by [`begin`] and returns the counted tally.
///
/// The window is not a barrier: a call on another thread that passed the
/// window check just before it closed may still be added after, so the
/// race can only add to a count, never hide one.
///
/// # Panics
///
/// If a foreign thread called the allocator in the window.
#[must_use]
#[expect(clippy::panic, reason = "a disturbed window fails the test")]
pub fn end() -> Tally {
    close();
    let main_calls = MAIN_CALLS.load(Ordering::SeqCst);
    let main_bytes = MAIN_BYTES.load(Ordering::SeqCst);
    if main_calls != 0 {
        eprintln!(
            "harness thread main: {main_bytes:+} bytes in {main_calls} allocator calls in the window (excluded)"
        );
    }
    let calls = FOREIGN_CALLS.load(Ordering::SeqCst);
    if calls != 0 {
        let len = FOREIGN_NAME_LEN.load(Ordering::SeqCst).min(NAME);
        let name: Vec<u8> = FOREIGN_NAME
            .iter()
            .take(len)
            .map(|b| b.load(Ordering::SeqCst))
            .collect();
        panic!(
            "measurement disturbed by thread {:?}, {:+} bytes in {calls} allocator calls",
            String::from_utf8_lossy(&name),
            FOREIGN_BYTES.load(Ordering::SeqCst),
        );
    }
    Tally {
        calls: COUNTED_CALLS.load(Ordering::SeqCst),
        bytes: COUNTED_BYTES.load(Ordering::SeqCst),
        worker_calls: WORKER_CALLS.load(Ordering::SeqCst),
        main_bytes,
    }
}

/// Closes the window if the measured code unwinds.
struct Unwound;

impl Drop for Unwound {
    fn drop(&mut self) {
        close();
    }
}

/// Runs `f` in a window on this thread; returns its result and the tally.
///
/// # Panics
///
/// If a foreign thread called the allocator in the window, or if `f`
/// panics (the window is closed first).
pub fn window<R>(f: impl FnOnce() -> R) -> (R, Tally) {
    begin();
    let guard = Unwound;
    let r = f();
    core::mem::forget(guard);
    (r, end())
}
