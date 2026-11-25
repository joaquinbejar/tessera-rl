// SPDX-License-Identifier: MIT OR Apache-2.0

//! Test-only allocation counting (feature `test-hooks`), for the
//! zero-allocation tests of the XLA FFI path (ADR-0095 point 5). An
//! extension's test binary installs it with
//! `#[global_allocator] static A: CountingAlloc = CountingAlloc;`, which is
//! safe code, so crates that forbid `unsafe` can use it.

#![expect(
    unsafe_code,
    reason = "audited module: a forwarding GlobalAlloc for tests"
)]

use core::cell::Cell;
use std::alloc::{GlobalAlloc, Layout, System};

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<u64> = const { Cell::new(0) };
}

/// Forwards to the system allocator and counts this thread's allocations
/// while [`count`] runs.
pub struct CountingAlloc;

// SAFETY: every method forwards to `System` with the caller's arguments;
// the counters are const-initialised thread-local `Cell`s without `Drop`,
// so reading them neither allocates nor recurses.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.with(Cell::get) {
            COUNT.with(|c| c.set(c.get() + 1));
        }
        // SAFETY: the caller's contract for `alloc` is forwarded unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract for `dealloc` is forwarded unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ARMED.with(Cell::get) {
            COUNT.with(|c| c.set(c.get() + 1));
        }
        // SAFETY: the caller's contract for `realloc` is forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// The allocations this thread makes while `f` runs, when [`CountingAlloc`]
/// is the global allocator (0 otherwise).
pub fn count(f: impl FnOnce()) -> u64 {
    COUNT.with(|c| c.set(0));
    ARMED.with(|a| a.set(true));
    f();
    ARMED.with(|a| a.set(false));
    COUNT.with(Cell::get)
}

/// pyffi's own unit tests count with it too (the handler's zero-allocation
/// test).
#[cfg(test)]
#[global_allocator]
static COUNTING: CountingAlloc = CountingAlloc;
