// SPDX-License-Identifier: MIT OR Apache-2.0

//! A counting global allocator for the core allocation tests, with the
//! per-thread attribution of `support/attribution.rs` (issue #43): inside a
//! window the measuring thread and the gang workers are counted, the
//! harness thread `main` is excluded and printed, and any other thread
//! fails the window. It is test code, not one of the audited `unsafe`
//! modules; its `unsafe` only forwards to the system allocator.
//!
//! [`scenarios`] runs several measured scenarios under one test, so that a
//! binary holds a single measuring test.

#![allow(dead_code, reason = "each test binary uses part of it")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[path = "attribution.rs"]
mod attribution;
pub use attribution::window;
use attribution::{delta, note};

struct Counting;

#[expect(unsafe_code, reason = "test-only counting allocator")]
// SAFETY: every method forwards to `System` with the same arguments; the
// attribution only touches atomics and const-initialized thread-locals, and
// reads `thread::current`, which does not allocate on the threads it runs on.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = note(delta(layout.size()), true);
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = note(-delta(layout.size()), false);
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _ = note(delta(layout.size()), true);
        // SAFETY: forwarded unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = note(delta(new_size) - delta(layout.size()), true);
        // SAFETY: forwarded unchanged; the caller upholds `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The message of a caught panic.
#[must_use]
pub fn panic_message(err: &(dyn core::any::Any + Send)) -> String {
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_else(|| "<non-string panic>".to_owned())
}

/// Runs every scenario in order, each in its own windows and with its own
/// pool, even after an earlier one failed; then checks that `expected`
/// scenarios ran and reports every failure together, by scenario name.
///
/// # Panics
///
/// If a scenario failed or the number run differs from `expected`.
pub fn scenarios(expected: usize, list: &[(&str, fn())]) {
    let mut failures = Vec::new();
    let mut ran = 0_usize;
    for &(name, scenario) in list {
        ran += 1;
        if let Err(err) = catch_unwind(AssertUnwindSafe(scenario)) {
            failures.push(format!("{name}: {}", panic_message(&*err)));
        }
    }
    assert_eq!(ran, expected, "scenarios run");
    assert!(
        failures.is_empty(),
        "{} of {ran} scenarios failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
