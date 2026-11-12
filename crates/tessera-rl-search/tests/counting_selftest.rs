// SPDX-License-Identifier: MIT OR Apache-2.0

//! A self-test of the counting allocator in `support/counting.rs`: the
//! measuring thread's own bytes, frees included, and the thread rule
//! (a named foreign thread fails the window, `main` is excluded). This
//! binary holds a single test, so no other test's thread runs inside a
//! window.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers: a failure is the assertion"
)]

use std::hint::black_box;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Barrier};
use std::thread::{Builder, JoinHandle};

#[path = "support/counting.rs"]
mod counting;
use counting::{measured, record, window};

const N: usize = 4_096;

/// A thread named `name`, spawned before the window, that allocates `N`
/// bytes between the two barrier waits the window makes and returns them,
/// so its allocation lands inside the window and is freed outside. One
/// first wait happens before the window: where std's `Mutex` and
/// `Condvar` box their pthread objects lazily (macOS: 112 bytes), that
/// one-time allocation must not land in the window.
fn parked(name: &str) -> (Arc<Barrier>, JoinHandle<Vec<u8>>) {
    let b = Arc::new(Barrier::new(2));
    let t = Arc::clone(&b);
    let h = Builder::new()
        .name(name.into())
        .spawn(move || {
            t.wait();
            t.wait();
            let v = black_box(vec![1_u8; N]);
            t.wait();
            v
        })
        .unwrap();
    b.wait();
    (b, h)
}

#[test]
fn test_counting_window_counts_the_measuring_thread_and_fails_on_named_foreign_threads() {
    // (1) Allocated and freed on the measuring thread: 0.
    let ((), live) = window(|| drop(black_box(vec![1_u8; N])));
    assert_eq!(live, 0, "allocate and free");

    // (2) Allocated and kept: exactly N, freed outside the window.
    let (kept, live) = window(|| black_box(Vec::<u8>::with_capacity(N)));
    assert_eq!(live, i64::try_from(N).unwrap(), "allocate and keep");
    drop(kept);

    // (3) A named foreign thread allocating in the window fails it.
    let (b, h) = parked("intruder");
    let err = catch_unwind(AssertUnwindSafe(|| {
        window(|| {
            b.wait();
            b.wait();
        })
    }))
    .unwrap_err();
    drop(h.join().unwrap());
    let msg = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| err.downcast_ref::<&str>().copied())
        .unwrap();
    assert!(
        msg.contains("measurement disturbed by thread \"intruder\""),
        "unexpected panic: {msg}"
    );

    // (4) The harness thread (`main`) is excluded: the window passes and
    // the measuring thread's count is unaffected.
    let (b, h) = parked("main");
    let ((), live) = window(|| {
        b.wait();
        b.wait();
    });
    assert_eq!(live, 0, "main excluded");
    assert_eq!(h.join().unwrap().len(), N);

    // The shared measured build runs under the same rule.
    let (live, bound) = measured(&record(), 1, 8, 8, false);
    assert!(i64::try_from(bound).unwrap() >= live);
}
