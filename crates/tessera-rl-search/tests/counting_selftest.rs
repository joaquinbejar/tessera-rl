// SPDX-License-Identifier: MIT OR Apache-2.0

//! A self-test of the per-thread attribution of `support/attribution.rs`
//! through the counting allocator of `support/counting.rs` (issue #43):
//! the measuring thread's own bytes, frees included, and the thread rule.
//! A gang worker's name is counted, `main` is excluded and printed, and
//! any other thread, named or not, fails the window. The measuring thread
//! is recognized by identity, not name: a window opened on a thread named
//! `main` counts that thread (libtest may run a test on its main thread). This binary holds
//! a single test, so no other test's thread runs inside a window.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers: a failure is the assertion"
)]
#![expect(
    clippy::print_stderr,
    reason = "test diagnostics: the measuring thread's name is reported"
)]

use std::hint::black_box;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Barrier};
use std::thread::{Builder, JoinHandle};

#[path = "support/counting.rs"]
mod counting;
use counting::{measured, panic_message, record, window};

const N: usize = 4_096;

/// A thread, named `name` or unnamed, spawned before the window, that
/// allocates `N` bytes between the two barrier waits the window makes and
/// returns them, so its allocation lands inside the window and is freed
/// outside (a free of the thread's own start-up may land in the window
/// too, so the tests compare allocation calls exactly and bytes by sign).
/// One first wait happens before the window: where std's `Mutex`
/// and `Condvar` box their pthread objects lazily (macOS), that one-time
/// allocation must not land in the window.
fn parked(name: Option<&str>) -> (Arc<Barrier>, JoinHandle<Vec<u8>>) {
    let b = Arc::new(Barrier::new(2));
    let t = Arc::clone(&b);
    let mut builder = Builder::new();
    if let Some(name) = name {
        builder = builder.name(name.into());
    }
    let h = builder
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

/// The window's panic message when a parked thread allocates in it.
fn disturbed(name: Option<&str>) -> String {
    let (b, h) = parked(name);
    let err = catch_unwind(AssertUnwindSafe(|| {
        window(|| {
            b.wait();
            b.wait();
        })
    }))
    .unwrap_err();
    drop(h.join().unwrap());
    panic_message(&*err)
}

#[test]
fn test_counting_window_counts_measuring_and_workers_and_fails_on_foreign_threads() {
    let n = i64::try_from(N).unwrap();
    let me = std::thread::current().name().map(str::to_owned);
    eprintln!("measuring thread: {me:?}");

    // (1) Allocated and freed on the measuring thread: 0 net, one call.
    let ((), t) = window(|| drop(black_box(vec![1_u8; N])));
    assert_eq!((t.bytes, t.calls), (0, 1), "allocate and free on {me:?}");

    // (2) Allocated and kept: exactly N, freed outside the window.
    let (kept, t) = window(|| black_box(Vec::<u8>::with_capacity(N)));
    assert_eq!((t.bytes, t.calls), (n, 1), "allocate and keep on {me:?}");
    drop(kept);

    // (3) A named foreign thread allocating in the window fails it.
    let msg = disturbed(Some("intruder"));
    assert!(
        msg.contains("measurement disturbed by thread \"intruder\""),
        "unexpected panic: {msg}"
    );

    // (4) An unnamed thread fails it too.
    let msg = disturbed(None);
    assert!(
        msg.contains("measurement disturbed by thread \"<unnamed>\""),
        "unexpected panic: {msg}"
    );

    // (5) The harness thread (`main`) is excluded: the window passes, its
    // bytes are reported apart and the counted tally is unaffected.
    let (b, h) = parked(Some("main"));
    let ((), t) = window(|| {
        b.wait();
        b.wait();
    });
    assert_eq!(h.join().unwrap().len(), N);
    assert_eq!((t.bytes, t.calls), (0, 0), "main excluded");
    assert!(t.main_bytes > 0, "main's bytes are reported apart: {t:?}");

    // (6) A thread named like a gang worker is counted.
    let (b, h) = parked(Some("tessera-worker-x"));
    let ((), t) = window(|| {
        b.wait();
        b.wait();
    });
    assert_eq!(h.join().unwrap().len(), N);
    assert_eq!((t.calls, t.worker_calls), (1, 1), "worker counted: {t:?}");
    assert!(t.bytes > 0, "worker's bytes counted: {t:?}");

    // (7) The measuring thread is known by identity, not by name: a window
    // opened on a thread named `main` (as when libtest runs the test on
    // its main thread) counts that thread's allocation instead of
    // excluding it.
    let t = Builder::new()
        .name("main".into())
        .spawn(|| window(|| black_box(Vec::<u8>::with_capacity(N))).1)
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(
        (t.calls, t.bytes, t.main_bytes),
        (1, n, 0),
        "measuring on main"
    );

    // The shared measured build runs under the same rule.
    let (live, bound) = measured(&record(), 1, 8, 8, false);
    assert!(i64::try_from(bound).unwrap() >= live);
}
