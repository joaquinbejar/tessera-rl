// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native timed waits (ADR-0056 decision 4): an early return, expiry at the
//! deadline and a wake before it. FFI calls, so native only (not Miri).

use core::time::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use super::Native;

/// Runs `f` on its own thread and fails, naming `what`, if it has not
/// finished within 30 s: a wait that ignores its timeout fails the test
/// instead of hanging it.
fn within(what: &str, f: impl FnOnce() + Send + 'static) {
    let (done, finished) = std::sync::mpsc::channel();
    let body = std::thread::spawn(move || {
        f();
        let _ = done.send(());
    });
    match finished.recv_timeout(Duration::from_secs(30)) {
        Ok(()) => body.join().unwrap(),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("{what}: did not finish within 30 s")
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            if let Err(payload) = body.join() {
                std::panic::resume_unwind(payload);
            }
        }
    }
}

fn native() -> Option<Native> {
    let found = Native::detect();
    if found.is_none() {
        tracing::warn!("no native wait primitive here; the timed-wait test is skipped");
    }
    found
}

/// A word that already differs from `expected` returns at once, whatever
/// the timeout.
#[test]
fn test_native_wait_timeout_changed_word_returns_early() {
    let Some(native) = native() else { return };
    within("a changed word returns early", move || {
        let word = AtomicU32::new(1);
        let start = Instant::now();
        native.wait_timeout(&word, 0, Duration::from_secs(60));
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "{:?}",
            start.elapsed()
        );
    });
}

/// An unchanged word with nobody waking it blocks until the deadline, not
/// for ever. The wait may return early (spuriously), so it loops on the
/// deadline as every caller does, and counts how often it returned.
#[test]
fn test_native_wait_timeout_unchanged_word_expires_at_the_deadline() {
    let Some(native) = native() else { return };
    within("an unchanged word expires", move || expires(native));
}

fn expires(native: Native) {
    let word = AtomicU32::new(0);
    let timeout = Duration::from_millis(50);
    // One call: it must not outlive its timeout by much.
    let start = Instant::now();
    native.wait_timeout(&word, 0, timeout);
    let one = start.elapsed();
    assert!(
        one < Duration::from_secs(5),
        "the timed wait ignored its timeout: {one:?}"
    );
    // The deadline loop of the callers ends at the deadline.
    let start = Instant::now();
    let deadline = start + timeout;
    let mut calls = 0u32;
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        native.wait_timeout(&word, 0, left);
        calls += 1;
    }
    let elapsed = start.elapsed();
    assert!(elapsed >= timeout, "{elapsed:?}");
    assert!(elapsed < Duration::from_secs(5), "{elapsed:?}");
    // A wait that returned at once every time would loop thousands of
    // times in 50 ms.
    assert!(calls < 1_000, "the wait did not block: {calls} calls");
    // A zero timeout returns at once.
    let start = Instant::now();
    native.wait_timeout(&word, 0, Duration::ZERO);
    assert!(start.elapsed() < Duration::from_secs(1));
}

/// A wake before the deadline ends the wait long before it.
#[test]
fn test_native_wait_timeout_wake_before_the_deadline_returns() {
    let Some(native) = native() else { return };
    within("a wake before the deadline", move || woken(native));
}

fn woken(native: Native) {
    let word = Arc::new(AtomicU32::new(0));
    let waker = {
        let word = Arc::clone(&word);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            word.store(1, Ordering::SeqCst);
            native.wake_all(&word);
        })
    };
    let start = Instant::now();
    let deadline = start + Duration::from_secs(30);
    while word.load(Ordering::SeqCst) == 0 {
        let Some(left) = deadline.checked_duration_since(Instant::now()) else {
            break;
        };
        native.wait_timeout(&word, 0, left);
    }
    let elapsed = start.elapsed();
    waker.join().unwrap();
    assert_eq!(word.load(Ordering::SeqCst), 1);
    assert!(elapsed < Duration::from_secs(10), "{elapsed:?}");
}
