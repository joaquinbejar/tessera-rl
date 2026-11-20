// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native timed waits (ADR-0056 decision 4): an early return, expiry at the
//! deadline and a wake before it. FFI calls, so native only (not Miri).

use core::time::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use super::Native;

/// How long a test may take before it fails by name instead of hanging.
/// It catches hangs and does not bound speed: a starved process on a
/// loaded CI host can run these tests many times slower (as in #301).
const WATCHDOG: Duration = Duration::from_secs(300);

/// The wait's deadline in [`woken`] and the long timeout in the early
/// return test, under [`WATCHDOG`].
const WAKE_DEADLINE: Duration = Duration::from_secs(240);

/// The bound on a wait that must end long before its deadline (a wake, a
/// changed word) or soon after a 50 ms one. Half of [`WAKE_DEADLINE`]: a
/// lost wake or an ignored early return runs to about the deadline and
/// fails by a wide margin, even if the backend rounds its timeout, while a
/// starved but correct run (as in #301) has minutes to spare. It is not a
/// speed claim. A wait that never returns fails by the watchdog, by name;
/// this bound catches a late return.
const LATE_RETURN: Duration = Duration::from_secs(120);

/// Runs `f` on its own thread and fails, naming `what`, if it has not
/// finished within [`WATCHDOG`]: a wait that ignores its timeout fails the
/// test instead of hanging it.
fn within(what: &str, f: impl FnOnce() + Send + 'static) {
    let (done, finished) = std::sync::mpsc::channel();
    let body = std::thread::spawn(move || {
        f();
        let _ = done.send(());
    });
    match finished.recv_timeout(WATCHDOG) {
        Ok(()) => body.join().unwrap(),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("{what}: did not finish within {WATCHDOG:?}")
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
        native.wait_timeout(&word, 0, WAKE_DEADLINE);
        assert!(start.elapsed() < LATE_RETURN, "{:?}", start.elapsed());
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
        one < LATE_RETURN,
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
    assert!(elapsed < LATE_RETURN, "{elapsed:?}");
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
    let deadline = start + WAKE_DEADLINE;
    while word.load(Ordering::SeqCst) == 0 {
        let Some(left) = deadline.checked_duration_since(Instant::now()) else {
            break;
        };
        native.wait_timeout(&word, 0, left);
    }
    let elapsed = start.elapsed();
    waker.join().unwrap();
    assert_eq!(word.load(Ordering::SeqCst), 1);
    assert!(elapsed < LATE_RETURN, "the wake was lost: {elapsed:?}");
}
