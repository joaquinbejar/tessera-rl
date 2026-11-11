// SPDX-License-Identifier: MIT OR Apache-2.0

//! The registered waiter of the park backend (ADR-0056 decision 3). Audited
//! `unsafe` module (part of `engine::sys`).
//!
//! Park wakes threads, not addresses, so a notifier needs the waiting
//! thread's handle. [`WaiterCell`] holds it for one wait only: the waiting
//! thread writes its own handle before it registers on the [`Signal`] and
//! clears it before the wait returns. This is not the stored waker that
//! ADR-0047 rejected: that one was written at submit time by the sending
//! thread.
//!
//! # Protocol
//!
//! `state` is `EMPTY`, `WRITING` or `READY`; `readers` counts notifier
//! leases. Every operation below is `SeqCst`.
//!
//! - **Writer** (one at a time; the receiver under `&mut self`). It writes
//!   the cell only while the signal has no waiter, after a compare-and-swap
//!   of `state` from `EMPTY` to `WRITING`; it stores its handle, then
//!   `state = READY`, then registers and rechecks. On leaving: it
//!   deregisters, takes the cell back with a compare-and-swap from `READY`
//!   to `WRITING`, waits until `readers == 0`, clears the cell, then stores
//!   `EMPTY`.
//! - **Notifier** (any worker, after `Signal::notify` returned true). It
//!   takes a lease (`readers += 1`), reads `state` with an idempotent RMW
//!   (`fetch_add(0)`), clones the handle only if `READY`, releases the
//!   lease and unparks its clone.
//!
//! Each side stores one variable and then reads the other (the notifier
//! `readers` then `state`, the writer `state` then `readers`). The
//! notifier's read of `state` and the writer's take-back are both RMWs on
//! `state`, so one comes first in its modification order. If the take-back
//! comes first, the notifier sees `WRITING` and does not read the cell.
//! Otherwise the take-back reads from the notifier's RMW (a release
//! sequence), so the lease happens before the writer's load of `readers`,
//! and the writer waits before it clears. ADR-0056 states the read as a
//! `SeqCst` load, which is also correct by the single total order; the RMW
//! makes the argument one of happens-before, which loom can check (loom
//! models `SeqCst` accesses as `AcqRel`). No read of the cell overlaps a clear or a
//! rewrite. A wake the notifier skips (it saw `WRITING`) belongs to a
//! receiver that already left its wait; the receiver's `park_timeout`
//! covers any other loss, so a lost unpark costs latency, never liveness.
//!
//! A notifier's unpark that lands after the receiver left its wait leaves
//! the park token set on the receiver's thread, so that thread's next
//! unrelated park returns early once. That is benign: every caller of
//! `park` rechecks its condition.
//!
//! The cell is preallocated, and cloning a `Thread` is a reference-count
//! increment, not an allocation.

use super::ring::Signal;
use super::sync::{AtomicU32, Ordering, UnsafeCell, spin_loop, thread};

const EMPTY: u32 = 0;
const WRITING: u32 = 1;
const READY: u32 = 2;

/// One waiting thread's handle, published for one wait (module docs).
#[derive(Debug)]
pub(crate) struct WaiterCell {
    thread: UnsafeCell<Option<thread::Thread>>,
    state: AtomicU32,
    readers: AtomicU32,
}

// SAFETY: the cell is written only by the one writer that holds `WRITING`
// with no lease outstanding, and read only under a lease that saw `READY`
// (module protocol); `Option<Thread>` is `Send + Sync`.
#[expect(unsafe_code, reason = "audited module: waiter cell")]
unsafe impl Sync for WaiterCell {}

impl Default for WaiterCell {
    fn default() -> Self {
        Self::new()
    }
}

impl WaiterCell {
    /// An empty cell.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            thread: UnsafeCell::new(None),
            state: AtomicU32::new(EMPTY),
            readers: AtomicU32::new(0),
        }
    }

    /// Writer: publishes the current thread's handle. Returns false, having
    /// written nothing, if the signal already has a waiter or the cell is
    /// not empty; the caller then sleeps on its timeout alone.
    fn publish(&self, signal: &Signal) -> bool {
        if signal.waiters() != 0
            || self
                .state
                .compare_exchange(EMPTY, WRITING, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
        {
            return false;
        }
        let me = thread::current();
        // SAFETY: this writer holds `WRITING`. A notifier reads the cell only
        // under a lease that saw `READY`, and the previous writer waited for
        // every lease to end before it stored `EMPTY`, which the
        // compare-and-swap above read; so nobody else accesses the cell.
        #[expect(unsafe_code, reason = "audited module: waiter cell write")]
        self.thread.with_mut(|cell| unsafe { *cell = Some(me) });
        self.state.store(READY, Ordering::SeqCst);
        true
    }

    /// Writer: takes the published handle back and clears it. Called after
    /// the writer deregistered from the signal.
    fn retract(&self) {
        // From this compare-and-swap on, no new notifier reads the cell.
        if self
            .state
            .compare_exchange(READY, WRITING, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }
        // A notifier that took its lease before the swap may still read.
        while self.readers.load(Ordering::SeqCst) != 0 {
            spin_loop();
        }
        // SAFETY: this writer holds `WRITING` and saw `readers == 0` after
        // its swap: every lease that saw `READY` has ended, and every later
        // lease sees `WRITING` and does not read (module protocol).
        #[expect(unsafe_code, reason = "audited module: waiter cell clear")]
        let old = self.thread.with_mut(|cell| unsafe { (*cell).take() });
        self.state.store(EMPTY, Ordering::SeqCst);
        // Released outside the protocol: dropping a handle is a
        // reference-count decrement.
        drop(old);
    }

    /// Notifier: unparks the published thread, if one is published. Call
    /// only after `Signal::notify` returned true.
    pub(crate) fn unpark(&self) {
        self.readers.fetch_add(1, Ordering::SeqCst);
        // An idempotent RMW, not a load: it and the writer's take-back
        // compare-and-swap are both RMWs on `state`, so one comes first in
        // its modification order. After the swap, this reads `WRITING` (or
        // a later writer's state). Before it, the swap reads from this RMW's
        // release sequence, so this lease happens before the writer's load
        // of `readers`. That holds by happens-before alone, which loom
        // checks. A plain `SeqCst` load is also correct, by the single total
        // order, but loom models `SeqCst` accesses as `AcqRel` and cannot
        // check that argument.
        let state = self.state.fetch_add(0, Ordering::SeqCst);
        let waiter = if state == READY {
            // SAFETY: this lease saw `READY`, so the writer has finished
            // writing and will not clear or rewrite the cell before the
            // lease ends (module protocol); this is a shared read.
            #[expect(unsafe_code, reason = "audited module: waiter cell read")]
            self.thread.with(|cell| unsafe { (*cell).clone() })
        } else {
            #[cfg(all(test, loom))]
            if state == WRITING {
                tests::LEASES_DURING_TAKE_BACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            None
        };
        self.readers.fetch_sub(1, Ordering::SeqCst);
        if let Some(waiter) = waiter {
            waiter.unpark();
        }
    }
}

/// One park-backend wait on `signal` (ADR-0056 decisions 2 and 3): publish
/// the handle, register, recheck `ready`, `sleep` if it does not hold, then
/// deregister and take the handle back. `sleep` is `park_timeout`: an
/// unpark ends it early, the timeout is the liveness fallback. Returns
/// whether `ready` held at the recheck.
pub(crate) fn park_wait(
    signal: &Signal,
    cell: &WaiterCell,
    ready: impl FnOnce() -> bool,
    sleep: impl FnOnce(),
) -> bool {
    let published = cell.publish(signal);
    let _value = signal.register();
    let done = ready();
    if !done {
        sleep();
    }
    signal.unregister();
    if published {
        cell.retract();
    }
    done
}

/// The completion wake of the Async mode (ADR-0056 decision 2): a counted
/// signal, the park backend's waiter cell, and the workers' backend. The
/// gang owns it, so a worker that dies outside its job (the backstop that
/// sets `dead`) notifies it too; the Async mode's workers notify it once
/// per pushed claim.
#[cfg(not(loom))]
#[derive(Debug, Default)]
pub(crate) struct Completion {
    signal: Signal,
    cell: WaiterCell,
    native: Option<super::Native>,
}

#[cfg(not(loom))]
impl Completion {
    /// A completion wake for the workers' backend (`None`: park).
    #[must_use]
    pub(crate) fn new(native: Option<super::Native>) -> Self {
        Self {
            signal: Signal::new(),
            cell: WaiterCell::new(),
            native,
        }
    }

    /// The native backend, `None` on park.
    #[must_use]
    pub(crate) fn native(&self) -> Option<super::Native> {
        self.native
    }

    /// The counted signal (for `BatchWaiter`'s own registration).
    #[must_use]
    pub(crate) fn signal(&self) -> &Signal {
        &self.signal
    }

    /// Announces a change made before this call and wakes a registered
    /// waiter. Nobody registered: two atomic RMWs, no syscall, no lock.
    #[inline]
    pub(crate) fn notify(&self) {
        if self.signal.notify() {
            self.wake();
        }
    }

    /// The wake through the backend, only when a waiter is registered. Out
    /// of line, so the notify inlined into a worker's loop stays two RMWs
    /// and a branch.
    #[cold]
    #[inline(never)]
    fn wake(&self) {
        match self.native {
            Some(native) => native.wake_all(self.signal.word()),
            None => self.cell.unpark(),
        }
    }

    /// One wait of the receiver (one thread at a time): registers,
    /// rechecks `ready`, and blocks if it does not hold. Native: on the
    /// signal's address, untimed for `None`, else at most the timeout; there
    /// is no periodic wake, so every exit condition must be notified. Park:
    /// a park of at most `min(timeout, park_fallback)`, the timeout being
    /// the liveness fallback of decision 3. May return early; the caller
    /// rechecks.
    pub(crate) fn wait(
        &self,
        ready: impl FnOnce() -> bool,
        timeout: Option<core::time::Duration>,
        park_fallback: core::time::Duration,
    ) {
        if let Some(native) = self.native {
            let value = self.signal.register();
            if !ready() {
                match timeout {
                    None => native.wait(self.signal.word(), value),
                    Some(t) => native.wait_timeout(self.signal.word(), value, t),
                }
            }
            self.signal.unregister();
        } else {
            let sleep = timeout.map_or(park_fallback, |t| t.min(park_fallback));
            park_wait(&self.signal, &self.cell, ready, || {
                thread::park_timeout(sleep);
            });
        }
    }
}

#[cfg(test)]
mod tests;
