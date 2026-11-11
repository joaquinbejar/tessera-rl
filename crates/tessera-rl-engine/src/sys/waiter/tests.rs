// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the park waiter cell (ADR-0056 decision 3). Under `cfg(loom)`
//! the model explores the interleavings; natively and under Miri the
//! threaded test runs real threads with `park_timeout`.

use super::{WaiterCell, park_wait};
use crate::sys::ring::Signal;
use crate::sys::sync::{Arc, AtomicU32, Ordering, thread};

/// Notifier leases that found the cell `WRITING` (loom only): proof that
/// the model reaches a lease inside a writer's take-back.
#[cfg(loom)]
pub(super) static LEASES_DURING_TAKE_BACK: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

struct Shared {
    signal: Signal,
    cell: WaiterCell,
    first: AtomicU32,
    second: AtomicU32,
}

/// A worker's completion: the change, then `notify`, then the wake through
/// the cell only if a waiter is registered.
fn notifier(shared: &Arc<Shared>, flag: fn(&Shared) -> &AtomicU32) -> thread::JoinHandle<()> {
    let shared = Arc::clone(shared);
    thread::spawn(move || {
        flag(&shared).store(1, Ordering::SeqCst);
        if shared.signal.notify() {
            shared.cell.unpark();
        }
    })
}

/// Waits on the park backend until `flag` is set. Under loom the sleep is
/// an untimed `park`, so a lost unpark deadlocks and loom reports it;
/// natively it is the real `park_timeout`.
fn wait_for(shared: &Shared, flag: fn(&Shared) -> &AtomicU32, mut on_registered: impl FnMut()) {
    loop {
        let done = park_wait(
            &shared.signal,
            &shared.cell,
            || {
                on_registered();
                flag(shared).load(Ordering::SeqCst) == 1
            },
            || {
                #[cfg(loom)]
                thread::park();
                #[cfg(not(loom))]
                thread::park_timeout(core::time::Duration::from_micros(100));
            },
        );
        if done || flag(shared).load(Ordering::SeqCst) == 1 {
            return;
        }
    }
}

/// One execution: a first wait on the main thread, with two notifiers
/// started after its registration (so each may pass `notify` while it is
/// registered and take its lease anywhere in its leaving sequence, both
/// leases at once); then, if `second_wait`, a second wait from another
/// thread, which rewrites the cell while those leases may still be live.
fn execution(second_wait: bool) {
    let shared = Arc::new(Shared {
        signal: Signal::new(),
        cell: WaiterCell::new(),
        first: AtomicU32::new(0),
        second: AtomicU32::new(0),
    });
    let mut notifiers = Vec::new();
    let mut started = false;
    wait_for(
        &shared,
        |s| &s.first,
        || {
            if !started {
                started = true;
                notifiers.push(notifier(&shared, |s| &s.first));
                notifiers.push(notifier(&shared, |s| &s.second));
            }
        },
    );
    if second_wait {
        let waiter = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || wait_for(&shared, |s| &s.second, || {}))
        };
        waiter.join().unwrap();
    }
    for n in notifiers {
        n.join().unwrap();
    }
}

/// ADR-0056 decision 3's model. Loom's `UnsafeCell` tracking fails the
/// model if a notifier's read of the cell overlaps the writer's clear or
/// rewrite, and an untimed `park` fails it (deadlock) if a wake is lost.
/// The first model has one wait, so a lease that finds `WRITING` can only
/// be inside its take-back: the counter proves loom reached it.
#[test]
fn test_waiter_cell_rewrite_never_races_a_notifier() {
    #[cfg(loom)]
    {
        LEASES_DURING_TAKE_BACK.store(0, std::sync::atomic::Ordering::Relaxed);
        loom::model(|| execution(false));
        assert!(
            LEASES_DURING_TAKE_BACK.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "the model never reached a lease during the take-back"
        );
        loom::model(|| execution(true));
    }
    #[cfg(not(loom))]
    {
        execution(false);
        execution(true);
    }
}

/// Natively and under Miri: real threads hammer the cell while one writer
/// publishes, waits and clears it round after round, so Miri checks every
/// unsafe read and write of the cell for data races.
#[cfg(not(loom))]
#[test]
fn test_waiter_cell_rounds_with_concurrent_notifiers_complete() {
    let rounds: u32 = if cfg!(miri) { 20 } else { 2_000 };
    let shared = Arc::new(Shared {
        signal: Signal::new(),
        cell: WaiterCell::new(),
        first: AtomicU32::new(0),
        second: AtomicU32::new(0),
    });
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let hammers: Vec<_> = (0..2)
        .map(|_| {
            let shared = Arc::clone(&shared);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    if shared.signal.notify() {
                        shared.cell.unpark();
                    }
                    thread::yield_now();
                }
            })
        })
        .collect();
    for _ in 0..rounds {
        shared.first.store(0, Ordering::SeqCst);
        let setter = notifier(&shared, |s| &s.first);
        wait_for(&shared, |s| &s.first, || {});
        setter.join().unwrap();
    }
    stop.store(true, Ordering::SeqCst);
    for h in hammers {
        h.join().unwrap();
    }
    assert_eq!(shared.signal.waiters(), 0);
}

/// A wait while another waiter is registered does not write the cell (it
/// sleeps on its timeout alone), and leaves it empty for the next writer.
#[cfg(not(loom))]
#[test]
fn test_waiter_cell_publish_with_a_registered_waiter_writes_nothing() {
    let signal = Signal::new();
    let cell = WaiterCell::new();
    let _v = signal.register();
    assert!(!cell.publish(&signal));
    signal.unregister();
    assert!(cell.publish(&signal));
    cell.retract();
    assert!(cell.publish(&signal), "the cell is empty again");
    cell.retract();
}
