// SPDX-License-Identifier: MIT OR Apache-2.0

//! Wake-based waits of the Async mode (ADR-0056 decisions 2 to 4), on the
//! park backend and on the native one (where this system has it; elsewhere
//! `WaitBackend::Native` falls back to park and the test still runs).
//!
//! Every pool here turns off the receiver's 100 µs liveness fallback, so a
//! wait ends only through a worker's wake: a missed wake hangs, and the
//! [`within`] watchdog fails the test by name instead.

use core::num::NonZeroUsize;
use core::sync::atomic::Ordering;
use core::time::Duration;
use std::time::Instant;

use super::async_jitter_tests::{
    GATES, N, Pool, ReleaseHeld, WalkConfig, all_ids, options, wait_for_gate, within,
};
use crate::buffer::batch::{GATE_AFTER_PUSH, GATE_HOLD_ID, GATE_OPEN, GATE_REACHED, HELD_ID};
use crate::sys::Faults;
use crate::{ActionInput, AsyncOptions, AutoReset, Reseed, WaitBackend};

type AsyncPoolOf = crate::AsyncPool<super::async_jitter_tests::Walker, crate::NoMasks>;

fn gates() -> std::sync::MutexGuard<'static, ()> {
    GATES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A reset pool on `wait` with `workers` workers, batches of `batch`, the
/// test gate `gate` and no liveness fallback.
fn wake_pool(wait: WaitBackend, workers: usize, batch: usize, gate: u32) -> AsyncPoolOf {
    GATE_REACHED.store(false, Ordering::Release);
    GATE_OPEN.store(false, Ordering::Release);
    let mut opts = AsyncOptions::new(
        NonZeroUsize::new(workers).unwrap(),
        NonZeroUsize::new(batch).unwrap(),
    );
    opts.sync.wait = wait;
    let mut pool = Pool::new(WalkConfig {}, options(AutoReset::NextStep))
        .unwrap()
        .async_faults(
            opts,
            Faults {
                jitter_us: gate,
                wake_fallback_off: true,
                ..Faults::default()
            },
        )
        .unwrap();
    drop(pool.reset_all(Reseed::Keep).unwrap());
    pool
}

fn ones(ids: &[u32]) -> Vec<i32> {
    vec![1; ids.len()]
}

fn send(pool: &mut AsyncPoolOf, ids: &[u32]) {
    let actions = ones(ids);
    pool.send(
        ids,
        ActionInput {
            discrete: &actions,
            continuous: &[],
        },
    )
    .unwrap();
}

/// A `recv` blocked while the only id is held before its step (so it is
/// still running and nothing is ready) is woken by the worker's completion
/// notify (decision 2), with no fallback poll.
fn recv_blocked_is_woken(wait: WaitBackend) {
    let _lock = gates();
    let _open = ReleaseHeld;
    within("a blocked recv is woken by the completion", move || {
        let _open = ReleaseHeld;
        HELD_ID.store(1, Ordering::Release);
        let mut pool = wake_pool(wait, 1, 1, GATE_HOLD_ID);
        send(&mut pool, &[1]);
        wait_for_gate();
        let opener = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(20));
            GATE_OPEN.store(true, Ordering::Release);
        });
        let token = pool.recv().unwrap();
        opener.join().unwrap();
        assert_eq!(token.ids().collect::<Vec<_>>(), [1]);
    });
}

#[test]
fn test_async_recv_blocked_park_is_woken_by_the_completion() {
    recv_blocked_is_woken(WaitBackend::Park);
}

#[test]
fn test_async_recv_blocked_native_is_woken_by_the_completion() {
    recv_blocked_is_woken(WaitBackend::Native);
}

/// With a worker held before stepping id 0, a blocking `recv` with batches
/// of one returns every other id: it waits for a ready batch, not for the
/// drain job, which cannot end while the worker is held (decision 2).
fn recv_returns_while_held(wait: WaitBackend) {
    let _lock = gates();
    let _release = ReleaseHeld;
    within("recv returns batches while a worker is held", move || {
        let release = ReleaseHeld;
        HELD_ID.store(0, Ordering::Release);
        let mut pool = wake_pool(wait, 2, 1, GATE_HOLD_ID);
        let (ids, _) = all_ids();
        send(&mut pool, &ids);
        wait_for_gate();
        let mut received = Vec::new();
        let mut failed = None;
        while received.len() + 1 < ids.len() {
            match pool.recv() {
                Ok(token) => received.extend(token.ids()),
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        drop(release);
        assert_eq!(failed, None);
        assert!(!received.contains(&0));
        assert_eq!(received.len(), ids.len() - 1);
        assert_eq!(pool.recv().unwrap().ids().collect::<Vec<_>>(), [0]);
    });
}

#[test]
fn test_async_recv_park_returns_batches_while_a_worker_is_held() {
    recv_returns_while_held(WaitBackend::Park);
}

#[test]
fn test_async_recv_native_returns_batches_while_a_worker_is_held() {
    recv_returns_while_held(WaitBackend::Native);
}

/// `recv_timeout` returns ready batches while a worker is held, and with
/// only the held id left it returns `None` at its deadline, neither early
/// nor much later (decisions 2 and 4).
fn recv_timeout_held(wait: WaitBackend) {
    let _lock = gates();
    let _release = ReleaseHeld;
    within("recv_timeout with a held worker", move || {
        let release = ReleaseHeld;
        HELD_ID.store(0, Ordering::Release);
        let mut pool = wake_pool(wait, 2, 1, GATE_HOLD_ID);
        let (ids, _) = all_ids();
        send(&mut pool, &ids);
        wait_for_gate();
        let mut received = 0;
        while received + 1 < ids.len() {
            let token = pool.recv_timeout(Duration::from_secs(20)).unwrap();
            received += token.map_or(0, |t| t.len());
        }
        let timeout = Duration::from_millis(50);
        let start = Instant::now();
        let none = pool.recv_timeout(timeout).unwrap().is_none();
        let elapsed = start.elapsed();
        drop(release);
        assert!(none, "only the held id was left");
        assert!(
            elapsed >= timeout,
            "returned before its deadline: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(10),
            "ignored its deadline: {elapsed:?}"
        );
        let last = pool.recv_timeout(Duration::from_secs(20)).unwrap().unwrap();
        assert_eq!(last.ids().collect::<Vec<_>>(), [0]);
        assert_eq!(received, usize::try_from(N).unwrap() - 1);
    });
}

#[test]
fn test_async_recv_timeout_park_returns_ready_batches_then_expires() {
    recv_timeout_held(WaitBackend::Park);
}

#[test]
fn test_async_recv_timeout_native_returns_ready_batches_then_expires() {
    recv_timeout_held(WaitBackend::Native);
}

/// A [`crate::BatchWaiter`] expires at its deadline with nothing ready, and
/// a waiter already waiting when the ready count moves returns true. The
/// worker is held after its push and before the count moves, so a notify
/// sent before the count would find the count unchanged (decision 2's
/// order). Native: the waiter blocks on the signal's address until the
/// wake. Park: it keeps its 100 µs sleep (issue #93).
fn waiter_wakes(wait: WaitBackend) {
    let _lock = gates();
    let _open = ReleaseHeld;
    within("a waiting BatchWaiter sees the batch", move || {
        let _open = ReleaseHeld;
        let mut pool = wake_pool(wait, 1, 1, GATE_AFTER_PUSH);
        let waiter = pool.waiter();
        let timeout = Duration::from_millis(50);
        let start = Instant::now();
        assert!(!waiter.wait_ready(timeout));
        let elapsed = start.elapsed();
        assert!(
            elapsed >= timeout,
            "returned before its deadline: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(10),
            "ignored its deadline: {elapsed:?}"
        );
        send(&mut pool, &[1]);
        wait_for_gate();
        let waiting = std::thread::spawn(move || waiter.wait_ready(Duration::from_secs(60)));
        std::thread::sleep(Duration::from_millis(20));
        GATE_OPEN.store(true, Ordering::Release);
        assert!(waiting.join().unwrap());
        assert_eq!(pool.recv().unwrap().len(), 1);
    });
}

#[test]
fn test_async_waiter_park_expires_then_sees_the_batch() {
    waiter_wakes(WaitBackend::Park);
}

#[test]
fn test_async_waiter_native_expires_then_is_woken_by_the_count() {
    waiter_wakes(WaitBackend::Native);
}

/// Opens [`DEATH_GATE`] on every exit path, so a held worker never keeps
/// the pool's `Drop` waiting.
struct OpenDeath;

impl Drop for OpenDeath {
    fn drop(&mut self) {
        DEATH_GATE.store(true, Ordering::Release);
    }
}

/// Holds the dying worker until the receiver is blocked.
static DEATH_GATE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
static DEATH_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The only worker dies outside its job (the gang's backstop) while `recv`
/// is blocked with no periodic wake: the backstop's notify ends the wait and
/// `recv` reports `WorkerDied` instead of hanging (ADR-0056 decision 2).
fn recv_blocked_sees_a_death(wait: WaitBackend) {
    let _serial = DEATH_SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    DEATH_GATE.store(false, Ordering::Release);
    let _open = OpenDeath;
    within("a blocked recv sees a worker's death", move || {
        let _open = OpenDeath;
        let mut opts = AsyncOptions::new(NonZeroUsize::MIN, NonZeroUsize::MIN);
        opts.sync.wait = wait;
        let mut pool = Pool::new(WalkConfig {}, options(AutoReset::NextStep))
            .unwrap()
            .async_faults(
                opts,
                Faults {
                    die_in_epoch: Some(0),
                    death_gate: Some(&DEATH_GATE),
                    wake_fallback_off: true,
                    ..Faults::default()
                },
            )
            .unwrap();
        drop(pool.reset_all(Reseed::Keep).unwrap());
        send(&mut pool, &[1]);
        let opener = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(20));
            DEATH_GATE.store(true, Ordering::Release);
        });
        let result = pool.recv().map(|t| t.len());
        opener.join().unwrap();
        assert_eq!(result, Err(crate::EngineError::WorkerDied));
    });
}

#[test]
fn test_async_recv_blocked_native_reports_a_worker_death() {
    recv_blocked_sees_a_death(WaitBackend::Native);
}

#[test]
fn test_async_recv_blocked_park_reports_a_worker_death() {
    recv_blocked_sees_a_death(WaitBackend::Park);
}

/// Opens the job-level panic gate on every exit path.
struct OpenJobPanic;

impl Drop for OpenJobPanic {
    fn drop(&mut self) {
        crate::buffer::batch::JOB_PANIC_GATE.store(true, Ordering::Release);
    }
}

/// A job-level panic: worker 0 panics in the drain job outside the per-step
/// catch, while worker 1 holds id 1, so the job stays in flight and `pump`
/// cannot reap it. With `before_recv` the panic has happened before `recv`
/// is called; otherwise it happens while `recv` is blocked. Either way
/// `recv` reports `Panicked` instead of blocking for ever with no periodic
/// wake (ADR-0056 decision 2).
/// When the job-level panic happens relative to `recv`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PanicAt {
    /// While `recv` is blocked.
    Blocked,
    /// Before `recv` is called.
    BeforeRecv,
    /// After `poll`'s failure checks and before the wait registers, so
    /// only the wait's predicate can see it (its notify came first).
    BeforeRegistration,
}

fn panic_now() {
    use crate::buffer::batch::{JOB_PANIC_GATE, JOB_PANIC_REACHED};
    JOB_PANIC_GATE.store(true, Ordering::Release);
    while !JOB_PANIC_REACHED.load(Ordering::Acquire) {
        std::thread::yield_now();
    }
    // The slot's flag and the notify follow the marker at once.
    std::thread::sleep(Duration::from_millis(20));
}

fn recv_sees_a_job_panic(wait: WaitBackend, at: PanicAt) {
    use crate::buffer::batch::{JOB_PANIC_GATE, JOB_PANIC_REACHED};
    let _lock = gates();
    JOB_PANIC_GATE.store(false, Ordering::Release);
    JOB_PANIC_REACHED.store(false, Ordering::Release);
    let _open = (ReleaseHeld, OpenJobPanic);
    within("recv sees a job-level panic", move || {
        HELD_ID.store(1, Ordering::Release);
        GATE_REACHED.store(false, Ordering::Release);
        GATE_OPEN.store(false, Ordering::Release);
        let mut opts = AsyncOptions::new(NonZeroUsize::new(2).unwrap(), NonZeroUsize::MIN);
        opts.sync.wait = wait;
        let mut pool = Pool::new(WalkConfig {}, options(AutoReset::NextStep))
            .unwrap()
            .async_faults(
                opts,
                Faults {
                    jitter_us: GATE_HOLD_ID,
                    wake_fallback_off: true,
                    job_panic: Some(0),
                    ..Faults::default()
                },
            )
            .unwrap();
        // Declared after the pool, so they drop first and the pool's
        // `Drop` never waits on a held worker.
        let _open = (ReleaseHeld, OpenJobPanic);
        drop(pool.reset_all(Reseed::Keep).unwrap());
        send(&mut pool, &[1]);
        wait_for_gate();
        let opener = match at {
            PanicAt::BeforeRecv => {
                panic_now();
                None
            }
            PanicAt::BeforeRegistration => {
                super::async_mode::WAIT_HOOK.set(Some(panic_now));
                None
            }
            PanicAt::Blocked => Some(std::thread::spawn(|| {
                std::thread::sleep(Duration::from_millis(20));
                JOB_PANIC_GATE.store(true, Ordering::Release);
            })),
        };
        let result = pool.recv().map(|t| t.len());
        if let Some(opener) = opener {
            opener.join().unwrap();
        }
        assert_eq!(result, Err(crate::EngineError::Panicked));
    });
}

#[test]
fn test_async_recv_blocked_native_reports_a_job_level_panic() {
    recv_sees_a_job_panic(WaitBackend::Native, PanicAt::Blocked);
}

#[test]
fn test_async_recv_blocked_park_reports_a_job_level_panic() {
    recv_sees_a_job_panic(WaitBackend::Park, PanicAt::Blocked);
}

#[test]
fn test_async_recv_after_a_job_level_panic_native_reports_it_at_once() {
    recv_sees_a_job_panic(WaitBackend::Native, PanicAt::BeforeRecv);
}

#[test]
fn test_async_recv_after_a_job_level_panic_park_reports_it_at_once() {
    recv_sees_a_job_panic(WaitBackend::Park, PanicAt::BeforeRecv);
}

#[test]
fn test_async_recv_job_level_panic_before_registration_native_ends_the_wait() {
    recv_sees_a_job_panic(WaitBackend::Native, PanicAt::BeforeRegistration);
}

/// The Park variant does NOT pin the wait predicate's job-panic term: with
/// that term removed it can still pass, because a park token left by an
/// earlier unpark ends one park early and `poll` then reports the panic. In
/// production that is acceptable: a Park wait is bounded by the accepted
/// 100 µs fallback, so the term costs latency at most, never liveness. It is
/// liveness-critical only on native, and the native variant above catches
/// its removal; do not read this test as coverage of that term.
#[test]
fn test_async_recv_job_level_panic_before_registration_park_ends_the_wait() {
    recv_sees_a_job_panic(WaitBackend::Park, PanicAt::BeforeRegistration);
}
