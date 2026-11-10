// SPDX-License-Identifier: MIT OR Apache-2.0

//! Loom models of the split-phase borrow window (ADR-0047 models 8 and 10)
//! and of the Async per-row snapshot (issue #85).
//!
//! The mode itself is not built under loom (its pool and kernels are too
//! large to model), so the models run the real live-flag functions of
//! [`super::live`] and the real gang (submit, completion and wait) over
//! two shadow rows, one per half, held in loom cells that report any
//! access not ordered by those protocols.

use core::num::NonZeroUsize;

use loom::cell::UnsafeCell;

use super::live::{claim, is_live, release};
use crate::sys::sync::{Arc, AtomicBool, Ordering, thread};
use crate::sys::{Faults, Gang, GangOptions, Job};

/// One shadow row per half.
struct Rows {
    cells: [UnsafeCell<u32>; 2],
    live: [AtomicBool; 2],
}

// SAFETY: the cells are accessed only under the borrow-window protocol the
// models check: a reader holds the half's live flag, a worker writes a half
// only after an admission saw the flag clear. Loom reports any access that
// the protocol does not order.
#[expect(
    unsafe_code,
    reason = "audited module: loom model of the borrow window"
)]
unsafe impl Sync for Rows {}

fn options() -> GangOptions {
    GangOptions {
        workers: NonZeroUsize::MIN,
        spin: 0,
        pin: None,
        mock_futex: false,
        faults: Faults::default(),
    }
}

fn read(rows: &Rows, half: usize) -> u32 {
    // SAFETY: the caller holds `half`'s live flag (a token), so no worker
    // writes the cell; loom checks this.
    #[expect(
        unsafe_code,
        reason = "audited module: loom model of the borrow window"
    )]
    let value = rows.cells[half].with(|p| unsafe { *p });
    value
}

/// Publishes a job that writes `value` into `half`'s row, after an
/// admission that saw the half's flag clear; waits for it.
fn send_and_wait(gang: &mut Gang, rows: &Arc<Rows>, half: usize, value: u32) {
    assert!(
        !is_live(&rows.live[half]),
        "admission needs the token ended"
    );
    let worker_rows = Arc::clone(rows);
    let f = move |_w: usize| {
        // SAFETY: the admission above saw the half's token ended (Acquire)
        // and no other job writes the half; loom checks the ordering.
        #[expect(
            unsafe_code,
            reason = "audited module: loom model of the borrow window"
        )]
        worker_rows.cells[half].with_mut(|p| unsafe { *p = value });
    };
    let job = Job::new(&f, None);
    // SAFETY: `job` and `f` live on this stack until `wait` returns below.
    #[expect(
        unsafe_code,
        reason = "audited module: loom model of the borrow window"
    )]
    unsafe {
        gang.submit(&job).unwrap();
    }
    assert!(!gang.wait().unwrap());
}

/// Model 8: a reader on half A's token while a worker writes half B; the
/// next token on B sees the worker's write.
#[test]
fn test_live_model_reader_on_a_while_worker_writes_b() {
    loom::model(|| {
        let rows = Arc::new(Rows {
            cells: [UnsafeCell::new(1), UnsafeCell::new(2)],
            live: [AtomicBool::new(false), AtomicBool::new(false)],
        });
        assert!(claim(&rows.live[0]), "token on A");
        let reader_rows = Arc::clone(&rows);
        let reader = thread::spawn(move || {
            let seen = read(&reader_rows, 0);
            release(&reader_rows.live[0]);
            seen
        });
        let mut gang = Gang::new(&options()).unwrap();
        send_and_wait(&mut gang, &rows, 1, 20);
        assert!(claim(&rows.live[1]), "token on B after recv");
        assert_eq!(read(&rows, 1), 20, "the completed write is visible");
        release(&rows.live[1]);
        assert_eq!(reader.join().unwrap(), 1);
        drop(gang);
    });
}

/// Model 10: a token on A is read and dropped on one thread; another
/// thread claims a new token on A (as `idle_token` does), sends it (its end
/// is the release before publication) and a worker writes A.
#[test]
fn test_live_model_token_dropped_on_one_thread_then_sent_on_another() {
    loom::model(|| {
        let rows = Arc::new(Rows {
            cells: [UnsafeCell::new(1), UnsafeCell::new(2)],
            live: [AtomicBool::new(false), AtomicBool::new(false)],
        });
        assert!(claim(&rows.live[0]), "token on A");
        let reader_rows = Arc::clone(&rows);
        let reader = thread::spawn(move || {
            let seen = read(&reader_rows, 0);
            release(&reader_rows.live[0]);
            seen
        });
        // Wait (Relaxed, so it orders nothing) until the reader's token has
        // ended, then claim A as `idle_token` does: the claim's Acquire alone
        // must order the reader's read before the worker's write.
        while rows.live[0].load(Ordering::Relaxed) {
            thread::yield_now();
        }
        assert!(claim(&rows.live[0]), "idle_token on A");
        // `send` consumes the new token: it ends before the job is published.
        release(&rows.live[0]);
        let mut gang = Gang::new(&options()).unwrap();
        send_and_wait(&mut gang, &rows, 0, 10);
        assert!(claim(&rows.live[0]), "token on A after recv");
        assert_eq!(read(&rows, 0), 10);
        assert_eq!(reader.join().unwrap(), 1);
        drop(gang);
    });
}

/// Issue #85: the caller snapshots an idle row (never admitted, so its
/// in-flight flag is clear) while a published job is still writing another
/// row, before the caller waits for it; after the wait (the `recv`), the
/// written row is read too and shows the worker's write.
#[test]
fn test_live_model_idle_row_snapshot_while_worker_writes_another() {
    loom::model(|| {
        let rows = Arc::new(Rows {
            cells: [UnsafeCell::new(1), UnsafeCell::new(2)],
            live: [AtomicBool::new(false), AtomicBool::new(false)],
        });
        let worker_rows = Arc::clone(&rows);
        let f = move |_w: usize| {
            // SAFETY: row 1 is admitted to this job alone; the caller reads
            // it only after `wait`. Loom checks the ordering.
            #[expect(
                unsafe_code,
                reason = "audited module: loom model of the borrow window"
            )]
            worker_rows.cells[1].with_mut(|p| unsafe { *p = 20 });
        };
        let job = Job::new(&f, None);
        let mut gang = Gang::new(&options()).unwrap();
        // SAFETY: `job` and `f` live on this stack until `wait` returns below.
        #[expect(
            unsafe_code,
            reason = "audited module: loom model of the borrow window"
        )]
        unsafe {
            gang.submit(&job).unwrap();
        }
        // The snapshot of idle row 0, concurrent with the worker on row 1.
        assert_eq!(read(&rows, 0), 1);
        assert!(!gang.wait().unwrap());
        assert_eq!(read(&rows, 1), 20, "the received row shows the write");
        drop(gang);
    });
}
