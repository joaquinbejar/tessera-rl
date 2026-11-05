// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the id-buffer pool. Under `cfg(loom)` every `model` explores
//! the interleavings; natively and under Miri they run once.

#[cfg(not(loom))]
use super::BufferError;
use super::IdBufferPool;
use crate::sys::sync::{Arc, AtomicBool, Ordering, thread};

fn model(f: impl Fn() + Sync + Send + 'static) {
    #[cfg(loom)]
    loom::model(f);
    #[cfg(not(loom))]
    f();
}

#[cfg(not(loom))]
#[test]
fn test_id_buffers_count_is_batches_plus_one_full() {
    assert_eq!(IdBufferPool::new(8, 3).unwrap().buffers(), 3);
    assert_eq!(IdBufferPool::new(2, 1).unwrap().buffers(), 3);
    assert_eq!(IdBufferPool::new(4, 4).unwrap().buffers(), 2);
}

#[cfg(not(loom))]
#[test]
fn test_id_buffers_bad_sizes_are_refused() {
    assert_eq!(
        IdBufferPool::new(4, 0).unwrap_err(),
        BufferError::Size {
            n: 4,
            batch_size: 0
        }
    );
    assert_eq!(
        IdBufferPool::new(2, 3).unwrap_err(),
        BufferError::Size {
            n: 2,
            batch_size: 3
        }
    );
}

#[cfg(not(loom))]
#[test]
fn test_id_buffers_batch_buffers_run_out_and_come_back() {
    let pool = IdBufferPool::new(4, 2).unwrap();
    let a = pool.acquire(false).unwrap();
    let b = pool.acquire(false).unwrap();
    assert_eq!(pool.acquire(false), None, "only floor(4 / 2) batch buffers");
    let full = pool.acquire(true).unwrap();
    assert_eq!(pool.acquire(true), None, "one full buffer");
    assert_ne!(a.index(), b.index());
    let index = a.index();
    let _ = pool.release(a);
    assert_eq!(pool.acquire(false).map(|h| h.index()), Some(index));
    let _ = pool.release(b);
    let _ = pool.release(full);
}

#[cfg(not(loom))]
#[test]
fn test_id_buffers_entries_stay_with_their_handle_across_another_acquire() {
    let pool = IdBufferPool::new(4, 2).unwrap();
    let a = pool.acquire(false).unwrap();
    pool.fill(&a, &[3, 1], &[7, 9]).unwrap();
    let b = pool.acquire(false).unwrap();
    pool.fill(&b, &[0, 2], &[5, 5]).unwrap();
    assert_eq!(pool.entries(&a).collect::<Vec<_>>(), [(3, 7), (1, 9)]);
    // Releasing in either order leaves the other buffer intact.
    let _ = pool.release(b);
    assert_eq!(pool.entries(&a).collect::<Vec<_>>(), [(3, 7), (1, 9)]);
    let _ = pool.release(a);
}

#[cfg(not(loom))]
#[test]
fn test_id_buffers_fill_that_does_not_fit_is_refused_and_writes_nothing() {
    let pool = IdBufferPool::new(4, 2).unwrap();
    let a = pool.acquire(false).unwrap();
    pool.fill(&a, &[1, 2], &[0, 0]).unwrap();
    assert_eq!(
        pool.fill(&a, &[1, 2, 3], &[0, 0, 0]),
        Err(BufferError::Length {
            len: 3,
            capacity: 2
        })
    );
    assert!(pool.fill(&a, &[4], &[0, 0]).is_err());
    assert_eq!(pool.entries(&a).collect::<Vec<_>>(), [(1, 0), (2, 0)]);
    let _ = pool.release(a);
}

#[cfg(not(loom))]
#[test]
fn test_id_buffers_release_reports_a_registered_waiter() {
    let pool = IdBufferPool::new(2, 1).unwrap();
    let a = pool.acquire(false).unwrap();
    assert!(!pool.release(a), "no waiter, no wake");
    let a = pool.acquire(false).unwrap();
    let _value = pool.signal().register();
    assert!(pool.release(a), "a registered waiter must be woken");
    pool.signal().unregister();
}

/// ADR-0047 loom model 12 at the primitive level: n = 2, batch size 1,
/// tokens A and B hold both batch buffers. A's drop clears its id flag,
/// then (maybe after a pause) frees its buffer. Meanwhile the caller sees
/// the id free, resends and completes it, and needs a batch buffer: it must
/// wait on the signal and then succeed, never fail or hang.
#[test]
fn test_id_buffers_drop_between_flag_clear_and_free_makes_acquire_wait() {
    model(|| {
        let pool = Arc::new(IdBufferPool::new(2, 1).unwrap());
        let id_live = Arc::new(AtomicBool::new(true));
        let a = pool.acquire(false).unwrap();
        let b = pool.acquire(false).unwrap();
        let dropper = {
            let pool = Arc::clone(&pool);
            let id_live = Arc::clone(&id_live);
            thread::spawn(move || {
                // The token's drop: clear its id flag, then free its buffer.
                id_live.store(false, Ordering::Release);
                let _wake = pool.release(a);
            })
        };
        // The caller: wait until id 0 is free to resend (its completion is
        // not modelled), then acquire a buffer for the new batch.
        while id_live.load(Ordering::Acquire) {
            thread::yield_now();
        }
        let handle = loop {
            if let Some(h) = pool.acquire(false) {
                break h;
            }
            let value = pool.signal().register();
            if let Some(h) = pool.acquire(false) {
                pool.signal().unregister();
                break h;
            }
            while pool.signal().word().load(Ordering::SeqCst) == value {
                thread::yield_now();
            }
            pool.signal().unregister();
        };
        assert_ne!(handle.index(), b.index(), "B's buffer stays held");
        dropper.join().unwrap();
        let _ = pool.release(handle);
        let _ = pool.release(b);
    });
}
