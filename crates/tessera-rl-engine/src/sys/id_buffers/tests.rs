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
    assert!(pool.release(a).is_ok());
    assert_eq!(pool.acquire(false).map(|h| h.index()), Some(index));
    assert!(pool.release(b).is_ok());
    assert!(pool.release(full).is_ok());
}

#[cfg(not(loom))]
#[test]
fn test_id_buffers_entries_stay_with_their_handle_across_another_acquire() {
    let pool = IdBufferPool::new(4, 2).unwrap();
    let a = pool.acquire(false).unwrap();
    pool.fill(&a, &[3, 1], &[7, 9]).unwrap();
    let b = pool.acquire(false).unwrap();
    pool.fill(&b, &[0, 2], &[5, 5]).unwrap();
    assert_eq!(
        pool.entries(&a).unwrap().collect::<Vec<_>>(),
        [(3, 7), (1, 9)]
    );
    // Releasing in either order leaves the other buffer intact.
    assert!(pool.release(b).is_ok());
    assert_eq!(
        pool.entries(&a).unwrap().collect::<Vec<_>>(),
        [(3, 7), (1, 9)]
    );
    assert!(pool.release(a).is_ok());
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
    assert_eq!(
        pool.entries(&a).unwrap().collect::<Vec<_>>(),
        [(1, 0), (2, 0)]
    );
    assert!(pool.release(a).is_ok());
}

#[cfg(not(loom))]
#[test]
fn test_id_buffers_release_reports_a_registered_waiter() {
    let pool = IdBufferPool::new(2, 1).unwrap();
    let a = pool.acquire(false).unwrap();
    assert!(!pool.release(a).unwrap(), "no waiter, no wake");
    let a = pool.acquire(false).unwrap();
    let _value = pool.signal().register();
    assert!(
        pool.release(a).unwrap(),
        "a registered waiter must be woken"
    );
    pool.signal().unregister();
}

/// ADR-0047 loom model 12 at the primitive level: n = 2, batch size 1,
/// tokens A and B hold both batch buffers. A's drop clears its id flag,
/// then (maybe after a pause) frees its buffer. Meanwhile the caller sees
/// the id free, resends and completes it, and needs a batch buffer: it must
/// wait on the signal and then succeed, never fail or hang. The wait is
/// counted: the caller blocks, and only a release that reports a
/// registered waiter wakes it.
#[test]
fn test_id_buffers_drop_between_flag_clear_and_free_makes_acquire_wait() {
    model(|| {
        let pool = Arc::new(IdBufferPool::new(2, 1).unwrap());
        let id_live = Arc::new(AtomicBool::new(true));
        let woken = Arc::new(AtomicBool::new(false));
        let a = pool.acquire(false).unwrap();
        let b = pool.acquire(false).unwrap();
        let dropper = {
            let pool = Arc::clone(&pool);
            let id_live = Arc::clone(&id_live);
            let woken = Arc::clone(&woken);
            thread::spawn(move || {
                // The token's drop: clear its id flag, then free its buffer,
                // waking the caller through the backend only if needed.
                id_live.store(false, Ordering::Release);
                if pool.release(a).unwrap() {
                    woken.store(true, Ordering::SeqCst);
                }
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
            let _value = pool.signal().register();
            if let Some(h) = pool.acquire(false) {
                pool.signal().unregister();
                break h;
            }
            // Blocked until the backend wakes this waiter.
            while !woken.load(Ordering::SeqCst) {
                thread::yield_now();
            }
            pool.signal().unregister();
        };
        assert_ne!(handle.index(), b.index(), "B's buffer stays held");
        dropper.join().unwrap();
        assert!(pool.release(handle).is_ok());
        assert!(pool.release(b).is_ok());
    });
}

#[cfg(not(loom))]
#[test]
fn test_id_buffers_foreign_handle_is_refused_and_changes_neither_pool() {
    let a_pool = IdBufferPool::new(1, 1).unwrap();
    let b_pool = IdBufferPool::new(1, 1).unwrap();
    let ha = a_pool.acquire(false).unwrap();
    let hb = b_pool.acquire(false).unwrap();
    assert_eq!(ha.index(), hb.index(), "same index, different pools");
    a_pool.fill(&ha, &[0], &[4]).unwrap();
    // B's handle cannot touch A: fill, entries and release are refused.
    assert_eq!(
        a_pool.fill(&hb, &[0], &[9]),
        Err(BufferError::ForeignHandle)
    );
    assert_eq!(a_pool.entries(&hb).err(), Some(BufferError::ForeignHandle));
    let hb = a_pool.release(hb).unwrap_err();
    // A's buffer is still held, with its contents.
    assert_eq!(a_pool.acquire(false), None, "A's live buffer was not freed");
    assert_eq!(a_pool.entries(&ha).unwrap().collect::<Vec<_>>(), [(0, 4)]);
    // B's handle still releases to B.
    assert!(b_pool.release(hb).is_ok());
    assert!(b_pool.acquire(false).is_some());
    assert!(a_pool.release(ha).is_ok());
}

/// A buffer filled on one thread and released is read back by the thread
/// that reacquires it: the release (Release) and the acquiring CAS
/// (Acquire) order the fill before the read.
#[test]
fn test_id_buffers_fill_release_reacquire_read_crosses_threads() {
    model(|| {
        let pool = Arc::new(IdBufferPool::new(1, 1).unwrap());
        let handle = pool.acquire(false).unwrap();
        let holder = {
            let pool = Arc::clone(&pool);
            thread::spawn(move || {
                pool.fill(&handle, &[9], &[3]).unwrap();
                let _ = pool.release(handle).unwrap();
            })
        };
        let again = loop {
            if let Some(h) = pool.acquire(false) {
                break h;
            }
            thread::yield_now();
        };
        assert_eq!(pool.entries(&again).unwrap().collect::<Vec<_>>(), [(9, 3)]);
        holder.join().unwrap();
        let _ = pool.release(again).unwrap();
    });
}

#[cfg(not(loom))]
#[test]
fn test_id_buffers_full_handle_is_the_last_buffer_and_holds_n_ids() {
    let pool = IdBufferPool::new(5, 2).unwrap();
    let full = pool.acquire(true).unwrap();
    assert_eq!(full.index(), pool.buffers() - 1, "the full buffer is last");
    let ids: Vec<u32> = (0..5).collect();
    pool.fill(&full, &ids, &[0; 5]).unwrap();
    assert_eq!(pool.entries(&full).unwrap().count(), 5);
    // Every batch buffer is still free, and none holds N ids.
    let batch = pool.acquire(false).unwrap();
    assert!(pool.fill(&batch, &ids, &[0; 5]).is_err());
    assert!(pool.release(batch).is_ok());
    assert!(pool.release(full).is_ok());
}

#[cfg(not(loom))]
#[test]
fn test_id_buffers_lengths_and_size_bound_give_typed_errors() {
    let pool = IdBufferPool::new(4, 2).unwrap();
    let a = pool.acquire(false).unwrap();
    assert_eq!(
        pool.fill(&a, &[1, 2], &[0]),
        Err(BufferError::LengthMismatch {
            ids: 2,
            generations: 1
        })
    );
    assert!(pool.release(a).is_ok());
    let over = crate::sys::ring::MAX_IDS + 1;
    assert_eq!(
        IdBufferPool::new(over, over).unwrap_err(),
        BufferError::Size {
            n: over,
            batch_size: over
        }
    );
}
