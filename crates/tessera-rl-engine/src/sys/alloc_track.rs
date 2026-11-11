// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation tracking of `debug-checks` (ADR-0004, ADR-0055). Audited
//! `unsafe`: the `GlobalAlloc` impl of [`TrackingAllocator`], which only
//! forwards to the system allocator.
//!
//! Counting is per thread: a chunk-step bracket ([`ChunkBracket`]) sets the
//! current thread's flag, and the allocator counts on that thread only
//! while the flag is set. The bracket adds its delta to the tally of the
//! pool whose chunk it ran, so an allocation is never seen by another pool.

use core::cell::Cell;
use core::sync::atomic::{AtomicU64, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

/// The current thread's flag and count. Only `Cell`s of plain integers and
/// no `Drop`, const-initialised: reading it neither allocates nor registers
/// a thread-local destructor, so the allocator may read it.
struct Track {
    active: Cell<bool>,
    count: Cell<u64>,
}

thread_local! {
    static TRACK: Track = const {
        Track {
            active: Cell::new(false),
            count: Cell::new(0),
        }
    };
}

/// Counts one allocation on this thread if a chunk step is running on it.
#[inline]
fn note() {
    // `try_with`: during thread teardown the slot may be gone; then nothing
    // is counted.
    let _ = TRACK.try_with(|t| {
        if t.active.get()
            && let Some(n) = t.count.get().checked_add(1)
        {
            t.count.set(n);
        }
    });
}

/// A counting wrapper of the system allocator for tests (ADR-0055).
///
/// Install it in a test binary with `#[global_allocator]`. `alloc`,
/// `alloc_zeroed` and `realloc` count one each while the current thread
/// runs an engine chunk step; `dealloc` does not count. It never panics.
/// The engine then fails the call that ran the chunk with
/// [`crate::EngineError::AllocationInChunkStep`] and poisons the pool.
#[derive(Debug, Default, Clone, Copy)]
pub struct TrackingAllocator;

// SAFETY: every method forwards to `System` with the same arguments, so the
// `GlobalAlloc` contract is upheld by the system allocator; `note` touches
// only a const-initialised thread-local without a destructor, which
// neither allocates nor panics.
#[expect(unsafe_code, reason = "audited module: counting allocator")]
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        // SAFETY: the caller's contract for `alloc` is passed through.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        // SAFETY: the caller's contract for `alloc_zeroed` is passed through.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract for `dealloc` is passed through.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        // SAFETY: the caller's contract for `realloc` is passed through.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// Allocations counted in the chunk steps of one pool since the last
/// [`AllocTally::take`].
#[derive(Debug, Default)]
pub(crate) struct AllocTally(AtomicU64);

impl AllocTally {
    /// Returns the count and resets it. The caller runs after the chunks'
    /// completion (a release by the worker, an acquire by the caller), so
    /// their additions are visible.
    pub(crate) fn take(&self) -> u64 {
        self.0.swap(0, Ordering::AcqRel)
    }
}

/// Brackets one chunk step on the current thread; on drop (also on
/// unwind) clears the flag and adds the thread's delta to the pool's tally.
pub(crate) struct ChunkBracket<'a> {
    tally: &'a AllocTally,
    before: u64,
}

impl<'a> ChunkBracket<'a> {
    /// Starts counting on this thread.
    pub(crate) fn enter(tally: &'a AllocTally) -> Self {
        let before = TRACK
            .try_with(|t| {
                t.active.set(true);
                t.count.get()
            })
            .unwrap_or(0);
        Self { tally, before }
    }
}

impl Drop for ChunkBracket<'_> {
    fn drop(&mut self) {
        let after = TRACK
            .try_with(|t| {
                t.active.set(false);
                t.count.get()
            })
            .unwrap_or(self.before);
        // The count only grows on this thread, so `after >= before`.
        if let Some(delta) = after.checked_sub(self.before)
            && delta > 0
        {
            self.tally.0.fetch_add(delta, Ordering::Relaxed);
        }
    }
}
