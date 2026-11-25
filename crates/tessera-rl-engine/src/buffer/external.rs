// SPDX-License-Identifier: MIT OR Apache-2.0

//! Caller-provided output memory (ADR-0084). Audited `unsafe` module (part
//! of `buffer`).
//!
//! An [`OutputAllocator`] hands the engine one range of memory for the
//! whole output storage of a pool, wrapped in an [`ExternalAlloc`] whose
//! `owner` keeps it alive. The engine checks the range (length and
//! alignment), zeroes it, and lays its regions out in it exactly as in its
//! own allocation (`StorageBuilder::finish`).

use core::ptr::NonNull;

/// A source of output memory for a pool (ADR-0084 item 1), set through
/// [`PoolOptions::output_allocator`](crate::PoolOptions::output_allocator).
///
/// The engine calls [`OutputAllocator::allocate`] once per output storage,
/// at construction, with `len` the layout size rounded up to
/// [`OUTPUT_ALIGN`](crate::OUTPUT_ALIGN) and `align = OUTPUT_ALIGN`. It
/// refuses a `None`, a shorter range or a misaligned pointer with
/// [`EngineError::OutputAllocation`](crate::EngineError::OutputAllocation).
pub trait OutputAllocator: Send + Sync + 'static {
    /// `len` bytes, aligned to `align`, or None.
    fn allocate(&self, len: usize, align: usize) -> Option<ExternalAlloc>;
}

/// One range of caller-provided memory and the owner that keeps it alive
/// (ADR-0084 item 1). The engine writes its outputs into `ptr..ptr + len`
/// and drops `owner` exactly once, on the thread that drops the last `Arc`
/// of the output storage (ADR-0084 item 4).
pub struct ExternalAlloc {
    ptr: NonNull<u8>,
    len: usize,
    /// `owner` of ADR-0084 item 1: keeps the memory alive and frees it on
    /// drop; held only for that drop.
    _owner: Box<dyn Send + Sync>,
}

impl core::fmt::Debug for ExternalAlloc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExternalAlloc")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl ExternalAlloc {
    /// Wraps `len` bytes at `ptr`, kept alive by `owner`.
    ///
    /// # Safety
    ///
    /// Until `owner` is dropped (ADR-0084 item 1):
    ///
    /// - `ptr..ptr + len` is valid and writable, and `ptr` is the start the
    ///   engine uses: when the allocator over-allocates and offsets for
    ///   alignment, `owner` keeps the original base pointer and frees that,
    ///   never `ptr`;
    /// - nothing but the engine ever writes to the range;
    /// - others only read it while a borrow window is open (ADR-0003 rule
    ///   2): a device copy reads a window's outputs and holds the window
    ///   open until it completes; no read happens outside a window;
    /// - the allocator's own handle to the memory never reaches user code:
    ///   users see the engine's exports, never the backing allocation.
    ///
    /// Dropping `owner` must be safe on any thread, without locking by the
    /// engine (ADR-0084 item 4).
    #[expect(unsafe_code, reason = "audited module: external output memory")]
    #[must_use]
    pub unsafe fn new(ptr: NonNull<u8>, len: usize, owner: Box<dyn Send + Sync>) -> Self {
        Self {
            ptr,
            len,
            _owner: owner,
        }
    }

    /// The first byte of the range.
    #[must_use]
    pub const fn ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    /// The length of the range, in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the range is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A heap allocator for tests: an [`OutputAllocator`] backed by a `Vec`
/// that over-allocates by two alignments and offsets the start past the
/// first aligned address, so the pointer the engine uses is never the
/// `Vec`'s base (ADR-0084 item 8). Counts allocations and owner drops and
/// records the thread of each drop.
#[cfg(any(test, feature = "test-hooks"))]
#[derive(Debug, Default)]
pub struct VecOutputAllocator {
    stats: std::sync::Arc<VecAllocStats>,
}

/// The counters of a [`VecOutputAllocator`], shared with its owners.
#[cfg(any(test, feature = "test-hooks"))]
#[derive(Debug, Default)]
pub struct VecAllocStats {
    allocations: core::sync::atomic::AtomicUsize,
    drops: core::sync::atomic::AtomicUsize,
    drop_threads: std::sync::Mutex<Vec<std::thread::ThreadId>>,
}

#[cfg(any(test, feature = "test-hooks"))]
impl VecAllocStats {
    /// Allocations handed out.
    #[must_use]
    pub fn allocations(&self) -> usize {
        self.allocations.load(core::sync::atomic::Ordering::SeqCst)
    }

    /// Owners dropped.
    #[must_use]
    pub fn drops(&self) -> usize {
        self.drops.load(core::sync::atomic::Ordering::SeqCst)
    }

    /// The thread of each owner drop, in order.
    #[must_use]
    pub fn drop_threads(&self) -> Vec<std::thread::ThreadId> {
        self.drop_threads
            .lock()
            .map(|t| t.clone())
            .unwrap_or_default()
    }
}

#[cfg(any(test, feature = "test-hooks"))]
impl VecOutputAllocator {
    /// A new allocator with zeroed counters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The counters, shared with every owner this allocator handed out.
    #[must_use]
    pub fn stats(&self) -> std::sync::Arc<VecAllocStats> {
        std::sync::Arc::clone(&self.stats)
    }
}

/// Keeps a [`VecOutputAllocator`]'s `Vec` alive and frees it, from its
/// base, on drop.
#[cfg(any(test, feature = "test-hooks"))]
struct VecOwner {
    _buf: Vec<u8>,
    stats: std::sync::Arc<VecAllocStats>,
}

#[cfg(any(test, feature = "test-hooks"))]
impl Drop for VecOwner {
    fn drop(&mut self) {
        self.stats
            .drops
            .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        if let Ok(mut threads) = self.stats.drop_threads.lock() {
            threads.push(std::thread::current().id());
        }
    }
}

#[cfg(any(test, feature = "test-hooks"))]
impl OutputAllocator for VecOutputAllocator {
    fn allocate(&self, len: usize, align: usize) -> Option<ExternalAlloc> {
        let total = align.checked_mul(2)?.checked_add(len)?;
        let mut buf = vec![0xA5_u8; total];
        let base = buf.as_mut_ptr();
        let mut offset = base.align_offset(align);
        if offset == 0 {
            // Never start at the base: the owner must free the base, not
            // the pointer the engine uses.
            offset = align;
        }
        if offset.checked_add(len)? > total {
            return None;
        }
        let ptr = NonNull::new(base.wrapping_add(offset))?;
        self.stats
            .allocations
            .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        let owner = VecOwner {
            _buf: buf,
            stats: std::sync::Arc::clone(&self.stats),
        };
        // SAFETY: `ptr..ptr + len` lies inside `buf`'s heap block
        // (`offset + len <= total`), which `owner` keeps alive and frees
        // from its own base; moving a `Vec` does not move its block. The
        // `Vec` is never touched again until its drop, so only the engine
        // writes the range and nothing else reads it.
        #[expect(unsafe_code, reason = "audited module: external output memory")]
        Some(unsafe { ExternalAlloc::new(ptr, len, Box::new(owner)) })
    }
}
