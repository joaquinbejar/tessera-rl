// SPDX-License-Identifier: MIT OR Apache-2.0

//! The preallocated id buffers of Async batch tokens (ADR-0047 section 4).
//!
//! There are `floor(n / batch_size)` buffers of `batch_size` ids and one
//! buffer of `n` ids (for the token of a full reset). Each buffer holds the
//! ids of one token and their generations, written once when the token is
//! made and unchanged while it lives.
//!
//! # Protocol
//!
//! - Only the caller acquires (the mode owner, one thread at a time); it
//!   finds a buffer whose `free` flag is set and clears it with
//!   an Acquire compare-and-swap, which orders the previous holder's reads
//!   before the new writes.
//! - A holder on any thread releases its buffer by setting `free` with
//!   Release, then notifying the pool's [`Signal`]; the release consumes
//!   the [`BufferHandle`].
//! - A token's drop clears its id flags first, then releases its buffer.
//!   Between those two steps the ids may be resent and completed, so a
//!   caller can find no free buffer for a moment. It then waits on the
//!   signal, with registration and a recheck; the dropper frees the
//!   buffer with no blocking work in between, so the wait is short.
//!
//! The capacity argument: the ids of live batch tokens are disjoint, and
//! while the full-reset token lives every id is covered, so no more than
//! `floor(n / batch_size)` batch buffers, or the one full buffer, are held
//! by live tokens at once. A buffer past its flag clear and before its
//! release only delays reuse. A forgotten token leaks its buffer, but its
//! ids stay covered, so the bound still holds.
//!
//! Ids and generations are atomics written and read `Relaxed`; the `free`
//! flag orders them between holders, and moving a handle to another thread
//! needs its own synchronization. So the pool needs no `unsafe`.

use super::ring::Signal;
use super::sync::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// An id-buffer pool operation that cannot complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BufferError {
    /// The sizes are not `1 <= batch_size <= n <= 2^31`, the bound of the
    /// id rings.
    #[error(
        "an id buffer pool needs 1 <= batch_size <= n <= 2^31, got batch_size {batch_size}, n {n}"
    )]
    Size {
        /// The number of instances.
        n: usize,
        /// The batch size.
        batch_size: usize,
    },
    /// The handle belongs to another pool. Neither pool is changed.
    #[error("the buffer handle belongs to another id buffer pool")]
    ForeignHandle,
    /// Every pool identity has been used (after `u64::MAX` pools).
    #[error("id buffer pool identities are exhausted")]
    PoolIdsExhausted,
    /// Ids and generations differ in length.
    #[error("{ids} ids but {generations} generations")]
    LengthMismatch {
        /// The number of ids given.
        ids: usize,
        /// The number of generations given.
        generations: usize,
    },
    /// The ids to store do not fit the buffer.
    #[error("{len} ids do not fit a buffer of {capacity}")]
    Length {
        /// The number of ids given.
        len: usize,
        /// The buffer's capacity.
        capacity: usize,
    },
}

/// One buffer: its free flag, its ids and their generations.
#[derive(Debug)]
struct IdBuffer {
    free: AtomicBool,
    len: AtomicU32,
    ids: Box<[AtomicU32]>,
    generations: Box<[AtomicU64]>,
}

impl IdBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            free: AtomicBool::new(true),
            len: AtomicU32::new(0),
            ids: (0..capacity).map(|_| AtomicU32::new(0)).collect(),
            generations: (0..capacity).map(|_| AtomicU64::new(0)).collect(),
        }
    }
}

/// The right to one buffer, from [`IdBufferPool::acquire`] until
/// [`IdBufferPool::release`]. It is neither `Clone` nor `Copy`.
#[derive(Debug, PartialEq, Eq)]
pub struct BufferHandle {
    pool: u64,
    index: usize,
}

impl BufferHandle {
    /// The buffer's index in its pool.
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }
}

/// The next pool identity. Only uniqueness matters, so this is a plain
/// process-wide counter, outside the loom model.
static NEXT_POOL: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// A fresh pool identity, advanced with checked arithmetic.
fn next_pool_id() -> Result<u64, BufferError> {
    use core::sync::atomic::Ordering::Relaxed;
    let mut id = NEXT_POOL.load(Relaxed);
    loop {
        let next = id.checked_add(1).ok_or(BufferError::PoolIdsExhausted)?;
        match NEXT_POOL.compare_exchange_weak(id, next, Relaxed, Relaxed) {
            Ok(_) => return Ok(id),
            Err(current) => id = current,
        }
    }
}

/// The pool of id buffers. Each pool has its own identity, carried by its
/// handles, so a handle from another pool is refused.
#[derive(Debug)]
pub struct IdBufferPool {
    id: u64,
    /// The batch buffers, then the full buffer.
    buffers: Box<[IdBuffer]>,
    batch_size: usize,
    n: usize,
    signal: Signal,
}

impl IdBufferPool {
    /// A pool for `n` instances and batches of `batch_size`.
    ///
    /// # Errors
    ///
    /// [`BufferError::Size`] unless `1 <= batch_size <= n <= 2^31`;
    /// [`BufferError::PoolIdsExhausted`] after `u64::MAX` pools.
    pub fn new(n: usize, batch_size: usize) -> Result<Self, BufferError> {
        if batch_size == 0 || batch_size > n || n > super::ring::MAX_IDS {
            return Err(BufferError::Size { n, batch_size });
        }
        let batches = n / batch_size;
        let buffers = (0..batches)
            .map(|_| IdBuffer::new(batch_size))
            .chain(core::iter::once(IdBuffer::new(n)))
            .collect();
        Ok(Self {
            id: next_pool_id()?,
            buffers,
            batch_size,
            n,
            signal: Signal::new(),
        })
    }

    /// The number of buffers: `floor(n / batch_size) + 1`.
    #[must_use]
    pub fn buffers(&self) -> usize {
        self.buffers.len()
    }

    /// The signal a caller waits on when no buffer is free.
    #[must_use]
    pub fn signal(&self) -> &Signal {
        &self.signal
    }

    /// Acquires a free buffer for a batch (`full == false`) or for all `n`
    /// ids (`full == true`). Meant for the caller only; the compare-and-swap
    /// keeps it correct even if two threads raced. `None` when every
    /// suitable buffer is held; the caller then waits on [`Self::signal`].
    pub fn acquire(&self, full: bool) -> Option<BufferHandle> {
        let last = self.buffers.len().checked_sub(1)?;
        let range = if full {
            last..self.buffers.len()
        } else {
            0..last
        };
        range.into_iter().find_map(|index| {
            let buffer = self.buffers.get(index)?;
            buffer
                .free
                .compare_exchange(true, false, Ordering::Acquire, Ordering::Relaxed)
                .ok()
                .map(|_| BufferHandle {
                    pool: self.id,
                    index,
                })
        })
    }

    /// Writes `ids` and their `generations` into the handle's buffer.
    ///
    /// # Errors
    ///
    /// [`BufferError::ForeignHandle`] if the handle belongs to another pool;
    /// [`BufferError::LengthMismatch`] if they differ in length;
    /// [`BufferError::Length`] if they do not fit.
    /// Nothing is written then.
    pub fn fill(
        &self,
        handle: &BufferHandle,
        ids: &[u32],
        generations: &[u64],
    ) -> Result<(), BufferError> {
        let buffer = self.buffer(handle)?;
        let capacity = buffer.ids.len();
        if ids.len() != generations.len() {
            return Err(BufferError::LengthMismatch {
                ids: ids.len(),
                generations: generations.len(),
            });
        }
        let error = BufferError::Length {
            len: ids.len(),
            capacity,
        };
        if ids.len() > capacity {
            return Err(error);
        }
        let len = u32::try_from(ids.len()).map_err(|_| error)?;
        for ((slot, gen_slot), (&id, &generation)) in buffer
            .ids
            .iter()
            .zip(buffer.generations.iter())
            .zip(ids.iter().zip(generations))
        {
            slot.store(id, Ordering::Relaxed);
            gen_slot.store(generation, Ordering::Relaxed);
        }
        buffer.len.store(len, Ordering::Relaxed);
        Ok(())
    }

    /// The ids and generations stored in the handle's buffer, in order. The
    /// iterator borrows both the pool and the handle.
    ///
    /// # Errors
    ///
    /// [`BufferError::ForeignHandle`] if the handle belongs to another pool.
    pub fn entries<'a>(
        &'a self,
        handle: &'a BufferHandle,
    ) -> Result<impl Iterator<Item = (u32, u64)> + 'a, BufferError> {
        let buffer = self.buffer(handle)?;
        let len = usize::try_from(buffer.len.load(Ordering::Relaxed)).unwrap_or(0);
        Ok(buffer
            .ids
            .iter()
            .zip(buffer.generations.iter())
            .take(len)
            .map(|(id, generation)| {
                (
                    id.load(Ordering::Relaxed),
                    generation.load(Ordering::Relaxed),
                )
            }))
    }

    /// Releases the handle's buffer from any thread: sets its `free` flag
    /// (Release), then notifies the signal. Returns whether a waiter is
    /// registered and must be woken.
    ///
    /// # Errors
    ///
    /// A handle from another pool is handed back unchanged, and neither
    /// pool is changed, so it can still be released to its own pool.
    #[must_use = "a registered waiter must be woken"]
    pub fn release(&self, handle: BufferHandle) -> Result<bool, BufferHandle> {
        let Ok(buffer) = self.buffer(&handle) else {
            return Err(handle);
        };
        buffer.free.store(true, Ordering::Release);
        Ok(self.signal.notify())
    }

    /// The batch size of the pool.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// The number of instances of the pool.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    fn buffer(&self, handle: &BufferHandle) -> Result<&IdBuffer, BufferError> {
        if handle.pool != self.id {
            return Err(BufferError::ForeignHandle);
        }
        self.buffers
            .get(handle.index)
            .ok_or(BufferError::ForeignHandle)
    }
}

#[cfg(test)]
mod tests;
