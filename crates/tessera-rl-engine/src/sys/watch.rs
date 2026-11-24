// SPDX-License-Identifier: MIT OR Apache-2.0

//! The live-watch request slot (ADR-0083 item 7, issue #378 stage 4d).
//! Audited `unsafe` module (part of `engine::sys`).
//!
//! A [`WatchSlot`] is a bounded single-producer single-consumer slot: the
//! viewer (one [`WatchHandle`]) asks for one instance's record, and the
//! pool answers at its next batch boundary, on the caller's thread, by
//! encoding the record into a reply buffer allocated once at attach.
//!
//! # Protocol
//!
//! `state` is one of `IDLE`, `REQUESTED`, `SERVING`, `READY` and `TAKING`:
//!
//! - **Handle** (one consumer, `&mut self`): `request` stores `instance`,
//!   then moves `IDLE | READY -> REQUESTED` with a `Release`
//!   compare-and-swap, and returns [`WatchError::Busy`] from any other
//!   state. `take` moves `READY -> TAKING` (`Acquire`), copies the buffer
//!   out, then stores `IDLE` (`Release`).
//! - **Pool** (the thread that owns the pool, at a batch boundary): the
//!   fast path is one `Relaxed` load of `state`; only on `REQUESTED` it
//!   moves `REQUESTED -> SERVING` (`Acquire`, so `instance` is visible),
//!   encodes the record into the buffer, stores the generation, then
//!   `READY` (`Release`). An instance not idle at this boundary goes back
//!   to `REQUESTED` for a later one.
//!
//! The pool only ever moves `REQUESTED` (to `SERVING`) and `SERVING`; the
//! handle only ever moves `IDLE`, `READY` and `TAKING`. So the buffer is
//! written only in `SERVING`, by the pool, and read only in `TAKING`, by
//! the handle; the two never overlap. `TAKING` is defence in depth: the
//! handle type is `!Sync`, `!Clone` and takes `&mut self`, so a request
//! during a take is already impossible; with `TAKING`, even a second
//! consumer could not make the pool write during a copy (the loom model
//! checks both, and that the model fails without `TAKING`).
//!
//! `flags` carries `DETACHED` (the pool dropped its side: requests and
//! takes return [`WatchError::Detached`]), `HANDLE_DROPPED` (read by the
//! pool only on a pending request, never on the fast path; the pool then
//! drops its side) and `POISONED` (the pool is poisoned and serves
//! nothing; `take` reports [`WatchError::Poisoned`]).
//!
//! The reply buffer, in an `UnsafeCell`, is the only `unsafe` here.

use core::marker::PhantomData;

use super::sync::{Arc, AtomicU32, AtomicU64, Ordering, UnsafeCell};

const IDLE: u32 = 0;
const REQUESTED: u32 = 1;
const SERVING: u32 = 2;
const READY: u32 = 3;
const TAKING: u32 = 4;

const DETACHED: u32 = 1;
const HANDLE_DROPPED: u32 = 1 << 1;
const POISONED: u32 = 1 << 2;

/// Why a [`WatchHandle`] operation was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum WatchError {
    /// A request is pending or being served, or a take is in progress.
    #[error("a watch request is already pending")]
    Busy,
    /// The pool detached the watch (or was dropped).
    #[error("the watch is detached from its pool")]
    Detached,
    /// The pool is poisoned and serves nothing (ADR-0028).
    #[error("the watched pool is poisoned")]
    Poisoned,
    /// The output buffer of `take` does not have the record's length.
    #[error("the buffer has {actual} bytes, a record has {expected}")]
    BufferLength {
        /// The record length ([`WatchHandle::record_len`]).
        expected: usize,
        /// The length given.
        actual: usize,
    },
    /// The requested instance is not in the pool.
    #[error("instance {id} out of range (num_envs {num_envs})")]
    InstanceOutOfRange {
        /// The id requested.
        id: u32,
        /// Instances in the pool.
        num_envs: u32,
    },
}

/// Returns a slot in `SERVING` to `REQUESTED` if the encoder unwinds
/// (REVIEW-4d-r1); forgotten on the normal path, which stores the next
/// state itself.
struct ServingGuard<'a> {
    state: &'a AtomicU32,
}

impl Drop for ServingGuard<'_> {
    fn drop(&mut self) {
        self.state.store(REQUESTED, Ordering::Release);
    }
}

/// What one pool-side service of the slot did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Served {
    /// No request was pending.
    Nothing,
    /// The requested instance is not idle at this boundary; the request
    /// stays pending.
    Pending,
    /// The record was encoded and is `READY`.
    Ready,
    /// The handle was dropped: the pool should drop its side.
    Abandoned,
}

/// The shared slot (module protocol).
#[derive(Debug)]
pub(crate) struct WatchSlot {
    state: AtomicU32,
    flags: AtomicU32,
    instance: AtomicU32,
    generation: AtomicU64,
    num_envs: u32,
    len: usize,
    buffer: UnsafeCell<Box<[u8]>>,
    /// The model's own mutant: `take` without `TAKING` (loom only).
    #[cfg(all(test, loom))]
    pub(crate) no_taking: bool,
}

// SAFETY: the buffer is written only by the pool in `SERVING` and read only
// by the consumer in `TAKING` (module protocol), and every other field is an
// atomic or immutable after construction; `Box<[u8]>` is `Send + Sync`.
#[expect(unsafe_code, reason = "audited module: watch slot")]
unsafe impl Sync for WatchSlot {}

impl WatchSlot {
    /// A slot for a pool of `num_envs` instances whose records encode to
    /// `len` bytes. Allocates the reply buffer; never on the step path.
    #[must_use]
    pub(crate) fn new(num_envs: u32, len: usize) -> Self {
        Self {
            state: AtomicU32::new(IDLE),
            flags: AtomicU32::new(0),
            instance: AtomicU32::new(0),
            generation: AtomicU64::new(0),
            num_envs,
            len,
            buffer: UnsafeCell::new(vec![0; len].into_boxed_slice()),
            #[cfg(all(test, loom))]
            no_taking: false,
        }
    }

    /// Pool: whether a request is pending. The fast path: one `Relaxed`
    /// load; a hit is confirmed by the `Acquire` swap in [`Self::serve`].
    #[inline]
    #[must_use]
    pub(crate) fn requested(&self) -> bool {
        self.state.load(Ordering::Relaxed) == REQUESTED
    }

    /// Pool: serves a pending request at a batch boundary. `encode` writes
    /// instance `id`'s record into the buffer and returns false if `id` is
    /// not idle at this boundary (the request then stays pending).
    pub(crate) fn serve(
        &self,
        generation: u64,
        encode: impl FnOnce(u32, &mut [u8]) -> bool,
    ) -> Served {
        if self.flags.load(Ordering::Acquire) & HANDLE_DROPPED != 0 {
            return Served::Abandoned;
        }
        if self
            .state
            .compare_exchange(REQUESTED, SERVING, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Served::Nothing;
        }
        let id = self.instance.load(Ordering::Relaxed);
        // If `encode` panics, the guard puts the slot back to `REQUESTED`
        // (REVIEW-4d-r1): never stuck in `SERVING`, and the half-written
        // buffer is never published, since only `READY` exposes it.
        let guard = ServingGuard { state: &self.state };
        // SAFETY: this thread holds `SERVING`, which only the pool enters
        // (from `REQUESTED`) and leaves; the consumer reads the buffer only
        // in `TAKING`, entered from `READY`, and `READY` is stored below only
        // after this write ends. So no read or other write overlaps it.
        #[expect(unsafe_code, reason = "audited module: watch buffer write")]
        let written = self.buffer.with_mut(|buf| encode(id, unsafe { &mut *buf }));
        core::mem::forget(guard);
        if written {
            self.generation.store(generation, Ordering::Relaxed);
            self.state.store(READY, Ordering::Release);
            Served::Ready
        } else {
            self.state.store(REQUESTED, Ordering::Release);
            Served::Pending
        }
    }

    /// Pool: marks the pool poisoned (or recovered).
    pub(crate) fn set_poisoned(&self, poisoned: bool) {
        if poisoned {
            self.flags.fetch_or(POISONED, Ordering::Release);
        } else {
            self.flags.fetch_and(!POISONED, Ordering::Release);
        }
    }

    /// Pool: its side is gone; the handle's calls return `Detached`.
    pub(crate) fn detach(&self) {
        self.flags.fetch_or(DETACHED, Ordering::Release);
    }

    /// Whether the handle was dropped (attach replaces such a slot).
    #[must_use]
    pub(crate) fn handle_dropped(&self) -> bool {
        self.flags.load(Ordering::Acquire) & HANDLE_DROPPED != 0
    }

    /// Consumer: asks for instance `instance`'s record (module protocol).
    pub(crate) fn request(&self, instance: u32) -> Result<(), WatchError> {
        if self.flags.load(Ordering::Acquire) & DETACHED != 0 {
            return Err(WatchError::Detached);
        }
        if instance >= self.num_envs {
            return Err(WatchError::InstanceOutOfRange {
                id: instance,
                num_envs: self.num_envs,
            });
        }
        let current = self.state.load(Ordering::Acquire);
        if current != IDLE && current != READY {
            return Err(WatchError::Busy);
        }
        // The pool reads `instance` only after its `Acquire` swap from
        // `REQUESTED`, which reads from the `Release` swap below.
        self.instance.store(instance, Ordering::Relaxed);
        self.state
            .compare_exchange(current, REQUESTED, Ordering::Release, Ordering::Relaxed)
            .map(|_| ())
            .map_err(|_| WatchError::Busy)
    }

    /// Consumer: copies a `READY` record into `out` and returns its
    /// generation; `None` if none is ready. Never blocks.
    pub(crate) fn take(&self, out: &mut [u8]) -> Result<Option<u64>, WatchError> {
        if out.len() != self.len {
            return Err(WatchError::BufferLength {
                expected: self.len,
                actual: out.len(),
            });
        }
        let flags = self.flags.load(Ordering::Acquire);
        if flags & DETACHED != 0 {
            return Err(WatchError::Detached);
        }
        if !self.enter_taking() {
            if flags & POISONED != 0 {
                return Err(WatchError::Poisoned);
            }
            return Ok(None);
        }
        // SAFETY: this consumer holds `TAKING` (entered from `READY` with
        // `Acquire`, which reads from the pool's `Release` store of `READY`
        // after its write); the pool writes only in `SERVING`, which it
        // enters only from `REQUESTED`, and `request` refuses to leave
        // `TAKING`. So no write overlaps this read.
        #[expect(unsafe_code, reason = "audited module: watch buffer read")]
        self.buffer.with(|buf| copy_into(out, unsafe { &*buf }));
        let generation = self.generation.load(Ordering::Relaxed);
        self.leave_taking();
        Ok(Some(generation))
    }

    /// `READY -> TAKING`.
    fn enter_taking(&self) -> bool {
        #[cfg(all(test, loom))]
        if self.no_taking {
            // The mutant: read in `READY`, which a request may leave.
            return self.state.load(Ordering::Acquire) == READY;
        }
        self.state
            .compare_exchange(READY, TAKING, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    /// `TAKING -> IDLE`.
    fn leave_taking(&self) {
        #[cfg(all(test, loom))]
        if self.no_taking {
            let _ = self
                .state
                .compare_exchange(READY, IDLE, Ordering::Release, Ordering::Relaxed);
            return;
        }
        self.state.store(IDLE, Ordering::Release);
    }

    /// The record length in bytes.
    #[must_use]
    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    /// Instances in the pool.
    #[must_use]
    pub(crate) const fn num_envs(&self) -> u32 {
        self.num_envs
    }

    /// Whether the pool detached.
    #[must_use]
    pub(crate) fn detached(&self) -> bool {
        self.flags.load(Ordering::Acquire) & DETACHED != 0
    }

    /// Consumer: its handle is gone.
    fn drop_handle(&self) {
        self.flags.fetch_or(HANDLE_DROPPED, Ordering::Release);
    }
}

/// Copies `src` into `out` (equal lengths, checked by the caller).
fn copy_into(out: &mut [u8], src: &[u8]) {
    for (o, s) in out.iter_mut().zip(src) {
        *o = *s;
    }
}

/// The viewer's side of a live watch (ADR-0083 item 7): one consumer.
///
/// `Send` but neither `Sync` nor `Clone`, and both operations take
/// `&mut self`, so one thread at a time requests and takes. Get one with
/// `EnvPool::attach_watch` (or the split-phase pools' `attach_watch`).
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<tessera_rl_engine::WatchHandle>();
/// ```
///
/// ```compile_fail
/// fn assert_clone<T: Clone>() {}
/// assert_clone::<tessera_rl_engine::WatchHandle>();
/// ```
///
/// ```
/// fn assert_send<T: Send>() {}
/// assert_send::<tessera_rl_engine::WatchHandle>();
/// ```
pub struct WatchHandle {
    slot: Arc<WatchSlot>,
    /// `!Sync`: a single consumer.
    _single: PhantomData<core::cell::Cell<()>>,
}

impl core::fmt::Debug for WatchHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WatchHandle")
            .field("record_len", &self.slot.len())
            .field("detached", &self.slot.detached())
            .finish_non_exhaustive()
    }
}

impl WatchHandle {
    pub(crate) fn new(slot: Arc<WatchSlot>) -> Self {
        Self {
            slot,
            _single: PhantomData,
        }
    }

    /// Asks for instance `instance`'s record at the pool's next batch
    /// boundary where it is idle. Replaces a ready record not yet taken.
    ///
    /// # Errors
    ///
    /// [`WatchError::Busy`] while a request is pending or being served,
    /// [`WatchError::Detached`] after the pool detached, or
    /// [`WatchError::InstanceOutOfRange`].
    pub fn request(&mut self, instance: u32) -> Result<(), WatchError> {
        self.slot.request(instance)
    }

    /// Copies the served record (canonical encoding, as
    /// `Record::to_bytes`) into `out` and returns its generation (the
    /// pool's boundary count since attach); `Ok(None)` if no record is ready
    /// yet. Never blocks.
    ///
    /// # Errors
    ///
    /// [`WatchError::BufferLength`] if `out` is not
    /// [`Self::record_len`] bytes, [`WatchError::Detached`] after the pool
    /// detached, or [`WatchError::Poisoned`] while the pool is poisoned.
    pub fn take(&mut self, out: &mut [u8]) -> Result<Option<u64>, WatchError> {
        self.slot.take(out)
    }

    /// The length of a record in bytes.
    #[must_use]
    pub fn record_len(&self) -> usize {
        self.slot.len()
    }

    /// Instances in the watched pool.
    #[must_use]
    pub fn num_envs(&self) -> u32 {
        self.slot.num_envs()
    }

    /// Whether the pool detached the watch.
    #[must_use]
    pub fn is_detached(&self) -> bool {
        self.slot.detached()
    }

    /// The slot, for the loom model's forced second consumer.
    #[cfg(test)]
    pub(crate) fn slot(&self) -> &WatchSlot {
        &self.slot
    }
}

impl Drop for WatchHandle {
    fn drop(&mut self) {
        self.slot.drop_handle();
    }
}

/// A new slot and its handle.
#[must_use]
pub(crate) fn watch_pair(num_envs: u32, len: usize) -> (Arc<WatchSlot>, WatchHandle) {
    let slot = Arc::new(WatchSlot::new(num_envs, len));
    let handle = WatchHandle::new(Arc::clone(&slot));
    (slot, handle)
}

#[cfg(test)]
mod tests;
