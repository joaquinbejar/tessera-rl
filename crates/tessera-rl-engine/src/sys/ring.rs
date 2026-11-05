// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded rings of instance ids for Async mode (ADR-0047 section 7).
//!
//! Both rings are the bounded queue of Vyukov: a power-of-two array of
//! slots, each with a sequence number, and two monotonic positions. A
//! producer claims position `p` when the slot's sequence equals `p`, writes
//! the id and stores `p + 1` (Release); a consumer claims `p` when the
//! sequence equals `p + 1`, reads the id and stores `p + capacity`
//! (Release). Positions are claimed with a compare-and-swap, so several
//! threads may share a side.
//!
//! - [`SubmitRing`] has one producer (the caller, `&mut self`) and many
//!   consumers (workers claim ids with `&self`).
//! - [`CompletionRing`] has many producers (workers, `&self`) and one
//!   consumer (the caller, `&mut self`).
//!
//! The id lives in an `AtomicU32` written and read `Relaxed`; the slot's
//! sequence store (Release) and load (Acquire) order it. So the rings need
//! no `unsafe`.
//!
//! # Capacity
//!
//! The capacity is the next power of two at or above `max(n, 2)`, where
//! `n` is the number of instances. The queue needs at least two slots: with
//! one, a slot's sequence `p + 1` would mean both "position `p` filled" and
//! "position `p + 1` free", and a second push would overwrite the first id.
//! At most `n` ids are ever live, because of backpressure (one step in
//! flight per instance). When `n` is a power of two a ring can be exactly
//! full; a push on a full ring returns [`RingError::Full`] and never
//! overwrites a slot.
//!
//! **Transient `Full` and empty.** An operation can be paused after it has
//! claimed a position and before it publishes the slot's new sequence. A
//! push then sees `Full` behind a pop that has taken an id but not yet freed
//! its slot, and a pop sees an empty ring behind a push that has reserved a
//! slot but not yet filled it. Both clear as soon as the paused thread
//! resumes. So "at most `n` live ids" does not make a push infallible: the
//! Async wrapper must treat `Full` from an admitted push as "retry after the
//! stalled operation finishes", never as success, and a consumer that sees
//! an empty ring must not conclude that nothing is in flight.
//!
//! Positions only grow, by one per push and per pop, and are advanced with
//! checked arithmetic: [`RingError::Exhausted`] after `usize::MAX`
//! operations, never a silent wrap.
//!
//! # Waiting
//!
//! A ring does not block. [`Signal`] is the counted-wake word a consumer
//! waits on: the producer changes the word after its push and reports
//! whether anyone is registered, so the caller wakes the waiter through the
//! gang's wait backend only when needed (see `sys::gang` for the
//! registration and recheck argument).

use super::sync::{AtomicU32, AtomicUsize, Ordering};

/// A ring operation that cannot complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RingError {
    /// The ring holds `capacity` ids; admission should have prevented this.
    #[error("the id ring is full ({capacity} ids)")]
    Full {
        /// The ring's capacity.
        capacity: usize,
    },
    /// A ring position cannot be advanced any further.
    #[error("the id ring positions are exhausted")]
    Exhausted,
    /// The ring was asked for zero ids or more than `u32` can address.
    #[error("an id ring needs between 1 and {max} ids, got {n}")]
    Size {
        /// The requested number of ids.
        n: usize,
        /// The largest accepted number.
        max: usize,
    },
}

/// One slot: its sequence and the id it holds.
#[derive(Debug)]
struct Slot {
    seq: AtomicUsize,
    id: AtomicU32,
}

/// The shared queue under both rings.
#[derive(Debug)]
struct IdRing {
    slots: Box<[Slot]>,
    mask: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

/// The largest number of ids a ring accepts: ids are `u32`, and the
/// capacity must stay a power of two that fits in `usize`.
pub(crate) const MAX_IDS: usize = 1 << 31;

impl IdRing {
    fn new(n: usize) -> Result<Self, RingError> {
        if n == 0 || n > MAX_IDS {
            return Err(RingError::Size { n, max: MAX_IDS });
        }
        let capacity = n
            .max(2)
            .checked_next_power_of_two()
            .ok_or(RingError::Size { n, max: MAX_IDS })?;
        let slots = (0..capacity)
            .map(|i| Slot {
                seq: AtomicUsize::new(i),
                id: AtomicU32::new(0),
            })
            .collect();
        Ok(Self {
            slots,
            mask: capacity - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        })
    }

    fn capacity(&self) -> usize {
        self.slots.len()
    }

    fn slot(&self, pos: usize) -> Option<&Slot> {
        self.slots.get(pos & self.mask)
    }

    fn push(&self, id: u32) -> Result<(), RingError> {
        let mut pos = self.tail.load(Ordering::Relaxed);
        loop {
            let slot = self.slot(pos).ok_or(RingError::Exhausted)?;
            let seq = slot.seq.load(Ordering::Acquire);
            match seq.cmp(&pos) {
                core::cmp::Ordering::Equal => {
                    let next = pos.checked_add(1).ok_or(RingError::Exhausted)?;
                    match self.tail.compare_exchange(
                        pos,
                        next,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            slot.id.store(id, Ordering::Relaxed);
                            slot.seq.store(next, Ordering::Release);
                            return Ok(());
                        }
                        Err(current) => pos = current,
                    }
                }
                // The slot still holds the id from one lap ago.
                core::cmp::Ordering::Less => {
                    return Err(RingError::Full {
                        capacity: self.capacity(),
                    });
                }
                core::cmp::Ordering::Greater => pos = self.tail.load(Ordering::Relaxed),
            }
        }
    }

    fn pop(&self) -> Result<Option<u32>, RingError> {
        let mut pos = self.head.load(Ordering::Relaxed);
        loop {
            let slot = self.slot(pos).ok_or(RingError::Exhausted)?;
            let seq = slot.seq.load(Ordering::Acquire);
            // Both positions are computed before the claim, so exhaustion is
            // reported without taking a position.
            let filled = pos.checked_add(1).ok_or(RingError::Exhausted)?;
            let free = pos
                .checked_add(self.capacity())
                .ok_or(RingError::Exhausted)?;
            match seq.cmp(&filled) {
                core::cmp::Ordering::Equal => match self.head.compare_exchange(
                    pos,
                    filled,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        let id = slot.id.load(Ordering::Relaxed);
                        slot.seq.store(free, Ordering::Release);
                        return Ok(Some(id));
                    }
                    Err(current) => pos = current,
                },
                core::cmp::Ordering::Less => return Ok(None),
                core::cmp::Ordering::Greater => pos = self.head.load(Ordering::Relaxed),
            }
        }
    }
}

/// The submission ring: the caller pushes, workers claim (SPMC).
#[derive(Debug)]
pub struct SubmitRing {
    ring: IdRing,
}

impl SubmitRing {
    /// A ring for `n` instances.
    ///
    /// # Errors
    ///
    /// [`RingError::Size`] if `n` is zero or larger than `2^31`.
    pub fn new(n: usize) -> Result<Self, RingError> {
        Ok(Self {
            ring: IdRing::new(n)?,
        })
    }

    /// The number of slots, a power of two at or above `max(n, 2)`.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.ring.capacity()
    }

    /// Admits `id`. Only the caller produces, so this takes `&mut self`.
    ///
    /// # Errors
    ///
    /// [`RingError::Full`] if the ring is full, including transiently behind
    /// a paused claim (see the module docs); [`RingError::Exhausted`] if its
    /// positions are exhausted. The ring is unchanged on an error.
    pub fn push(&mut self, id: u32) -> Result<(), RingError> {
        self.ring.push(id)
    }

    /// Claims the oldest admitted id, if any. Workers share this side.
    ///
    /// # Errors
    ///
    /// [`RingError::Exhausted`] if the positions are exhausted.
    pub fn claim(&self) -> Result<Option<u32>, RingError> {
        self.ring.pop()
    }

    /// [`Self::push`] through a shared reference, for a mode whose ring sits
    /// in its shared allocation. The queue is safe for any number of
    /// producers; the caller keeps the single-producer protocol (only the
    /// mode's `&mut self` methods push).
    #[cfg_attr(
        loom,
        allow(dead_code, reason = "the split-phase modes are not built under loom")
    )]
    pub(crate) fn push_shared(&self, id: u32) -> Result<(), RingError> {
        self.ring.push(id)
    }
}

/// The completion ring: workers push, the caller pops (MPSC).
#[derive(Debug)]
pub struct CompletionRing {
    ring: IdRing,
}

impl CompletionRing {
    /// A ring for `n` instances.
    ///
    /// # Errors
    ///
    /// [`RingError::Size`] if `n` is zero or larger than `2^31`.
    pub fn new(n: usize) -> Result<Self, RingError> {
        Ok(Self {
            ring: IdRing::new(n)?,
        })
    }

    /// The number of slots, a power of two at or above `max(n, 2)`.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.ring.capacity()
    }

    /// Reports `id` complete. Workers share this side.
    ///
    /// # Errors
    ///
    /// [`RingError::Full`] if the ring is full, [`RingError::Exhausted`] if
    /// its positions are exhausted.
    pub fn push(&self, id: u32) -> Result<(), RingError> {
        self.ring.push(id)
    }

    /// Pops the oldest completed id, if any. Only the caller consumes.
    ///
    /// # Errors
    ///
    /// [`RingError::Exhausted`] if the positions are exhausted.
    pub fn pop(&mut self) -> Result<Option<u32>, RingError> {
        self.ring.pop()
    }

    /// [`Self::pop`] through a shared reference, for a mode whose ring sits
    /// in its shared allocation. The queue is safe for any number of
    /// consumers; the caller keeps the single-consumer protocol (only the
    /// mode's `&mut self` methods pop).
    #[cfg_attr(
        loom,
        allow(dead_code, reason = "the split-phase modes are not built under loom")
    )]
    pub(crate) fn pop_shared(&self) -> Result<Option<u32>, RingError> {
        self.ring.pop()
    }
}

/// A counted-wake word (ADR-0047 section 7): a waiter registers and
/// rechecks its condition, a notifier changes the word and learns whether
/// a wake is needed.
///
/// Both sides use read-modify-writes on `parked` (`SeqCst`), so whichever
/// comes second in its modification order sees the other: either the
/// waiter's recheck sees the change, or the notifier sees the waiter. A
/// waiter blocks only while the word still holds the value it read before
/// registering, through the gang's wait backend.
#[derive(Debug, Default)]
pub struct Signal {
    word: AtomicU32,
    parked: AtomicU32,
}

impl Signal {
    /// A signal with no waiter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The word a waiter blocks on.
    #[must_use]
    pub fn word(&self) -> &AtomicU32 {
        &self.word
    }

    /// Registers a waiter and returns the word's value to block on. The
    /// waiter must recheck its condition after this and call
    /// [`Signal::unregister`] when it stops waiting.
    #[must_use = "the value is what the waiter blocks on"]
    pub fn register(&self) -> u32 {
        let value = self.word.load(Ordering::SeqCst);
        self.parked.fetch_add(1, Ordering::SeqCst);
        value
    }

    /// Ends a registration.
    pub fn unregister(&self) {
        self.parked.fetch_sub(1, Ordering::SeqCst);
    }

    /// Announces a change made before this call. Returns whether a waiter
    /// is registered, that is, whether a wake through the backend is
    /// needed.
    #[must_use = "a registered waiter must be woken"]
    pub fn notify(&self) -> bool {
        self.word.fetch_add(1, Ordering::SeqCst);
        self.parked.fetch_add(0, Ordering::SeqCst) > 0
    }
}

#[cfg(test)]
mod tests;
