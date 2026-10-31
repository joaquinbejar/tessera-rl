// SPDX-License-Identifier: MIT OR Apache-2.0

//! One shared allocation for every output buffer of a pool, with a write
//! generation (phase 1b B3a). Audited `unsafe` module (part of `buffer`).
//!
//! The pool owns one [`Buf`] per output region; borrowers (the Python
//! layer's borrowed views) hold an `Arc<OutputStorage>`, so the memory
//! outlives the pool, and read it only through [`OutputStorage::region`].
//!
//! # Invariants
//!
//! 1. The allocation is made once, zeroed, and never resized or freed
//!    before the last `Arc` is dropped. Every element type is valid for the
//!    all-zero bit pattern ([`Element`]).
//! 2. Regions are disjoint: offsets come from one running sum, each region
//!    starts aligned to 8 bytes, which covers every [`Element`] alignment.
//! 3. Exactly one [`Buf`] exists per region (`Buf` is not `Clone` and is
//!    only created by [`StorageBuilder::finish`]), so `&mut Buf` is the only
//!    way to get a `&mut` slice of its region.
//! 4. The engine increments the generation before any write to the
//!    outputs ([`Buf`] slices are written only through
//!    `Outputs::view_all`, which calls [`OutputStorage::begin_write`]).
//!    Borrowers may read a region only between engine calls, which the
//!    caller serializes (the borrow window of ADR-0003); comparing the
//!    generation tells a borrower whether its view is stale.

use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use std::sync::Arc;

use tessera_rl_core::Bool8;

use crate::sys::sync::{AtomicU64, Ordering};

/// The element type of an output region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OutputDtype {
    /// Raw little-endian bytes (observation leaves; the leaf dtype is in
    /// the pool layout).
    U8,
    /// `Bool8`: one byte, 0 or 1.
    Bool8,
    /// `f32`.
    F32,
    /// `u32`.
    U32,
}

/// An output region of a pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OutputField {
    /// Observation leaf `n`, `[N, A, leaf bytes]`.
    Observation(usize),
    /// Terminal observation leaf `n` (`SameStep` mode only).
    TerminalObservation(usize),
    /// Rewards, `[N, A]`.
    Rewards,
    /// Per-agent terminated flags, `[N, A]`.
    AgentTerminated,
    /// Per-agent truncated flags, `[N, A]`.
    AgentTruncated,
    /// Agents that died in the last call, `[N, A]`.
    Died,
    /// Agents that spawned in the last call, `[N, A]`.
    Spawned,
    /// Invalid agent events of the last call, `[N, A]`.
    InvalidAgentEvent,
    /// Alive mask, `[N, A]`.
    Alive,
    /// Acting mask, `[N, A]`.
    Acting,
    /// Slot generations, `[N, A]`.
    Generations,
    /// Episode terminated flags, `[N]`.
    EpisodeTerminated,
    /// Episode truncated flags, `[N]`.
    EpisodeTruncated,
    /// Episode returns, `[N]`.
    EpisodeReturn,
    /// Episode lengths, `[N]`.
    EpisodeLength,
    /// Illegal actions per ended episode, `[N]`.
    EpisodeIllegalActions,
    /// Invalid agent events per ended episode, `[N]`.
    EpisodeInvalidAgentEvents,
    /// Action masks, `[N, A, mask length]`.
    Masks,
}

/// A read-only description of one region: where it is, how long, of what.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputRegion {
    /// First byte of the region; valid while an `Arc<OutputStorage>` is
    /// held. Never written through: the engine is the only writer.
    pub ptr: *const u8,
    /// Number of elements.
    pub len: usize,
    /// Element type.
    pub dtype: OutputDtype,
}

/// Element types of output regions. Sealed: each is plain data, at most
/// 8-byte aligned, and valid for the all-zero bit pattern (invariant 1).
pub(crate) trait Element: Copy + sealed::Sealed + 'static {
    const DTYPE: OutputDtype;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for tessera_rl_core::Bool8 {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
}

impl Element for u8 {
    const DTYPE: OutputDtype = OutputDtype::U8;
}
impl Element for Bool8 {
    const DTYPE: OutputDtype = OutputDtype::Bool8;
}
impl Element for f32 {
    const DTYPE: OutputDtype = OutputDtype::F32;
}
impl Element for u32 {
    const DTYPE: OutputDtype = OutputDtype::U32;
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    field: OutputField,
    offset: usize,
    len: usize,
    dtype: OutputDtype,
}

/// The single allocation holding every output buffer of a pool.
pub struct OutputStorage {
    words: Box<[UnsafeCell<u64>]>,
    entries: Vec<Entry>,
    generation: AtomicU64,
}

impl core::fmt::Debug for OutputStorage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OutputStorage")
            .field("bytes", &self.bytes())
            .field("regions", &self.entries.len())
            .field("generation", &self.generation())
            .finish_non_exhaustive()
    }
}

// SAFETY: the words are only written through the unique `Buf` of each
// region (invariant 3) under `&mut`, and read by borrowers only between
// engine calls (invariant 4); the generation is atomic.
#[expect(unsafe_code, reason = "audited module: shared output storage")]
unsafe impl Sync for OutputStorage {}
// SAFETY: as for `Sync`; the storage owns its allocation.
#[expect(unsafe_code, reason = "audited module: shared output storage")]
unsafe impl Send for OutputStorage {}

impl OutputStorage {
    /// The write generation: incremented before every engine call that
    /// writes outputs. A view taken at generation `g` is stale once this
    /// differs from `g`.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Total size of the allocation in bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.words.len().saturating_mul(8)
    }

    /// The region of `field`, if the pool has it.
    #[must_use]
    pub fn region(&self, field: OutputField) -> Option<OutputRegion> {
        let e = self.entries.iter().find(|e| e.field == field)?;
        let base = UnsafeCell::raw_get(self.words.as_ptr()).cast::<u8>();
        Some(OutputRegion {
            // `wrapping_add` keeps this safe; `offset` is within the
            // allocation by construction (invariant 2).
            ptr: base.wrapping_add(e.offset).cast_const(),
            len: e.len,
            dtype: e.dtype,
        })
    }

    /// Marks the start of a write to the outputs (invariant 4).
    pub(crate) fn begin_write(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
    }
}

/// A typed handle to one region, reserved before the allocation exists.
#[derive(Debug)]
pub(crate) struct Slot<T> {
    index: usize,
    _t: core::marker::PhantomData<T>,
}

/// Lays out the regions of an [`OutputStorage`].
#[derive(Debug, Default)]
pub(crate) struct StorageBuilder {
    entries: Vec<Entry>,
    bytes: usize,
}

impl StorageBuilder {
    /// Reserves `len` elements of `T` for `field`; `None` on overflow.
    pub(crate) fn reserve<T: Element>(
        &mut self,
        field: OutputField,
        len: usize,
    ) -> Option<Slot<T>> {
        let offset = self.bytes.checked_next_multiple_of(8)?;
        let size = len.checked_mul(core::mem::size_of::<T>())?;
        self.bytes = offset.checked_add(size)?;
        self.entries.push(Entry {
            field,
            offset,
            len,
            dtype: T::DTYPE,
        });
        Some(Slot {
            index: self.entries.len().checked_sub(1)?,
            _t: core::marker::PhantomData,
        })
    }

    /// Allocates the zeroed storage.
    pub(crate) fn finish(self) -> Option<Arc<OutputStorage>> {
        let words = self.bytes.checked_next_multiple_of(8)? / 8;
        let words = (0..words).map(|_| UnsafeCell::new(0)).collect();
        Some(Arc::new(OutputStorage {
            words,
            entries: self.entries,
            generation: AtomicU64::new(0),
        }))
    }
}

/// The engine's unique, writable handle to one region (invariant 3). It
/// dereferences to the region's slice and keeps the storage alive.
pub(crate) struct Buf<T> {
    storage: Arc<OutputStorage>,
    ptr: NonNull<T>,
    len: usize,
}

impl<T> core::fmt::Debug for Buf<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Buf")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

// SAFETY: a `Buf` is the unique handle to its region (invariant 3); moving
// it to another thread moves that exclusivity with it.
#[expect(unsafe_code, reason = "audited module: shared output storage")]
unsafe impl<T: Send> Send for Buf<T> {}
// SAFETY: `&Buf` only yields `&[T]`; writes need `&mut Buf`.
#[expect(unsafe_code, reason = "audited module: shared output storage")]
unsafe impl<T: Sync> Sync for Buf<T> {}

impl<T: Element> Buf<T> {
    /// The handle of `slot`; each slot is taken once (invariant 3).
    #[expect(
        clippy::needless_pass_by_value,
        reason = "taking the slot by value is what makes each region's Buf unique"
    )]
    pub(crate) fn take(storage: &Arc<OutputStorage>, slot: Slot<T>) -> Option<Self> {
        let e = storage.entries.get(slot.index)?;
        let base = UnsafeCell::raw_get(storage.words.as_ptr()).cast::<u8>();
        let ptr = NonNull::new(base.wrapping_add(e.offset).cast::<T>())?;
        debug_assert!(ptr.as_ptr().is_aligned(), "regions are 8-byte aligned");
        Some(Self {
            storage: Arc::clone(storage),
            ptr,
            len: e.len,
        })
    }

    /// The storage this region lives in.
    pub(crate) fn storage(&self) -> &Arc<OutputStorage> {
        &self.storage
    }
}

impl<T> Deref for Buf<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: `ptr..ptr + len` is an aligned region of the live
        // allocation (invariants 1, 2), initialized (zeroed, then written
        // only as `T`), and no `&mut` to it exists while `&self` is held
        // (invariant 3).
        #[expect(unsafe_code, reason = "audited module: shared output storage")]
        unsafe {
            core::slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }
}

impl<T> DerefMut for Buf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as for `deref`; `&mut self` is exclusive to the region
        // (invariant 3), and the storage words are `UnsafeCell`, so writing
        // through a pointer derived from a shared `Arc` is allowed.
        #[expect(unsafe_code, reason = "audited module: shared output storage")]
        unsafe {
            core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
        }
    }
}

#[cfg(test)]
#[cfg(not(loom))]
mod tests {
    use super::*;

    fn storage() -> (Arc<OutputStorage>, Buf<f32>, Buf<Bool8>, Buf<u8>) {
        let mut builder = StorageBuilder::default();
        let rewards = builder.reserve::<f32>(OutputField::Rewards, 5).unwrap();
        let alive = builder.reserve::<Bool8>(OutputField::Alive, 3).unwrap();
        let obs = builder
            .reserve::<u8>(OutputField::Observation(0), 7)
            .unwrap();
        let storage = builder.finish().unwrap();
        let rewards = Buf::take(&storage, rewards).unwrap();
        let alive = Buf::take(&storage, alive).unwrap();
        let obs = Buf::take(&storage, obs).unwrap();
        (storage, rewards, alive, obs)
    }

    #[test]
    fn test_storage_regions_disjoint_aligned_and_zeroed() {
        let (s, r, f, o) = storage();
        assert!(r.iter().all(|x| x.to_bits() == 0));
        assert!(f.iter().all(|x| !x.get()));
        assert!(o.iter().all(|x| *x == 0));
        let rr = s.region(OutputField::Rewards).unwrap();
        let fr = s.region(OutputField::Alive).unwrap();
        let or = s.region(OutputField::Observation(0)).unwrap();
        assert_eq!((rr.len, rr.dtype), (5, OutputDtype::F32));
        for reg in [rr, fr, or] {
            assert_eq!(reg.ptr.align_offset(8), 0);
        }
        assert!(rr.ptr.wrapping_add(20) <= fr.ptr);
        assert!(fr.ptr.wrapping_add(3) <= or.ptr);
        assert_eq!(s.region(OutputField::Masks), None);
    }

    #[test]
    fn test_storage_writes_visible_through_region_and_outlive_bufs() {
        let (s, mut r, mut f, mut o) = storage();
        s.begin_write();
        r.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        f.fill(Bool8::TRUE);
        o.fill(9);
        drop((r, f, o));
        assert_eq!(s.generation(), 1);
        let reg = s.region(OutputField::Rewards).unwrap();
        // SAFETY: the region is 5 aligned, initialized `f32`s of a live
        // storage, and no `Buf` (the only writer) exists any more.
        #[expect(unsafe_code, reason = "test of the audited module")]
        #[expect(clippy::cast_ptr_alignment, reason = "regions are 8-byte aligned")]
        let rewards = unsafe { core::slice::from_raw_parts(reg.ptr.cast::<f32>(), reg.len) };
        assert_eq!(rewards, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let reg = s.region(OutputField::Observation(0)).unwrap();
        // SAFETY: as above, 7 bytes.
        #[expect(unsafe_code, reason = "test of the audited module")]
        let obs = unsafe { core::slice::from_raw_parts(reg.ptr, reg.len) };
        assert_eq!(obs, &[9; 7]);
    }

    #[test]
    fn test_storage_reserve_overflow_returns_none() {
        let mut b = StorageBuilder::default();
        assert!(b.reserve::<f32>(OutputField::Rewards, usize::MAX).is_none());
    }
}

#[cfg(test)]
#[cfg(loom)]
mod loom_tests {
    use super::*;
    use crate::sys::sync::thread;

    /// `storage_generation`: a reader that sees a generation also sees
    /// every write made before the matching `begin_write` was published by
    /// the next one, and the generation never goes backwards.
    #[test]
    fn test_storage_generation_monotonic_across_threads() {
        loom::model(|| {
            let mut b = StorageBuilder::default();
            let _ = b.reserve::<u32>(OutputField::Generations, 1);
            let s = b.finish().unwrap();
            let reader = {
                let s = Arc::clone(&s);
                thread::spawn(move || {
                    let a = s.generation();
                    let b = s.generation();
                    assert!(a <= b);
                    b
                })
            };
            s.begin_write();
            s.begin_write();
            let seen = reader.join().unwrap();
            assert!(seen <= 2);
            assert_eq!(s.generation(), 2);
        });
    }
}
