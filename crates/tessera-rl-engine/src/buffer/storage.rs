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
//! 2. Regions are disjoint: offsets come from one running sum. The
//!    allocation starts at an [`OUTPUT_ALIGN`] (16 KiB) boundary;
//!    observation leaves start at [`OUTPUT_ALIGN`] and every other region at
//!    [`OUTPUT_MIN_ALIGN`] (64 bytes) (SPEC 5.2), which covers every
//!    [`Element`] alignment. Padding between regions stays zero and is never
//!    part of a region.
//! 3. Exactly one [`Buf`] exists per region: `Buf` is not `Clone` and is
//!    only created by [`Regions::take`], which checks the slot's builder,
//!    dtype and bounds and hands each region out once, so `&mut Buf` is the only
//!    way to get a `&mut` slice of its region.
//! 4. The engine increments the generation before any write to the
//!    outputs ([`Buf`] slices are written only through
//!    `Outputs::view_all`, which calls [`OutputStorage::begin_write`]).
//!    Borrowers may read a region only between engine calls, which the
//!    caller serializes (the borrow window of ADR-0003); comparing the
//!    generation tells a borrower whether its view is stale. It is a
//!    diagnostic, not a synchronization protocol: it does not make reads
//!    that overlap an engine call sound.

use core::alloc::Layout;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use std::sync::Arc;

use tessera_rl_core::Bool8;

use crate::error::EngineError;
use crate::sys::sync::{AtomicU64, Ordering};

/// Alignment of the output allocation and of the observation leaves: 16
/// KiB, a multiple of the page size on every supported platform (4 KiB on
/// x86-64, 16 KiB on Apple silicon). A constant, so layouts are identical
/// everywhere (SPEC 5.2).
pub const OUTPUT_ALIGN: usize = 16 * 1024;

/// Alignment of every other output region: a cache line (SPEC 5.2). Padding
/// those small regions to a page would multiply the memory of small pools,
/// such as the scratch pools of search trees (ADR-0045 bound, #355).
pub const OUTPUT_MIN_ALIGN: usize = 64;

/// Start alignment of `field`'s region: [`OUTPUT_ALIGN`] for observation
/// and terminal observation leaves, [`OUTPUT_MIN_ALIGN`] for the rest.
#[must_use]
const fn region_align(field: OutputField) -> usize {
    match field {
        OutputField::Observation(_) | OutputField::TerminalObservation(_) => OUTPUT_ALIGN,
        _ => OUTPUT_MIN_ALIGN,
    }
}

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
    /// Host pools only: aborted episodes, `[N]` (ADR-0065).
    Aborted,
    /// Host pools only: abort codes, `[N]`, 0 when not aborted (ADR-0065).
    AbortCode,
    /// Host pools only: instances pending a reset, `[N]` (ADR-0065).
    PendingReset,
    /// Refusing host pools only: refused agents, `[N, A]` (ADR-0070).
    Refused,
    /// Refusing host pools only: refuse codes, `[N, A]`, 0 when not refused
    /// (ADR-0070).
    RefuseCode,
    /// Privileged state leaf `n`, `[N, leaf bytes]`: one row per instance,
    /// only in a pool that opted in (ADR-0079 item 7).
    PrivilegedState(usize),
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
    align: usize,
    len: usize,
    dtype: OutputDtype,
}

/// The single allocation holding every output buffer of a pool.
pub struct OutputStorage {
    /// Start of the zeroed allocation of `layout`, owned by this storage.
    base: NonNull<u8>,
    layout: Layout,
    bytes: usize,
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

impl Drop for OutputStorage {
    fn drop(&mut self) {
        // SAFETY: `base` was returned by `alloc_zeroed(self.layout)` in
        // `StorageBuilder::finish` and is freed only here, once; no `Buf`
        // outlives the storage (each holds an `Arc` to it).
        #[expect(unsafe_code, reason = "audited module: shared output storage")]
        unsafe {
            std::alloc::dealloc(self.base.as_ptr(), self.layout);
        }
    }
}

// SAFETY: the bytes are only written through the unique `Buf` of each
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
    ///
    /// Scope: a staleness diagnostic for readers whose reads are already
    /// serialized with engine calls (the borrow windows of ADR-0003). It is
    /// not a seqlock and not a permission to read: reading a region while an
    /// engine call runs is a data race, and checking the generation after
    /// such a read does not make it sound.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Bytes from the start of the allocation to the end of the last
    /// region (the allocation itself is this rounded up to
    /// [`OUTPUT_ALIGN`], at least one page).
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The region of `field`, if the pool has it. Rows of a poisoned pool have
    /// no defined contents.
    #[must_use]
    pub fn region(&self, field: OutputField) -> Option<OutputRegion> {
        let e = self.entries.iter().find(|e| e.field == field)?;
        let base = self.base.as_ptr();
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
/// It names its builder, so it cannot be redeemed against another storage.
#[derive(Debug)]
pub(crate) struct Slot<T> {
    builder: u64,
    index: usize,
    _t: core::marker::PhantomData<T>,
}

/// Distinct ids for builders (not modeled by loom: identity only).
static BUILDERS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Lays out the regions of an [`OutputStorage`].
#[derive(Debug)]
pub(crate) struct StorageBuilder {
    id: u64,
    entries: Vec<Entry>,
    bytes: usize,
}

/// Takes the next id from `counter`, refusing to wrap.
fn next_id(counter: &core::sync::atomic::AtomicU64) -> Result<u64, EngineError> {
    use core::sync::atomic::Ordering::Relaxed;
    // A CAS loop rather than `fetch_update`/`try_update`, which differ
    // between the MSRV and the current toolchain.
    let mut id = counter.load(Relaxed);
    loop {
        let next = id.checked_add(1).ok_or(EngineError::StorageIdsExhausted)?;
        match counter.compare_exchange_weak(id, next, Relaxed, Relaxed) {
            Ok(_) => return Ok(id),
            Err(seen) => id = seen,
        }
    }
}

impl StorageBuilder {
    /// A builder with a fresh id; ids are distinct for the process lifetime.
    pub(crate) fn new() -> Result<Self, EngineError> {
        Ok(Self {
            id: next_id(&BUILDERS)?,
            entries: Vec::new(),
            bytes: 0,
        })
    }

    /// Reserves `len` elements of `T` for `field`; `None` on overflow.
    pub(crate) fn reserve<T: Element>(
        &mut self,
        field: OutputField,
        len: usize,
    ) -> Option<Slot<T>> {
        let align = region_align(field);
        let offset = self.bytes.checked_next_multiple_of(align)?;
        let size = len.checked_mul(core::mem::size_of::<T>())?;
        self.bytes = offset.checked_add(size)?;
        self.entries.push(Entry {
            field,
            offset,
            align,
            len,
            dtype: T::DTYPE,
        });
        Some(Slot {
            builder: self.id,
            index: self.entries.len().checked_sub(1)?,
            _t: core::marker::PhantomData,
        })
    }

    /// Allocates the zeroed storage, [`OUTPUT_ALIGN`]-aligned and at least
    /// one page long. Its regions are handed out only by the returned
    /// [`Regions`], once each. `None` on overflow or when the allocator
    /// returns null (never `handle_alloc_error`).
    pub(crate) fn finish(self) -> Option<Regions> {
        let size = self
            .bytes
            .checked_next_multiple_of(OUTPUT_ALIGN)?
            .max(OUTPUT_ALIGN);
        let layout = Layout::from_size_align(size, OUTPUT_ALIGN).ok()?;
        // SAFETY: `layout` has a non-zero size (at least one page).
        #[expect(unsafe_code, reason = "audited module: shared output storage")]
        let base = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })?;
        let taken = vec![false; self.entries.len()];
        Some(Regions {
            builder: self.id,
            storage: Arc::new(OutputStorage {
                base,
                layout,
                bytes: self.bytes,
                entries: self.entries,
                generation: AtomicU64::new(0),
            }),
            taken,
        })
    }
}

/// The only source of [`Buf`]s: one per region of its storage, at most once
/// (invariant 3).
#[derive(Debug)]
pub(crate) struct Regions {
    builder: u64,
    storage: Arc<OutputStorage>,
    taken: Vec<bool>,
}

impl Regions {
    /// The storage being handed out.
    #[cfg(test)]
    pub(crate) fn storage(&self) -> &Arc<OutputStorage> {
        &self.storage
    }

    /// The unique handle of `slot`; `None` if the slot is from another
    /// builder, already taken, or does not match its entry's dtype and
    /// bounds (checked in every build, not only in debug).
    #[expect(
        clippy::needless_pass_by_value,
        reason = "consuming the slot documents that each region is taken once"
    )]
    pub(crate) fn take<T: Element>(&mut self, slot: Slot<T>) -> Option<Buf<T>> {
        if slot.builder != self.builder {
            return None;
        }
        let taken = self.taken.get_mut(slot.index)?;
        let e = self.storage.entries.get(slot.index)?;
        let end = e
            .len
            .checked_mul(core::mem::size_of::<T>())?
            .checked_add(e.offset)?;
        if *taken || e.dtype != T::DTYPE || end > self.storage.bytes || e.offset % e.align != 0 {
            return None;
        }
        *taken = true;
        let ptr = NonNull::new(
            self.storage
                .base
                .as_ptr()
                .wrapping_add(e.offset)
                .cast::<T>(),
        )?;
        Some(Buf {
            storage: Arc::clone(&self.storage),
            ptr,
            len: e.len,
        })
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

impl<T> Buf<T> {
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
        // (invariant 3), and `ptr` derives from the allocation's raw
        // pointer, never from a shared reference to its bytes, so writing
        // through it while the `Arc` is shared is allowed.
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
        let mut builder = StorageBuilder::new().unwrap();
        let rewards = builder.reserve::<f32>(OutputField::Rewards, 5).unwrap();
        let alive = builder.reserve::<Bool8>(OutputField::Alive, 3).unwrap();
        let obs = builder
            .reserve::<u8>(OutputField::Observation(0), 7)
            .unwrap();
        let mut regions = builder.finish().unwrap();
        let rewards = regions.take(rewards).unwrap();
        let alive = regions.take(alive).unwrap();
        let obs = regions.take(obs).unwrap();
        (Arc::clone(regions.storage()), rewards, alive, obs)
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
        assert_eq!(
            rr.ptr.align_offset(OUTPUT_ALIGN),
            0,
            "first region at the base"
        );
        assert_eq!(fr.ptr.align_offset(OUTPUT_MIN_ALIGN), 0);
        assert_eq!(or.ptr.align_offset(OUTPUT_ALIGN), 0);
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
        #[expect(clippy::cast_ptr_alignment, reason = "regions are page-aligned")]
        let rewards = unsafe { core::slice::from_raw_parts(reg.ptr.cast::<f32>(), reg.len) };
        assert_eq!(rewards, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let reg = s.region(OutputField::Observation(0)).unwrap();
        // SAFETY: as above, 7 bytes.
        #[expect(unsafe_code, reason = "test of the audited module")]
        let obs = unsafe { core::slice::from_raw_parts(reg.ptr, reg.len) };
        assert_eq!(obs, &[9; 7]);
    }

    #[test]
    fn test_storage_slot_of_another_builder_or_taken_twice_is_refused() {
        let mut a = StorageBuilder::new().unwrap();
        let mut b = StorageBuilder::new().unwrap();
        let from_b = b.reserve::<f32>(OutputField::Rewards, 1 << 20).unwrap();
        let first = a.reserve::<f32>(OutputField::Rewards, 1).unwrap();
        let mut regions = a.finish().unwrap();
        // Same index 0, but another builder (and out of this storage's
        // bounds): refused, not an out-of-bounds view.
        assert!(regions.take(from_b).is_none());
        assert!(regions.take(first).is_some());
        let again = Slot::<f32> {
            builder: regions.builder,
            index: 0,
            _t: core::marker::PhantomData,
        };
        assert!(regions.take(again).is_none());
        let wrong_dtype = Slot::<u32> {
            builder: regions.builder,
            index: 0,
            _t: core::marker::PhantomData,
        };
        assert!(regions.take(wrong_dtype).is_none());
    }

    #[test]
    fn test_storage_id_counter_near_limit_fails_instead_of_wrapping() {
        let counter = core::sync::atomic::AtomicU64::new(u64::MAX - 1);
        assert_eq!(next_id(&counter), Ok(u64::MAX - 1));
        assert_eq!(next_id(&counter), Err(EngineError::StorageIdsExhausted));
        assert_eq!(next_id(&counter), Err(EngineError::StorageIdsExhausted));
        assert_eq!(
            counter.load(core::sync::atomic::Ordering::Relaxed),
            u64::MAX
        );
    }

    #[test]
    fn test_output_storage_empty_layout_allocates_one_page_ok() {
        let regions = StorageBuilder::new().unwrap().finish().unwrap();
        let s = regions.storage();
        assert_eq!(s.bytes(), 0);
        assert_eq!(s.layout.size(), OUTPUT_ALIGN);
        assert_eq!(s.base.as_ptr().align_offset(OUTPUT_ALIGN), 0);
    }

    #[test]
    fn test_output_storage_region_align_observations_page_rest_cache_line() {
        for f in [
            OutputField::Observation(0),
            OutputField::Observation(2),
            OutputField::TerminalObservation(0),
        ] {
            assert_eq!(region_align(f), OUTPUT_ALIGN, "{f:?}");
        }
        for f in [
            OutputField::Rewards,
            OutputField::Masks,
            OutputField::AbortCode,
            OutputField::PendingReset,
        ] {
            assert_eq!(region_align(f), OUTPUT_MIN_ALIGN, "{f:?}");
        }
    }

    #[test]
    fn test_output_storage_padding_zero_and_unexposed_true() {
        let (s, mut rewards, mut alive, mut obs) = storage();
        s.begin_write();
        rewards.fill(f32::from_bits(u32::MAX));
        alive.fill(Bool8::TRUE);
        obs.fill(0xFF);
        drop((rewards, alive, obs));
        let mut inside = vec![false; s.layout.size()];
        for field in [
            OutputField::Rewards,
            OutputField::Alive,
            OutputField::Observation(0),
        ] {
            let entry = s.entries.iter().find(|e| e.field == field).unwrap();
            let size = match entry.dtype {
                OutputDtype::F32 | OutputDtype::U32 => 4,
                OutputDtype::U8 | OutputDtype::Bool8 => 1,
            };
            let end = entry.offset + entry.len * size;
            for byte in &mut inside[entry.offset..end] {
                assert!(!*byte, "regions overlap");
                *byte = true;
            }
            // A region never reaches the start of the next one.
            if let Some(next) = s.entries.iter().find(|n| n.offset > entry.offset) {
                assert!(end <= next.offset);
            }
        }
        // SAFETY: the whole allocation is live and initialized (zeroed,
        // then written only through the regions), and no `Buf` exists.
        #[expect(unsafe_code, reason = "test of the audited module")]
        let all = unsafe { core::slice::from_raw_parts(s.base.as_ptr(), s.layout.size()) };
        for (i, byte) in all.iter().enumerate() {
            if !inside[i] {
                assert_eq!(*byte, 0, "padding byte {i} was written");
            }
        }
    }

    /// The privileged state regions (ADR-0079 item 7) are ordinary `u8`
    /// regions laid out after the others: zeroed, cache-line aligned,
    /// disjoint from the observation leaf before them, and written only
    /// through their own `Buf`.
    #[test]
    fn test_storage_privileged_regions_disjoint_zeroed_and_writable() {
        let mut builder = StorageBuilder::new().unwrap();
        let obs = builder
            .reserve::<u8>(OutputField::Observation(0), 6)
            .unwrap();
        let first = builder
            .reserve::<u8>(OutputField::PrivilegedState(0), 9)
            .unwrap();
        let second = builder
            .reserve::<u8>(OutputField::PrivilegedState(1), 3)
            .unwrap();
        let mut regions = builder.finish().unwrap();
        let s = Arc::clone(regions.storage());
        let mut obs = regions.take(obs).unwrap();
        let mut first = regions.take(first).unwrap();
        let mut second = regions.take(second).unwrap();
        assert!(first.iter().chain(second.iter()).all(|b| *b == 0));
        assert_eq!(
            region_align(OutputField::PrivilegedState(0)),
            OUTPUT_MIN_ALIGN
        );
        s.begin_write();
        obs.fill(1);
        first.fill(2);
        second.fill(3);
        drop((obs, first, second));
        let o = s.region(OutputField::Observation(0)).unwrap();
        let p0 = s.region(OutputField::PrivilegedState(0)).unwrap();
        let p1 = s.region(OutputField::PrivilegedState(1)).unwrap();
        assert_eq!((p0.len, p0.dtype), (9, OutputDtype::U8));
        assert_eq!(p0.ptr.align_offset(OUTPUT_MIN_ALIGN), 0);
        assert!(o.ptr.wrapping_add(o.len) <= p0.ptr);
        assert!(p0.ptr.wrapping_add(p0.len) <= p1.ptr);
        // SAFETY: each region is initialized bytes of a live storage, and
        // no `Buf` (the only writer) exists any more.
        #[expect(unsafe_code, reason = "test of the audited module")]
        let (ob, b0, b1) = unsafe {
            (
                core::slice::from_raw_parts(o.ptr, o.len),
                core::slice::from_raw_parts(p0.ptr, p0.len),
                core::slice::from_raw_parts(p1.ptr, p1.len),
            )
        };
        assert_eq!(ob, &[1; 6]);
        assert_eq!(b0, &[2; 9]);
        assert_eq!(b1, &[3; 3]);
        assert_eq!(s.region(OutputField::PrivilegedState(2)), None);
    }

    #[test]
    fn test_storage_reserve_overflow_returns_none() {
        let mut b = StorageBuilder::new().unwrap();
        assert!(b.reserve::<f32>(OutputField::Rewards, usize::MAX).is_none());
    }
}

#[cfg(test)]
#[cfg(loom)]
mod loom_tests {
    use super::*;
    use crate::sys::sync::thread;

    /// The generation never goes backwards as seen from another thread,
    /// and ends at the number of writes. It proves monotonicity only, not
    /// publication of the region payload (readers are serialized with the
    /// engine by the caller, see `OutputStorage::generation`).
    #[test]
    fn test_storage_generation_is_monotonic_across_threads() {
        loom::model(|| {
            let mut b = StorageBuilder::new().unwrap();
            let _ = b.reserve::<u32>(OutputField::Generations, 1);
            let s = Arc::clone(b.finish().unwrap().storage());
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
