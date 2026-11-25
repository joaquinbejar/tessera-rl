// SPDX-License-Identifier: MIT OR Apache-2.0

//! Disjoint chunk views shared with worker threads (ADR-0002, ADR-0005).
//! Audited `unsafe` module.
//!
//! Sync mode splits a pool-wide [`Rows`] into its raw parts and hands each
//! worker `&mut` views of *its own* chunks only. The split-phase modes
//! (ADR-0047) also build views of a Pipelined half or of one instance
//! ([`RawChunk::range_view`], [`RawChunk::instance_view`]): a caller views
//! only the rows a live token covers, while workers write the rows of an
//! admitted job or claim.
//!
//! # Invariants
//!
//! 1. The buffers, and in the split-phase modes the shared context that
//!    holds them, never move or reallocate while any job or claim is
//!    outstanding: the pool never pushes to or resizes them, so pointers
//!    taken when a job is published stay valid until it completes. The
//!    pointers are to the heap data (record arrays, and the shared output
//!    storage of [`storage`]), not to the pool, so moving the pool, or a
//!    mode wrapper, while no reference is live is fine.
//! 2. Chunk ranges are disjoint by construction: chunk `c` covers instances
//!    `c * size .. min((c + 1) * size, n)`, and each chunk of a job is run
//!    by exactly one worker, so no two workers, and no two chunks of one
//!    worker at a time, ever view the same element (checked in debug builds
//!    and by tests). Which worker runs a chunk depends on the schedule
//!    (ADR-0076):
//!    - `Static`: worker `w` takes the chunks `c` with `c % workers == w`.
//!    - `Dynamic`: workers take chunk indices from one counter per job with
//!      `fetch_add`; an atomic read-modify-write hands out each value once,
//!      so `Relaxed` is enough for uniqueness. The counter lives in the
//!      job's frame (or, for a Pipelined half, in that half's work record,
//!      reset before each publication), and outlives every use because the
//!      job is waited for before the frame ends or the half is published
//!      again. Visibility of chunk writes to the caller is ordered by the
//!      job's completion, as under `Static`.
//! 3. The caller forms references only to rows covered by a live token,
//!    and workers only to the rows of an admitted job or claim. In Sync
//!    mode this is the whole-epoch rule: `par_chunks` takes the pool-wide
//!    chunk by value (borrowing the pool mutably through `&mut self` of
//!    every public entry point) and returns only after the gang's `run`
//!    returns, so the caller touches nothing between the epoch's Release
//!    and its Acquire of `pending == 0`.
//! 4. A row belongs to at most one live token or one admitted job or claim
//!    at a time. Admission is ordered after the end of the token that
//!    covered the row (the token's live flag, cleared with Release and
//!    checked with Acquire), and the workers' writes before completion are
//!    ordered before the next token is created (Release on completion,
//!    Acquire in `recv`). In Sync mode no reference survives the epoch:
//!    worker views are created and dropped inside the job, and the caller's
//!    output views are returned only after `run`.

#[cfg(not(loom))]
pub(crate) mod batch;
#[cfg_attr(
    loom,
    allow(dead_code, reason = "the split-phase modes are not built under loom")
)]
mod generations;
pub(crate) mod live;
#[cfg(all(test, loom))]
mod live_model;
#[cfg(not(loom))]
pub(crate) mod pipeline;
pub(crate) mod rows;
mod storage;

use core::ops::Range;

use tessera_rl_core::{Bool8, LaneTuple};

use crate::error::EngineError;
use crate::layout::{MAX_LEAVES, SetSpans};
use crate::options::Schedule;
use crate::outputs::OutputsView;
use crate::pool::Rows;
use crate::sys::Gang;
use crate::sys::sync::{AtomicBool, AtomicUsize, Ordering};
use crate::transform::kernels::StackMode;
use crate::transform::rows::{Shared, TransformRows};
use crate::transform::stats::Moments;

use generations::HALVES;
#[cfg(not(loom))]
pub(crate) use generations::{GenerationError, Generations};
pub(crate) use storage::{Buf, StorageBuilder};
pub use storage::{
    OUTPUT_ALIGN, OUTPUT_MIN_ALIGN, OutputDtype, OutputField, OutputRegion, OutputStorage,
};

struct Raw<T> {
    ptr: *mut T,
    len: usize,
}

// A pointer and a length: copying them copies no `T`, so no `T: Copy`
// bound (the host pool's instances are not `Copy`, ADR-0062).
impl<T> Clone for Raw<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Raw<T> {}

impl<T> Raw<T> {
    fn new(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    /// `per` elements per instance; an empty buffer (unused in this mode)
    /// yields empty views.
    ///
    /// # Safety
    ///
    /// The module invariants: the buffer is alive and unaliased for the
    /// epoch, and `instances` is a range no other live view covers.
    #[expect(unsafe_code, reason = "audited module: disjoint chunk views")]
    unsafe fn view<'a>(self, instances: &Range<usize>, per: usize) -> &'a mut [T] {
        // Products stay below the buffer length for valid ranges; an
        // overflow can only come from an out-of-range request, which is
        // clamped to the end of the buffer (an empty view).
        let start = instances
            .start
            .checked_mul(per)
            .map_or(self.len, |s| s.min(self.len));
        let end = instances
            .end
            .checked_mul(per)
            .map_or(self.len, |e| e.min(self.len))
            .max(start);
        // SAFETY: `start <= len`, so the offset stays inside (or one past)
        // the live allocation (invariant 1).
        let first = unsafe { self.ptr.add(start) };
        // SAFETY: `start <= end <= len` keeps the view inside the buffer,
        // which is live and never reallocated during the epoch (invariant
        // 1); the caller guarantees no other view overlaps it (invariant 2).
        unsafe { core::slice::from_raw_parts_mut(first, end - start) }
    }
}

/// The raw parts of a pool-wide chunk of headers `H` and states `S`, and
/// of the columns of a lane-wise state's field tuple `T` (ADR-0090).
struct RawChunk<H, S, T: LaneTuple = ()> {
    instances: usize,
    agents: usize,
    dh: usize,
    ch: usize,
    mask_len: usize,
    leaf_bytes: Raw<usize>,
    sets: SetSpans,
    headers: Raw<H>,
    states: Raw<S>,
    columns: T::Raw,
    discrete: Raw<i32>,
    continuous: Raw<f32>,
    penalties: Raw<f32>,
    penalized: Raw<Bool8>,
    observations: [Raw<u8>; MAX_LEAVES],
    terminal: [Raw<u8>; MAX_LEAVES],
    rewards: Raw<f32>,
    agent_terminated: Raw<Bool8>,
    agent_truncated: Raw<Bool8>,
    died: Raw<Bool8>,
    spawned: Raw<Bool8>,
    invalid_agent_event: Raw<Bool8>,
    alive: Raw<Bool8>,
    acting: Raw<Bool8>,
    generations: Raw<u32>,
    episode_terminated: Raw<Bool8>,
    episode_truncated: Raw<Bool8>,
    episode_return: Raw<f32>,
    episode_length: Raw<f32>,
    episode_illegal_actions: Raw<f32>,
    episode_invalid_agent_events: Raw<f32>,
    masks: Raw<Bool8>,
    aborted: Raw<Bool8>,
    abort_code: Raw<u32>,
    pending_reset: Raw<Bool8>,
    refused: Raw<Bool8>,
    refuse_code: Raw<u32>,
    transform: RawTransform,
    /// One partial per instance (ADR-0081 item 3); empty when frozen.
    episode_partials: Raw<crate::adaptive::EpisodePartial>,
    /// One abort count per instance of a host pool; empty otherwise.
    episode_aborts: Raw<u64>,
    league_outcomes: Raw<i8>,
    league_slots: u64,
}

/// The raw transform parts of a pool-wide view (ADR-0078).
#[derive(Clone, Copy)]
struct RawTransform {
    /// The pool's shared transform parts, or null without transforms.
    shared: *const Shared,
    /// `instance_bytes` per instance.
    arena: Raw<u8>,
    /// `normalized` moments per chunk; empty when the pass does not
    /// accumulate.
    accum: Raw<Moments>,
    /// `scratch_bytes` per worker.
    scratch: Raw<u8>,
    mode: StackMode,
}

/// The raw parts of an `Environment` pool's chunk, for the split-phase
/// modes.
#[cfg(not(loom))]
type EnvRawChunk<E> = RawChunk<
    <E as tessera_rl_core::Environment>::Header,
    <E as tessera_rl_core::Environment>::State,
    crate::pool::Fields<E>,
>;

// SAFETY: a `RawChunk` is shared by workers only to build disjoint views
// (invariant 2), each of which is used by one thread at a time; the
// elements are `Send` (`H` and `S` are required `Send` here; the columns'
// elements are `LaneField`s, which are `Send`; the others are plain
// numbers), so handing out `&mut` views of disjoint ranges to other threads
// is sound.
#[expect(unsafe_code, reason = "audited module: shared raw chunk")]
unsafe impl<H: Send, S: Send, T: LaneTuple> Sync for RawChunk<H, S, T> {}

impl<H, S, T: LaneTuple> RawChunk<H, S, T> {
    #[expect(
        clippy::too_many_lines,
        reason = "one raw view per output buffer, in field order"
    )]
    fn new(chunk: Rows<'_, H, S, T>, dh: usize, ch: usize) -> Self {
        let Rows {
            headers,
            states,
            mut columns,
            discrete,
            continuous,
            penalties,
            penalized,
            outputs,
        } = chunk;
        let OutputsView {
            agents,
            mask_len,
            leaf_bytes,
            sets,
            observations,
            terminal_observations,
            rewards,
            agent_terminated,
            agent_truncated,
            died,
            spawned,
            invalid_agent_event,
            alive,
            acting,
            generations,
            episode_terminated,
            episode_truncated,
            episode_return,
            episode_length,
            episode_illegal_actions,
            episode_invalid_agent_events,
            masks,
            aborted,
            abort_code,
            pending_reset,
            refused,
            refuse_code,
            transform,
            episode_partials,
            episode_aborts,
            league_outcomes,
            league_slots,
            base: _,
        } = outputs;
        let leaves = |views: [&mut [u8]; MAX_LEAVES]| views.map(Raw::new);
        // `leaf_bytes` is only read; a read-only view is stored as raw too.
        let leaf_bytes = Raw {
            ptr: leaf_bytes.as_ptr().cast_mut(),
            len: leaf_bytes.len(),
        };
        Self {
            instances: headers.len(),
            agents,
            dh,
            ch,
            mask_len,
            leaf_bytes,
            sets,
            headers: Raw::new(headers),
            states: Raw::new(states),
            columns: T::view_raw(&mut columns),
            discrete: Raw::new(discrete),
            continuous: Raw::new(continuous),
            penalties: Raw::new(penalties),
            penalized: Raw::new(penalized),
            observations: leaves(observations),
            terminal: leaves(terminal_observations),
            rewards: Raw::new(rewards),
            agent_terminated: Raw::new(agent_terminated),
            agent_truncated: Raw::new(agent_truncated),
            died: Raw::new(died),
            spawned: Raw::new(spawned),
            invalid_agent_event: Raw::new(invalid_agent_event),
            alive: Raw::new(alive),
            acting: Raw::new(acting),
            generations: Raw::new(generations),
            episode_terminated: Raw::new(episode_terminated),
            episode_truncated: Raw::new(episode_truncated),
            episode_return: Raw::new(episode_return),
            episode_length: Raw::new(episode_length),
            episode_illegal_actions: Raw::new(episode_illegal_actions),
            episode_invalid_agent_events: Raw::new(episode_invalid_agent_events),
            masks: Raw::new(masks),
            aborted: Raw::new(aborted),
            abort_code: Raw::new(abort_code),
            pending_reset: Raw::new(pending_reset),
            refused: Raw::new(refused),
            refuse_code: Raw::new(refuse_code),
            transform: RawTransform {
                shared: transform
                    .shared
                    .map_or(core::ptr::null(), core::ptr::from_ref),
                arena: Raw::new(transform.arena),
                accum: Raw::new(transform.accum),
                scratch: Raw::new(transform.scratch),
                mode: transform.mode,
            },
            episode_partials: Raw::new(episode_partials),
            episode_aborts: Raw::new(episode_aborts),
            league_outcomes: Raw::new(league_outcomes),
            league_slots,
        }
    }

    /// The views of the instances in `r`.
    ///
    /// # Safety
    ///
    /// Module invariants; in particular no other live view covers `r`.
    #[cfg(test)]
    #[expect(unsafe_code, reason = "audited module: disjoint chunk views")]
    unsafe fn chunk<'a>(&self, r: &Range<usize>) -> Rows<'a, H, S, T> {
        // SAFETY: forwarded; no worker scratch is handed out.
        unsafe { self.chunk_on(r, None) }
    }

    /// [`Self::chunk`] run by `worker`, which gets its transform scratch.
    ///
    /// # Safety
    ///
    /// As [`Self::chunk`]; besides, `worker`'s scratch is used by no other
    /// live view (each worker runs one view at a time).
    #[expect(unsafe_code, reason = "audited module: disjoint chunk views")]
    unsafe fn chunk_on<'a>(&self, r: &Range<usize>, worker: Option<usize>) -> Rows<'a, H, S, T> {
        let a = self.agents;
        // SAFETY: `leaf_bytes` is never written during an epoch, so a shared
        // view of it alongside the workers' views is sound (invariant 1).
        let leaf_bytes: &'a [usize] =
            unsafe { core::slice::from_raw_parts(self.leaf_bytes.ptr, self.leaf_bytes.len) };
        let mut observations: [&mut [u8]; MAX_LEAVES] = Default::default();
        let mut terminal: [&mut [u8]; MAX_LEAVES] = Default::default();
        for (((view, term), raw), (raw_term, &bytes)) in observations
            .iter_mut()
            .zip(terminal.iter_mut())
            .zip(self.observations)
            .zip(self.terminal.into_iter().zip(leaf_bytes))
        {
            // SAFETY: invariants 1 to 4 for range `r` (one leaf row per agent).
            *view = unsafe { raw.view(r, a * bytes) };
            // SAFETY: as above, for the terminal buffer of the same leaf.
            *term = unsafe { raw_term.view(r, a * bytes) };
        }
        // SAFETY: invariants 1 to 4 for range `r` (1 elements per instance).
        let headers = unsafe { self.headers.view(r, 1) };
        // SAFETY: invariants 1 to 4 for range `r` (1 elements per instance).
        let states = unsafe { self.states.view(r, 1) };
        // SAFETY: `LaneTuple::view`'s preconditions, from this module's
        // invariants: disjoint ranges (invariants 2 to 4: no other live view
        // covers `r`), no reallocation (invariant 1: the columns live in the
        // pool and are never resized during the epoch), and bounds (`r` lies
        // within the instances, which the padded columns cover; the columns
        // are empty for an Array-of-Structs state, which yields empty views).
        let columns = unsafe { T::view(self.columns, r) };
        // SAFETY: invariants 1 to 4 for range `r` (a * self.dh elements per instance).
        let discrete = unsafe { self.discrete.view(r, a * self.dh) };
        // SAFETY: invariants 1 to 4 for range `r` (a * self.ch elements per instance).
        let continuous = unsafe { self.continuous.view(r, a * self.ch) };
        // SAFETY: invariants 1 to 4 for range `r` (a elements per instance).
        let penalties = unsafe { self.penalties.view(r, a) };
        // SAFETY: invariants 1 to 4 for range `r` (a elements per instance).
        let penalized = unsafe { self.penalized.view(r, a) };
        // SAFETY: invariants 1 to 4 for range `r` (a elements per instance).
        let rewards = unsafe { self.rewards.view(r, a) };
        // SAFETY: invariants 1 to 4 for range `r` (a elements per instance).
        let agent_terminated = unsafe { self.agent_terminated.view(r, a) };
        // SAFETY: invariants 1 to 4 for range `r` (a elements per instance).
        let agent_truncated = unsafe { self.agent_truncated.view(r, a) };
        // SAFETY: invariants 1 to 4 for range `r` (a elements per instance).
        let died = unsafe { self.died.view(r, a) };
        // SAFETY: invariants 1 to 4 for range `r` (a elements per instance).
        let spawned = unsafe { self.spawned.view(r, a) };
        // SAFETY: invariants 1 to 4 for range `r` (a elements per instance).
        let invalid_agent_event = unsafe { self.invalid_agent_event.view(r, a) };
        // SAFETY: invariants 1 to 4 for range `r` (a elements per instance).
        let alive = unsafe { self.alive.view(r, a) };
        // SAFETY: invariants 1 to 4 for range `r` (a elements per instance).
        let acting = unsafe { self.acting.view(r, a) };
        // SAFETY: invariants 1 to 4 for range `r` (a elements per instance).
        let generations = unsafe { self.generations.view(r, a) };
        // SAFETY: invariants 1 to 4 for range `r` (1 elements per instance).
        let episode_terminated = unsafe { self.episode_terminated.view(r, 1) };
        // SAFETY: invariants 1 to 4 for range `r` (1 elements per instance).
        let episode_truncated = unsafe { self.episode_truncated.view(r, 1) };
        // SAFETY: invariants 1 to 4 for range `r` (1 elements per instance).
        let episode_return = unsafe { self.episode_return.view(r, 1) };
        // SAFETY: invariants 1 to 4 for range `r` (1 elements per instance).
        let episode_length = unsafe { self.episode_length.view(r, 1) };
        // SAFETY: invariants 1 to 4 for range `r` (1 elements per instance).
        let episode_illegal_actions = unsafe { self.episode_illegal_actions.view(r, 1) };
        // SAFETY: invariants 1 to 4 for range `r` (1 elements per instance).
        let episode_invalid_agent_events = unsafe { self.episode_invalid_agent_events.view(r, 1) };
        // SAFETY: invariants 1 to 4 for range `r` (a * self.mask_len elements per instance).
        let masks = unsafe { self.masks.view(r, a * self.mask_len) };
        // SAFETY: invariants 1 to 4 for range `r` (1 element per instance;
        // an empty buffer outside host pools, which yields empty views).
        let aborted = unsafe { self.aborted.view(r, 1) };
        // SAFETY: as for `aborted`.
        let abort_code = unsafe { self.abort_code.view(r, 1) };
        // SAFETY: as for `aborted`.
        let pending_reset = unsafe { self.pending_reset.view(r, 1) };
        // SAFETY: invariants 1 to 4 for range `r` (a elements per instance;
        // an empty buffer outside refusing host pools, which yields empty
        // views).
        let refused = unsafe { self.refused.view(r, a) };
        // SAFETY: as for `refused`.
        let refuse_code = unsafe { self.refuse_code.view(r, a) };
        // SAFETY: the transform parts follow invariants 1 to 4 (see
        // `transform_view`).
        let transform = unsafe { self.transform_view(r, worker) };
        // SAFETY: invariants 1 to 4 for range `r` (1 element per instance;
        // an empty buffer when frozen or outside a step pass, which yields
        // empty views). Partials are per instance, so any disjoint ranges,
        // whole chunks or Async's claimed ids, give disjoint views.
        let episode_partials = unsafe { self.episode_partials.view(r, 1) };
        // SAFETY: as for `episode_partials` (empty outside host pools).
        let episode_aborts = unsafe { self.episode_aborts.view(r, 1) };
        // SAFETY: as for `episode_partials` (1 element per instance; empty
        // without a league).
        let league_outcomes = unsafe { self.league_outcomes.view(r, 1) };
        Rows {
            headers,
            states,
            columns,
            discrete,
            continuous,
            penalties,
            penalized,
            outputs: OutputsView {
                agents: a,
                mask_len: self.mask_len,
                leaf_bytes,
                sets: self.sets,
                observations,
                terminal_observations: terminal,
                rewards,
                agent_terminated,
                agent_truncated,
                died,
                spawned,
                invalid_agent_event,
                alive,
                acting,
                generations,
                episode_terminated,
                episode_truncated,
                episode_return,
                episode_length,
                episode_illegal_actions,
                episode_invalid_agent_events,
                masks,
                aborted,
                abort_code,
                pending_reset,
                refused,
                refuse_code,
                transform,
                episode_partials,
                episode_aborts,
                league_outcomes,
                league_slots: self.league_slots,
                base: r.start,
            },
        }
    }

    /// The transform parts of the view of `r` run by `worker`.
    ///
    /// # Safety
    ///
    /// As [`Self::chunk_on`]. Accumulators are per chunk: a pass that
    /// accumulates hands out views of whole chunks only (Sync, Inline and
    /// Pipelined; an Async pool never accumulates, ADR-0078 erratum 1), so
    /// no two live views share a chunk's accumulator.
    #[expect(unsafe_code, reason = "audited module: disjoint chunk views")]
    unsafe fn transform_view<'a>(
        &self,
        r: &Range<usize>,
        worker: Option<usize>,
    ) -> TransformRows<'a> {
        let t = self.transform;
        if t.shared.is_null() {
            return TransformRows::default();
        }
        // SAFETY: the shared parts live in the pool and are not written
        // during an epoch (invariant 1), so a shared view is sound.
        let shared: &'a Shared = unsafe { &*t.shared };
        let size = shared.chunk_size.max(1);
        let first = r.start / size;
        let chunks = accum_chunks(r, size, self.instances);
        // An accumulating pass hands out whole chunks only, so it never
        // skips accumulation silently (REVIEW-headA-r2 note).
        debug_assert!(
            t.accum.len == 0 || r.is_empty() || chunks.is_some(),
            "an accumulating view of {r:?} splits a chunk of {size}"
        );
        // SAFETY: invariants 1 to 4 for range `r` (`instance_bytes` per
        // instance).
        let arena = unsafe { t.arena.view(r, shared.instance_bytes) };
        // SAFETY: invariants 1 and 3. A view gets accumulators only when the
        // pass accumulates (`t.accum` is empty otherwise: frozen, Async,
        // re-observations) and `r` covers whole chunks (`accum_chunks`); in
        // that case no other live view covers any of those chunks, because
        // the accumulating passes (Sync, Inline, Pipelined) hand out disjoint
        // views of whole chunks. Any other range gets an empty view, so no two
        // live `&mut` slices of accumulators can overlap.
        let accum = match chunks {
            // SAFETY: as above.
            Some(c) => unsafe { t.accum.view(&c, shared.plan.normalized) },
            None => &mut [],
        };
        let scratch = match worker {
            // SAFETY: invariant 1; worker `w`'s scratch is used by its one
            // live view only.
            Some(w) => unsafe { t.scratch.view(&(w..w + 1), shared.scratch_bytes) },
            None => &mut [],
        };
        TransformRows {
            shared: Some(shared),
            arena,
            accum,
            base: r.start - first * size,
            scratch,
            mode: t.mode,
        }
    }
}

#[cfg_attr(
    loom,
    allow(dead_code, reason = "the split-phase modes are not built under loom")
)]
impl<H, S, T: LaneTuple> RawChunk<H, S, T> {
    /// The views of the instances in `r`, or `None` if `r` is reversed or
    /// reaches past the last instance. No slice covers a row outside `r`.
    ///
    /// # Safety
    ///
    /// Invariants 1, 3 and 4: every row of `r` is covered by a live token
    /// held by the caller (or belongs to the calling worker's admitted job
    /// or claim), so no other live view or write covers it, and the buffers
    /// stay alive and in place for `'a`.
    #[expect(unsafe_code, reason = "audited module: token-guarded views")]
    unsafe fn range_view<'a>(&self, r: Range<usize>) -> Option<Rows<'a, H, S, T>> {
        // SAFETY: forwarded; no worker scratch.
        unsafe { self.range_view_on(r, None) }
    }

    /// [`Self::range_view`] run by `worker` (its transform scratch).
    ///
    /// # Safety
    ///
    /// As [`Self::range_view`] and [`Self::chunk_on`].
    #[expect(unsafe_code, reason = "audited module: token-guarded views")]
    unsafe fn range_view_on<'a>(
        &self,
        r: Range<usize>,
        worker: Option<usize>,
    ) -> Option<Rows<'a, H, S, T>> {
        if r.start > r.end || r.end > self.instances {
            return None;
        }
        // SAFETY: `r` lies inside the instances, so every per-instance view
        // stays inside its buffer; the caller guarantees the rows are
        // exclusively its own for `'a` (invariants 1, 3 and 4).
        Some(unsafe { self.chunk_on(&r, worker) })
    }

    /// The views of instance `id` alone, or `None` if `id` is out of range.
    ///
    /// # Safety
    ///
    /// As [`Self::range_view`], for the one row `id`.
    #[expect(unsafe_code, reason = "audited module: token-guarded views")]
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "used by the Async mode (ADR-0047)")
    )]
    unsafe fn instance_view<'a>(&self, id: usize) -> Option<Rows<'a, H, S, T>> {
        let end = id.checked_add(1)?;
        // SAFETY: forwarded to the caller's guarantee for row `id`.
        unsafe { self.range_view(id..end) }
    }
}

/// The two Pipelined halves of `instances` with chunks of `size`
/// (ADR-0047 section 8): split at the chunk boundary nearest `instances / 2`,
/// ties to the lower boundary. `None` if either half would be empty, which
/// the mode reports as `HalfTooSmall`.
#[cfg_attr(
    loom,
    allow(dead_code, reason = "the split-phase modes are not built under loom")
)]
#[must_use]
pub(crate) fn half_ranges(
    instances: usize,
    size: core::num::NonZeroUsize,
) -> Option<[Range<usize>; HALVES]> {
    let size = size.get();
    let chunks = instances.div_ceil(size);
    let last = chunks.checked_sub(1).filter(|&k| k >= 1)?;
    // Interior boundaries are `k * size` for `k` in `1..chunks`. The nearest
    // to `instances / 2` is `k0 = instances / (2 * size)` or `k0 + 1`;
    // compare `|2 * boundary - instances|` to stay in integers.
    let k0 = instances / size.checked_mul(2)?;
    let distance = |k: usize| -> Option<usize> {
        Some(k.checked_mul(size)?.checked_mul(2)?.abs_diff(instances))
    };
    let low = k0.clamp(1, last);
    let high = k0.saturating_add(1).clamp(1, last);
    let k = if distance(high)? < distance(low)? {
        high
    } else {
        low
    };
    let split = k.checked_mul(size)?;
    Some([0..split, split..instances])
}

/// The chunks whose accumulators a view of `r` may hold: `Some` only when
/// `r` is non-empty and covers whole chunks of `size` (its end a chunk
/// boundary or the last instance), so that two disjoint such views never
/// share a chunk (ADR-0078, REVIEW-headA-r1).
pub(crate) fn accum_chunks(
    r: &Range<usize>,
    size: usize,
    instances: usize,
) -> Option<Range<usize>> {
    let size = size.max(1);
    let aligned = r.start < r.end
        && r.start.is_multiple_of(size)
        && (r.end.is_multiple_of(size) || r.end == instances);
    aligned.then(|| r.start / size..r.end.div_ceil(size))
}

/// The instances of chunk `c`.
pub(crate) fn chunk_range(c: usize, size: usize, instances: usize) -> Range<usize> {
    // Overflow means `c` is past the last chunk: an empty range at the end.
    let start = c.checked_mul(size).map_or(instances, |s| s.min(instances));
    start
        ..start
            .checked_add(size)
            .map_or(instances, |e| e.min(instances))
}

/// Runs `f` on every chunk of `size` instances of `all`, spread over the
/// gang's workers, `size` instances per chunk, under `schedule`
/// (invariant 2). `f` runs once per
/// chunk; a panic in it is caught per chunk and the other chunks still run.
/// Returns `true` if any chunk panicked. `dh` and `ch` are the discrete and
/// continuous heads per agent.
pub(crate) fn par_chunks<H, S, T, F>(
    gang: &mut Gang,
    all: Rows<'_, H, S, T>,
    (size, schedule): (usize, Schedule),
    heads: (usize, usize),
    f: F,
) -> Result<bool, EngineError>
where
    H: Send,
    S: Send,
    T: LaneTuple,
    F: Fn(&mut Rows<'_, H, S, T>, usize) + Sync,
{
    // The worker count comes from the gang itself, so the assignment below
    // is disjoint by construction (invariant 2).
    let workers = gang.workers();
    let raw = RawChunk::new(all, heads.0, heads.1);
    let chunks = raw.instances.div_ceil(size.max(1));
    let panicked = AtomicBool::new(false);
    // The Dynamic claim counter: in this frame, which outlives every worker's
    // use of it because `gang.run` returns only after the job completes.
    let next = AtomicUsize::new(0);
    let run = |c: usize, w: usize| {
        let range = chunk_range(c, size, raw.instances);
        // SAFETY: chunk `c` is run by this worker only (Static: `c % workers
        // == w`; Dynamic: the claim counter handed `c` out once), and chunk
        // ranges do not overlap (invariant 2); the pool is borrowed for the
        // whole epoch (invariant 3) and this view is dropped at the end of
        // the call (invariant 4).
        #[expect(unsafe_code, reason = "audited module: disjoint chunk views")]
        let mut chunk = unsafe { raw.chunk_on(&range, Some(w)) };
        let ok = crate::sys::catch(|| f(&mut chunk, range.start));
        if !ok {
            panicked.store(true, Ordering::Relaxed);
        }
    };
    let job = |w: usize| {
        if w >= workers {
            return;
        }
        match schedule {
            Schedule::Static => {
                for c in (w..chunks).step_by(workers) {
                    debug_assert!(c % workers == w, "chunk {c} is not worker {w}'s");
                    run(c, w);
                }
            }
            Schedule::Dynamic => loop {
                let c = next.fetch_add(1, Ordering::Relaxed);
                if c >= chunks {
                    break;
                }
                run(c, w);
            },
        }
    };
    let gang_panicked = gang.run(&job)?;
    Ok(gang_panicked || panicked.load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::io::{Actions, ObsWriter};
    use tessera_rl_core::space::Space;
    use tessera_rl_core::{
        AgentSlot, Bool8, EnvConfig, Environment, NoAdaptive, RecordHeader, SpaceError, StateInit,
        StepOut, Tier,
    };
    use tessera_rl_num::rng::{ObsRng, StepRng};

    use super::{Raw, RawChunk, chunk_range, half_ranges};

    fn nz(n: usize) -> core::num::NonZeroUsize {
        core::num::NonZeroUsize::new(n).unwrap()
    }
    use crate::layout::Leaf;
    use crate::outputs::Outputs;
    use crate::pool::Chunk;

    /// A minimal environment so a real `Chunk` can be built in unit tests.
    struct Probe;

    #[derive(Debug, Clone, Copy, Default, PartialEq)]
    #[repr(C)]
    struct ProbeState(u32);

    impl StateInit for ProbeState {
        fn state_default() -> Self {
            Self(0)
        }
    }

    impl tessera_rl_core::encoding::Canonical for ProbeState {
        const ENCODED_LEN: usize = 4;
        fn encode<S: tessera_rl_core::encoding::Sink>(
            &self,
            out: &mut tessera_rl_core::encoding::Encoder<S>,
        ) {
            self.0.encode(out);
        }
        fn decode(
            input: &mut tessera_rl_core::encoding::Decoder<'_>,
        ) -> Result<Self, tessera_rl_core::DecodeError> {
            u32::decode(input).map(Self)
        }
    }

    impl tessera_rl_core::EnvState for ProbeState {
        type Fields = ();
        fn to_fields(&self) -> Self::Fields {}
        fn from_fields((): Self::Fields) -> Self {
            <Self as tessera_rl_core::StateInit>::state_default()
        }
    }

    struct ProbeConfig;

    impl EnvConfig for ProbeConfig {}

    impl Environment for Probe {
        type State = ProbeState;
        type Config = ProbeConfig;
        type Obs = ();
        type Action = ();
        type Header = RecordHeader<2, NoAdaptive>;
        const MAX_AGENTS: usize = 2;
        const TIER: Tier = Tier::CrossPlatform;
        fn observation_space(_: &ProbeConfig) -> Result<Space, SpaceError> {
            Ok(Space::Discrete { n: 2 })
        }
        fn action_space(_: &ProbeConfig) -> Result<Space, SpaceError> {
            Ok(Space::Discrete { n: 2 })
        }
        fn reset(_: &ProbeConfig, _: &mut StepRng, _: &mut ProbeState) {}
        fn step(
            _: &ProbeConfig,
            _: &mut ProbeState,
            _: &Actions<'_, ()>,
            _: &mut StepRng,
            _: &mut StepOut<'_>,
        ) {
        }
        fn observe(
            _: &ProbeConfig,
            _: &ProbeState,
            _: AgentSlot,
            _: &mut ObsRng,
            _: &mut ObsWriter<'_, '_, ()>,
        ) {
        }
    }

    /// Two threads write through the views of disjoint, ragged chunks into
    /// every buffer family (run under Miri in the nightly job); every
    /// element ends up written by exactly its owner, and empty optional
    /// buffers give empty views.
    /// Steps 6 chunks of one instance on 2 workers. Chunk 0 waits until the
    /// other 5 are done, or `timeout`. Returns whether it timed out and the
    /// number of chunks run.
    #[cfg(not(loom))]
    fn chunk0_waits_for_the_rest(
        schedule: crate::options::Schedule,
        timeout: std::time::Duration,
    ) -> (bool, usize) {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        const N: usize = 6;
        let leaves = [Leaf { bytes: 1 }];
        let mut outputs = Outputs::new(N, 2, &leaves, false, 2).unwrap();
        let mut headers = vec![RecordHeader::<2, NoAdaptive>::state_default(); N];
        let mut states = vec![ProbeState(0); N];
        let mut discrete = vec![0i32; N * 2];
        let mut continuous: Vec<f32> = Vec::new();
        let mut penalties: Vec<f32> = Vec::new();
        let mut penalized: Vec<Bool8> = Vec::new();
        let all = Chunk::<Probe> {
            headers: &mut headers,
            states: &mut states,
            columns: (),
            discrete: &mut discrete,
            continuous: &mut continuous,
            penalties: &mut penalties,
            penalized: &mut penalized,
            outputs: outputs.view_all(),
        };
        let mut gang = crate::sys::Gang::new(&crate::sys::GangOptions {
            workers: nz(2),
            spin: 64,
            pin: None,
            native: None,
            faults: crate::sys::Faults::default(),
            adaptive_cap_ns: None,
        })
        .unwrap();
        let done = AtomicUsize::new(0);
        let ran = AtomicUsize::new(0);
        let timed_out = AtomicBool::new(false);
        let panicked = super::par_chunks(&mut gang, all, (1, schedule), (1, 0), |_, first| {
            ran.fetch_add(1, Ordering::Relaxed);
            if first == 0 {
                let start = std::time::Instant::now();
                while done.load(Ordering::Acquire) < N - 1 {
                    if start.elapsed() > timeout {
                        timed_out.store(true, Ordering::Relaxed);
                        return;
                    }
                    std::thread::yield_now();
                }
            } else {
                done.fetch_add(1, Ordering::Release);
            }
        })
        .unwrap();
        assert!(!panicked);
        (
            timed_out.load(Ordering::Relaxed),
            ran.load(Ordering::Relaxed),
        )
    }

    #[cfg(not(loom))]
    #[test]
    fn test_par_chunks_dynamic_lets_another_worker_take_a_blocked_workers_chunks() {
        use crate::options::Schedule;
        // Dynamic: whichever worker is not blocked in chunk 0 claims the rest.
        let (timed_out, ran) =
            chunk0_waits_for_the_rest(Schedule::Dynamic, std::time::Duration::from_secs(30));
        assert!(
            !timed_out,
            "Dynamic must not leave chunks behind a blocked worker"
        );
        assert_eq!(ran, 6);
        // Static: chunks 2 and 4 belong to chunk 0's worker, so the wait times
        // out (the assignment really differs, not only the outputs' timing).
        let (timed_out, ran) =
            chunk0_waits_for_the_rest(Schedule::Static, std::time::Duration::from_millis(300));
        assert!(
            timed_out,
            "Static runs chunk 0's worker's chunks only after it"
        );
        assert_eq!(ran, 6);
    }

    #[test]
    #[expect(unsafe_code, reason = "exercises the audited views")]
    fn test_raw_chunk_views_cover_every_buffer_from_two_threads() {
        const N: usize = 7;
        let leaves = [Leaf { bytes: 3 }, Leaf { bytes: 1 }];
        let mut outputs = Outputs::new(N, 2, &leaves, true, 2).unwrap();
        let mut headers = vec![RecordHeader::<2, NoAdaptive>::state_default(); N];
        let mut states = vec![ProbeState(0); N];
        let mut discrete = vec![0i32; N * 2];
        let mut continuous: Vec<f32> = Vec::new();
        let mut penalties: Vec<f32> = Vec::new();
        let mut penalized: Vec<Bool8> = Vec::new();
        let all = Chunk::<Probe> {
            headers: &mut headers,
            states: &mut states,
            columns: (),
            discrete: &mut discrete,
            continuous: &mut continuous,
            penalties: &mut penalties,
            penalized: &mut penalized,
            outputs: outputs.view_all(),
        };
        let raw = RawChunk::<<Probe as Environment>::Header, ProbeState>::new(all, 1, 0);
        // Size 3 over 7 instances: chunks 0..3, 3..6 and the ragged 6..7.
        std::thread::scope(|scope| {
            for w in 0..2usize {
                let raw = &raw;
                scope.spawn(move || {
                    for c in (w..3).step_by(2) {
                        let range = chunk_range(c, 3, N);
                        // SAFETY: worker `w` owns chunk `c`; chunks are
                        // disjoint and the buffers outlive the scope.
                        let chunk = unsafe { raw.chunk(&range) };
                        let mark = u32::try_from(w + 1).unwrap();
                        for s in chunk.states.iter_mut() {
                            s.0 = mark;
                        }
                        for h in chunk.headers.iter_mut() {
                            h.engine.episode_steps = mark;
                        }
                        chunk.discrete.fill(i32::try_from(mark).unwrap());
                        assert!(chunk.continuous.is_empty() && chunk.penalties.is_empty());
                        let out = chunk.outputs;
                        for leaf in out
                            .observations
                            .into_iter()
                            .chain(out.terminal_observations)
                        {
                            leaf.fill(u8::try_from(mark).unwrap());
                        }
                        #[expect(clippy::cast_precision_loss, reason = "tiny marks")]
                        let f = mark as f32;
                        out.rewards.fill(f);
                        out.episode_return.fill(f);
                        out.episode_length.fill(f);
                        out.episode_illegal_actions.fill(f);
                        out.episode_invalid_agent_events.fill(f);
                        out.generations.fill(mark);
                        for flags in [
                            out.agent_terminated,
                            out.agent_truncated,
                            out.died,
                            out.spawned,
                            out.invalid_agent_event,
                            out.alive,
                            out.acting,
                            out.episode_terminated,
                            out.episode_truncated,
                            out.masks,
                        ] {
                            flags.fill(Bool8::TRUE);
                        }
                    }
                });
            }
        });
        let owner = |i: usize| u32::try_from((i / 3) % 2 + 1).unwrap();
        for i in 0..N {
            assert_eq!(states[i].0, owner(i));
            assert_eq!(headers[i].engine.episode_steps, owner(i));
            assert_eq!(
                &discrete[i * 2..i * 2 + 2],
                &[i32::try_from(owner(i)).unwrap(); 2]
            );
            assert_eq!(&outputs.generations()[i * 2..i * 2 + 2], &[owner(i); 2]);
        }
        assert!(outputs.masks().iter().all(|m| m.get()));
        assert!(outputs.invalid_agent_event().iter().all(|m| m.get()));
        for i in 0..N {
            #[expect(clippy::cast_precision_loss, reason = "tiny marks")]
            let mark = owner(i) as f32;
            assert_eq!(outputs.episode_invalid_agent_events()[i], mark);
        }
        assert!(outputs.episode_truncated().iter().all(|m| m.get()));
        let first_leaf = outputs.observation(0).unwrap();
        for i in 0..N {
            let row = &first_leaf[i * 6..(i + 1) * 6];
            assert!(row.iter().all(|&b| u32::from(b) == owner(i)));
        }
    }

    /// Threadless check of the raw views (run under Miri in the nightly
    /// job): disjoint views of one buffer live at the same time, writes go
    /// to the right elements, and ranges are clamped to the buffer.
    #[test]
    #[expect(unsafe_code, reason = "exercises the audited views")]
    fn test_raw_views_are_disjoint_and_in_bounds() {
        let mut data = vec![0u32; 12];
        let raw = Raw::new(&mut data);
        // SAFETY: ranges 0..2 and 2..4 (3 elements each) do not overlap,
        // and `data` outlives both views.
        let first = unsafe { raw.view(&(0..2), 3) };
        // SAFETY: as above; this view does not overlap `first`.
        let second = unsafe { raw.view(&(2..4), 3) };
        first.fill(1);
        second.fill(2);
        // SAFETY: a range past the end is clamped to an empty view.
        let empty = unsafe { raw.view(&(9..20), 3) };
        assert_eq!(empty, &mut [] as &mut [u32]);
        assert_eq!(data, [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]);
    }

    #[test]
    fn test_chunk_ranges_are_disjoint_and_cover_every_instance() {
        for (n, size, workers) in [(10, 3, 2), (64, 64, 7), (5, 8, 3), (100, 7, 4)] {
            let chunks = n / size + usize::from(n % size != 0);
            let mut owner = vec![None; n];
            for w in 0..workers {
                for c in (w..chunks).step_by(workers) {
                    for i in chunk_range(c, size, n) {
                        assert_eq!(owner[i], None, "instance {i} seen twice");
                        owner[i] = Some(w);
                    }
                }
            }
            assert!(owner.iter().all(Option::is_some));
        }
    }

    /// Builds a probe pool of `n` instances, runs `f` on its raw parts and
    /// returns the outputs and states for checking.
    fn with_raw(
        n: usize,
        f: impl FnOnce(&RawChunk<<Probe as Environment>::Header, ProbeState>),
    ) -> (Outputs, Vec<ProbeState>) {
        let leaves = [Leaf { bytes: 3 }, Leaf { bytes: 1 }];
        let mut outputs = Outputs::new(n, 2, &leaves, true, 2).unwrap();
        let mut headers = vec![RecordHeader::<2, NoAdaptive>::state_default(); n];
        let mut states = vec![ProbeState(0); n];
        let mut discrete = vec![0i32; n * 2];
        let mut continuous: Vec<f32> = Vec::new();
        let mut penalties: Vec<f32> = Vec::new();
        let mut penalized: Vec<Bool8> = Vec::new();
        let raw = RawChunk::<<Probe as Environment>::Header, ProbeState>::new(
            Chunk::<Probe> {
                headers: &mut headers,
                states: &mut states,
                columns: (),
                discrete: &mut discrete,
                continuous: &mut continuous,
                penalties: &mut penalties,
                penalized: &mut penalized,
                outputs: outputs.view_all(),
            },
            1,
            0,
        );
        f(&raw);
        (outputs, states)
    }

    #[test]
    fn test_half_ranges_split_at_the_nearest_chunk_boundary() {
        // (instances, chunk size, expected split)
        for (n, size, split) in [
            (7, 3, 3),    // boundaries 3 and 6; 3.5 is nearest 3
            (8, 2, 4),    // exact middle
            (9, 3, 3),    // 3 and 6 tie around 4.5: the lower one
            (10, 4, 4),   // boundaries 4 and 8
            (6, 4, 4),    // one interior boundary, uneven halves
            (100, 7, 49), // N not a multiple of the chunk size
            (1024, 64, 512),
        ] {
            let [a, b] = half_ranges(n, nz(size)).unwrap();
            assert_eq!(
                (a.clone(), b.clone()),
                (0..split, split..n),
                "n {n} size {size}"
            );
            assert_eq!(split % size, 0, "the split is a chunk boundary");
        }
    }

    #[test]
    fn test_half_ranges_without_two_chunks_is_none() {
        for (n, size) in [(0, 4), (1, 1), (3, 4), (4, 4)] {
            assert_eq!(half_ranges(n, nz(size)), None, "n {n} size {size}");
        }
        assert_eq!(half_ranges(2, nz(1)), Some([0..1, 1..2]));
    }

    /// Range and instance views stay inside the instances: reversed or
    /// out-of-range requests are refused, the empty range at the end is a
    /// valid empty view (run under Miri in the nightly job).
    #[test]
    #[expect(unsafe_code, reason = "exercises the audited views")]
    fn test_raw_chunk_range_view_refuses_out_of_range_rows() {
        const N: usize = 5;
        let _unused = with_raw(N, |raw| {
            // SAFETY: each view is dropped before the next is built, so no
            // two live views overlap, and the buffers outlive them.
            assert!(unsafe { raw.range_view(0..N + 1) }.is_none());
            #[expect(clippy::reversed_empty_ranges, reason = "the refused case")]
            let reversed = 3..2;
            // SAFETY: as above.
            assert!(unsafe { raw.range_view(reversed) }.is_none());
            // SAFETY: as above.
            assert!(unsafe { raw.instance_view(N) }.is_none());
            // SAFETY: as above.
            assert!(unsafe { raw.instance_view(usize::MAX) }.is_none());
            // SAFETY: as above.
            let end = unsafe { raw.range_view(N..N) }.unwrap();
            assert!(end.states.is_empty() && end.outputs.rewards.is_empty());
            // SAFETY: as above; `end` is no longer used.
            let last = unsafe { raw.instance_view(N - 1) }.unwrap();
            assert_eq!(last.states.len(), 1);
            assert_eq!(last.discrete.len(), 2);
            assert_eq!(last.outputs.rewards.len(), 2);
            assert_eq!(last.outputs.observations[0].len(), 6);
            assert_eq!(last.outputs.masks.len(), 4);
        });
    }

    /// Two threads write at once: the first through one instance view at a
    /// time over half A, the second through one view of all of half B. With
    /// N not a multiple of the chunk size, every element ends up written by
    /// exactly its owner (run under Miri in the nightly job).
    #[test]
    #[expect(unsafe_code, reason = "exercises the audited views")]
    fn test_raw_chunk_half_and_instance_views_never_overlap() {
        const N: usize = 7;
        let halves = half_ranges(N, nz(3)).unwrap();
        let [first_half, second_half] = halves.clone();
        let (outputs, states) = with_raw(N, |raw| {
            std::thread::scope(|scope| {
                // Half A: one view per instance, built and dropped in turn.
                scope.spawn(move || {
                    for id in first_half {
                        // SAFETY: half A's rows belong to this thread only
                        // and each instance view is dropped before the next.
                        let one = unsafe { raw.instance_view(id) }.unwrap();
                        one.states[0].0 = u32::try_from(id).unwrap() + 1;
                        one.outputs.rewards.fill(1.0);
                        one.outputs.observations[0].fill(1);
                    }
                });
                // Half B: one view of the whole half.
                scope.spawn(move || {
                    // SAFETY: half B's rows belong to this thread only.
                    let half = unsafe { raw.range_view(second_half) }.unwrap();
                    for s in half.states.iter_mut() {
                        s.0 = 100;
                    }
                    half.outputs.rewards.fill(2.0);
                    half.outputs.observations[0].fill(2);
                });
            });
        });
        let [lower, _] = halves;
        for (i, state_now) in states.iter().enumerate() {
            let (state, reward, byte) = if lower.contains(&i) {
                (u32::try_from(i).unwrap() + 1, 1.0, 1)
            } else {
                (100, 2.0, 2)
            };
            assert_eq!(state_now.0, state, "instance {i}");
            assert_eq!(&outputs.rewards()[i * 2..i * 2 + 2], &[reward; 2]);
            let row = &outputs.observation(0).unwrap()[i * 6..(i + 1) * 6];
            assert!(row.iter().all(|&v| v == byte), "instance {i}");
        }
    }
}

#[cfg(test)]
mod accum_tests {
    use super::accum_chunks;

    #[test]
    fn test_accum_chunks_whole_chunks_only() {
        // 10 instances in chunks of 4: 0..4, 4..8, 8..10.
        assert_eq!(accum_chunks(&(0..4), 4, 10), Some(0..1));
        assert_eq!(accum_chunks(&(4..10), 4, 10), Some(1..3));
        assert_eq!(accum_chunks(&(0..10), 4, 10), Some(0..3));
        // A split chunk (Async runs of claimed ids) gets no accumulators.
        assert_eq!(accum_chunks(&(2..4), 4, 10), None);
        assert_eq!(accum_chunks(&(4..6), 4, 10), None);
        assert_eq!(accum_chunks(&(3..5), 4, 10), None);
        assert_eq!(accum_chunks(&(4..4), 4, 10), None);
    }
}
