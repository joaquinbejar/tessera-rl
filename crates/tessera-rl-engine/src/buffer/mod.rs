// SPDX-License-Identifier: MIT OR Apache-2.0

//! Disjoint chunk views shared with worker threads (ADR-0002, ADR-0005).
//! Audited `unsafe` module.
//!
//! Sync mode splits a pool-wide [`Chunk`] into its raw parts and hands each
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
//!    `c * size .. min((c + 1) * size, n)`, and worker `w` takes the chunks
//!    `c` with `c % workers == w`, so no two workers, and no two chunks of
//!    one worker at a time, ever view the same element (checked in debug
//!    builds and by tests).
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

use tessera_rl_core::{Bool8, Environment};

use crate::error::EngineError;
use crate::layout::MAX_LEAVES;
use crate::outputs::OutputsView;
use crate::pool::Chunk;
use crate::sys::Gang;
use crate::sys::sync::{AtomicBool, Ordering};

use generations::HALVES;
#[cfg(not(loom))]
pub(crate) use generations::{GenerationError, Generations};
pub(crate) use storage::{Buf, StorageBuilder};
pub use storage::{OutputDtype, OutputField, OutputRegion, OutputStorage};

#[derive(Clone, Copy)]
struct Raw<T> {
    ptr: *mut T,
    len: usize,
}

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

/// The raw parts of a pool-wide chunk.
struct RawChunk<E: Environment> {
    instances: usize,
    agents: usize,
    dh: usize,
    ch: usize,
    mask_len: usize,
    leaf_bytes: Raw<usize>,
    headers: Raw<E::Header>,
    states: Raw<E::State>,
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
}

// SAFETY: a `RawChunk` is shared by workers only to build disjoint views
// (invariant 2), each of which is used by one thread at a time; the
// elements are `Send` (`E::Header` and `E::State` are required `Send` by
// Sync mode, the others are plain numbers), so handing out `&mut` views of
// disjoint ranges to other threads is sound.
#[expect(unsafe_code, reason = "audited module: shared raw chunk")]
unsafe impl<E: Environment> Sync for RawChunk<E>
where
    E::Header: Send,
    E::State: Send,
{
}

impl<E: Environment> RawChunk<E> {
    fn new(chunk: Chunk<'_, E>, dh: usize, ch: usize) -> Self {
        let Chunk {
            headers,
            states,
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
            headers: Raw::new(headers),
            states: Raw::new(states),
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
        }
    }

    /// The views of the instances in `r`.
    ///
    /// # Safety
    ///
    /// Module invariants; in particular no other live view covers `r`.
    #[expect(unsafe_code, reason = "audited module: disjoint chunk views")]
    unsafe fn chunk<'a>(&self, r: &Range<usize>) -> Chunk<'a, E> {
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
        Chunk {
            headers,
            states,
            discrete,
            continuous,
            penalties,
            penalized,
            outputs: OutputsView {
                agents: a,
                mask_len: self.mask_len,
                leaf_bytes,
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
            },
        }
    }
}

#[cfg_attr(
    loom,
    allow(dead_code, reason = "the split-phase modes are not built under loom")
)]
impl<E: Environment> RawChunk<E> {
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
    unsafe fn range_view<'a>(&self, r: Range<usize>) -> Option<Chunk<'a, E>> {
        if r.start > r.end || r.end > self.instances {
            return None;
        }
        // SAFETY: `r` lies inside the instances, so every per-instance view
        // stays inside its buffer; the caller guarantees the rows are
        // exclusively its own for `'a` (invariants 1, 3 and 4).
        Some(unsafe { self.chunk(&r) })
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
    unsafe fn instance_view<'a>(&self, id: usize) -> Option<Chunk<'a, E>> {
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
/// gang's workers (chunk `c` on worker `c % workers`). `f` runs once per
/// chunk; a panic in it is caught per chunk. Returns `true` if any chunk
/// panicked. `dh` and `ch` are the discrete and continuous heads per agent.
pub(crate) fn par_chunks<E, F>(
    gang: &mut Gang,
    all: Chunk<'_, E>,
    size: usize,
    heads: (usize, usize),
    f: F,
) -> Result<bool, EngineError>
where
    E: Environment,
    E::Header: Send,
    E::State: Send,
    F: Fn(&mut Chunk<'_, E>, usize) + Sync,
{
    // The worker count comes from the gang itself, so the assignment below
    // is disjoint by construction (invariant 2).
    let workers = gang.workers();
    let raw = RawChunk::new(all, heads.0, heads.1);
    let chunks = raw.instances.div_ceil(size.max(1));
    let panicked = AtomicBool::new(false);
    let job = |w: usize| {
        if w >= workers {
            return;
        }
        for c in (w..chunks).step_by(workers) {
            let range = chunk_range(c, size, raw.instances);
            debug_assert!(c % workers == w, "chunk {c} is not worker {w}'s");
            // SAFETY: chunk `c` belongs to worker `w` only, and chunk ranges
            // do not overlap (invariant 2); the pool is borrowed for the
            // whole epoch (invariant 3) and this view is dropped at the end
            // of the iteration (invariant 4).
            #[expect(unsafe_code, reason = "audited module: disjoint chunk views")]
            let mut chunk = unsafe { raw.chunk(&range) };
            let ok = crate::sys::catch(|| f(&mut chunk, range.start));
            if !ok {
                panicked.store(true, Ordering::Relaxed);
            }
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

    impl tessera_rl_core::EnvState for ProbeState {}

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
            discrete: &mut discrete,
            continuous: &mut continuous,
            penalties: &mut penalties,
            penalized: &mut penalized,
            outputs: outputs.view_all(),
        };
        let raw = RawChunk::<Probe>::new(all, 1, 0);
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
    fn with_raw(n: usize, f: impl FnOnce(&RawChunk<Probe>)) -> (Outputs, Vec<ProbeState>) {
        let leaves = [Leaf { bytes: 3 }, Leaf { bytes: 1 }];
        let mut outputs = Outputs::new(n, 2, &leaves, true, 2).unwrap();
        let mut headers = vec![RecordHeader::<2, NoAdaptive>::state_default(); n];
        let mut states = vec![ProbeState(0); n];
        let mut discrete = vec![0i32; n * 2];
        let mut continuous: Vec<f32> = Vec::new();
        let mut penalties: Vec<f32> = Vec::new();
        let mut penalized: Vec<Bool8> = Vec::new();
        let raw = RawChunk::<Probe>::new(
            Chunk {
                headers: &mut headers,
                states: &mut states,
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
