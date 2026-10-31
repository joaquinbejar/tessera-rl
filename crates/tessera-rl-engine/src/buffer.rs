// SPDX-License-Identifier: MIT OR Apache-2.0

//! Disjoint chunk views shared with worker threads (ADR-0002, ADR-0005).
//! Audited `unsafe` module.
//!
//! Sync mode splits a pool-wide [`Chunk`] into its raw parts and hands each
//! worker `&mut` views of *its own* chunks only.
//!
//! # Invariants
//!
//! 1. The buffers are never reallocated after construction: the pool never
//!    pushes to or resizes them, so pointers taken at the start of an epoch
//!    stay valid for the whole epoch. The pointers are to the heap data of
//!    the `Vec`s, not to the pool, so moving the pool between calls is fine.
//! 2. Chunk ranges are disjoint by construction: chunk `c` covers instances
//!    `c * size .. min((c + 1) * size, n)`, and worker `w` takes the chunks
//!    `c` with `c % workers == w`, so no two workers, and no two chunks of
//!    one worker at a time, ever view the same element (checked in debug
//!    builds and by tests).
//! 3. The caller touches nothing between the epoch's Release and its
//!    Acquire of `pending == 0`: `par_chunks` takes the pool-wide chunk by
//!    value (borrowing the pool mutably through `&mut self` of every public
//!    entry point) and returns only after the gang's `run` returns.
//! 4. No reference to the buffers survives the epoch: worker views are
//!    created and dropped inside the job, and the caller's output views are
//!    returned only after `run`.

use core::ops::Range;

use tessera_rl_core::{Bool8, Environment};

use crate::error::EngineError;
use crate::layout::MAX_LEAVES;
use crate::outputs::OutputsView;
use crate::pool::Chunk;
use crate::sys::Gang;
use crate::sys::sync::{AtomicBool, Ordering};

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
            // SAFETY: invariants 1 and 2 for range `r` (one leaf row per agent).
            *view = unsafe { raw.view(r, a * bytes) };
            // SAFETY: as above, for the terminal buffer of the same leaf.
            *term = unsafe { raw_term.view(r, a * bytes) };
        }
        // SAFETY: invariants 1 and 2 for range `r` (1 elements per instance).
        let headers = unsafe { self.headers.view(r, 1) };
        // SAFETY: invariants 1 and 2 for range `r` (1 elements per instance).
        let states = unsafe { self.states.view(r, 1) };
        // SAFETY: invariants 1 and 2 for range `r` (a * self.dh elements per instance).
        let discrete = unsafe { self.discrete.view(r, a * self.dh) };
        // SAFETY: invariants 1 and 2 for range `r` (a * self.ch elements per instance).
        let continuous = unsafe { self.continuous.view(r, a * self.ch) };
        // SAFETY: invariants 1 and 2 for range `r` (a elements per instance).
        let penalties = unsafe { self.penalties.view(r, a) };
        // SAFETY: invariants 1 and 2 for range `r` (a elements per instance).
        let penalized = unsafe { self.penalized.view(r, a) };
        // SAFETY: invariants 1 and 2 for range `r` (a elements per instance).
        let rewards = unsafe { self.rewards.view(r, a) };
        // SAFETY: invariants 1 and 2 for range `r` (a elements per instance).
        let agent_terminated = unsafe { self.agent_terminated.view(r, a) };
        // SAFETY: invariants 1 and 2 for range `r` (a elements per instance).
        let agent_truncated = unsafe { self.agent_truncated.view(r, a) };
        // SAFETY: invariants 1 and 2 for range `r` (a elements per instance).
        let died = unsafe { self.died.view(r, a) };
        // SAFETY: invariants 1 and 2 for range `r` (a elements per instance).
        let spawned = unsafe { self.spawned.view(r, a) };
        // SAFETY: invariants 1 and 2 for range `r` (a elements per instance).
        let invalid_agent_event = unsafe { self.invalid_agent_event.view(r, a) };
        // SAFETY: invariants 1 and 2 for range `r` (a elements per instance).
        let alive = unsafe { self.alive.view(r, a) };
        // SAFETY: invariants 1 and 2 for range `r` (a elements per instance).
        let acting = unsafe { self.acting.view(r, a) };
        // SAFETY: invariants 1 and 2 for range `r` (a elements per instance).
        let generations = unsafe { self.generations.view(r, a) };
        // SAFETY: invariants 1 and 2 for range `r` (1 elements per instance).
        let episode_terminated = unsafe { self.episode_terminated.view(r, 1) };
        // SAFETY: invariants 1 and 2 for range `r` (1 elements per instance).
        let episode_truncated = unsafe { self.episode_truncated.view(r, 1) };
        // SAFETY: invariants 1 and 2 for range `r` (1 elements per instance).
        let episode_return = unsafe { self.episode_return.view(r, 1) };
        // SAFETY: invariants 1 and 2 for range `r` (1 elements per instance).
        let episode_length = unsafe { self.episode_length.view(r, 1) };
        // SAFETY: invariants 1 and 2 for range `r` (1 elements per instance).
        let episode_illegal_actions = unsafe { self.episode_illegal_actions.view(r, 1) };
        // SAFETY: invariants 1 and 2 for range `r` (1 elements per instance).
        let episode_invalid_agent_events = unsafe { self.episode_invalid_agent_events.view(r, 1) };
        // SAFETY: invariants 1 and 2 for range `r` (a * self.mask_len elements per instance).
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

    use super::{Raw, RawChunk, chunk_range};
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
                        out.episode_invalid_spawns.fill(f);
                        out.generations.fill(mark);
                        for flags in [
                            out.agent_terminated,
                            out.agent_truncated,
                            out.died,
                            out.spawned,
                            out.invalid_spawn,
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
            assert_eq!(&outputs.generations[i * 2..i * 2 + 2], &[owner(i); 2]);
        }
        assert!(outputs.masks.iter().all(|m| m.get()));
        assert!(outputs.invalid_spawn.iter().all(|m| m.get()));
        for i in 0..N {
            #[expect(clippy::cast_precision_loss, reason = "tiny marks")]
            let mark = owner(i) as f32;
            assert_eq!(outputs.episode_invalid_spawns[i], mark);
        }
        assert!(outputs.episode_truncated.iter().all(|m| m.get()));
        let first_leaf = &outputs.observations[0];
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
}
