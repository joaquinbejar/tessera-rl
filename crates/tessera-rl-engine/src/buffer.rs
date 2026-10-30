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
        let start = instances.start.saturating_mul(per).min(self.len);
        let end = instances.end.saturating_mul(per).min(self.len);
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
    let start = c.saturating_mul(size).min(instances);
    start..start.saturating_add(size).min(instances)
}

/// Runs `f` on every chunk of `size` instances of `all`, spread over the
/// gang's workers (chunk `c` on worker `c % workers`). `f` runs once per
/// chunk; a panic in it is caught per chunk. Returns `true` if any chunk
/// panicked. `dh` and `ch` are the discrete and continuous heads per agent.
pub(crate) fn par_chunks<E, F>(
    gang: &Gang,
    workers: usize,
    all: Chunk<'_, E>,
    size: usize,
    heads: (usize, usize),
    f: F,
) -> bool
where
    E: Environment,
    E::Header: Send,
    E::State: Send,
    F: Fn(&mut Chunk<'_, E>, usize) + Sync,
{
    let raw = RawChunk::new(all, heads.0, heads.1);
    let chunks = raw.instances.div_ceil(size.max(1));
    let panicked = AtomicBool::new(false);
    let job = |w: usize| {
        for c in (w..chunks).step_by(workers.max(1)) {
            let range = chunk_range(c, size, raw.instances);
            debug_assert!(c % workers.max(1) == w, "chunk {c} is not worker {w}'s");
            // SAFETY: chunk `c` belongs to worker `w` only, and chunk ranges
            // do not overlap (invariant 2); the pool is borrowed for the
            // whole epoch (invariant 3) and this view is dropped at the end
            // of the iteration (invariant 4).
            #[expect(unsafe_code, reason = "audited module: disjoint chunk views")]
            let mut chunk = unsafe { raw.chunk(&range) };
            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                f(&mut chunk, range.start);
            }))
            .is_ok();
            if !ok {
                panicked.store(true, Ordering::Relaxed);
            }
        }
    };
    let gang_panicked = gang.run(&job);
    gang_panicked || panicked.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::{Raw, chunk_range};

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
