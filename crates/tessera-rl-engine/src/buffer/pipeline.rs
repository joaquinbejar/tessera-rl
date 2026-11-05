// SPDX-License-Identifier: MIT OR Apache-2.0

//! The shared allocation of the Pipelined mode, its half jobs and its
//! tokens (ADR-0047 sections 1, 2 and 7). Audited `unsafe` module.
//!
//! [`Mode`] lives in one `Arc` and owns the moved `EnvPool`. It holds the
//! raw parts of the pool's buffers ([`RawChunk`]), one gang [`Job`] per
//! half and the per-half flags. Workers reach it only through a job's data
//! pointer, taken from `Arc::as_ptr` (never from a `&mut`), so later
//! shared borrows of the allocation leave it valid; they hold no `Arc`.
//!
//! A [`HalfToken`] is the caller's exclusive right to read one half's rows
//! (invariants 3 and 4 of the module): it sets the half's live flag when it
//! is created and clears it with Release in `Drop`; a send checks the flag
//! with Acquire and consumes the token, so a worker writes a half's rows
//! only after every view of them has ended. Views of a half are built
//! directly as shared slices, never as `&mut`, so a token can hand out
//! several at once.
//!
//! The raw parts are taken from a `&mut` of the pool. Any later `&mut` of
//! the pool (the writers of ADR-0047 section 3, through `Arc::get_mut`)
//! invalidates them, so every such writer calls [`Mode::rebuild`] before a
//! job or a token uses them again.

use core::ops::Range;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use std::sync::Arc;

use tessera_rl_core::{Bool8, Environment, RecordLayout as _};

use super::{HALVES, Raw, RawChunk, chunk_range};
use crate::masks::MaskProvider;
use crate::pool::{EnvPool, step_chunk};
use crate::sys::Gang;
use crate::sys::Job;

/// Per-half flags shared by the caller, the tokens, the workers and a
/// [`Waiter`].
#[derive(Debug, Default)]
pub(crate) struct Ready {
    /// A worker finished the half's last chunk.
    pub(crate) ready: [AtomicBool; HALVES],
}

/// The work of one half: its panic flag and the workers still running it.
#[derive(Debug, Default)]
struct HalfWork {
    panicked: AtomicBool,
    remaining: AtomicUsize,
}

/// The one shared allocation of a Pipelined pool (ADR-0047 section 2).
pub(crate) struct Mode<E: Environment, M: MaskProvider<E>> {
    pub(crate) pool: EnvPool<E, M>,
    raw: RawChunk<E>,
    halves: [Range<usize>; HALVES],
    size: usize,
    workers: usize,
    work: [HalfWork; HALVES],
    jobs: [Job; HALVES],
    /// A live token covers the half.
    pub(crate) live: [AtomicBool; HALVES],
    /// The half is sent and not yet received.
    pub(crate) in_flight: [AtomicBool; HALVES],
    pub(crate) ready: Arc<Ready>,
}

// SAFETY: `Mode` is shared by the caller, its tokens and the workers. The
// raw parts and the jobs point into the pool's heap buffers and into the
// `Mode` itself; every access through them follows the module invariants
// (a half's rows are touched by one token holder or one admitted job at a
// time, ordered by the live and in-flight flags and the gang's
// Release/Acquire on completion). The rest is the pool, read only while
// shared (`Config: Sync`), and atomics. Headers and states move between
// threads (`Send`) and headers are read through shared token views
// (`Sync`).
#[expect(unsafe_code, reason = "audited module: shared mode allocation")]
unsafe impl<E: Environment, M: MaskProvider<E>> Send for Mode<E, M>
where
    E::Config: Send + Sync,
    E::Header: Send + Sync,
    E::State: Send,
{
}

// SAFETY: as for `Send` above; shared `&Mode` also reaches the states
// (snapshots read them while tokens are alive), so they must be `Sync`.
#[expect(unsafe_code, reason = "audited module: shared mode allocation")]
unsafe impl<E: Environment, M: MaskProvider<E>> Sync for Mode<E, M>
where
    E::Config: Send + Sync,
    E::Header: Send + Sync,
    E::State: Send + Sync,
{
}

impl<E, M> Mode<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
{
    /// Moves `pool` into a new shared allocation with `halves` of chunks of
    /// `size` instances, run by `workers` workers.
    pub(crate) fn new(
        mut pool: EnvPool<E, M>,
        halves: [Range<usize>; HALVES],
        size: usize,
        workers: usize,
    ) -> Arc<Self> {
        let raw = Self::parts(&mut pool);
        let mut arc = Arc::new(Self {
            pool,
            raw,
            halves,
            size: size.max(1),
            workers,
            work: Default::default(),
            jobs: placeholders(),
            live: Default::default(),
            in_flight: Default::default(),
            ready: Arc::new(Ready::default()),
        });
        // The jobs' data pointer comes from the `Arc` itself, not from the
        // `&mut` below, so it stays valid under later shared borrows.
        let data = Arc::as_ptr(&arc).cast::<()>();
        if let Some(mode) = Arc::get_mut(&mut arc) {
            mode.jobs = half_jobs::<E, M>(data);
        }
        arc
    }

    /// Raw parts of every buffer of `pool`, from a fresh `&mut`.
    fn parts(pool: &mut EnvPool<E, M>) -> RawChunk<E> {
        let (chunk, (dh, ch)) = pool.chunk_all();
        RawChunk::new(chunk, dh, ch)
    }

    /// Takes the raw parts again after a writer used `&mut` of the pool.
    pub(crate) fn rebuild(&mut self) {
        self.raw = Self::parts(&mut self.pool);
    }

    /// The two halves.
    pub(crate) fn halves(&self) -> &[Range<usize>; HALVES] {
        &self.halves
    }

    /// Gives the pool back (`into_pool`).
    pub(crate) fn into_pool(self) -> EnvPool<E, M> {
        self.pool
    }

    /// Validates and copies the actions of `token`'s half with `f`, given
    /// the half's headers and action rows (ADR-0047 section 1).
    pub(crate) fn with_half_rows<R>(
        &self,
        token: &HalfToken<E, M>,
        f: impl FnOnce(usize, &[E::Header], &mut [i32], &mut [f32]) -> R,
    ) -> Option<R> {
        if !core::ptr::eq(Arc::as_ptr(&token.mode), self) {
            return None;
        }
        let range = self.halves.get(token.half)?.clone();
        let first = range.start;
        let a = self.raw.agents;
        // Only the action rows are borrowed mutably; the headers (and the
        // masks a caller may read through `token`) stay shared views.
        // SAFETY: `token` covers the half, so no worker and no other view
        // touches its action rows while `token` is borrowed (invariants 3
        // and 4); the buffers are in place (invariant 1, `rebuild` after
        // writers).
        #[expect(unsafe_code, reason = "audited module: token-guarded views")]
        let discrete = unsafe { self.raw.discrete.view(&range, a * self.raw.dh) };
        // SAFETY: as above, the continuous action rows of the half.
        #[expect(unsafe_code, reason = "audited module: token-guarded views")]
        let continuous = unsafe { self.raw.continuous.view(&range, a * self.raw.ch) };
        Some(f(first, token.headers(), discrete, continuous))
    }
}

/// A placeholder job that is never published.
#[expect(unsafe_code, reason = "audited module: placeholder job")]
unsafe fn never(_: *const (), _: usize) {}

/// The two half jobs of the `Mode<E, M>` at `data`.
#[expect(unsafe_code, reason = "audited module: half jobs")]
fn half_jobs<E: Environment, M: MaskProvider<E>>(data: *const ()) -> [Job; HALVES] {
    // SAFETY: `run_half::<E, M, H>` reads `data` as a `Mode<E, M>`, which
    // it is (taken from `Arc::as_ptr` in `Mode::new`); the caller keeps the
    // allocation alive until every job it publishes is waited for.
    [
        unsafe { Job::from_raw(run_half::<E, M, 0>, data) },
        unsafe { Job::from_raw(run_half::<E, M, 1>, data) },
    ]
}

/// Jobs that are replaced before the allocation is shared.
#[expect(unsafe_code, reason = "audited module: placeholder job")]
fn placeholders() -> [Job; HALVES] {
    // SAFETY: `never` does nothing and the null data is never read; these
    // jobs are overwritten in `Mode::new` before any is published.
    [unsafe { Job::from_raw(never, core::ptr::null()) }, unsafe {
        Job::from_raw(never, core::ptr::null())
    }]
}

/// Steps half `H` on `worker` (ADR-0047 section 7).
///
/// # Safety
///
/// `data` points to a live `Mode<E, M>` whose half `H` was published by
/// `publish`: no token covers it and no other job writes it.
#[expect(unsafe_code, reason = "audited module: half jobs")]
unsafe fn run_half<E: Environment, M: MaskProvider<E>, const H: usize>(
    data: *const (),
    worker: usize,
) {
    // SAFETY: `data` is the `Mode` of the allocation the caller keeps alive
    // until this job is waited for; only shared access is taken here.
    let mode = unsafe { &*data.cast::<Mode<E, M>>() };
    let (Some(range), Some(work)) = (mode.halves.get(H), mode.work.get(H)) else {
        return;
    };
    let workers = mode.workers;
    if worker < workers {
        let size = mode.size;
        let ctx = mode.pool.step_ctx();
        let first = range.start / size;
        let last = range.end.div_ceil(size);
        for c in (first..last).skip(worker).step_by(workers.max(1)) {
            let r = chunk_range(c, size, mode.raw.instances);
            let r = r.start.max(range.start)..r.end.min(range.end);
            // SAFETY: chunk `c` lies in half `H`, which only this job writes
            // (the half's token was consumed by `send`), and each chunk
            // belongs to one worker (invariants 2, 3 and 4).
            let Some(mut chunk) = (unsafe { mode.raw.range_view(r) }) else {
                continue;
            };
            if !crate::sys::catch(|| step_chunk::<E, M>(&ctx, &mut chunk)) {
                work.panicked.store(true, Ordering::Relaxed);
            }
        }
    }
    // The last worker signals the waiter (Release: after its writes).
    if work.remaining.fetch_sub(1, Ordering::AcqRel) == 1
        && let Some(flag) = mode.ready.ready.get(H)
    {
        flag.store(true, Ordering::Release);
    }
}

/// Publishes half `half` of `mode` to `gang`.
///
/// The caller keeps `mode` alive (it holds an `Arc`) and the gang alive
/// until the job is waited for, and has consumed the half's token.
pub(crate) fn publish<E: Environment, M: MaskProvider<E>>(
    gang: &mut Gang,
    mode: &Arc<Mode<E, M>>,
    half: usize,
) -> Result<(), crate::error::EngineError> {
    let (Some(job), Some(work), Some(ready)) = (
        mode.jobs.get(half),
        mode.work.get(half),
        mode.ready.ready.get(half),
    ) else {
        return Err(crate::error::EngineError::InvalidHalf {
            half: u32::try_from(half).unwrap_or(u32::MAX),
        });
    };
    work.panicked.store(false, Ordering::Relaxed);
    work.remaining.store(gang.workers(), Ordering::Relaxed);
    ready.store(false, Ordering::Relaxed);
    // SAFETY: `job` and its data live in the `Arc` allocation, which the
    // caller keeps alive until the `wait` that returns this job (its `Drop`
    // drains every job first); the half's token was consumed, so no view of
    // its rows is live (module invariants).
    #[expect(unsafe_code, reason = "audited module: half job publication")]
    let submitted = unsafe { gang.submit(job) };
    submitted
}

/// Whether a chunk of half `half` panicked in its last job.
pub(crate) fn half_panicked<E: Environment, M: MaskProvider<E>>(
    mode: &Mode<E, M>,
    half: usize,
) -> bool {
    mode.work
        .get(half)
        .is_some_and(|w| w.panicked.load(Ordering::Relaxed))
}

/// The caller's exclusive right to read one returned half (ADR-0047
/// section 1). Results borrow the token, never the pool; dropping it, or
/// sending it, ends every view of the half.
pub struct HalfToken<E: Environment, M: MaskProvider<E>> {
    mode: Arc<Mode<E, M>>,
    half: usize,
}

impl<E: Environment, M: MaskProvider<E>> core::fmt::Debug for HalfToken<E, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HalfToken")
            .field("half", &self.half)
            .finish_non_exhaustive()
    }
}

impl<E: Environment, M: MaskProvider<E>> Drop for HalfToken<E, M> {
    fn drop(&mut self) {
        if let Some(flag) = self.mode.live.get(self.half) {
            // Every read of the half's rows happens before the next send's
            // Acquire of this flag.
            super::live::release(flag);
        }
    }
}

impl<E: Environment, M: MaskProvider<E>> HalfToken<E, M> {
    /// A new token for `half`, which must be returned and not covered;
    /// `None` if a token already covers it.
    pub(crate) fn new(mode: &Arc<Mode<E, M>>, half: usize) -> Option<Self> {
        let flag = mode.live.get(half)?;
        // Acquire: the half's completed writes are visible to the views.
        if !super::live::claim(flag) {
            return None;
        }
        Some(Self {
            mode: Arc::clone(mode),
            half,
        })
    }

    /// Whether `self` belongs to `mode`.
    pub(crate) fn belongs_to(&self, mode: &Arc<Mode<E, M>>) -> bool {
        Arc::ptr_eq(&self.mode, mode)
    }

    /// The half index, 0 or 1.
    #[must_use]
    pub fn half(&self) -> u32 {
        u32::try_from(self.half).unwrap_or(u32::MAX)
    }

    /// The instance ids of the half.
    #[must_use]
    pub fn ids(&self) -> Range<u32> {
        let r = self.range();
        u32::try_from(r.start).unwrap_or(u32::MAX)..u32::try_from(r.end).unwrap_or(u32::MAX)
    }

    fn range(&self) -> Range<usize> {
        self.mode.halves.get(self.half).cloned().unwrap_or(0..0)
    }

    /// A shared view of `per` elements per instance of the half.
    fn view<T>(&self, raw: &Raw<T>, per: usize) -> &[T] {
        let r = self.range();
        let start = r.start.checked_mul(per).map_or(raw.len, |s| s.min(raw.len));
        let end = r
            .end
            .checked_mul(per)
            .map_or(raw.len, |e| e.min(raw.len))
            .max(start);
        #[expect(unsafe_code, reason = "audited module: token-guarded views")]
        // SAFETY: `start <= len`, inside (or one past) the live buffer.
        let first = unsafe { raw.ptr.add(start) };
        #[expect(unsafe_code, reason = "audited module: token-guarded views")]
        // SAFETY: `start <= end <= len` keeps the view inside the live
        // buffer (invariant 1); the token covers the half, so no worker
        // writes these rows while `self` is borrowed and every other view of
        // them is shared (invariants 3 and 4).
        let view = unsafe { core::slice::from_raw_parts(first.cast_const(), end - start) };
        view
    }

    fn agents(&self) -> usize {
        self.mode.raw.agents
    }

    /// The half's record headers, `[half]`.
    pub(crate) fn headers(&self) -> &[E::Header] {
        self.view(&self.mode.raw.headers, 1)
    }

    /// The step counter of instance `id` (episode index, step index), if
    /// `id` is in the half.
    #[must_use]
    pub fn step_counter(&self, id: u32) -> Option<(u32, u32)> {
        let i = usize::try_from(id).ok()?;
        let r = self.range();
        let header = self.headers().get(i.checked_sub(r.start)?)?;
        let identity = &header.engine().identity;
        Some((identity.episode_index(), identity.step_index()))
    }

    /// Observations of leaf `leaf`, `[half, A, leaf bytes]`.
    #[must_use]
    pub fn observation(&self, leaf: usize) -> Option<&[u8]> {
        let raw_leaf = self.mode.raw.observations.get(leaf).copied()?;
        let bytes = self.leaf_bytes(leaf)?;
        Some(self.view(&raw_leaf, self.agents().checked_mul(bytes)?))
    }

    /// Final observations of leaf `leaf`, `[half, A, leaf bytes]`
    /// (`SameStep` auto-reset).
    #[must_use]
    pub fn terminal_observation(&self, leaf: usize) -> Option<&[u8]> {
        let raw_leaf = self.mode.raw.terminal.get(leaf).copied()?;
        let bytes = self.leaf_bytes(leaf)?;
        Some(self.view(&raw_leaf, self.agents().checked_mul(bytes)?))
    }

    fn leaf_bytes(&self, leaf: usize) -> Option<usize> {
        let raw = self.mode.raw.leaf_bytes;
        // SAFETY: `leaf_bytes` is never written after construction, so a
        // shared view of it is sound alongside any worker (invariant 1).
        #[expect(unsafe_code, reason = "audited module: read-only layout")]
        let bytes = unsafe { core::slice::from_raw_parts(raw.ptr.cast_const(), raw.len) };
        bytes.get(leaf).copied()
    }

    /// Rewards, `[half, A]`.
    #[must_use]
    pub fn rewards(&self) -> &[f32] {
        self.view(&self.mode.raw.rewards, self.agents())
    }

    /// Agents terminated in the last step, `[half, A]`.
    #[must_use]
    pub fn agent_terminated(&self) -> &[Bool8] {
        self.view(&self.mode.raw.agent_terminated, self.agents())
    }

    /// Agents truncated in the last step, `[half, A]`.
    #[must_use]
    pub fn agent_truncated(&self) -> &[Bool8] {
        self.view(&self.mode.raw.agent_truncated, self.agents())
    }

    /// Alive agent slots, `[half, A]`.
    #[must_use]
    pub fn alive(&self) -> &[Bool8] {
        self.view(&self.mode.raw.alive, self.agents())
    }

    /// Acting agent slots, `[half, A]`.
    #[must_use]
    pub fn acting(&self) -> &[Bool8] {
        self.view(&self.mode.raw.acting, self.agents())
    }

    /// Agent slot generations, `[half, A]`.
    #[must_use]
    pub fn generations(&self) -> &[u32] {
        self.view(&self.mode.raw.generations, self.agents())
    }

    /// Episodes terminated in the last step, `[half]`.
    #[must_use]
    pub fn episode_terminated(&self) -> &[Bool8] {
        self.view(&self.mode.raw.episode_terminated, 1)
    }

    /// Episodes truncated in the last step, `[half]`.
    #[must_use]
    pub fn episode_truncated(&self) -> &[Bool8] {
        self.view(&self.mode.raw.episode_truncated, 1)
    }

    /// Episode returns, `[half]`.
    #[must_use]
    pub fn episode_return(&self) -> &[f32] {
        self.view(&self.mode.raw.episode_return, 1)
    }

    /// Episode lengths, `[half]`.
    #[must_use]
    pub fn episode_length(&self) -> &[f32] {
        self.view(&self.mode.raw.episode_length, 1)
    }

    /// Illegal actions in the episode, `[half]`.
    #[must_use]
    pub fn episode_illegal_actions(&self) -> &[f32] {
        self.view(&self.mode.raw.episode_illegal_actions, 1)
    }

    /// Invalid agent events in the episode, `[half]`.
    #[must_use]
    pub fn episode_invalid_agent_events(&self) -> &[f32] {
        self.view(&self.mode.raw.episode_invalid_agent_events, 1)
    }

    /// Action masks, `[half, A, mask length]`.
    #[must_use]
    pub fn masks(&self) -> &[Bool8] {
        let per = self.agents().saturating_mul(self.mode.raw.mask_len);
        self.view(&self.mode.raw.masks, per)
    }

    /// Mask values per agent.
    #[cfg(feature = "debug-checks")]
    pub(crate) fn mask_len(&self) -> usize {
        self.mode.raw.mask_len
    }

    /// Invalid agent events of the last step, `[half, A]`.
    #[cfg(feature = "debug-checks")]
    pub(crate) fn invalid_agent_event(&self) -> &[Bool8] {
        self.view(&self.mode.raw.invalid_agent_event, self.agents())
    }
}

/// A handle that waits, without popping, until a half is ready
/// (ADR-0047 section 9); it borrows nothing of the pool.
#[derive(Debug, Clone)]
pub struct Waiter {
    pub(crate) ready: Arc<Ready>,
}

impl Waiter {
    /// Waits up to `timeout` until a sent half is ready; returns whether
    /// one is. Polls in short bounded steps.
    #[must_use]
    pub fn wait_ready(&self, timeout: core::time::Duration) -> bool {
        let start = std::time::Instant::now();
        loop {
            if self.ready.ready.iter().any(|f| f.load(Ordering::Acquire)) {
                return true;
            }
            if start.elapsed() >= timeout {
                return false;
            }
            std::thread::park_timeout(core::time::Duration::from_micros(100));
        }
    }
}

/// The half's headers with their instance ids, for the mode's turn check.
pub(crate) fn turn_rows<E: Environment, M: MaskProvider<E>>(
    token: &HalfToken<E, M>,
) -> impl Iterator<Item = (usize, &E::Header)> {
    let start = token.range().start;
    token
        .headers()
        .iter()
        .enumerate()
        .map(move |(i, h)| (start + i, h))
}
