// SPDX-License-Identifier: MIT OR Apache-2.0

//! The shared allocation of the Async mode, its drain job and its batch
//! tokens (ADR-0047 sections 1, 2, 4 and 7). Audited `unsafe` module.
//!
//! [`BatchMode`] lives in one `Arc` and owns the moved `EnvPool`, its raw
//! parts, the submission and completion rings, the id-buffer pool of the
//! tokens and the per-id flags. Workers reach it only through the drain
//! job's data pointer, taken from `Arc::as_ptr`; they hold no `Arc`.
//!
//! The drain job claims runs of ids from the submission ring (dynamic
//! claiming, ADR-0047 section 7; a claim takes its share of the ids it
//! finds, issue #83), steps each run of consecutive ids as one chunk and
//! pushes the claimed ids to the completion ring. Every step reads and
//! writes only its own instance's rows, so an instance's outputs do not
//! depend on how ids are grouped into claims or chunks. A job ends when the
//! submission ring is empty; the caller publishes another when ids remain
//! unclaimed.
//!
//! A [`BatchToken`] is the caller's exclusive right to read the rows of its
//! ids (invariants 3 and 4 of the module): it sets each id's live flag when
//! it is made and clears them with Release in `Drop`, then frees its id
//! buffer, then releases its `Arc` (ADR-0047 section 4). A send checks the
//! flags with Acquire before it admits an id. Views of an id are built as
//! shared slices of that id's rows alone, never as `&mut` and never over
//! another id.
//!
//! The raw parts are taken from a `&mut` of the pool; any later `&mut` of
//! the pool (the writers of ADR-0047 section 3, through `Arc::get_mut`)
//! invalidates them, so every such writer calls [`BatchMode::rebuild`].

use core::ops::Range;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use std::sync::Arc;

use tessera_rl_core::{Bool8, Environment, RecordLayout as _};

use super::rows::{Admitted, RawRows};
use super::{Raw, RawChunk};
use crate::error::EngineError;
use crate::masks::MaskProvider;
use crate::pool::{EnvPool, step_chunk};
use crate::record::Record;
use crate::sys::id_buffers::{BufferHandle, IdBufferPool};
use crate::sys::ring::{CompletionRing, RingError, SubmitRing};
use crate::sys::{Gang, Job};
use crate::validate::check_send_ids;

/// The one shared allocation of an Async pool (ADR-0047 section 2).
pub(crate) struct BatchMode<E: Environment, M: MaskProvider<E>> {
    pub(crate) pool: EnvPool<E, M>,
    raw: RawChunk<E>,
    rows: RawRows<E::Header>,
    workers: usize,
    pub(crate) submit: SubmitRing,
    pub(crate) complete: CompletionRing,
    pub(crate) buffers: IdBufferPool,
    job: Job,
    /// A live token covers the id.
    pub(crate) live: Box<[AtomicBool]>,
    /// The id is sent (or reset) and not yet returned by `recv`.
    pub(crate) in_flight: Box<[AtomicBool]>,
    /// A step of a claimed id panicked (sticky until recovery).
    pub(crate) panicked: AtomicBool,
    /// Ids a worker finished (pushed to completion), for quiescence.
    pub(crate) stepped: AtomicUsize,
    /// Ids pushed to the completion ring and not yet popped, plus ids the
    /// caller staged for its next batch: what a [`BatchWaiter`] waits for.
    /// It wraps below zero for a moment between a pop and the push's count.
    pub(crate) ready: Arc<ReadyCount>,
    /// Test-only jitter before each claimed step, in microseconds.
    #[cfg(test)]
    jitter_us: u32,
}

/// The number of ids ready for a batch, shared with a [`BatchWaiter`].
#[derive(Debug, Default)]
pub(crate) struct ReadyCount {
    pub(crate) count: AtomicUsize,
    pub(crate) batch_size: AtomicUsize,
}

// SAFETY: `BatchMode` is shared by the caller, its tokens and the workers.
// The raw parts and the job point into the pool's heap buffers and into the
// `BatchMode` itself; every access through them follows the module
// invariants (an id's rows are touched by one token holder or one claiming
// worker at a time, ordered by the live and in-flight flags, the ring
// sequences and the gang's Release/Acquire on completion). The rest is the
// pool, read only while shared, the rings and the id buffers (atomics).
#[expect(unsafe_code, reason = "audited module: shared mode allocation")]
unsafe impl<E: Environment, M: MaskProvider<E>> Send for BatchMode<E, M>
where
    E::Config: Send + Sync,
    E::Header: Send + Sync,
    E::State: Send,
{
}

// SAFETY: as for `Send` above; shared `&BatchMode` also reaches the states
// (snapshots read them while tokens are alive), so they must be `Sync`.
#[expect(unsafe_code, reason = "audited module: shared mode allocation")]
unsafe impl<E: Environment, M: MaskProvider<E>> Sync for BatchMode<E, M>
where
    E::Config: Send + Sync,
    E::Header: Send + Sync,
    E::State: Send + Sync,
{
}

impl<E, M> BatchMode<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
{
    /// Moves `pool` into a new shared allocation for `n` instances, batches
    /// of `batch_size` and `workers` workers.
    pub(crate) fn new(
        mut pool: EnvPool<E, M>,
        n: usize,
        batch_size: usize,
        workers: usize,
        #[cfg(test)] jitter_us: u32,
    ) -> Result<Arc<Self>, EngineError> {
        let ring_err = |_| EngineError::Options {
            option: "num_envs",
            reason: "too many instances for the Async rings",
        };
        let submit = SubmitRing::new(n).map_err(ring_err)?;
        let complete = CompletionRing::new(n).map_err(ring_err)?;
        let buffers = IdBufferPool::new(n, batch_size).map_err(|_| EngineError::Options {
            option: "batch_size",
            reason: "must be between 1 and num_envs",
        })?;
        let (raw, rows) = Self::parts(&mut pool);
        let flags = |_| AtomicBool::new(false);
        let mut arc = Arc::new(Self {
            pool,
            raw,
            rows,
            workers,
            submit,
            complete,
            buffers,
            job: placeholder(),
            live: (0..n).map(flags).collect(),
            in_flight: (0..n).map(flags).collect(),
            panicked: AtomicBool::new(false),
            stepped: AtomicUsize::new(0),
            ready: Arc::new(ReadyCount {
                count: AtomicUsize::new(0),
                batch_size: AtomicUsize::new(batch_size),
            }),
            #[cfg(test)]
            jitter_us,
        });
        // The job's data pointer comes from the `Arc` itself, not from the
        // `&mut` below, so it stays valid under later shared borrows.
        let data = Arc::as_ptr(&arc).cast::<()>();
        if let Some(mode) = Arc::get_mut(&mut arc) {
            mode.job = drain_job::<E, M>(data);
        }
        Ok(arc)
    }

    /// Raw parts of every buffer of `pool`, from a fresh `&mut`.
    fn parts(pool: &mut EnvPool<E, M>) -> (RawChunk<E>, RawRows<E::Header>) {
        let (chunk, (dh, ch)) = pool.chunk_all();
        let raw = RawChunk::new(chunk, dh, ch);
        let rows = RawRows::from_parts(
            raw.instances,
            raw.agents,
            (dh, ch),
            Raw {
                ptr: raw.headers.ptr,
                len: raw.headers.len,
            },
            Raw {
                ptr: raw.discrete.ptr,
                len: raw.discrete.len,
            },
            Raw {
                ptr: raw.continuous.ptr,
                len: raw.continuous.len,
            },
        );
        (raw, rows)
    }

    /// Takes the raw parts again after a writer used `&mut` of the pool.
    pub(crate) fn rebuild(&mut self) {
        let (raw, rows) = Self::parts(&mut self.pool);
        self.raw = raw;
        self.rows = rows;
    }

    /// Gives the pool back (`into_pool`).
    pub(crate) fn into_pool(self) -> EnvPool<E, M> {
        self.pool
    }

    /// Admits `ids` to a send and runs `f` on their rows alone: the ids are
    /// checked first (at least one, in range, each once, none in flight and
    /// none covered by a live token), so the rows handed to `f` are this
    /// call's alone while it runs (invariants 3 and 4). Only the caller sets
    /// those flags, under `&mut` of the mode's wrapper, so they cannot
    /// change during `f`.
    ///
    /// With `owner`, the ids are exactly that token's (a resend): they are
    /// covered by it instead of being uncovered. The token is held by value
    /// for the whole call, so no view of its rows can be alive, and it is
    /// handed back with the result.
    pub(crate) fn admit<R>(
        &self,
        ids: &[u32],
        seen: &mut [Bool8],
        owner: Option<BatchToken<E, M>>,
        f: impl FnOnce(&mut Admitted<'_, E::Header>) -> Result<R, EngineError>,
    ) -> (Result<R, EngineError>, Option<BatchToken<E, M>>) {
        let checked = match &owner {
            None => check_send_ids(ids, seen, &self.in_flight, &self.live),
            Some(token) => self.check_owned(token, ids, seen),
        };
        if let Err(e) = checked {
            return (Err(e), owner);
        }
        // SAFETY: every id passed the checks above: no worker touches its
        // rows, and no live view of them exists (no token covers them, or the
        // one that does is held by value here); `ids` repeats no id; the
        // buffers stay in place while `&self` is held (invariants 1, 3, 4).
        #[expect(unsafe_code, reason = "audited module: admitted rows")]
        let mut rows = unsafe { self.rows.admitted(ids) };
        (f(&mut rows), owner)
    }

    /// The checks of a resend: `ids` are exactly `token`'s, in its order,
    /// and none is in flight.
    fn check_owned(
        &self,
        token: &BatchToken<E, M>,
        ids: &[u32],
        seen: &mut [Bool8],
    ) -> Result<(), EngineError> {
        if ids.is_empty() {
            return Err(EngineError::EmptySend);
        }
        crate::validate::check_ids(ids, seen)?;
        if !token.ids().eq(ids.iter().copied()) {
            return Err(EngineError::WrongPool);
        }
        for &id in ids {
            let i = usize::try_from(id).unwrap_or(usize::MAX);
            if self
                .in_flight
                .get(i)
                .is_none_or(|f| f.load(Ordering::Acquire))
            {
                return Err(EngineError::InFlight { id });
            }
        }
        Ok(())
    }

    /// Instance `id`'s masks, `[A, mask length]`, for the debug-checks
    /// legality pass of a send. `owned` says the send's caller holds the
    /// token that covers `id` by value (a resend), so no view of it is alive;
    /// otherwise `id` must be uncovered. `None` if `id` is in flight or its
    /// coverage does not match (then nobody else is known to leave its rows
    /// alone, and the caller must refuse the send rather than skip the
    /// check).
    #[cfg(feature = "debug-checks")]
    pub(crate) fn send_masks(&self, id: usize, owned: bool) -> Option<&[Bool8]> {
        let set = |flags: &[AtomicBool]| flags.get(id).map(|f| f.load(Ordering::Acquire));
        if set(&self.in_flight) != Some(false) || set(&self.live) != Some(owned) {
            return None;
        }
        let per = self.raw.agents.saturating_mul(self.raw.mask_len);
        // SAFETY: `id` is not in flight and is either uncovered or covered
        // by a token its caller holds by value (checked above), so no worker
        // and no view writes or reads its rows mutably while `&self` is held
        // (invariants 3 and 4).
        #[expect(unsafe_code, reason = "audited module: admitted rows")]
        let view = unsafe { shared_view(&self.raw.masks, &(id..id.saturating_add(1)), per) };
        Some(view)
    }

    /// The record (header and state) of instance `id`, read through the raw
    /// parts of that one row (issue #85, ADR-0047 section 3: snapshots only
    /// read). Unlike [`EnvPool::snapshot`], it forms no slice over the pool's
    /// record arrays, so it is sound while workers write other rows.
    ///
    /// # Errors
    ///
    /// [`EngineError::InFlight`] if `id` is sent (or reset) and not yet
    /// received, whether it waits in the ring, is being stepped, or has
    /// completed but not been returned by `recv`: a worker may write its
    /// rows. [`EngineError::InstanceOutOfRange`] for an unknown id.
    pub(crate) fn idle_record(&self, id: u32) -> Result<Record<E>, EngineError> {
        let i = usize::try_from(id).unwrap_or(usize::MAX);
        let (Some(flag), true, true) = (
            self.in_flight.get(i),
            i < self.raw.headers.len,
            i < self.raw.states.len,
        ) else {
            return Err(EngineError::InstanceOutOfRange {
                id,
                num_envs: u32::try_from(self.raw.instances).unwrap_or(u32::MAX),
            });
        };
        if flag.load(Ordering::Acquire) {
            return Err(EngineError::InFlight { id });
        }
        Ok(Record {
            // SAFETY: see `read_idle_row`; row `i` is not in flight (checked
            // above) and is below the headers' length.
            #[expect(unsafe_code, reason = "audited module: idle row snapshot")]
            header: unsafe { read_idle_row(&self.raw.headers, i) },
            // SAFETY: as above, for the states.
            #[expect(unsafe_code, reason = "audited module: idle row snapshot")]
            state: unsafe { read_idle_row(&self.raw.states, i) },
        })
    }

    /// Mask values per agent.
    #[cfg(feature = "debug-checks")]
    pub(crate) const fn mask_len(&self) -> usize {
        self.raw.mask_len
    }
}

/// A shared view of `per` elements per instance of `r`.
///
/// # Safety
///
/// No `&mut` of these rows is alive and no worker writes them while the view
/// is (invariants 1, 3 and 4).
#[expect(unsafe_code, reason = "audited module: token-guarded views")]
unsafe fn shared_view<'a, T>(raw: &Raw<T>, r: &Range<usize>, per: usize) -> &'a [T] {
    let start = r.start.checked_mul(per).map_or(raw.len, |s| s.min(raw.len));
    let end = r
        .end
        .checked_mul(per)
        .map_or(raw.len, |e| e.min(raw.len))
        .max(start);
    // SAFETY: `start <= len`, inside (or one past) the live buffer.
    let first = unsafe { raw.ptr.add(start) };
    // SAFETY: `start <= end <= len` keeps the view inside the live buffer
    // (invariant 1); the caller guarantees nobody writes these rows while it
    // lives (invariants 3 and 4).
    unsafe { core::slice::from_raw_parts(first.cast_const(), end - start) }
}

/// Copies element `i` of `raw` (one row of a per-instance record array),
/// forming no reference to the buffer or to any other row.
///
/// # Safety
///
/// `i < raw.len`, the buffer is live (invariant 1), and row `i` is not in
/// flight while the caller holds the `&BatchMode` that `raw` belongs to:
/// only the caller sets the in-flight flags, under `&mut` of the mode's
/// wrapper, so the flag cannot be set during the read, and no worker holds
/// or forms a reference to row `i` (workers touch only claimed, in-flight
/// rows: invariants 3 and 4). The row's last step happened before its
/// completion push (Release), which `recv` popped (Acquire) on the caller's
/// thread before it cleared the flag. A live token may cover row `i`, but
/// its views are shared, so a read alongside them is sound. Workers may hold
/// `&mut` views of other rows meanwhile; this read does not alias them.
#[expect(unsafe_code, reason = "audited module: idle row snapshot")]
unsafe fn read_idle_row<T: Copy>(raw: &Raw<T>, i: usize) -> T {
    debug_assert!(i < raw.len, "row {i} out of {}", raw.len);
    // SAFETY: `i < len` keeps the offset inside the live buffer (caller).
    let row = unsafe { raw.ptr.add(i) };
    // SAFETY: `row` is aligned and initialised (an element of the live
    // buffer), and no worker writes it during the read (caller).
    unsafe { row.read() }
}

/// A placeholder job that is never published.
#[expect(unsafe_code, reason = "audited module: placeholder job")]
unsafe fn never(_: *const (), _: usize) {}

/// A job replaced before the allocation is shared.
#[expect(unsafe_code, reason = "audited module: placeholder job")]
fn placeholder() -> Job {
    // SAFETY: `never` does nothing and the null data is never read; the job
    // is overwritten in `BatchMode::new` before it is published.
    unsafe { Job::from_raw(never, core::ptr::null()) }
}

/// The drain job of the `BatchMode<E, M>` at `data`.
#[expect(unsafe_code, reason = "audited module: drain job")]
fn drain_job<E: Environment, M: MaskProvider<E>>(data: *const ()) -> Job {
    // SAFETY: `run_drain::<E, M>` reads `data` as a `BatchMode<E, M>`, which
    // it is (taken from `Arc::as_ptr` in `BatchMode::new`); the caller keeps
    // the allocation alive until every job it publishes is waited for.
    unsafe { Job::from_raw(run_drain::<E, M>, data) }
}

/// Claims and steps ids until the submission ring is empty (ADR-0047
/// section 7).
///
/// # Safety
///
/// `data` points to a live `BatchMode<E, M>`; every id in its submission
/// ring was admitted by a send: no token covers it and no other worker
/// steps it (each id is claimed once).
#[expect(unsafe_code, reason = "audited module: drain job")]
unsafe fn run_drain<E: Environment, M: MaskProvider<E>>(data: *const (), worker: usize) {
    // SAFETY: `data` is the `BatchMode` of the allocation the caller keeps
    // alive until this job is waited for; only shared access is taken here.
    let mode = unsafe { &*data.cast::<BatchMode<E, M>>() };
    if worker >= mode.workers {
        return;
    }
    let ctx = mode.pool.step_ctx();
    let mut run = [0u32; CLAIM_RUN];
    loop {
        let claimed = match mode.submit.claim_run(&mut run, mode.workers) {
            Ok(0) => return,
            Ok(k) => run.get(..k).unwrap_or(&[]),
            Err(_) => {
                mode.panicked.store(true, Ordering::Relaxed);
                return;
            }
        };
        let mut rest = claimed;
        while let Some(&first) = rest.first() {
            let len = consecutive(rest);
            #[cfg(test)]
            for &id in rest.get(..len).unwrap_or(&[]) {
                jitter(mode, id);
            }
            let start = usize::try_from(first).unwrap_or(usize::MAX);
            let end = start.saturating_add(len);
            // SAFETY: ids `start..end` were claimed by this worker alone (the
            // ring hands each admitted id out once, and `rest` repeats none)
            // and no token covers them (the send checked each live flag with
            // Acquire before admitting it): invariants 3, 4.
            let view = unsafe { mode.raw.range_view(start..end) };
            if let Some(mut chunk) = view
                && !crate::sys::catch(|| step_chunk::<E, M>(&ctx, &mut chunk))
            {
                mode.panicked.store(true, Ordering::Relaxed);
            }
            rest = rest.get(len..).unwrap_or(&[]);
        }
        // The ids' writes happen before their completion entries (the ring's
        // Release on each slot sequence; the caller pops with Acquire).
        complete(mode, claimed);
    }
}

/// The most ids one claim of the drain job takes from the submission ring
/// (issue #83). A claim takes `ceil(k / workers)` of the `k` ids it finds,
/// up to this many, so a backlog is split among the workers.
const CLAIM_RUN: usize = 64;

/// The length of the leading run of consecutive ids of `ids` (`a, a + 1,
/// ...`), at least one if `ids` is not empty. Such a run is stepped as one
/// chunk; each instance's step reads and writes only its own rows, so the
/// result per instance does not depend on how ids are grouped.
fn consecutive(ids: &[u32]) -> usize {
    let mut len = usize::from(!ids.is_empty());
    while let (Some(&prev), Some(&next)) = (ids.get(len.wrapping_sub(1)), ids.get(len)) {
        if prev.checked_add(1) != Some(next) {
            break;
        }
        len += 1;
    }
    len
}

/// Test-only: sleeps a pseudo-random time before stepping `id`, so that
/// completion order varies far more than send order alone makes it.
#[cfg(test)]
fn jitter<E: Environment, M: MaskProvider<E>>(mode: &BatchMode<E, M>, id: u32) {
    if mode.jitter_us == 0 || mode.jitter_us >= GATE_AFTER_PUSH {
        return;
    }
    let salt = mode.stepped.load(Ordering::Relaxed) as u64;
    let mut z = (u64::from(id) << 32 ^ salt).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    let micros = z % u64::from(mode.jitter_us);
    std::thread::sleep(core::time::Duration::from_micros(micros));
}

/// Pushes the stepped `ids` to the completion ring, in order, retrying
/// while it is transiently full behind a paused pop (ADR-0047 section 7, the
/// ring's module docs).
///
/// The ids count as stepped before any of their entries is visible: a
/// caller that received every id must find nothing running, whatever this
/// worker does next (ADR-0047 section 3). Only ring bookkeeping remains,
/// which no row reader touches, and a writer drains the job before it
/// starts.
fn complete<E: Environment, M: MaskProvider<E>>(mode: &BatchMode<E, M>, ids: &[u32]) {
    mode.stepped.fetch_add(ids.len(), Ordering::Release);
    #[cfg(test)]
    {
        hold_at(mode, GATE_BEFORE_PUSH);
        for &id in ids {
            jitter(mode, id);
        }
    }
    let mut rest = ids;
    while !rest.is_empty() {
        match mode.complete.push_run(rest) {
            Ok(k) => rest = rest.get(k..).unwrap_or(&[]),
            Err(RingError::Full { .. }) => std::thread::yield_now(),
            Err(_) => {
                mode.panicked.store(true, Ordering::Relaxed);
                break;
            }
        }
    }
    #[cfg(test)]
    {
        hold_at(mode, GATE_AFTER_PUSH);
        for &id in ids {
            jitter(mode, id);
        }
    }
    mode.ready.count.fetch_add(ids.len(), Ordering::Release);
}

/// A `jitter_us` that holds a worker in [`complete`] before its push.
#[cfg(test)]
pub(crate) const GATE_BEFORE_PUSH: u32 = u32::MAX;
/// A `jitter_us` that holds a worker in [`complete`] after its push.
#[cfg(test)]
pub(crate) const GATE_AFTER_PUSH: u32 = u32::MAX - 1;
/// Set by a worker held at a gate.
#[cfg(test)]
pub(crate) static GATE_REACHED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);
/// Opens the gates.
#[cfg(test)]
pub(crate) static GATE_OPEN: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Holds this worker while the mode is gated at `gate`, to force a schedule.
#[cfg(test)]
fn hold_at<E: Environment, M: MaskProvider<E>>(mode: &BatchMode<E, M>, gate: u32) {
    if mode.jitter_us == gate {
        GATE_REACHED.store(true, Ordering::Release);
        while !GATE_OPEN.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
    }
}

/// Publishes the drain job of `mode` to `gang`.
///
/// The caller keeps `mode` alive (it holds an `Arc`) and the gang alive
/// until the job is waited for; every id in the submission ring is admitted.
pub(crate) fn publish<E: Environment, M: MaskProvider<E>>(
    gang: &mut Gang,
    mode: &Arc<BatchMode<E, M>>,
) -> Result<(), EngineError> {
    // SAFETY: the job and its data live in the `Arc` allocation, which the
    // caller keeps alive until the `wait` that returns the job (its `Drop`
    // drains every job first); the ids it claims were admitted (module
    // invariants).
    #[expect(unsafe_code, reason = "audited module: drain job publication")]
    let submitted = unsafe { gang.submit(&mode.job) };
    submitted
}

/// The caller's exclusive right to read the rows of a batch of ids
/// (ADR-0047 section 1). Results borrow the token, never the pool; dropping
/// it, or sending it, ends every view of its ids.
pub struct BatchToken<E: Environment, M: MaskProvider<E>> {
    // Dropped last: `Drop` clears the flags and frees the buffer first.
    mode: Arc<BatchMode<E, M>>,
    handle: Option<BufferHandle>,
}

impl<E: Environment, M: MaskProvider<E>> core::fmt::Debug for BatchToken<E, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BatchToken")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

impl<E: Environment, M: MaskProvider<E>> Drop for BatchToken<E, M> {
    fn drop(&mut self) {
        let Some(handle) = self.handle.take() else {
            return;
        };
        // 1. The flags: every read through the token happens before the next
        //    admission's Acquire of an id's flag.
        if let Ok(entries) = self.mode.buffers.entries(&handle) {
            for (id, _) in entries {
                let i = usize::try_from(id).unwrap_or(usize::MAX);
                if let Some(flag) = self.mode.live.get(i) {
                    super::live::release(flag);
                }
            }
        }
        // 2. The buffer (a waiting `recv` polls, so the wake hint is not
        //    needed). 3. The `Arc`, when `self.mode` drops after this.
        let _ = self.mode.buffers.release(handle);
    }
}

impl<E: Environment, M: MaskProvider<E>> BatchToken<E, M> {
    /// A token for `ids` with their export `generations`, all of which are
    /// returned and uncovered. `None` if no buffer of the right size is free
    /// (a drop is still freeing one) or an id is already covered.
    pub(crate) fn new(
        mode: &Arc<BatchMode<E, M>>,
        ids: &[u32],
        generations: &[u64],
        full: bool,
    ) -> Option<Self> {
        let handle = mode.buffers.acquire(full)?;
        if mode.buffers.fill(&handle, ids, generations).is_err() {
            let _ = mode.buffers.release(handle);
            return None;
        }
        // Claim every id (Acquire: the completed writes of each id are
        // visible to the views). A failed claim undoes the earlier ones.
        for (k, &id) in ids.iter().enumerate() {
            let flag = mode.live.get(usize::try_from(id).unwrap_or(usize::MAX));
            if !flag.is_some_and(super::live::claim) {
                for &done in ids.get(..k).unwrap_or(&[]) {
                    if let Some(f) = mode.live.get(usize::try_from(done).unwrap_or(usize::MAX)) {
                        super::live::release(f);
                    }
                }
                let _ = mode.buffers.release(handle);
                return None;
            }
        }
        Some(Self {
            mode: Arc::clone(mode),
            handle: Some(handle),
        })
    }

    /// Whether `self` belongs to `mode`.
    pub(crate) fn belongs_to(&self, mode: &Arc<BatchMode<E, M>>) -> bool {
        Arc::ptr_eq(&self.mode, mode)
    }

    /// The number of ids in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries().count()
    }

    /// Whether the batch is empty (never, for a token from the mode).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn entries(&self) -> impl Iterator<Item = (u32, u64)> + '_ {
        self.handle
            .as_ref()
            .and_then(|h| self.mode.buffers.entries(h).ok())
            .into_iter()
            .flatten()
    }

    /// The ids of the batch, in the order they completed.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.entries().map(|(id, _)| id)
    }

    /// Every id of the batch with its rows, in the order they completed. Each
    /// row costs O(1), where the per-id accessors search the batch.
    pub fn rows(&self) -> impl Iterator<Item = BatchRow<'_, E, M>> + '_ {
        self.entries().map(|(id, generation)| BatchRow {
            token: self,
            id,
            generation,
        })
    }

    /// The export generation of `id` when the batch was received
    /// (ADR-0047 section 5), if `id` is in the batch.
    #[must_use]
    pub fn generation(&self, id: u32) -> Option<u64> {
        self.entries().find(|&(i, _)| i == id).map(|(_, g)| g)
    }

    /// Whether `id` is in the batch.
    #[must_use]
    pub fn contains(&self, id: u32) -> bool {
        self.entries().any(|(i, _)| i == id)
    }

    /// A shared view of `per` elements of `id`'s rows.
    fn view<T>(&self, raw: &Raw<T>, id: u32, per: usize) -> Option<&[T]> {
        if !self.contains(id) {
            return None;
        }
        self.view_covered(raw, id, per)
    }

    /// As [`Self::view`], for an id taken from this token's own entries.
    fn view_covered<T>(&self, raw: &Raw<T>, id: u32, per: usize) -> Option<&[T]> {
        let i = usize::try_from(id).ok()?;
        let r = (i < self.mode.raw.instances).then(|| i..i + 1)?;
        // SAFETY: the token covers `id`, so no worker writes its rows while
        // `self` is borrowed and every other view of them is shared
        // (invariants 3 and 4); the buffers are in place (invariant 1).
        #[expect(unsafe_code, reason = "audited module: token-guarded views")]
        let view = unsafe { shared_view(raw, &r, per) };
        Some(view)
    }

    fn agents(&self) -> usize {
        self.mode.raw.agents
    }

    fn header(&self, id: u32) -> Option<&E::Header> {
        self.view(&self.mode.raw.headers, id, 1)?.first()
    }

    /// The step counter of `id` (episode index, step index), if `id` is in
    /// the batch.
    #[must_use]
    pub fn step_counter(&self, id: u32) -> Option<(u32, u32)> {
        let identity = &self.header(id)?.engine().identity;
        Some((identity.episode_index(), identity.step_index()))
    }

    fn leaf_bytes(&self, leaf: usize) -> Option<usize> {
        let raw = self.mode.raw.leaf_bytes;
        // SAFETY: `leaf_bytes` is never written after construction, so a
        // shared view of it is sound alongside any worker (invariant 1).
        #[expect(unsafe_code, reason = "audited module: read-only layout")]
        let bytes = unsafe { core::slice::from_raw_parts(raw.ptr.cast_const(), raw.len) };
        bytes.get(leaf).copied()
    }

    /// Observations of leaf `leaf` of `id`, `[A, leaf bytes]`.
    #[must_use]
    pub fn observation(&self, leaf: usize, id: u32) -> Option<&[u8]> {
        let raw = self.mode.raw.observations.get(leaf).copied()?;
        let per = self.agents().checked_mul(self.leaf_bytes(leaf)?)?;
        self.view(&raw, id, per)
    }

    /// Final observations of leaf `leaf` of `id`, `[A, leaf bytes]`
    /// (`SameStep` auto-reset).
    #[must_use]
    pub fn terminal_observation(&self, leaf: usize, id: u32) -> Option<&[u8]> {
        let raw = self.mode.raw.terminal.get(leaf).copied()?;
        let per = self.agents().checked_mul(self.leaf_bytes(leaf)?)?;
        self.view(&raw, id, per)
    }

    /// Rewards of `id`, `[A]`.
    #[must_use]
    pub fn rewards(&self, id: u32) -> Option<&[f32]> {
        self.view(&self.mode.raw.rewards, id, self.agents())
    }

    /// Agents of `id` terminated in the last step, `[A]`.
    #[must_use]
    pub fn agent_terminated(&self, id: u32) -> Option<&[Bool8]> {
        self.view(&self.mode.raw.agent_terminated, id, self.agents())
    }

    /// Agents of `id` truncated in the last step, `[A]`.
    #[must_use]
    pub fn agent_truncated(&self, id: u32) -> Option<&[Bool8]> {
        self.view(&self.mode.raw.agent_truncated, id, self.agents())
    }

    /// Alive agent slots of `id`, `[A]`.
    #[must_use]
    pub fn alive(&self, id: u32) -> Option<&[Bool8]> {
        self.view(&self.mode.raw.alive, id, self.agents())
    }

    /// Acting agent slots of `id`, `[A]`.
    #[must_use]
    pub fn acting(&self, id: u32) -> Option<&[Bool8]> {
        self.view(&self.mode.raw.acting, id, self.agents())
    }

    /// Agent slot generations of `id`, `[A]`.
    #[must_use]
    pub fn generations(&self, id: u32) -> Option<&[u32]> {
        self.view(&self.mode.raw.generations, id, self.agents())
    }

    /// Action masks of `id`, `[A, mask length]`.
    #[must_use]
    pub fn masks(&self, id: u32) -> Option<&[Bool8]> {
        let per = self.agents().saturating_mul(self.mode.raw.mask_len);
        self.view(&self.mode.raw.masks, id, per)
    }

    fn one<T: Copy>(&self, raw: &Raw<T>, id: u32) -> Option<T> {
        self.view(raw, id, 1)?.first().copied()
    }

    /// Whether `id`'s episode terminated in the last step.
    #[must_use]
    pub fn episode_terminated(&self, id: u32) -> Option<Bool8> {
        self.one(&self.mode.raw.episode_terminated, id)
    }

    /// Whether `id`'s episode was truncated in the last step.
    #[must_use]
    pub fn episode_truncated(&self, id: u32) -> Option<Bool8> {
        self.one(&self.mode.raw.episode_truncated, id)
    }

    /// The episode return of `id`.
    #[must_use]
    pub fn episode_return(&self, id: u32) -> Option<f32> {
        self.one(&self.mode.raw.episode_return, id)
    }

    /// The episode length of `id`.
    #[must_use]
    pub fn episode_length(&self, id: u32) -> Option<f32> {
        self.one(&self.mode.raw.episode_length, id)
    }

    /// Illegal actions in `id`'s episode.
    #[must_use]
    pub fn episode_illegal_actions(&self, id: u32) -> Option<f32> {
        self.one(&self.mode.raw.episode_illegal_actions, id)
    }

    /// Invalid agent events in `id`'s episode.
    #[must_use]
    pub fn episode_invalid_agent_events(&self, id: u32) -> Option<f32> {
        self.one(&self.mode.raw.episode_invalid_agent_events, id)
    }

    /// Invalid agent events of `id` in the last step, `[A]`.
    #[cfg(feature = "debug-checks")]
    pub(crate) fn invalid_agent_event(&self, id: u32) -> Option<&[Bool8]> {
        self.view(&self.mode.raw.invalid_agent_event, id, self.agents())
    }
}

/// One id of a [`BatchToken`] with its rows. It can only come from the token's
/// own entries, so it exposes only that token's rows (ADR-0047 section 1).
pub struct BatchRow<'a, E: Environment, M: MaskProvider<E>> {
    token: &'a BatchToken<E, M>,
    id: u32,
    generation: u64,
}

impl<'a, E: Environment, M: MaskProvider<E>> BatchRow<'a, E, M> {
    /// The instance id.
    #[must_use]
    pub const fn id(&self) -> u32 {
        self.id
    }

    /// The export generation of the id when the batch was received.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    fn view<T>(&self, raw: &'a Raw<T>, per: usize) -> Option<&'a [T]> {
        self.token.view_covered(raw, self.id, per)
    }

    fn agent_view<T>(&self, raw: &'a Raw<T>) -> Option<&'a [T]> {
        self.view(raw, self.token.agents())
    }

    /// Observations of leaf `leaf`, `[A, leaf bytes]`.
    #[must_use]
    pub fn observation(&self, leaf: usize) -> Option<&'a [u8]> {
        let raw = self.token.mode.raw.observations.get(leaf)?;
        let per = self
            .token
            .agents()
            .checked_mul(self.token.leaf_bytes(leaf)?)?;
        self.view(raw, per)
    }

    /// Final observations of leaf `leaf`, `[A, leaf bytes]` (`SameStep`
    /// auto-reset).
    #[must_use]
    pub fn terminal_observation(&self, leaf: usize) -> Option<&'a [u8]> {
        let raw = self.token.mode.raw.terminal.get(leaf)?;
        let per = self
            .token
            .agents()
            .checked_mul(self.token.leaf_bytes(leaf)?)?;
        self.view(raw, per)
    }

    /// Rewards, `[A]`.
    #[must_use]
    pub fn rewards(&self) -> Option<&'a [f32]> {
        self.agent_view(&self.token.mode.raw.rewards)
    }

    /// Agents terminated in the last step, `[A]`.
    #[must_use]
    pub fn agent_terminated(&self) -> Option<&'a [Bool8]> {
        self.agent_view(&self.token.mode.raw.agent_terminated)
    }

    /// Agents truncated in the last step, `[A]`.
    #[must_use]
    pub fn agent_truncated(&self) -> Option<&'a [Bool8]> {
        self.agent_view(&self.token.mode.raw.agent_truncated)
    }

    /// Alive agent slots, `[A]`.
    #[must_use]
    pub fn alive(&self) -> Option<&'a [Bool8]> {
        self.agent_view(&self.token.mode.raw.alive)
    }

    /// Acting agent slots, `[A]`.
    #[must_use]
    pub fn acting(&self) -> Option<&'a [Bool8]> {
        self.agent_view(&self.token.mode.raw.acting)
    }

    /// Action masks, `[A, mask length]`.
    #[must_use]
    pub fn masks(&self) -> Option<&'a [Bool8]> {
        let raw = &self.token.mode.raw;
        self.view(&raw.masks, self.token.agents().saturating_mul(raw.mask_len))
    }

    /// Whether the episode terminated in the last step.
    #[must_use]
    pub fn episode_terminated(&self) -> Option<Bool8> {
        self.view(&self.token.mode.raw.episode_terminated, 1)?
            .first()
            .copied()
    }

    /// Whether the episode was truncated in the last step.
    #[must_use]
    pub fn episode_truncated(&self) -> Option<Bool8> {
        self.view(&self.token.mode.raw.episode_truncated, 1)?
            .first()
            .copied()
    }

    /// The header, for the mode's turn check.
    pub(crate) fn header(&self) -> Option<&'a E::Header> {
        self.view(&self.token.mode.raw.headers, 1)?.first()
    }
}

/// A handle that waits, without popping, until a batch is ready (ADR-0047
/// section 9); it borrows nothing of the pool.
#[derive(Debug, Clone)]
pub struct BatchWaiter {
    pub(crate) ready: Arc<ReadyCount>,
}

impl BatchWaiter {
    /// Waits up to `timeout` until a full batch of ids is ready; returns
    /// whether one is. Polls in short bounded steps.
    #[must_use]
    pub fn wait_ready(&self, timeout: core::time::Duration) -> bool {
        let start = std::time::Instant::now();
        loop {
            let need = self.ready.batch_size.load(Ordering::Relaxed);
            // Signed: a receive can pop an id before its worker counts it, so
            // the count is briefly below zero, which means nothing is ready.
            if self.ready.count.load(Ordering::Acquire).cast_signed() >= need.cast_signed() {
                return true;
            }
            if start.elapsed() >= timeout {
                return false;
            }
            // A sleep, not a park: nobody unparks this thread, and on macOS a
            // park traps in the child of a multithreaded parent (issue #93).
            std::thread::sleep(core::time::Duration::from_micros(100));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::consecutive;

    /// A run of consecutive ids ends at the first gap, a descent or a
    /// repeat, and at `u32::MAX`, which has no successor.
    #[test]
    fn test_consecutive_run_ends_at_the_first_gap_expected() {
        assert_eq!(consecutive(&[]), 0);
        assert_eq!(consecutive(&[7]), 1);
        assert_eq!(consecutive(&[3, 4, 5, 7, 8]), 3);
        assert_eq!(consecutive(&[5, 4]), 1);
        assert_eq!(consecutive(&[5, 5]), 1);
        assert_eq!(consecutive(&[u32::MAX - 1, u32::MAX, 0]), 2);
    }
}
