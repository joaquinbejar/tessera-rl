// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Async mode (ADR-0006, ADR-0047): ids are sent in any subsets and
//! received in batches of `batch_size`, the first ready.
//!
//! [`EnvPool::async_mode`] moves the pool into one shared allocation and
//! takes its workers. [`AsyncPool::send`] admits ids with their actions into
//! the submission ring; workers claim them one at a time and push each to
//! the completion ring; [`AsyncPool::recv`] returns the first `batch_size`
//! completed ids as a [`BatchToken`]. Results borrow the token, and an id
//! cannot be sent again while a token covers it (ADR-0047 section 1).
//! Determinism is per instance: batch membership depends on timing, an
//! instance's trajectory does not (ADR-0006).
//!
//! Every caller-side writer (`reset_all`, `reset`, `restore`,
//! `restore_all`, recovery, `into_pool`) needs no admitted id still
//! running and no live token (ADR-0047 section 3). `reset(ids)` and
//! `restore(ids)` return nothing: their ids are delivered by the next
//! `recv`, in a normal batch.
//!
//! The pool has no pool-level adaptive subsystem, so there is no unfrozen
//! adaptive state to refuse yet; and no environment uses lane (`SoA`) layouts, so
//! [`EngineError::LaneEnvUnsupported`] has no trigger yet (ADR-0047
//! section 8). Both refusals belong where those features land.
//!
//! # Snapshots
//!
//! `snapshot` needs whole-pool quiescence today; per-row snapshot through
//! raw parts is tracked in issue #85. This is a recorded deviation from
//! design D11, which asks only that the snapshotted rows be quiescent. The
//! snapshotted id itself must also not be in flight (`InFlight`).
//!
//! # Performance
//!
//! Async is currently much slower per round than Sync. Workers claim one
//! instance at a time and the waits poll in 100 µs steps. An indicative
//! local measurement (`benches/async_step.rs`, a loaded Mac): a round of
//! 256 instances received in batches of 64 took about 183 µs on the park
//! backend and 58 µs on the native one, against 15 µs and 9 µs for one Sync
//! step of the same pool. Batch claiming and wake-based waits are tracked in
//! issue #83.

use core::num::NonZeroUsize;
use core::time::Duration;

use std::sync::Arc;
use std::time::Instant;

use tessera_rl_core::{AdaptiveValues, Bool8, Environment, RecordLayout, TurnStructure};

use super::pipelined::Workers;
use crate::buffer::Generations;
use crate::buffer::batch::{BatchMode, BatchToken, BatchWaiter, publish};
use crate::error::EngineError;
use crate::masks::MaskProvider;
use crate::options::SyncOptions;
use crate::pool::{EnvPool, Reseed, SyncRuntime};
use crate::record::Record;
use crate::sys::ring::RingError;
use crate::validate::{
    ActionInput, check_recv, check_send_ids, check_subset_actions, copy_subset_actions,
};

/// Reads the export generations of an Async pool's ids from any thread
/// (ADR-0047 section 5). A generation moves before the send that moves it
/// admits its id.
#[derive(Debug, Clone)]
pub struct IdGenerations(Arc<Generations>);

impl IdGenerations {
    /// The generation of `id`; `None` out of range.
    #[must_use]
    pub fn id(&self, id: u32) -> Option<u64> {
        self.0.id(usize::try_from(id).ok()?).ok()
    }
}

#[cfg(test)]
std::thread_local! {
    /// Runs inside the next `snapshot` of this thread, between its poison
    /// check and its quiescence check, which then waits for the running ids.
    pub(crate) static SNAPSHOT_HOOK: core::cell::Cell<Option<fn()>> =
        const { core::cell::Cell::new(None) };
}

/// Options of the Async mode: the workers, as in Sync mode, and the batch
/// size of `recv`.
#[derive(Debug, Clone)]
pub struct AsyncOptions {
    /// Workers, spin, core set and wait backend (as [`SyncOptions`]).
    pub sync: SyncOptions,
    /// Ids per received batch, between 1 and the number of instances.
    pub batch_size: NonZeroUsize,
}

impl AsyncOptions {
    /// Options with `workers` workers, batches of `batch_size` and the Sync
    /// defaults.
    #[must_use]
    pub const fn new(workers: NonZeroUsize, batch_size: NonZeroUsize) -> Self {
        Self {
            sync: SyncOptions::new(workers),
            batch_size,
        }
    }
}

/// `EnvPool::async_mode` refused; the pool is handed back when the refusal
/// came before it was converted.
pub struct AsyncBuildError<E: Environment, M: MaskProvider<E>> {
    /// Why.
    pub error: EngineError,
    /// The pool, unchanged, if it was not consumed.
    pub pool: Option<Box<EnvPool<E, M>>>,
}

impl<E: Environment, M: MaskProvider<E>> core::fmt::Debug for AsyncBuildError<E, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AsyncBuildError")
            .field("error", &self.error)
            .field("pool", &self.pool.is_some())
            .finish()
    }
}

/// A `send_batch` that was refused. With the token handed back (`Some`),
/// nothing changed. `None` only if publication failed after validation and
/// the token had ended: the ids' action rows then hold the new actions and
/// their generations have moved, and they are admitted to the submission
/// ring; a later `recv` returns them, or reports why it cannot.
#[derive(Debug)]
pub struct SendBatchError<E: Environment, M: MaskProvider<E>> {
    /// Why the send was refused.
    pub error: EngineError,
    /// The token, unchanged.
    pub token: Option<BatchToken<E, M>>,
}

/// `AsyncPool::into_pool` refused because a token is alive; the wrapper is
/// handed back, still working.
#[derive(Debug)]
pub struct AsyncIntoPoolError<E: Environment, M: MaskProvider<E>> {
    /// The pool, unchanged.
    pub pool: Box<AsyncPool<E, M>>,
}

/// A pool in Async mode (ADR-0047). It never derefs to [`EnvPool`];
/// [`Self::into_pool`] gives the pool back.
pub struct AsyncPool<E: Environment, M: MaskProvider<E>> {
    // Declared first, so it drops first: its `Drop` drains every job and
    // joins the workers before the mode's `Arc` below is released.
    runtime: Workers,
    mode: Arc<BatchMode<E, M>>,
    generations: Arc<Generations>,
    seen: Vec<Bool8>,
    /// Completed ids popped for the next batch.
    stage: Vec<u32>,
    stage_gens: Vec<u64>,
    /// Ids of a token being sent back (`send_batch`).
    scratch: Vec<u32>,
    /// Ids admitted to the submission ring since the last full writer.
    submitted: usize,
    /// Ids sent or reset and not yet returned by `recv`.
    outstanding: usize,
    batch: usize,
    poisoned: bool,
}

impl<E: Environment, M: MaskProvider<E>> core::fmt::Debug for AsyncPool<E, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AsyncPool")
            .field("batch_size", &self.batch)
            .field("outstanding", &self.outstanding)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl<E, M> EnvPool<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
    E::Config: Send + Sync,
    E::Header: Send + Sync,
    E::State: Send + Sync,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    /// Moves the pool into Async mode with its own workers (ADR-0047).
    ///
    /// # Errors
    ///
    /// [`EngineError::Options`] if the pool already runs in Sync mode or
    /// `batch_size` exceeds the number of instances, with the pool handed
    /// back; any error of [`EnvPool::with_workers`], without it.
    pub fn async_mode(
        self,
        options: AsyncOptions,
    ) -> Result<AsyncPool<E, M>, AsyncBuildError<E, M>> {
        self.async_inner(
            options,
            #[cfg(test)]
            crate::sys::Faults::default(),
        )
    }

    /// [`Self::async_mode`] with test-only fault injection.
    #[cfg(test)]
    pub(crate) fn async_faults(
        self,
        options: AsyncOptions,
        faults: crate::sys::Faults,
    ) -> Result<AsyncPool<E, M>, AsyncBuildError<E, M>> {
        self.async_inner(options, faults)
    }

    fn async_inner(
        self,
        options: AsyncOptions,
        #[cfg(test)] faults: crate::sys::Faults,
    ) -> Result<AsyncPool<E, M>, AsyncBuildError<E, M>> {
        let refuse = |pool: Self, option, reason| AsyncBuildError {
            error: EngineError::Options { option, reason },
            pool: Some(Box::new(pool)),
        };
        if self.workers() != 0 {
            return Err(refuse(
                self,
                "workers",
                "the pool already runs in Sync mode",
            ));
        }
        let n = usize::try_from(self.options.num_envs.get()).unwrap_or(usize::MAX);
        let batch = options.batch_size.get();
        if batch > n {
            return Err(refuse(self, "batch_size", "must not exceed num_envs"));
        }
        let lost = |error| AsyncBuildError { error, pool: None };
        let mut pool = self
            .with_workers_inner(
                options.sync,
                #[cfg(test)]
                faults,
            )
            .map_err(lost)?;
        let Some(runtime) = pool.take_runtime() else {
            return Err(lost(EngineError::WorkerDied));
        };
        let workers = runtime.workers;
        // An already poisoned pool stays poisoned: only a full reset or
        // restore of the wrapper recovers it, as everywhere else.
        let poisoned = pool.is_poisoned();
        let mode = BatchMode::new(
            pool,
            n,
            batch,
            workers,
            #[cfg(test)]
            faults.jitter_us,
        )
        .map_err(lost)?;
        Ok(AsyncPool {
            runtime: Workers(Some(runtime)),
            mode,
            generations: Arc::new(Generations::new(n)),
            seen: vec![Bool8::FALSE; n],
            stage: Vec::with_capacity(n),
            stage_gens: Vec::with_capacity(n),
            scratch: Vec::with_capacity(n),
            submitted: 0,
            outstanding: 0,
            batch,
            poisoned,
        })
    }
}

fn id_of(index: usize) -> u32 {
    u32::try_from(index).unwrap_or(u32::MAX)
}

impl<E, M> AsyncPool<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
    E::Config: Send + Sync,
    E::Header: Send + Sync,
    E::State: Send + Sync,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    /// The environment configuration.
    #[must_use]
    pub fn config(&self) -> &E::Config {
        self.mode.pool.config()
    }

    /// Number of instances.
    #[must_use]
    pub fn num_envs(&self) -> u32 {
        self.mode.pool.options.num_envs.get()
    }

    /// Ids per received batch.
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.batch
    }

    /// Whether a panic or a dead worker poisoned the pool.
    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
            || self.dead()
            || self
                .mode
                .panicked
                .load(core::sync::atomic::Ordering::Acquire)
            || self.runtime.0.as_ref().is_some_and(|rt| rt.gang.poisoned())
    }

    /// The export generation of instance `id` (ADR-0047 section 5): it
    /// moves on every accepted send and every write of the instance's rows.
    ///
    /// # Errors
    ///
    /// [`EngineError::InstanceOutOfRange`] for an unknown id.
    pub fn generation(&self, id: u32) -> Result<u64, EngineError> {
        self.generations
            .id(usize::try_from(id).unwrap_or(usize::MAX))
            .map_err(|_| self.out_of_range(id))
    }

    fn out_of_range(&self, id: u32) -> EngineError {
        EngineError::InstanceOutOfRange {
            id,
            num_envs: self.num_envs(),
        }
    }

    /// A handle reading the ids' generations from any thread.
    #[must_use]
    pub fn id_generations(&self) -> IdGenerations {
        IdGenerations(Arc::clone(&self.generations))
    }

    /// A [`BatchWaiter`] that waits for a full batch from any thread.
    #[must_use]
    pub fn waiter(&self) -> BatchWaiter {
        BatchWaiter {
            ready: Arc::clone(&self.mode.ready),
        }
    }

    fn runtime(&mut self) -> Result<&mut SyncRuntime, EngineError> {
        self.runtime.0.as_mut().ok_or(EngineError::WorkerDied)
    }

    fn dead(&self) -> bool {
        self.runtime.0.as_ref().is_none_or(|rt| rt.gang.dead())
    }

    /// Ids admitted to the submission ring and not yet stepped.
    fn running(&self) -> usize {
        self.submitted.saturating_sub(
            self.mode
                .stepped
                .load(core::sync::atomic::Ordering::Acquire),
        )
    }

    fn first_live(&self) -> Option<u32> {
        self.mode
            .live
            .iter()
            .position(crate::buffer::live::is_live)
            .map(id_of)
    }

    /// Waits for every job in flight (recovery and `into_pool`). A drained
    /// job that panicked poisons the wrapper.
    fn drain(&mut self) {
        let mut failed = self.runtime.drain();
        // A send that found two jobs already past their last claim published
        // nothing, so admitted ids can wait in the ring with no job left:
        // publish until every admitted id is stepped (or no worker is left).
        // A panicked claim (positions exhausted) makes no progress: stop.
        while self.running() != 0
            && !self.dead()
            && !self
                .mode
                .panicked
                .load(core::sync::atomic::Ordering::Acquire)
        {
            let mode = Arc::clone(&self.mode);
            let published = self.runtime().and_then(|rt| publish(&mut rt.gang, &mode));
            drop(mode);
            if published.is_err() {
                break;
            }
            failed |= self.runtime.drain();
        }
        if failed
            || self
                .mode
                .panicked
                .load(core::sync::atomic::Ordering::Acquire)
        {
            self.poisoned = true;
        }
    }

    /// Exclusive access to the pool for a caller-side writer (ADR-0047
    /// section 3): no admitted id still running and no live token. A token
    /// dropping on another thread may still hold its `Arc`: that is
    /// `Borrowed` too.
    fn exclusive(&mut self, recovery: bool) -> Result<&mut BatchMode<E, M>, EngineError> {
        if self.dead() {
            return Err(EngineError::WorkerDied);
        }
        if self.running() != 0 && !recovery {
            // Make progress on admitted ids first, so a caller retrying
            // after `NotQuiescent` is never stuck.
            self.pump(false)?;
            return Err(EngineError::NotQuiescent);
        }
        // Every id has finished (or this is a recovery): the jobs end at
        // once on an empty ring, so draining them is bounded.
        self.drain();
        if self.running() != 0 {
            return Err(EngineError::WorkerDied);
        }
        if let Some(id) = self.first_live() {
            return Err(EngineError::Borrowed { id });
        }
        Arc::get_mut(&mut self.mode).ok_or(EngineError::Borrowed { id: 0 })
    }

    /// Runs a caller-side writer on the pool. The raw parts are taken again
    /// whatever the outcome; a writer that poisoned the pool poisons the
    /// wrapper.
    fn write(
        &mut self,
        recovery: bool,
        f: impl FnOnce(&mut EnvPool<E, M>) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        let mode = self.exclusive(recovery)?;
        let result = f(&mut mode.pool);
        let poisoned = mode.pool.is_poisoned();
        mode.rebuild();
        if result.is_err() && poisoned {
            self.poisoned = true;
        }
        result
    }

    /// After a successful full writer: every id gets a new generation, the
    /// rings and the stage are emptied, and the poison is cleared.
    fn after_full_write(&mut self) -> Result<(), EngineError> {
        for id in 0..self.generations.len() {
            self.generations
                .advance_id(id)
                .map_err(|_| EngineError::SequenceExhausted)?;
        }
        // Discard every completed, unreturned id: the writer rewrote its rows.
        while self
            .mode
            .complete
            .pop_shared()
            .map_err(|_| EngineError::SequenceExhausted)?
            .is_some()
        {}
        for flag in &*self.mode.in_flight {
            flag.store(false, core::sync::atomic::Ordering::Release);
        }
        self.stage.clear();
        self.submitted = 0;
        self.outstanding = 0;
        self.mode
            .stepped
            .store(0, core::sync::atomic::Ordering::Release);
        self.mode
            .ready
            .count
            .store(0, core::sync::atomic::Ordering::Release);
        self.mode
            .panicked
            .store(false, core::sync::atomic::Ordering::Release);
        if let Some(rt) = self.runtime.0.as_mut() {
            rt.gang.clear_poison()?;
        }
        self.poisoned = false;
        Ok(())
    }

    /// Resets every instance and returns one token covering all of them.
    /// Also the recovery after a panic: it drains first. Completed ids not
    /// yet received are discarded.
    ///
    /// # Errors
    ///
    /// [`EngineError::NotQuiescent`] while a sent id still runs on a healthy
    /// pool, [`EngineError::Borrowed`] while a token is alive,
    /// [`EngineError::WorkerDied`] after a worker died, or any error of
    /// [`EnvPool::reset_all`].
    pub fn reset_all(&mut self, reseed: Reseed<'_>) -> Result<BatchToken<E, M>, EngineError> {
        let recovery = self.is_poisoned();
        self.write(recovery, |pool| pool.reset_all(reseed).map(|_| ()))?;
        self.after_full_write()?;
        let n = self.generations.len();
        self.scratch.clear();
        self.scratch.extend((0..n).map(id_of));
        self.stage_gens.clear();
        for id in 0..n {
            let g = self
                .generations
                .id(id)
                .map_err(|_| EngineError::SequenceExhausted)?;
            self.stage_gens.push(g);
        }
        BatchToken::new(&self.mode, &self.scratch, &self.stage_gens, true)
            .ok_or(EngineError::Borrowed { id: 0 })
    }

    /// Restores every instance from `records`. Completed ids not yet
    /// received are discarded; there is no token afterwards until ids are
    /// sent and received (or [`Self::reset`] or [`Self::restore`] deliver
    /// some).
    ///
    /// # Errors
    ///
    /// As [`Self::reset_all`], or any error of [`EnvPool::restore_all`].
    pub fn restore_all(&mut self, records: &[Record<E>]) -> Result<(), EngineError> {
        let recovery = self.is_poisoned();
        self.write(recovery, |pool| pool.restore_all(records))?;
        self.after_full_write()
    }

    /// Checks ids for a partial writer: the same rules as a send.
    fn check_partial(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        if self.is_poisoned() {
            return Err(EngineError::Poisoned);
        }
        check_send_ids(ids, &mut self.seen, &self.mode.in_flight, &self.mode.live)
    }

    /// After a successful partial writer: the ids get new generations and
    /// are delivered by the next `recv`.
    fn deliver(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        for &id in ids {
            let i = usize::try_from(id).unwrap_or(usize::MAX);
            self.generations
                .advance_id(i)
                .map_err(|_| EngineError::SequenceExhausted)?;
            if let Some(flag) = self.mode.in_flight.get(i) {
                flag.store(true, core::sync::atomic::Ordering::Release);
            }
            self.push_completion(id)?;
            self.outstanding = self.outstanding.saturating_add(1);
        }
        Ok(())
    }

    fn push_completion(&self, id: u32) -> Result<(), EngineError> {
        loop {
            match self.mode.complete.push(id) {
                Ok(()) => break,
                Err(RingError::Full { .. }) => std::thread::yield_now(),
                Err(_) => return Err(EngineError::SequenceExhausted),
            }
        }
        self.mode
            .ready
            .count
            .fetch_add(1, core::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Resets the instances `ids`; they are delivered by the next `recv`
    /// (ADR-0047 section 3).
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`], [`EngineError::EmptySend`], an id error
    /// ([`EngineError::InFlight`], [`EngineError::Borrowed`], duplicates or
    /// out of range), [`EngineError::NotQuiescent`] while a sent id still
    /// runs, [`EngineError::Borrowed`] while any token is alive, or any
    /// error of [`EnvPool::reset`]. Nothing changes on an error.
    pub fn reset(&mut self, ids: &[u32], reseed: Reseed<'_>) -> Result<(), EngineError> {
        self.check_partial(ids)?;
        self.write(false, |pool| pool.reset(ids, reseed).map(|_| ()))?;
        self.deliver(ids)
    }

    /// Restores the instances `ids` from `records` (one each, in order);
    /// they are delivered by the next `recv`.
    ///
    /// # Errors
    ///
    /// As [`Self::reset`], [`EngineError::RecordCount`] if the counts
    /// differ, or any error of [`EnvPool::restore`].
    pub fn restore(&mut self, ids: &[u32], records: &[Record<E>]) -> Result<(), EngineError> {
        self.check_partial(ids)?;
        if records.len() != ids.len() {
            return Err(EngineError::RecordCount {
                records: records.len(),
                instances: ids.len(),
            });
        }
        self.write(false, |pool| {
            for (&id, record) in ids.iter().zip(records) {
                pool.restore(id, record)?;
            }
            Ok(())
        })?;
        self.deliver(ids)
    }

    /// The record of instance `id`. The id itself must not be in flight
    /// (`InFlight`), and no sent id may still be running anywhere: the
    /// snapshot needs whole-pool quiescence today (see the module docs).
    ///
    /// # Errors
    ///
    /// [`EngineError::WorkerDied`] after a worker died,
    /// [`EngineError::Poisoned`], [`EngineError::InFlight`] for a sent id
    /// not yet received, [`EngineError::NotQuiescent`] while any sent id
    /// still runs (the call makes progress on them first, so a retry
    /// succeeds), or any error of [`EnvPool::snapshot`].
    pub fn snapshot(&mut self, id: u32) -> Result<Record<E>, EngineError> {
        if self.dead() {
            return Err(EngineError::WorkerDied);
        }
        if self.is_poisoned() {
            return Err(EngineError::Poisoned);
        }
        let i = usize::try_from(id).unwrap_or(usize::MAX);
        if self
            .mode
            .in_flight
            .get(i)
            .is_some_and(|f| f.load(core::sync::atomic::Ordering::Acquire))
        {
            return Err(EngineError::InFlight { id });
        }
        #[cfg(test)]
        if let Some(hook) = SNAPSHOT_HOOK.take() {
            hook();
            while self.running() != 0 {
                std::thread::yield_now();
            }
        }
        // `EnvPool::snapshot` reads through slices over every instance, so
        // no worker may be writing any row meanwhile.
        if self.running() != 0 {
            self.pump(false)?;
            return Err(EngineError::NotQuiescent);
        }
        // Again: a step that panicked after the check above has finished now
        // (its panic flag is ordered before `stepped`), leaving a row partly
        // written.
        if self.is_poisoned() {
            return Err(EngineError::Poisoned);
        }
        self.mode.pool.snapshot(id)
    }

    /// Captures every record into `out`; no sent id may still be running
    /// or waiting to be received.
    ///
    /// # Errors
    ///
    /// [`EngineError::NotQuiescent`] while ids are outstanding,
    /// [`EngineError::Poisoned`], or any error of [`EnvPool::snapshot_all`].
    pub fn snapshot_all(&self, out: &mut [Record<E>]) -> Result<(), EngineError> {
        if self.outstanding != 0 {
            return Err(EngineError::NotQuiescent);
        }
        if self.is_poisoned() {
            return Err(EngineError::Poisoned);
        }
        self.mode.pool.snapshot_all(out)
    }

    /// Starts the next step of the instances `ids` with `actions` (one row
    /// block per id, in the order of `ids`).
    ///
    /// With two jobs already in flight, this waits for the oldest to finish
    /// before publishing, so a send can block for as long as that job runs.
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`], [`EngineError::WorkerDied`],
    /// [`EngineError::EmptySend`], [`EngineError::InFlight`],
    /// [`EngineError::Borrowed`], [`EngineError::SequenceExhausted`] or a
    /// validation error; nothing changed then.
    pub fn send(&mut self, ids: &[u32], actions: ActionInput<'_>) -> Result<(), EngineError> {
        self.admit(ids, actions, None).0?;
        self.launch(ids)
    }

    /// Publishes admitted ids whose covering token (if any) has ended.
    fn launch(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        // Validation passed: the generations move before publication, so a
        // view is stale before any worker writes (ADR-0047 section 5).
        for &id in ids {
            self.generations
                .advance_id(usize::try_from(id).unwrap_or(usize::MAX))
                .map_err(|_| EngineError::SequenceExhausted)?;
        }
        for &id in ids {
            if let Some(flag) = self
                .mode
                .in_flight
                .get(usize::try_from(id).unwrap_or(usize::MAX))
            {
                flag.store(true, core::sync::atomic::Ordering::Release);
            }
            self.push_submission(id)?;
        }
        self.submitted = self.submitted.saturating_add(ids.len());
        self.outstanding = self.outstanding.saturating_add(ids.len());
        self.pump(true)
    }

    /// Test-only: a send whose ids are admitted to the ring with no drain
    /// job published, the state a send leaves when two jobs are already past
    /// their last claim.
    #[cfg(test)]
    pub(crate) fn send_unpublished(
        &mut self,
        ids: &[u32],
        actions: ActionInput<'_>,
    ) -> Result<(), EngineError> {
        self.admit(ids, actions, None).0?;
        for &id in ids {
            self.generations
                .advance_id(usize::try_from(id).unwrap_or(usize::MAX))
                .map_err(|_| EngineError::SequenceExhausted)?;
            if let Some(flag) = self
                .mode
                .in_flight
                .get(usize::try_from(id).unwrap_or(usize::MAX))
            {
                flag.store(true, core::sync::atomic::Ordering::Release);
            }
            self.push_submission(id)?;
        }
        self.submitted = self.submitted.saturating_add(ids.len());
        self.outstanding = self.outstanding.saturating_add(ids.len());
        Ok(())
    }

    /// Admits `id`, retrying while the ring is transiently full behind a
    /// paused claim (ADR-0047 section 7): at most N ids are outstanding, so
    /// the slot frees as soon as the claimer resumes.
    fn push_submission(&self, id: u32) -> Result<(), EngineError> {
        loop {
            match self.mode.submit.push_shared(id) {
                Ok(()) => return Ok(()),
                Err(RingError::Full { .. }) => std::thread::yield_now(),
                Err(_) => return Err(EngineError::SequenceExhausted),
            }
        }
    }

    fn admit(
        &mut self,
        ids: &[u32],
        actions: ActionInput<'_>,
        owner: Option<BatchToken<E, M>>,
    ) -> (Result<(), EngineError>, Option<BatchToken<E, M>>) {
        if let Err(e) = self.precheck(ids) {
            return (Err(e), owner);
        }
        let mode = &self.mode;
        let layout = mode.pool.action_layout();
        let auto_reset = mode.pool.options.auto_reset;
        #[cfg(feature = "debug-checks")]
        let resend = owner.is_some();
        mode.admit(ids, &mut self.seen, owner, |rows| {
            crate::entry::canonical(|| {
                // The same order as Sync: the boundary checks first, then
                // (debug-checks) the legality pass, then the copy, so the same
                // input gives the same error (ADR-0047 ruling 6).
                check_subset_actions(layout, auto_reset, rows, actions)?;
                #[cfg(feature = "debug-checks")]
                if M::ENABLED {
                    legality(mode, rows, ids, layout, actions, resend)?;
                }
                copy_subset_actions(layout, rows, actions)
            })
        })
    }

    /// The checks of a send that need no rows.
    fn precheck(&self, ids: &[u32]) -> Result<(), EngineError> {
        if self.is_poisoned() {
            return Err(EngineError::Poisoned);
        }
        if self.dead() {
            return Err(EngineError::WorkerDied);
        }
        for &id in ids {
            let g = self
                .generations
                .id(usize::try_from(id).unwrap_or(usize::MAX))
                .map_err(|_| self.out_of_range(id))?;
            // A generation that cannot advance is an admission failure,
            // found before anything is written.
            if g == u64::MAX {
                return Err(EngineError::SequenceExhausted);
            }
        }
        Ok(())
    }

    /// Sends every id of `token` with `actions` (one row block per id, in
    /// the token's order), consuming the token.
    ///
    /// # Errors
    ///
    /// Hands the token back with [`EngineError::WrongPool`] for another
    /// pool's token, [`EngineError::Poisoned`], [`EngineError::WorkerDied`],
    /// [`EngineError::SequenceExhausted`] or a validation error; nothing
    /// changed then. See [`SendBatchError`] for the one case without it.
    pub fn send_batch(
        &mut self,
        token: BatchToken<E, M>,
        actions: ActionInput<'_>,
    ) -> Result<(), SendBatchError<E, M>> {
        if !token.belongs_to(&self.mode) {
            return Err(SendBatchError {
                error: EngineError::WrongPool,
                token: Some(token),
            });
        }
        let mut ids = core::mem::take(&mut self.scratch);
        ids.clear();
        ids.extend(token.ids());
        let (admitted, token) = self.admit(&ids, actions, Some(token));
        let result = if let Err(error) = admitted {
            Err(SendBatchError { error, token })
        } else {
            // The token ends here (Release on its flags), before the ids are
            // published.
            drop(token);
            self.launch(&ids)
                .map_err(|error| SendBatchError { error, token: None })
        };
        self.scratch = ids;
        result
    }

    /// Reaps finished jobs, then publishes a drain job when admitted ids
    /// have not been stepped. With `after_push` (a send), a job is always
    /// published after the send's last push, waiting for the oldest job
    /// first if two are in flight: a job published after a push claims that
    /// id, so admitted ids can never wait in the ring with no job left to
    /// claim them. Otherwise a job is published only when none is in flight
    /// (`recv`, writers and snapshots, which make progress on whatever a
    /// send left).
    fn pump(&mut self, after_push: bool) -> Result<(), EngineError> {
        let running = self.running();
        let mode = Arc::clone(&self.mode);
        let rt = self.runtime()?;
        let mut panicked = false;
        while rt.gang.in_flight() > 0 {
            match rt.gang.try_wait()? {
                Some(p) => panicked |= p,
                None => break,
            }
        }
        if after_push && rt.gang.in_flight() >= 2 {
            panicked |= rt.gang.wait()?;
        }
        let result = if running > 0 && (after_push || rt.gang.in_flight() == 0) {
            publish(&mut rt.gang, &mode)
        } else {
            Ok(())
        };
        if panicked {
            mode.panicked
                .store(true, core::sync::atomic::Ordering::Release);
        }
        drop(mode);
        result
    }

    /// Waits for the first `batch_size` completed ids and returns them as a
    /// token.
    ///
    /// # Errors
    ///
    /// [`EngineError::NothingInFlight`] or [`EngineError::BatchUnreachable`]
    /// (never waits), [`EngineError::Panicked`] if a step panicked (the pool
    /// is then poisoned), [`EngineError::WorkerDied`], or
    /// [`EngineError::TurnOrder`] for a `Sequential` turn violation.
    pub fn recv(&mut self) -> Result<BatchToken<E, M>, EngineError> {
        loop {
            if let Some(token) = self.poll(true)? {
                return Ok(token);
            }
        }
    }

    /// As [`Self::recv`], but returns `None` at once if no full batch is
    /// ready.
    ///
    /// # Errors
    ///
    /// As [`Self::recv`].
    pub fn try_recv(&mut self) -> Result<Option<BatchToken<E, M>>, EngineError> {
        self.poll(false)
    }

    /// As [`Self::recv`], waiting at most `timeout`.
    ///
    /// # Errors
    ///
    /// As [`Self::recv`].
    pub fn recv_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<BatchToken<E, M>>, EngineError> {
        let start = Instant::now();
        loop {
            if let Some(token) = self.poll(false)? {
                return Ok(Some(token));
            }
            if start.elapsed() >= timeout {
                return Ok(None);
            }
            std::thread::park_timeout(Duration::from_micros(100));
        }
    }

    /// Poisons the wrapper and reports a dead worker or a step's panic.
    fn check_failed(&mut self) -> Result<(), EngineError> {
        if self.dead() {
            self.poisoned = true;
            return Err(EngineError::WorkerDied);
        }
        if self
            .mode
            .panicked
            .load(core::sync::atomic::Ordering::Acquire)
            || self.runtime.0.as_ref().is_some_and(|rt| rt.gang.poisoned())
        {
            self.poisoned = true;
            return Err(EngineError::Panicked);
        }
        Ok(())
    }

    /// One step of a receive: gather completed ids, then either hand out a
    /// batch or make progress (blocking on the oldest job with `block`).
    fn poll(&mut self, block: bool) -> Result<Option<BatchToken<E, M>>, EngineError> {
        // A panic not yet reported is reported once as `Panicked` below; a
        // reported one is `Poisoned`.
        if self.poisoned {
            return Err(EngineError::Poisoned);
        }
        // Progress first: a refusal below must not leave admitted ids
        // unstepped.
        if !self.dead() {
            self.pump(false)?;
        }
        // A failure outranks the reachability refusal: with too few ids for a
        // batch, `BatchUnreachable` would otherwise hide the panic for good.
        self.check_failed()?;
        let outstanding = u32::try_from(self.outstanding).unwrap_or(u32::MAX);
        let staged = u32::try_from(self.stage.len()).unwrap_or(u32::MAX);
        let batch = u32::try_from(self.batch).unwrap_or(u32::MAX);
        check_recv(outstanding.saturating_sub(staged), staged, batch)?;
        while self.stage.len() < self.batch {
            match self.mode.complete.pop_shared() {
                Ok(Some(id)) => self.stage.push(id),
                Ok(None) => break,
                Err(_) => return Err(EngineError::SequenceExhausted),
            }
        }
        self.check_failed()?;
        if self.stage.len() == self.batch {
            return self.hand_out();
        }
        if self
            .mode
            .panicked
            .load(core::sync::atomic::Ordering::Acquire)
        {
            self.poisoned = true;
            return Err(EngineError::Panicked);
        }
        if self.dead() {
            self.poisoned = true;
            return Err(EngineError::WorkerDied);
        }
        self.pump(false)?;
        if block {
            let rt = self.runtime()?;
            if rt.gang.in_flight() > 0 {
                if rt.gang.wait()? {
                    self.mode
                        .panicked
                        .store(true, core::sync::atomic::Ordering::Release);
                }
            } else {
                std::thread::park_timeout(Duration::from_micros(100));
            }
        }
        Ok(None)
    }

    /// Turns the staged batch into a token (when a buffer is free).
    fn hand_out(&mut self) -> Result<Option<BatchToken<E, M>>, EngineError> {
        self.stage_gens.clear();
        for &id in &self.stage {
            let g = self
                .generations
                .id(usize::try_from(id).unwrap_or(usize::MAX))
                .map_err(|_| EngineError::SequenceExhausted)?;
            self.stage_gens.push(g);
        }
        // `None`: a dropping token has cleared its flags but not yet freed
        // its buffer (ADR-0047 section 4); poll again.
        let Some(token) = BatchToken::new(&self.mode, &self.stage, &self.stage_gens, false) else {
            return Ok(None);
        };
        for &id in &self.stage {
            if let Some(flag) = self
                .mode
                .in_flight
                .get(usize::try_from(id).unwrap_or(usize::MAX))
            {
                flag.store(false, core::sync::atomic::Ordering::Release);
            }
        }
        self.outstanding = self.outstanding.saturating_sub(self.stage.len());
        self.mode
            .ready
            .count
            .fetch_sub(self.stage.len(), core::sync::atomic::Ordering::AcqRel);
        self.stage.clear();
        self.check_turns(&token)?;
        #[cfg(feature = "debug-checks")]
        self.check_agent_events(&token)?;
        Ok(Some(token))
    }

    /// The turn order of a `Sequential` environment on the returned ids.
    fn check_turns(&mut self, token: &BatchToken<E, M>) -> Result<(), EngineError> {
        if E::TURNS != TurnStructure::Sequential {
            return Ok(());
        }
        for row in token.rows() {
            let id = row.id();
            let Some(header) = row.header() else {
                continue;
            };
            if header.engine().pending_reset.get() {
                continue;
            }
            let acting = header.agents().iter().filter(|r| r.acting.get()).count();
            if acting != 1 {
                self.poisoned = true;
                return Err(EngineError::TurnOrder {
                    instance: id,
                    violation: tessera_rl_core::InvariantViolation {
                        code: if acting == 0 {
                            "a Sequential environment named no acting agent"
                        } else {
                            "a Sequential environment named more than one acting agent"
                        },
                    },
                });
            }
        }
        Ok(())
    }

    #[cfg(feature = "debug-checks")]
    fn check_agent_events(&mut self, token: &BatchToken<E, M>) -> Result<(), EngineError> {
        for id in token.ids() {
            if let Some(slot) = token
                .invalid_agent_event(id)
                .and_then(|row| row.iter().position(|f| f.get()))
            {
                self.poisoned = true;
                return Err(EngineError::InvalidAgentEvent {
                    instance: id,
                    agent: u32::try_from(slot).unwrap_or(u32::MAX),
                });
            }
        }
        Ok(())
    }

    /// Gives the pool back, with its workers, in Sync mode.
    ///
    /// # Errors
    ///
    /// [`AsyncIntoPoolError`] while a token is alive; the wrapper is handed
    /// back untouched.
    pub fn into_pool(mut self) -> Result<EnvPool<E, M>, AsyncIntoPoolError<E, M>> {
        // Tokens are only made under `&mut self`, so the count can only
        // fall while this runs: check before draining anything.
        if Arc::strong_count(&self.mode) > 1 {
            return Err(AsyncIntoPoolError {
                pool: Box::new(self),
            });
        }
        self.drain();
        let poisoned = self.is_poisoned();
        let Self {
            mut runtime, mode, ..
        } = self;
        let rt = runtime.0.take();
        match Arc::try_unwrap(mode) {
            Ok(mode) => {
                let mut pool = mode.into_pool();
                if let Some(rt) = rt {
                    pool.put_runtime(rt);
                }
                // A failed step left rows partly written: the pool stays
                // poisoned, so only a full reset or restore reads it again.
                if poisoned {
                    pool.poisoned = true;
                }
                Ok(pool)
            }
            // Unreachable: the count was 1 and no token can be made here.
            Err(mode) => {
                let n = mode.live.len();
                Err(AsyncIntoPoolError {
                    pool: Box::new(Self {
                        runtime: Workers(rt),
                        mode,
                        generations: Arc::new(Generations::new(n)),
                        seen: vec![Bool8::FALSE; n],
                        stage: Vec::new(),
                        stage_gens: Vec::new(),
                        scratch: Vec::new(),
                        submitted: 0,
                        outstanding: 0,
                        batch: 1,
                        poisoned: true,
                    }),
                })
            }
        }
    }

    /// Drains and joins the workers (same as dropping).
    pub fn close(self) {
        drop(self);
    }
}

/// The debug-checks legality pass of an Async send, id by id, before
/// anything is copied (as Sync's step and Pipelined's send, ADR-0047).
#[cfg(feature = "debug-checks")]
fn legality<E: Environment, M: MaskProvider<E>, R: crate::validate::AdmittedRows<E::Header>>(
    mode: &BatchMode<E, M>,
    rows: &mut R,
    ids: &[u32],
    layout: &crate::layout::ActionLayout,
    actions: ActionInput<'_>,
    owned: bool,
) -> Result<(), EngineError> {
    let per = <E::Header as RecordLayout>::AGENTS.saturating_mul(layout.discrete.len());
    for (position, &id) in ids.iter().enumerate() {
        let i = usize::try_from(id).unwrap_or(usize::MAX);
        // Never skip the check: an id whose masks cannot be read refuses the
        // send (the admission checks make this unreachable).
        let Some(masks) = mode.send_masks(i, owned) else {
            return Err(EngineError::Borrowed { id });
        };
        let Some(row) = rows.row(position) else {
            return Err(EngineError::InstanceOutOfRange {
                id,
                num_envs: u32::try_from(rows.instances()).unwrap_or(u32::MAX),
            });
        };
        let start = position.saturating_mul(per);
        let input = actions
            .discrete
            .get(start..start.saturating_add(per))
            .unwrap_or(&[]);
        crate::pool::legality_rows(
            core::slice::from_ref(row.header),
            i,
            input,
            &layout.discrete,
            masks,
            mode.mask_len(),
        )?;
    }
    Ok(())
}
