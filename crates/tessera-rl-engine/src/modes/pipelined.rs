// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Pipelined mode (ADR-0006, ADR-0047): two fixed halves, each sent
//! with its actions and received with its results while the other half
//! runs.
//!
//! [`EnvPool::pipelined`] moves the pool into one shared allocation and
//! takes its workers. [`PipelinedPool::reset_all`] returns a
//! [`HalfToken`] per half; [`PipelinedPool::send`] consumes a token and
//! starts that half's next step; [`PipelinedPool::recv`] returns the oldest
//! sent half (FIFO) as a new token. Results borrow the token, so a half's
//! outputs stay readable across the other half's send, and the borrow
//! window ends when the token is sent or dropped (ADR-0047 section 1).
//!
//! Every operation that writes rows from the caller (`reset_all`,
//! `restore_all`, recovery, `into_pool`) needs no work in flight and no
//! live token (ADR-0047 section 3). Cycles, the two-cycle admission lag and
//! the per-half reduction partials follow ADR-0047 section 6; the pool has
//! no pool-level adaptive subsystem yet, so the reduction is a test-only
//! sum of rewards.

use core::ops::Range;
use core::time::Duration;

use std::sync::Arc;

use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout, TurnStructure};

use crate::buffer::pipeline::{HalfToken, Mode, Waiter, half_panicked, publish, turn_rows};
use crate::buffer::{GenerationError, Generations, half_ranges};
use crate::error::EngineError;
use crate::masks::MaskProvider;
use crate::options::SyncOptions;
use crate::pool::{EnvPool, Reseed, SyncRuntime, chunk_size};
use crate::record::Record;
use crate::validate::{
    ActionInput, check_cycle, check_range_actions, check_recv, copy_range_actions,
};

const HALVES: usize = 2;

/// Options of the Pipelined mode: the workers, as in Sync mode.
#[derive(Debug, Clone)]
pub struct PipelinedOptions {
    /// Workers, spin, core set and wait backend (as [`SyncOptions`]).
    pub sync: SyncOptions,
}

impl PipelinedOptions {
    /// Options with `workers` workers and the Sync defaults.
    #[must_use]
    pub const fn new(workers: core::num::NonZeroUsize) -> Self {
        Self {
            sync: SyncOptions::new(workers),
        }
    }
}

/// A send that was refused. With the token handed back (`Some`), nothing
/// changed. `None` only if the gang refused the job after validation and
/// the token had ended: then the half's action rows hold the new actions
/// and its generation has moved (any old view reads as stale), but no row
/// of outputs or state changed; [`PipelinedPool::idle_token`] gives a new
/// token.
#[derive(Debug)]
pub struct SendError<E: Environment, M: MaskProvider<E>> {
    /// Why the send was refused.
    pub error: EngineError,
    /// The token, unchanged.
    pub token: Option<HalfToken<E, M>>,
}

/// `EnvPool::pipelined` refused; the pool is handed back when the refusal
/// came before it was converted (`HalfTooSmall`, already in Sync mode).
pub struct PipelinedBuildError<E: Environment, M: MaskProvider<E>> {
    /// Why.
    pub error: EngineError,
    /// The pool, unchanged, if it was not consumed.
    pub pool: Option<Box<EnvPool<E, M>>>,
}

impl<E: Environment, M: MaskProvider<E>> core::fmt::Debug for PipelinedBuildError<E, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PipelinedBuildError")
            .field("error", &self.error)
            .field("pool", &self.pool.is_some())
            .finish()
    }
}

/// Why `into_pool` refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntoPoolErrorKind {
    /// A token is still alive, so the pool's rows cannot be handed out.
    LiveTokens,
}

/// `into_pool` refused; the wrapper is handed back, still working.
#[derive(Debug)]
pub struct IntoPoolError<E: Environment, M: MaskProvider<E>> {
    /// The pool, unchanged.
    pub pool: Box<PipelinedPool<E, M>>,
    /// Why.
    pub kind: IntoPoolErrorKind,
}

/// The cycle state of a Pipelined pool, captured with its records
/// (ADR-0047 section 6, ADR-0009).
#[derive(Debug, Clone, PartialEq)]
pub struct CycleState {
    sent: [u64; HALVES],
    done: [u64; HALVES],
    committed: Option<u64>,
    partials: [[f64; 2]; HALVES],
    versions: [f64; 3],
}

impl CycleState {
    const fn initial() -> Self {
        Self {
            sent: [0; HALVES],
            done: [0; HALVES],
            committed: None,
            partials: [[0.0; 2]; HALVES],
            versions: [0.0; 3],
        }
    }
}

/// Reads the export generations of a Pipelined pool's halves from any
/// thread (ADR-0047 section 5). A generation moves before the send that
/// moves it publishes its job.
#[derive(Debug, Clone)]
pub struct HalfGenerations(Arc<Generations>);

impl HalfGenerations {
    /// The generation of half `half`; `None` for a half other than 0 or 1.
    #[must_use]
    pub fn half(&self, half: usize) -> Option<u64> {
        self.0.half(half).ok()
    }
}

/// A pool in Pipelined mode (ADR-0047). It never derefs to [`EnvPool`];
/// [`Self::into_pool`] gives the pool back.
pub struct PipelinedPool<E: Environment, M: MaskProvider<E>> {
    // Declared first, so it drops first: its `Drop` drains every job and
    // joins the workers before the mode's `Arc` below is released.
    runtime: Workers,
    mode: Arc<Mode<E, M>>,
    generations: Arc<Generations>,
    /// Sent halves in send order (FIFO), oldest first.
    order: [usize; HALVES],
    queued: usize,
    cycles: CycleState,
    poisoned: bool,
}

/// The mode's workers; dropping them drains every job in flight, then
/// shuts the gang down and joins it.
#[derive(Debug)]
pub(crate) struct Workers(pub(crate) Option<SyncRuntime>);

impl Workers {
    /// Waits for every job in flight; returns whether any of them panicked
    /// (or could not be waited for).
    pub(crate) fn drain(&mut self) -> bool {
        let mut failed = false;
        if let Some(rt) = self.0.as_mut() {
            while rt.gang.in_flight() > 0 {
                let Ok(panicked) = rt.gang.wait() else {
                    failed = true;
                    break;
                };
                failed |= panicked;
            }
        }
        failed
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        let _ = self.drain();
    }
}

impl<E: Environment, M: MaskProvider<E>> core::fmt::Debug for PipelinedPool<E, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PipelinedPool")
            .field("queued", &self.queued)
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
    /// Moves the pool into Pipelined mode with its own workers
    /// (ADR-0047). The two halves split at the chunk boundary nearest
    /// `N / 2`.
    ///
    /// # Errors
    ///
    /// [`EngineError::HalfTooSmall`] if the pool has fewer than two chunks
    /// and [`EngineError::Options`] if it already runs in Sync mode, both
    /// with the pool handed back; any error of [`EnvPool::with_workers`],
    /// without it.
    pub fn pipelined(
        self,
        options: PipelinedOptions,
    ) -> Result<PipelinedPool<E, M>, PipelinedBuildError<E, M>> {
        self.pipelined_inner(
            options,
            #[cfg(test)]
            crate::sys::Faults::default(),
        )
    }

    /// [`Self::pipelined`] with test-only fault injection.
    #[cfg(test)]
    pub(crate) fn pipelined_faults(
        self,
        options: PipelinedOptions,
        faults: crate::sys::Faults,
    ) -> Result<PipelinedPool<E, M>, PipelinedBuildError<E, M>> {
        self.pipelined_inner(options, faults)
    }

    fn pipelined_inner(
        self,
        options: PipelinedOptions,
        #[cfg(test)] faults: crate::sys::Faults,
    ) -> Result<PipelinedPool<E, M>, PipelinedBuildError<E, M>> {
        let refuse = |pool: Self, error| PipelinedBuildError {
            error,
            pool: Some(Box::new(pool)),
        };
        if self.workers() != 0 {
            return Err(refuse(
                self,
                EngineError::Options {
                    option: "workers",
                    reason: "the pool already runs in Sync mode",
                },
            ));
        }
        let size = chunk_size(&self.options);
        let chunk = self.options.chunk_size.get();
        let n = self.num_envs_usize();
        let halves = core::num::NonZeroUsize::new(size).and_then(|s| half_ranges(n, s));
        let Some(halves) = halves else {
            return Err(refuse(
                self,
                EngineError::HalfTooSmall { chunk_size: chunk },
            ));
        };
        let lost = |error| PipelinedBuildError { error, pool: None };
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
        Ok(PipelinedPool {
            runtime: Workers(Some(runtime)),
            mode: Mode::new(pool, halves, size, workers),
            generations: Arc::new(Generations::new(n)),
            order: [0; HALVES],
            queued: 0,
            cycles: CycleState::initial(),
            poisoned,
        })
    }

    fn num_envs_usize(&self) -> usize {
        usize::try_from(self.options.num_envs.get()).unwrap_or(usize::MAX)
    }
}

fn gen_error(e: GenerationError) -> EngineError {
    match e {
        GenerationError::Exhausted { .. } => EngineError::SequenceExhausted,
        GenerationError::OutOfRange { index, .. } => EngineError::InvalidHalf {
            half: u32::try_from(index).unwrap_or(u32::MAX),
        },
    }
}

impl<E, M> PipelinedPool<E, M>
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

    /// The instance ids of the two halves.
    #[must_use]
    pub fn halves(&self) -> [Range<u32>; HALVES] {
        self.mode.halves().clone().map(|r| {
            u32::try_from(r.start).unwrap_or(u32::MAX)..u32::try_from(r.end).unwrap_or(u32::MAX)
        })
    }

    /// Whether a panic or a dead worker poisoned the pool.
    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned || self.dead() || self.runtime.0.as_ref().is_some_and(|rt| rt.gang.poisoned())
    }

    /// The export generation of half `half` (ADR-0047 section 5): it moves
    /// on every accepted send and every write of the half's rows.
    ///
    /// # Errors
    ///
    /// [`EngineError::InvalidHalf`] for a half other than 0 or 1.
    pub fn generation(&self, half: u32) -> Result<u64, EngineError> {
        let index = usize::try_from(half).unwrap_or(usize::MAX);
        self.generations.half(index).map_err(gen_error)
    }

    /// A handle reading the halves' generations from any thread.
    #[must_use]
    pub fn half_generations(&self) -> HalfGenerations {
        HalfGenerations(Arc::clone(&self.generations))
    }

    /// The oldest sent half not yet received, the one `recv` returns next.
    #[must_use]
    pub fn oldest_half(&self) -> Option<u32> {
        if self.queued == 0 {
            return None;
        }
        self.order.first().and_then(|&h| u32::try_from(h).ok())
    }

    /// A [`Waiter`] that waits for a ready half from any thread.
    #[must_use]
    pub fn waiter(&self) -> Waiter {
        Waiter {
            ready: Arc::clone(&self.mode.ready),
        }
    }

    fn runtime(&mut self) -> Result<&mut SyncRuntime, EngineError> {
        self.runtime.0.as_mut().ok_or(EngineError::WorkerDied)
    }

    fn dead(&self) -> bool {
        self.runtime.0.as_ref().is_none_or(|rt| rt.gang.dead())
    }

    fn any_live(&self) -> bool {
        self.mode.live.iter().any(crate::buffer::live::is_live)
    }

    /// Waits for every job in flight and discards it (recovery and
    /// `into_pool`). A drained job that panicked poisons the wrapper.
    fn drain(&mut self) {
        let mut failed = self.runtime.drain();
        for (h, (flag, ready)) in self
            .mode
            .in_flight
            .iter()
            .zip(&self.mode.ready.ready)
            .enumerate()
        {
            failed |=
                flag.load(core::sync::atomic::Ordering::Acquire) && half_panicked(&self.mode, h);
            flag.store(false, core::sync::atomic::Ordering::Release);
            ready.store(false, core::sync::atomic::Ordering::Release);
        }
        if failed {
            self.poisoned = true;
        }
        self.queued = 0;
    }

    /// Exclusive access to the pool for a caller-side writer (ADR-0047
    /// section 3): no work in flight and no live token. A token dropping on
    /// another thread may still hold its `Arc`: that is `Borrowed` too.
    fn exclusive(&mut self, recovery: bool) -> Result<&mut Mode<E, M>, EngineError> {
        if self.dead() {
            return Err(EngineError::WorkerDied);
        }
        if self.queued != 0 {
            if !recovery {
                return Err(EngineError::NotQuiescent);
            }
            self.drain();
        }
        if self.any_live() {
            return Err(EngineError::HalfBorrowed {
                half: self.live_half(),
            });
        }
        Arc::get_mut(&mut self.mode).ok_or(EngineError::HalfBorrowed { half: 0 })
    }

    fn live_half(&self) -> u8 {
        u8::from(
            !self
                .mode
                .live
                .first()
                .is_some_and(crate::buffer::live::is_live),
        )
    }

    /// After a successful writer: both halves and every instance get a new
    /// generation, the cycles take `cycles`, and the poison is cleared.
    fn after_write(&mut self, cycles: CycleState) -> Result<(), EngineError> {
        for h in 0..HALVES {
            self.generations.advance_half(h).map_err(gen_error)?;
        }
        for id in 0..self.generations.len() {
            self.generations.advance_id(id).map_err(gen_error)?;
        }
        self.cycles = cycles;
        if let Some(rt) = self.runtime.0.as_mut() {
            rt.gang.clear_poison()?;
        }
        self.poisoned = false;
        Ok(())
    }

    fn tokens(&self) -> Result<[HalfToken<E, M>; HALVES], EngineError> {
        let a = HalfToken::new(&self.mode, 0).ok_or(EngineError::HalfBorrowed { half: 0 })?;
        let b = HalfToken::new(&self.mode, 1).ok_or(EngineError::HalfBorrowed { half: 1 })?;
        Ok([a, b])
    }

    /// Resets every instance and returns a token for each half. Also the
    /// recovery after a panic: it drains first.
    ///
    /// # Errors
    ///
    /// [`EngineError::NotQuiescent`] with work in flight on a healthy pool,
    /// [`EngineError::HalfBorrowed`] while a token is alive,
    /// [`EngineError::WorkerDied`] after a worker died, or any error of
    /// [`EnvPool::reset_all`].
    pub fn reset_all(
        &mut self,
        reseed: Reseed<'_>,
    ) -> Result<[HalfToken<E, M>; HALVES], EngineError> {
        self.write(|pool| pool.reset_all(reseed).map(|_| ()))?;
        self.after_write(CycleState::initial())?;
        self.tokens()
    }

    /// Restores every instance from `records` and the cycle state from
    /// `cycles` (as captured by [`Self::snapshot_all`]). Get tokens with
    /// [`Self::idle_token`] afterwards.
    ///
    /// # Errors
    ///
    /// As [`Self::reset_all`], or any error of [`EnvPool::restore_all`].
    pub fn restore_all(
        &mut self,
        records: &[Record<E>],
        cycles: &CycleState,
    ) -> Result<(), EngineError> {
        self.write(|pool| pool.restore_all(records))?;
        self.after_write(cycles.clone())
    }

    /// Runs a caller-side writer on the pool (ADR-0047 section 3). The raw
    /// parts are taken again whatever the outcome (the `&mut` invalidated
    /// them). A failure changes no generation, cycle or poison, except that
    /// a writer which poisoned the pool (a panic) poisons the wrapper.
    fn write(
        &mut self,
        f: impl FnOnce(&mut EnvPool<E, M>) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        let recovery = self.is_poisoned();
        let mode = self.exclusive(recovery)?;
        let result = f(&mut mode.pool);
        let poisoned = mode.pool.is_poisoned();
        mode.rebuild();
        if result.is_err() && poisoned {
            self.poisoned = true;
        }
        result
    }

    /// Captures every record into `out` and returns the cycle state.
    /// Tokens may be alive (snapshots only read); no work may be in flight.
    ///
    /// # Errors
    ///
    /// [`EngineError::NotQuiescent`] with work in flight,
    /// [`EngineError::Poisoned`] on a poisoned pool, or any error of
    /// [`EnvPool::snapshot_all`].
    pub fn snapshot_all(&self, out: &mut [Record<E>]) -> Result<CycleState, EngineError> {
        if self.queued != 0 {
            return Err(EngineError::NotQuiescent);
        }
        if self.is_poisoned() {
            return Err(EngineError::Poisoned);
        }
        self.mode.pool.snapshot_all(out)?;
        Ok(self.cycles.clone())
    }

    /// A token for a returned half that no token covers (after
    /// [`Self::restore_all`], or after a token was dropped).
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`] on a poisoned pool,
    /// [`EngineError::InvalidHalf`], [`EngineError::HalfInFlight`] or
    /// [`EngineError::HalfBorrowed`].
    pub fn idle_token(&self, half: u32) -> Result<HalfToken<E, M>, EngineError> {
        // A poisoned pool's rows may be partly written: no new views.
        if self.is_poisoned() {
            return Err(EngineError::Poisoned);
        }
        let index = match half {
            0 => 0,
            1 => 1,
            _ => return Err(EngineError::InvalidHalf { half }),
        };
        let short = u8::try_from(half).unwrap_or(u8::MAX);
        if self
            .mode
            .in_flight
            .get(index)
            .is_some_and(|f| f.load(core::sync::atomic::Ordering::Acquire))
        {
            return Err(EngineError::HalfInFlight { half: short });
        }
        HalfToken::new(&self.mode, index).ok_or(EngineError::HalfBorrowed { half: short })
    }

    /// Starts the next step of `token`'s half with `actions` (one row block
    /// per instance of the half). Consumes the token.
    ///
    /// # Errors
    ///
    /// Hands the token back with [`EngineError::WrongPool`],
    /// [`EngineError::Poisoned`], [`EngineError::WorkerDied`],
    /// [`EngineError::CycleAhead`], [`EngineError::SequenceExhausted`] or a
    /// validation error; nothing changed. See [`SendError`] for the one
    /// case without the token.
    pub fn send(
        &mut self,
        token: HalfToken<E, M>,
        actions: ActionInput<'_>,
    ) -> Result<(), SendError<E, M>> {
        let half = match self.admit(&token, actions) {
            Ok(half) => half,
            Err(error) => {
                return Err(SendError {
                    error,
                    token: Some(token),
                });
            }
        };
        // Validation passed: the generation moves before publication, so a
        // view is stale before any worker writes (ADR-0047 section 5).
        if let Err(e) = self.generations.advance_half(half) {
            return Err(SendError {
                error: gen_error(e),
                token: Some(token),
            });
        }
        // The token ends here (Release on its live flag), before the job is
        // published.
        drop(token);
        self.launch(half).map_err(|error| SendError {
            error,
            token: HalfToken::new(&self.mode, half),
        })
    }

    fn admit(
        &mut self,
        token: &HalfToken<E, M>,
        actions: ActionInput<'_>,
    ) -> Result<usize, EngineError> {
        if !token.belongs_to(&self.mode) {
            return Err(EngineError::WrongPool);
        }
        // Dead first: `is_poisoned` includes it, and the death is the error.
        if self.dead() {
            return Err(EngineError::WorkerDied);
        }
        if self.is_poisoned() {
            return Err(EngineError::Poisoned);
        }
        let half = usize::try_from(token.half()).unwrap_or(usize::MAX);
        let sent = self.cycles.sent.get(half).copied().unwrap_or(u64::MAX);
        check_cycle(sent, self.cycles.committed)?;
        // A generation that cannot advance is an admission failure, found
        // before anything is written.
        if self.generations.half(half).map_err(gen_error)? == u64::MAX {
            return Err(EngineError::SequenceExhausted);
        }
        let mode = &self.mode;
        let layout = mode.pool.action_layout();
        let auto_reset = mode.pool.options.auto_reset;
        mode.with_half_rows(token, |first, headers, discrete, continuous| {
            crate::entry::canonical(|| {
                // The same order as Sync: the boundary checks first, then
                // (debug-checks) the legality pass, then the copy, so the
                // same input gives the same error (ADR-0047 ruling 6).
                check_range_actions(
                    layout,
                    auto_reset,
                    first,
                    headers,
                    actions,
                    (discrete.len(), continuous.len()),
                )?;
                #[cfg(feature = "debug-checks")]
                if M::ENABLED {
                    crate::pool::legality_rows(
                        headers,
                        first,
                        actions.discrete,
                        &layout.discrete,
                        token.masks(),
                        token.mask_len(),
                    )?;
                }
                copy_range_actions(layout, headers, actions, discrete, continuous);
                Ok::<(), EngineError>(())
            })
        })
        .ok_or(EngineError::WrongPool)??;
        Ok(half)
    }

    /// Publishes an admitted half whose token has ended.
    fn launch(&mut self, half: usize) -> Result<(), EngineError> {
        if let Some(flag) = self.mode.in_flight.get(half) {
            flag.store(true, core::sync::atomic::Ordering::Release);
        }
        let mode = Arc::clone(&self.mode);
        let published = match self.runtime() {
            Ok(rt) => publish(&mut rt.gang, &mode, half),
            Err(e) => Err(e),
        };
        drop(mode);
        if let Err(e) = published {
            if let Some(flag) = self.mode.in_flight.get(half) {
                flag.store(false, core::sync::atomic::Ordering::Release);
            }
            return Err(e);
        }
        if let Some(slot) = self.order.get_mut(self.queued) {
            *slot = half;
        }
        self.queued += 1;
        if let Some(s) = self.cycles.sent.get_mut(half) {
            *s = s.saturating_add(1);
        }
        Ok(())
    }

    /// Waits for the oldest sent half and returns it as a token.
    ///
    /// # Errors
    ///
    /// [`EngineError::NothingInFlight`] (never waits),
    /// [`EngineError::Panicked`] if a chunk of the half panicked (the pool
    /// is then poisoned), [`EngineError::WorkerDied`], or
    /// [`EngineError::TurnOrder`] for a `Sequential` turn violation.
    pub fn recv(&mut self) -> Result<HalfToken<E, M>, EngineError> {
        self.check_recv()?;
        let panicked = self.runtime()?.gang.wait()?;
        self.finish(panicked)
    }

    /// As [`Self::recv`], but returns `None` at once if the oldest half is
    /// still running.
    ///
    /// # Errors
    ///
    /// As [`Self::recv`].
    pub fn try_recv(&mut self) -> Result<Option<HalfToken<E, M>>, EngineError> {
        self.check_recv()?;
        match self.runtime()?.gang.try_wait()? {
            None => Ok(None),
            Some(panicked) => self.finish(panicked).map(Some),
        }
    }

    /// As [`Self::recv`], waiting at most `timeout`; `None` if the oldest
    /// half is still running then.
    ///
    /// # Errors
    ///
    /// As [`Self::recv`].
    pub fn recv_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<HalfToken<E, M>>, EngineError> {
        let start = std::time::Instant::now();
        loop {
            if let Some(token) = self.try_recv()? {
                return Ok(Some(token));
            }
            if start.elapsed() >= timeout {
                return Ok(None);
            }
            std::thread::park_timeout(Duration::from_micros(100));
        }
    }

    fn check_recv(&self) -> Result<(), EngineError> {
        if self.dead() {
            return Err(EngineError::WorkerDied);
        }
        if self.is_poisoned() {
            return Err(EngineError::Poisoned);
        }
        let queued = u32::try_from(self.queued).unwrap_or(u32::MAX);
        check_recv(queued, 0, 1)
    }

    fn finish(&mut self, panicked: bool) -> Result<HalfToken<E, M>, EngineError> {
        let half = self.order.first().copied().unwrap_or(0);
        self.order.rotate_left(1);
        self.queued = self.queued.saturating_sub(1);
        if let Some(flag) = self.mode.in_flight.get(half) {
            flag.store(false, core::sync::atomic::Ordering::Release);
        }
        if let Some(flag) = self.mode.ready.ready.get(half) {
            flag.store(false, core::sync::atomic::Ordering::Release);
        }
        if self.dead() {
            self.poisoned = true;
            return Err(EngineError::WorkerDied);
        }
        if panicked || half_panicked(&self.mode, half) {
            self.poisoned = true;
            return Err(EngineError::Panicked);
        }
        let token = HalfToken::new(&self.mode, half).ok_or(EngineError::HalfBorrowed {
            half: u8::try_from(half).unwrap_or(u8::MAX),
        })?;
        self.check_turns(&token)?;
        #[cfg(feature = "debug-checks")]
        self.check_agent_events(&token)?;
        crate::entry::canonical(|| self.reduce(half, &token));
        Ok(token)
    }

    /// The turn order of a `Sequential` environment on the returned half
    /// (as `EnvPool::step`).
    fn check_turns(&mut self, token: &HalfToken<E, M>) -> Result<(), EngineError> {
        if E::TURNS != TurnStructure::Sequential {
            return Ok(());
        }
        for (i, header) in turn_rows(token) {
            if header.engine().pending_reset.get() {
                continue;
            }
            let acting = header.agents().iter().filter(|r| r.acting.get()).count();
            if acting != 1 {
                self.poisoned = true;
                return Err(EngineError::TurnOrder {
                    instance: u32::try_from(i).unwrap_or(u32::MAX),
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
    fn check_agent_events(&mut self, token: &HalfToken<E, M>) -> Result<(), EngineError> {
        let a = E::MAX_AGENTS.max(1);
        let first = usize::try_from(token.ids().start).unwrap_or(0);
        if let Some(k) = token.invalid_agent_event().iter().position(|f| f.get()) {
            self.poisoned = true;
            return Err(EngineError::InvalidAgentEvent {
                instance: u32::try_from(first + k / a).unwrap_or(u32::MAX),
                agent: u32::try_from(k % a).unwrap_or(u32::MAX),
            });
        }
        Ok(())
    }

    /// The per-half reduction of the half's cycle, in instance order, and
    /// the commit of every cycle both halves have finished (ADR-0047
    /// section 6). The reduction is a test-only sum of rewards.
    fn reduce(&mut self, half: usize, token: &HalfToken<E, M>) {
        let mut sum = 0.0_f64;
        for &r in token.rewards() {
            sum += f64::from(r);
        }
        let c = &mut self.cycles;
        let Some(done) = c.done.get_mut(half) else {
            return;
        };
        let cycle = *done;
        if let Some(p) = c
            .partials
            .get_mut(half)
            .and_then(|p| p.get_mut(slot2(cycle)))
        {
            *p = sum;
        }
        *done = done.saturating_add(1);
        let finished = c.done.iter().copied().min().unwrap_or(0);
        let mut next = c.committed.map_or(0, |k| k.saturating_add(1));
        while next < finished {
            let a = c
                .partials
                .first()
                .and_then(|p| p.get(slot2(next)))
                .copied()
                .unwrap_or(0.0);
            let b = c
                .partials
                .get(1)
                .and_then(|p| p.get(slot2(next)))
                .copied()
                .unwrap_or(0.0);
            if let Some(v) = c.versions.get_mut(slot3(next)) {
                *v = a + b;
            }
            c.committed = Some(next);
            next = next.saturating_add(1);
        }
    }

    /// The committed context of cycle `cycle` (test-only reduction), if it
    /// is among the last three committed.
    #[doc(hidden)]
    #[must_use]
    pub fn context_version(&self, cycle: u64) -> Option<f64> {
        let done = self.cycles.committed?;
        if cycle > done || done - cycle >= 3 {
            return None;
        }
        self.cycles.versions.get(slot3(cycle)).copied()
    }

    /// Gives the pool back, with its workers, in Sync mode.
    ///
    /// # Errors
    ///
    /// [`IntoPoolError`] with [`IntoPoolErrorKind::LiveTokens`] while a
    /// token is alive; the wrapper is handed back untouched.
    pub fn into_pool(mut self) -> Result<EnvPool<E, M>, IntoPoolError<E, M>> {
        // Tokens are only made under `&mut self`, so the count can only
        // fall while this runs: check before draining anything.
        if Arc::strong_count(&self.mode) > 1 {
            return Err(IntoPoolError {
                pool: Box::new(self),
                kind: IntoPoolErrorKind::LiveTokens,
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
                // A failed job left rows partly written: the pool stays
                // poisoned, so only a full reset or restore reads it again.
                if poisoned {
                    pool.poisoned = true;
                }
                Ok(pool)
            }
            // Unreachable: the count was 1 and no token can be made here.
            Err(mode) => Err(IntoPoolError {
                pool: Box::new(Self {
                    runtime: Workers(rt),
                    mode,
                    generations: Arc::new(Generations::new(0)),
                    order: [0; HALVES],
                    queued: 0,
                    cycles: CycleState::initial(),
                    poisoned: true,
                }),
                kind: IntoPoolErrorKind::LiveTokens,
            }),
        }
    }

    /// Drains and joins the workers (same as dropping).
    pub fn close(self) {
        drop(self);
    }
}

const fn slot2(cycle: u64) -> usize {
    (cycle % 2) as usize
}

const fn slot3(cycle: u64) -> usize {
    (cycle % 3) as usize
}
