// SPDX-License-Identifier: MIT OR Apache-2.0

//! `EnvPool`: instance records, construction and the Inline mode
//! (SPEC 5, 6; ADR-0006, ADR-0028, ADR-0036).
//!
//! Records are stored as two parallel arrays, headers and states; chunks
//! are ranges of `chunk_size` records. Inline mode runs every chunk on the
//! calling thread inside `fp::with_canonical` (ADR-0037). After
//! construction, `reset` and `step` never allocate.

use core::marker::PhantomData;

use tessera_rl_core::AgentRow;
use tessera_rl_core::error::ConfigError;
use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::{
    ActionMask, AdaptiveValues, AgentSlot, Bool8, EnvConfig, Environment, EpisodeEnd, RecordLayout,
    StateInit, StepOut, TurnStructure,
};
use tessera_rl_num::fp;
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{ObsRng, RngAuthority, StepRng};

use crate::archive::ArchiveSource;
use crate::buffer::par_chunks;
use crate::error::EngineError;
use crate::layout::{
    ActionLayout, MAX_GROUPED_HEADS, MAX_LEAVES, MaskHeadShape, MaskLayout, obs_leaves,
};
use crate::masks::{IllegalActionPolicy, MaskProvider, Masked, NoMasks};
#[cfg(not(loom))]
use crate::options::WaitBackend;
use crate::options::{AutoReset, PoolOptions, SyncOptions};
use crate::outputs::{Outputs, OutputsView};
use crate::registry::Registration;
use crate::sys::sync::{AtomicBool, Ordering};
use crate::sys::{Gang, GangOptions, available_cores};
use crate::validate::{
    ActionInput, check_and_copy_actions, check_and_copy_actions_fast, check_ids,
};

/// Whether an entry point is accepted on a poisoned pool (ADR-0028).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Poison {
    /// Rejected with `EngineError::Poisoned`.
    Reject,
    /// Accepted: a full reset or restore that may clear the poison.
    Recover,
}

/// How `reset` reseeds the instances it resets (SPEC 8.1).
#[derive(Debug, Clone, Copy)]
pub enum Reseed<'a> {
    /// Keep the seeds; the episode index advances.
    Keep,
    /// The k-th reset instance, in the order given (ADR-0007), gets
    /// `seed + k`, modulo 2^64; for a full reset, instance i gets
    /// `seed + i`.
    Base(Seed),
    /// The k-th reset instance gets `seeds[k]`.
    List(&'a [Seed]),
    /// The k-th reset instance gets `seeds[k]`, or keeps its seed (its
    /// episode index advances, as with `Keep`) where `seeds[k]` is `None`:
    /// Gymnasium's per-environment `None`, in one call (SPEC 8.1). The
    /// whole request (ids, seed count, the episode counters of the kept
    /// instances) is validated before anything is written; on an error no
    /// instance or output changes.
    Each(&'a [Option<Seed>]),
}

impl Reseed<'_> {
    /// The seed of the k-th reset instance; `None` keeps its seed.
    pub(crate) fn seed(self, k: usize) -> Option<Seed> {
        match self {
            Self::Keep => None,
            // `k` counts reset instances, at most `u32::MAX`.
            Self::Base(base) => Some(base.offset(u64::try_from(k).unwrap_or(u64::MAX))),
            Self::List(seeds) => seeds.get(k).copied(),
            Self::Each(seeds) => seeds.get(k).copied().flatten(),
        }
    }

    /// The number of seeds given, for lists.
    const fn len(self) -> Option<usize> {
        match self {
            Self::List(seeds) => Some(seeds.len()),
            Self::Each(seeds) => Some(seeds.len()),
            Self::Keep | Self::Base(_) => None,
        }
    }
}

/// A pool of `num_envs` instances of `E`.
#[derive(Debug)]
pub struct EnvPool<E: Environment, M: MaskProvider<E> = NoMasks> {
    pub(crate) cfg: E::Config,
    pub(crate) options: PoolOptions,
    pub(crate) headers: Vec<E::Header>,
    pub(crate) states: Vec<E::State>,
    actions: ActionLayout,
    discrete: Vec<i32>,
    continuous: Vec<f32>,
    pub(crate) outputs: Outputs,
    pub(crate) seen: Vec<Bool8>,
    pub(crate) authority: RngAuthority,
    pub(crate) poisoned: bool,
    penalties: Vec<f32>,
    penalized: Vec<Bool8>,
    pub(crate) branch_counter: u64,
    /// The attached archive source (ADR-0075); `None` for fresh resets.
    pub(crate) reset_source: Option<ArchiveSource<E>>,
    /// Whether every instance will step next time: none pending reset and
    /// every agent slot alive and acting (env-step plan r3 section 3.3,
    /// facts 1 and 4). Set by a step that ends so; cleared by every step's
    /// start, reset, restore, fork, import and split-phase hand-over.
    pub(crate) fast_clean: bool,
    /// An upper bound on every instance's `episode_index` and
    /// `episode_steps` (facts 2 and 3), kept without scanning: 0 at
    /// construction, raised by one per step and per reset call. `None` after
    /// a restore, fork, import or split-phase hand-over, which can write any
    /// counter; the next slow-path step then scans the headers once.
    pub(crate) counter_bound: Option<u32>,
    sync: Option<SyncRuntime>,
    /// Allocations counted in this pool's chunk steps (ADR-0055).
    #[cfg(feature = "debug-checks")]
    pub(crate) allocs: crate::sys::alloc_track::AllocTally,
    _masks: PhantomData<M>,
}

#[cfg_attr(
    loom,
    allow(dead_code, reason = "the split-phase modes are not built under loom")
)]
impl<E: Environment, M: MaskProvider<E>> EnvPool<E, M> {
    /// Takes the workers out of the pool (the split-phase modes own them,
    /// ADR-0047); the pool is left in Inline mode.
    pub(crate) fn take_runtime(&mut self) -> Option<SyncRuntime> {
        self.forget_fast_facts();
        self.sync.take()
    }

    /// Gives the workers back (`into_pool` of a split-phase mode).
    pub(crate) fn put_runtime(&mut self, runtime: SyncRuntime) {
        self.forget_fast_facts();
        self.sync = Some(runtime);
    }

    /// The read-only context of a step, for the split-phase modes.
    pub(crate) fn step_ctx(&self) -> StepCtx<'_, E> {
        StepCtx {
            cfg: &self.cfg,
            authority: &self.authority,
            layout: &self.actions,
            policy: self.options.illegal_action,
            limit: self
                .options
                .max_episode_steps
                .map(core::num::NonZeroU32::get),
            auto_reset: self.options.auto_reset,
            archive: self.reset_source.as_ref(),
            #[cfg(feature = "debug-checks")]
            allocs: &self.allocs,
        }
    }

    /// Where discrete head `head`'s block lies in an agent's mask row, and
    /// its shape (SPEC 11.3); `None` without masks or past the last head.
    #[must_use]
    pub fn mask_head_shape(&self, head: usize) -> Option<MaskHeadShape> {
        self.actions.mask.shape(head, &self.actions.discrete)
    }

    /// The action layout of the pool.
    pub(crate) fn action_layout(&self) -> &ActionLayout {
        &self.actions
    }

    /// Forgets the validation fast path's facts: for code that writes headers
    /// outside a step (restore, fork, import, split-phase modes).
    pub(crate) fn forget_fast_facts(&mut self) {
        self.fast_clean = false;
        self.counter_bound = None;
    }

    /// Notes a reset call: it clears `fast_clean` and raises every
    /// `episode_index` by at most one.
    fn note_reset(&mut self) {
        self.fast_clean = false;
        self.counter_bound = self.counter_bound.and_then(|b| b.checked_add(1));
    }

    /// Test hook (the fuzz target `fast_path`): forgets the validation fast
    /// path's facts, so the next step takes the slow path.
    #[cfg(feature = "test-hooks")]
    #[doc(hidden)]
    pub fn forget_validation_fast_path(&mut self) {
        self.forget_fast_facts();
    }

    /// Whether the next step may take the validation fast path: the facts
    /// hold, and every counter is below `u32::MAX` (the bound is an upper
    /// bound on all of them), so none of the slow path's `CounterExhausted`
    /// checks, which fire at `u32::MAX` exactly, can fail.
    fn fast_path_ready(&self) -> bool {
        self.fast_clean && self.counter_bound.is_some_and(|b| b < u32::MAX)
    }

    /// Test-only: whether the next step may take the validation fast path.
    #[cfg(test)]
    pub(crate) fn fast_path_armed(&self) -> bool {
        self.fast_path_ready()
    }

    /// The pool-wide chunk and the discrete and continuous heads per agent,
    /// for the split-phase modes to take raw parts of (ADR-0047).
    pub(crate) fn chunk_all(&mut self) -> (Chunk<'_, E>, (usize, usize)) {
        self.forget_fast_facts();
        let heads = (self.actions.discrete.len(), self.actions.continuous.len());
        let chunk = Rows {
            headers: &mut self.headers,
            states: &mut self.states,
            discrete: &mut self.discrete,
            continuous: &mut self.continuous,
            penalties: &mut self.penalties,
            penalized: &mut self.penalized,
            outputs: self.outputs.view_all(),
        };
        (chunk, heads)
    }
}

/// Workers of a pool in Sync mode. Dropping it shuts them down and joins
/// them.
#[derive(Debug)]
pub(crate) struct SyncRuntime {
    pub(crate) gang: Gang,
    pub(crate) workers: usize,
    pub(crate) native: bool,
    /// Whether the workers were pinned to cores.
    pub(crate) pinned: bool,
    /// Present only for a runtime that counts as an active engine (ADR-0005).
    /// Held for its `Drop`, which unregisters; tests read it.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "held for its Drop, which unregisters")
    )]
    pub(crate) registration: Option<Registration>,
}

/// When a pool's workers pin, and whether the pool registers as an active
/// engine (ADR-0005).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pinning {
    /// `EnvPool`: always registers; pins to `cpu_set`, or to the first
    /// reported cores when no other engine is active.
    Auto,
    /// The host pool (ADR-0062 item 2): pins and registers only with an
    /// explicit `cpu_set`. Without one its workers are unpinned and it does
    /// not count as an active engine, so a pool started after it still pins
    /// as usual, possibly to cores the host pool's workers also run on.
    ExplicitOnly,
}

/// The adaptive cap in nanoseconds, `None` for the fixed policy
/// (ADR-0074). A cap above [`crate::MAX_ADAPTIVE_CAP`] is refused.
pub(crate) fn adaptive_cap_ns(policy: crate::WaitPolicy) -> Result<Option<u64>, EngineError> {
    match policy {
        crate::WaitPolicy::Fixed => Ok(None),
        crate::WaitPolicy::Adaptive { cap } if cap <= crate::MAX_ADAPTIVE_CAP => {
            u64::try_from(cap.as_nanos())
                .map(Some)
                .map_err(|_| EngineError::Options {
                    option: "wait_policy",
                    reason: "the adaptive cap must be at most 1 s",
                })
        }
        crate::WaitPolicy::Adaptive { .. } => Err(EngineError::Options {
            option: "wait_policy",
            reason: "the adaptive cap must be at most 1 s",
        }),
    }
}

/// Whether a pool's workers count as an active engine (ADR-0005): every
/// `EnvPool`, and a host pool only with an explicit `cpu_set` (ADR-0062
/// item 2).
pub(crate) fn registers(pinning: Pinning, explicit_cpu_set: bool) -> bool {
    pinning == Pinning::Auto || explicit_cpu_set
}

/// The cores the workers pin to, `None` for unpinned. A pure function of
/// the explicit `cpu_set`, the cores the OS reports (`None` where it reports
/// none, as on macOS), the worker count, the number of other active engines
/// and the pool's kind:
///
/// - an explicit `cpu_set` pins to it, for either kind, if it names one
///   reported core per worker; otherwise it is refused;
/// - without one, an `EnvPool` pins to the first reported cores when no
///   other engine is active and there are enough of them, else runs
///   unpinned;
/// - without one, a host pool runs unpinned. It does not register either
///   ([`registers`]), so an `EnvPool` started after it may auto-pin to cores
///   the host pool's unpinned workers also run on.
pub(crate) fn pin_plan(
    cpu_set: Option<Vec<usize>>,
    available: Option<&[usize]>,
    workers: usize,
    others: usize,
    pinning: Pinning,
) -> Result<Option<Vec<usize>>, EngineError> {
    match cpu_set {
        Some(cores) if cores.len() < workers => Err(EngineError::Options {
            option: "cpu_set",
            reason: "needs one core per worker",
        }),
        Some(cores) => {
            // Only ids the OS reports (and, on Windows, below 64).
            let known = |id: &usize| available.is_some_and(|a| a.contains(id));
            if cores.iter().all(known) {
                Ok(Some(cores))
            } else {
                Err(EngineError::Options {
                    option: "cpu_set",
                    reason: "contains a core id the OS does not report",
                })
            }
        }
        None if pinning == Pinning::ExplicitOnly => Ok(None),
        None if others == 0 => match available {
            Some(cores) if cores.len() >= workers => {
                Ok(Some(cores.iter().copied().take(workers).collect()))
            }
            _ => {
                tracing::warn!(
                    workers,
                    "fewer reported cores than workers; workers run unpinned"
                );
                Ok(None)
            }
        },
        None => {
            tracing::warn!(
                others,
                "another engine with workers is active; workers run unpinned (set cpu_set to partition cores)"
            );
            Ok(None)
        }
    }
}

impl SyncRuntime {
    /// Starts the workers of `options` under `pinning`. Shared by `EnvPool`
    /// and the host pool (ADR-0062).
    pub(crate) fn start(
        options: SyncOptions,
        pinning: Pinning,
        #[cfg(test)] faults: crate::sys::Faults,
    ) -> Result<Self, EngineError> {
        let workers = options.workers.get();
        let adaptive_cap_ns = adaptive_cap_ns(options.wait_policy)?;
        let available = available_cores();
        let (registration, others) = if registers(pinning, options.cpu_set.is_some()) {
            let (registration, others) = Registration::register();
            (Some(registration), others)
        } else {
            (None, 0)
        };
        let pin = pin_plan(
            options.cpu_set,
            available.as_deref(),
            workers,
            others,
            pinning,
        )?;
        #[cfg(not(loom))]
        let native = select_native(options.wait);
        #[cfg(loom)]
        let native: Option<()> = None;
        let pinned = pin.is_some();
        let gang = Gang::new(&GangOptions {
            workers: options.workers,
            spin: options.spin,
            adaptive_cap_ns,
            pin,
            #[cfg(not(loom))]
            native,
            #[cfg(loom)]
            mock_futex: false,
            #[cfg(test)]
            faults,
        })?;
        Ok(Self {
            gang,
            workers,
            native: native.is_some(),
            pinned,
            registration,
        })
    }
}

impl<E> EnvPool<E, NoMasks>
where
    E: Environment,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    /// Builds a pool without action masks: validates the config, the
    /// options and the spaces, and initializes every record. Instances need
    /// a `reset` before stepping.
    ///
    /// # Errors
    ///
    /// Returns an [`EngineError`] for an invalid config, options or space.
    pub fn new(cfg: E::Config, options: PoolOptions) -> Result<Self, EngineError> {
        // Config validation, spaces, initial values: all FP-sensitive.
        crate::entry::canonical(|| Self::build(cfg, options))
    }
}

impl<E> EnvPool<E, Masked>
where
    E: ActionMask,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    /// Builds a pool with action masks from `ActionMask`: masks are written
    /// in the observation pass, and illegal actions follow
    /// `options.illegal_action`.
    ///
    /// # Errors
    ///
    /// Returns an [`EngineError`] for an invalid config, options or space.
    pub fn with_masks(cfg: E::Config, options: PoolOptions) -> Result<Self, EngineError> {
        // Config validation, spaces, initial values: all FP-sensitive.
        crate::entry::canonical(|| Self::build(cfg, options))
    }
}

impl<E, M> EnvPool<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    fn build(cfg: E::Config, options: PoolOptions) -> Result<Self, EngineError> {
        const {
            assert!(
                <E::Header as RecordLayout>::AGENTS == E::MAX_AGENTS,
                "Environment::Header must have MAX_AGENTS agent rows"
            );
        }
        cfg.validate()?;
        if let Some((id, d)) = tessera_rl_core::envs::deprecation_of::<E>() {
            tracing::warn!(
                id,
                since = d.since(),
                note = d.note(),
                "building a pool of a deprecated environment (SPEC 16.1, ADR-0080)"
            );
        }
        if !(1..=E::MAX_AGENTS).contains(&E::initial_agents(&cfg)) {
            return Err(ConfigError {
                field: "initial_agents",
                reason: "must be in 1..=MAX_AGENTS",
            }
            .into());
        }
        options.validate()?;
        let obs_space = E::observation_space(&cfg)?;
        obs_space.validate()?;
        let action_space = E::action_space(&cfg)?;
        action_space.validate()?;
        let mut actions = ActionLayout::from_space(&action_space)?;
        let leaves = obs_leaves(&obs_space)?;

        let num_envs = usize::try_from(options.num_envs.get()).map_err(|_| too_large())?;
        let rows = num_envs.checked_mul(E::MAX_AGENTS).ok_or_else(too_large)?;
        let mask_len = mask_len_of::<E, M>(&cfg, &mut actions, rows, options.mask_budget_bytes)?;
        let outputs = Outputs::new(
            num_envs,
            E::MAX_AGENTS,
            &leaves,
            options.auto_reset == AutoReset::SameStep,
            mask_len,
        )?;
        let adaptive = <E::Header as RecordLayout>::Adaptive::from_config(&cfg);
        let headers = (0..num_envs)
            .map(|i| {
                let mut header = E::Header::state_default();
                // `i < num_envs <= u32::MAX`, so the conversion never fails.
                let seed = options.seed.offset(u64::try_from(i).unwrap_or(u64::MAX));
                let engine = header.engine_mut();
                engine.identity = RngIdentity::new(seed, LogicalId::DEFAULT);
                engine.fresh = Bool8::TRUE;
                engine.pending_reset = Bool8::TRUE;
                *header.adaptive_mut() = adaptive;
                header
            })
            .collect();
        Ok(Self {
            cfg,
            options,
            headers,
            states: (0..num_envs).map(|_| E::State::state_default()).collect(),
            discrete: vec![
                0;
                rows.checked_mul(actions.discrete.len())
                    .ok_or_else(too_large)?
            ],
            continuous: vec![
                0.0;
                rows.checked_mul(actions.continuous.len())
                    .ok_or_else(too_large)?
            ],
            actions,
            outputs,
            seen: vec![Bool8::FALSE; num_envs],
            authority: RngAuthority::engine_only(),
            poisoned: false,
            penalties: vec![0.0; if M::ENABLED { rows } else { 0 }],
            penalized: vec![Bool8::FALSE; if M::ENABLED { rows } else { 0 }],
            branch_counter: 0,
            reset_source: None,
            fast_clean: false,
            counter_bound: Some(0),
            sync: None,
            #[cfg(feature = "debug-checks")]
            allocs: crate::sys::alloc_track::AllocTally::default(),
            _masks: PhantomData,
        })
    }

    /// Switches the pool to Sync mode: `step` and `reset_all` run chunks on
    /// persistent worker threads and return when every instance is done.
    /// Results are bit-identical to Inline mode for any worker count.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::NestedEngine`] when called from an engine
    /// worker, or [`EngineError::Options`] for an invalid `cpu_set`.
    pub fn with_workers(self, options: SyncOptions) -> Result<Self, EngineError> {
        self.with_workers_inner(
            options,
            #[cfg(test)]
            crate::sys::Faults::default(),
        )
    }

    /// [`Self::with_workers`] with test-only fault injection.
    #[cfg(test)]
    pub(crate) fn with_workers_faults(
        self,
        options: SyncOptions,
        faults: crate::sys::Faults,
    ) -> Result<Self, EngineError> {
        self.with_workers_inner(options, faults)
    }

    pub(crate) fn with_workers_inner(
        mut self,
        options: SyncOptions,
        #[cfg(test)] faults: crate::sys::Faults,
    ) -> Result<Self, EngineError> {
        self.sync = Some(SyncRuntime::start(
            options,
            Pinning::Auto,
            #[cfg(test)]
            faults,
        )?);
        Ok(self)
    }

    /// Whether the workers block with the native address wait (otherwise
    /// park and unpark, or Inline mode).
    #[must_use]
    pub fn uses_native_wait(&self) -> bool {
        self.sync.as_ref().is_some_and(|s| s.native)
    }

    /// Number of worker threads; 0 in Inline mode.
    #[must_use]
    pub fn workers(&self) -> usize {
        self.sync.as_ref().map_or(0, |s| s.workers)
    }

    /// Whether the workers were pinned to cores (ADR-0005); `false` in
    /// Inline mode.
    #[must_use]
    pub fn pinned_workers(&self) -> bool {
        self.sync.as_ref().is_some_and(|s| s.pinned)
    }

    /// Times a worker found its FP environment changed at a batch start and
    /// reset it (ADR-0037).
    #[must_use]
    pub fn fp_resets(&self) -> u64 {
        self.sync.as_ref().map_or(0, |s| s.gang.fp_resets())
    }

    /// Whether a panic in environment code poisoned the pool (ADR-0028).
    #[must_use]
    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// The pool options.
    #[must_use]
    pub const fn options(&self) -> &PoolOptions {
        &self.options
    }

    /// The environment configuration of the pool.
    #[must_use]
    pub const fn config(&self) -> &E::Config {
        &self.cfg
    }

    /// Runs author code that reads only the configuration (for example
    /// `Determinize::sample` for a search root) under the pool's
    /// containment: inside the canonical floating-point environment, with
    /// a panic caught and its payload disposed of safely (ADR-0028,
    /// ADR-0037). Returns `None` if `f` panicked. The pool is not touched,
    /// so it is not poisoned. With `panic = "abort"` a panic aborts.
    pub fn contain<R>(&self, f: impl FnOnce(&E::Config) -> R) -> Option<R> {
        let cfg = &self.cfg;
        let mut out = None;
        if crate::sys::catch(|| out = Some(crate::entry::canonical(|| f(cfg)))) {
            out
        } else {
            None
        }
    }

    /// The outputs of the last call.
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`] on a poisoned pool (ADR-0028), whose rows may
    /// be partly written, until [`Self::reset_all`] or a full restore recovers
    /// it. Runs no environment code.
    pub const fn outputs(&self) -> Result<&Outputs, EngineError> {
        if self.poisoned {
            return Err(EngineError::Poisoned);
        }
        Ok(&self.outputs)
    }

    /// Number of discrete and continuous action heads per agent.
    #[must_use]
    pub fn action_heads(&self) -> (usize, usize) {
        (self.actions.discrete.len(), self.actions.continuous.len())
    }

    /// The single entry path of every public method that compares floats,
    /// runs environment code or mutates records: it applies the poison
    /// policy, then runs `f` in the canonical FP environment (ADR-0028,
    /// ADR-0037).
    pub(crate) fn guarded<R>(
        &mut self,
        poison: Poison,
        f: impl FnOnce(&mut Self) -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        if poison == Poison::Reject && self.poisoned {
            return Err(EngineError::Poisoned);
        }
        crate::entry::canonical(|| f(self))
    }

    /// [`EnvPool::guarded`] for read-only methods; they are always rejected
    /// on a poisoned pool.
    pub(crate) fn guarded_ref<R>(
        &self,
        f: impl FnOnce(&Self) -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        if self.poisoned {
            return Err(EngineError::Poisoned);
        }
        crate::entry::canonical(|| f(self))
    }

    /// The record header of instance `id`; `None` for an unknown id or a
    /// poisoned pool, whose records may be partly written.
    #[must_use]
    pub fn header(&self, id: u32) -> Option<&E::Header> {
        if self.poisoned {
            return None;
        }
        self.headers.get(usize::try_from(id).ok()?)
    }

    /// The author state of instance `id`; `None` for an unknown id or a
    /// poisoned pool.
    #[must_use]
    pub fn state(&self, id: u32) -> Option<&E::State> {
        if self.poisoned {
            return None;
        }
        self.states.get(usize::try_from(id).ok()?)
    }

    /// Sets the lineage of `id`, so instances sharing a seed draw
    /// independent streams (ADR-0007). Takes effect immediately, from the
    /// next draw of the current episode.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Poisoned`] on a poisoned pool, or
    /// [`EngineError::InstanceOutOfRange`] for an unknown id.
    pub fn set_lineage(&mut self, id: u32, lineage: LogicalId) -> Result<(), EngineError> {
        self.guarded(Poison::Reject, |pool| pool.set_lineage_inner(id, lineage))
    }

    fn set_lineage_inner(&mut self, id: u32, lineage: LogicalId) -> Result<(), EngineError> {
        let num_envs = self.options.num_envs.get();
        let header = usize::try_from(id)
            .ok()
            .and_then(|i| self.headers.get_mut(i))
            .ok_or(EngineError::InstanceOutOfRange { id, num_envs })?;
        let identity = header.engine().identity;
        header.engine_mut().identity = RngIdentity::from_parts(
            identity.seed(),
            identity.branch_salt(),
            lineage,
            identity.episode_index(),
            identity.step_index(),
        );
        Ok(())
    }

    /// Resets every instance.
    ///
    /// # Errors
    ///
    /// Returns an [`EngineError`] if a seed list has the wrong length or an
    /// episode counter is exhausted; nothing is mutated then.
    ///
    /// On a poisoned pool this is the recovery path: the poison is cleared
    /// only if every instance resets without panicking.
    pub fn reset_all(&mut self, reseed: Reseed<'_>) -> Result<&Outputs, EngineError> {
        self.guarded(Poison::Recover, |pool| pool.reset_all_inner(reseed))?;
        Ok(&self.outputs)
    }

    fn reset_all_inner(&mut self, reseed: Reseed<'_>) -> Result<(), EngineError> {
        self.note_reset();
        let n = self.headers.len();
        self.check_reset(0..n, n, reseed)?;
        let Self {
            cfg,
            options,
            actions: layout,
            headers,
            states,
            discrete,
            continuous,
            outputs,
            authority,
            poisoned,
            penalties,
            penalized,
            sync,
            reset_source,
            ..
        } = self;
        let archive = reset_source.as_ref();
        let all = Chunk::<E> {
            headers,
            states,
            discrete,
            continuous,
            penalties,
            penalized,
            outputs: outputs.view_all(),
        };
        if let Some(rt) = sync {
            let heads = (layout.discrete.len(), layout.continuous.len());
            let grain = (chunk_size(options), options.schedule);
            let panicked = par_chunks(&mut rt.gang, all, grain, heads, |chunk, first| {
                reset_chunk::<E, M>(cfg, authority, archive, chunk, reseed, first);
            })?;
            if panicked {
                *poisoned = true;
                return Err(EngineError::Panicked);
            }
        } else {
            let mut chunk = all;
            run_kernels(poisoned, || {
                reset_chunk::<E, M>(cfg, authority, archive, &mut chunk, reseed, 0);
            })?;
        }
        self.poisoned = false;
        #[cfg(feature = "debug-checks")]
        self.discard_allocations();
        self.check_turns()?;
        #[cfg(feature = "debug-checks")]
        self.check_nan(0..n)?;
        Ok(())
    }

    /// Resets the instances `ids`, in the order given; other instances keep
    /// their state and outputs.
    ///
    /// # Errors
    ///
    /// Returns an [`EngineError`] for unknown or duplicate ids, a seed list
    /// of the wrong length or an exhausted episode counter; nothing is
    /// mutated then.
    ///
    /// Rejected with [`EngineError::Poisoned`] on a poisoned pool.
    pub fn reset(&mut self, ids: &[u32], reseed: Reseed<'_>) -> Result<&Outputs, EngineError> {
        self.guarded(Poison::Reject, |pool| pool.reset_inner(ids, reseed))?;
        Ok(&self.outputs)
    }

    fn reset_inner(&mut self, ids: &[u32], reseed: Reseed<'_>) -> Result<(), EngineError> {
        self.note_reset();
        check_ids(ids, &mut self.seen)?;
        let indices = ids.iter().filter_map(|&id| usize::try_from(id).ok());
        self.check_reset(indices.clone(), ids.len(), reseed)?;
        #[cfg(not(feature = "debug-checks"))]
        self.apply_reset(indices, reseed, None)?;
        #[cfg(feature = "debug-checks")]
        self.apply_reset(indices.clone(), reseed, None)?;
        self.check_turns()?;
        #[cfg(feature = "debug-checks")]
        self.check_nan(indices)?;
        Ok(())
    }

    /// Resets the instances `ids` from the archive entries `entries`, one
    /// per id, in the order given (ADR-0075 item 5): each an archive reset
    /// (item 3) from the named entry, with no reseed. Other instances keep
    /// their state and outputs.
    ///
    /// # Errors
    ///
    /// [`EngineError::Archive`] without an attached archive or with an
    /// entry out of range, [`EngineError::RecordCount`] when `entries` and
    /// `ids` differ in length, and the errors of [`EnvPool::reset`] for the
    /// ids; nothing is mutated then. Rejected with
    /// [`EngineError::Poisoned`] on a poisoned pool.
    pub fn reset_from(&mut self, ids: &[u32], entries: &[u32]) -> Result<&Outputs, EngineError> {
        self.guarded(Poison::Reject, |pool| pool.reset_from_inner(ids, entries))?;
        Ok(&self.outputs)
    }

    fn reset_from_inner(&mut self, ids: &[u32], entries: &[u32]) -> Result<(), EngineError> {
        let len = self
            .reset_source
            .as_ref()
            .map(|s| s.archive().len())
            .ok_or(EngineError::Archive {
                reason: "reset_from needs an attached archive",
            })?;
        if entries.len() != ids.len() {
            return Err(EngineError::RecordCount {
                records: entries.len(),
                instances: ids.len(),
            });
        }
        if entries
            .iter()
            .any(|&e| !usize::try_from(e).is_ok_and(|e| e < len))
        {
            return Err(EngineError::Archive {
                reason: "an archive entry out of range",
            });
        }
        self.note_reset();
        check_ids(ids, &mut self.seen)?;
        let indices = ids.iter().filter_map(|&id| usize::try_from(id).ok());
        self.check_reset(indices.clone(), ids.len(), Reseed::Keep)?;
        #[cfg(not(feature = "debug-checks"))]
        self.apply_reset(indices, Reseed::Keep, Some(entries))?;
        #[cfg(feature = "debug-checks")]
        self.apply_reset(indices.clone(), Reseed::Keep, Some(entries))?;
        self.check_turns()?;
        #[cfg(feature = "debug-checks")]
        self.check_nan(indices)?;
        Ok(())
    }

    fn check_reset(
        &self,
        indices: impl Iterator<Item = usize>,
        count: usize,
        reseed: Reseed<'_>,
    ) -> Result<(), EngineError> {
        check_reset(&self.headers, indices, count, reseed)
    }

    fn apply_reset(
        &mut self,
        indices: impl Iterator<Item = usize>,
        reseed: Reseed<'_>,
        entries: Option<&[u32]>,
    ) -> Result<(), EngineError> {
        let Self {
            cfg,
            headers,
            states,
            outputs,
            authority,
            poisoned,
            reset_source,
            ..
        } = self;
        let archive = reset_source.as_ref();
        run_kernels(poisoned, || {
            let outputs = &mut outputs.view_all();
            for (k, i) in indices.enumerate() {
                let (Some(header), Some(state)) = (headers.get_mut(i), states.get_mut(i)) else {
                    continue;
                };
                let seed = reseed.seed(k);
                let entry = entries
                    .and_then(|e| e.get(k))
                    .and_then(|&e| usize::try_from(e).ok());
                reset_one::<E>(cfg, authority, archive, header, state, seed, entry);
                clear_instance(outputs, i);
                write_agent_masks(outputs, i, header);
                observe_instance::<E, M>(cfg, authority, header, state, outputs, i);
            }
        })
    }

    /// Checks that every required action (alive, acting, not pending reset)
    /// is legal under the current masks, without mutating anything. Always
    /// `Ok` for pools without masks. The `debug-checks` feature runs it on
    /// every step and rejects the call instead of replacing the actions.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::IllegalAction`] for the first illegal action,
    /// [`EngineError::Poisoned`] on a poisoned pool, or a validation error
    /// for malformed actions.
    pub fn check_actions(&self, actions: ActionInput<'_>) -> Result<(), EngineError> {
        // Float comparisons: the same canonical environment as `step`.
        self.guarded_ref(|pool| {
            crate::validate::check_actions_only(
                &pool.actions,
                pool.options.auto_reset,
                &pool.headers,
                actions,
            )?;
            pool.legality(actions.discrete)
        })
    }

    #[cfg(feature = "debug-checks")]
    fn check_actions_inner(&self) -> Result<(), EngineError> {
        self.legality(&self.discrete)
    }

    fn legality(&self, discrete: &[i32]) -> Result<(), EngineError> {
        if !M::ENABLED {
            return Ok(());
        }
        legality_rows(
            &self.headers,
            0,
            discrete,
            &self.actions.mask,
            self.outputs.masks(),
            self.outputs.mask_len,
        )
    }

    /// Steps every instance with `actions` (Inline mode).
    ///
    /// # Errors
    ///
    /// Returns an [`EngineError`] if the actions have the wrong shape, a
    /// required action is invalid, an instance is pending reset or a step
    /// counter is exhausted; nothing is mutated then. Returns
    /// [`EngineError::Poisoned`] on a poisoned pool, and
    /// [`EngineError::Panicked`] if environment code panics during the
    /// call, which poisons the pool.
    pub fn step(&mut self, actions: ActionInput<'_>) -> Result<&Outputs, EngineError> {
        // The validation boundary compares floats too: it runs in the
        // canonical environment with the kernels (ADR-0037).
        self.guarded(Poison::Reject, |pool| pool.step_inner(actions))?;
        Ok(&self.outputs)
    }

    fn step_inner(&mut self, actions: ActionInput<'_>) -> Result<(), EngineError> {
        // Validation fast path (env-step plan r3 section 3.3): while the
        // facts hold, the per-instance header checks cannot fail and every
        // slot is required, so one range pass over the flat buffers and one
        // copy replace them. A bad value reruns the slow path, whose error is
        // then reported exactly as before.
        let armed = self.fast_path_ready();
        let fast = armed
            && check_and_copy_actions_fast(
                &self.actions,
                actions,
                &mut self.discrete,
                &mut self.continuous,
            )?;
        if !fast {
            check_and_copy_actions(
                &self.actions,
                self.options.auto_reset,
                &self.headers,
                actions,
                &mut self.discrete,
                &mut self.continuous,
            )?;
            if self.counter_bound.is_none() {
                self.counter_bound = Some(counter_max(&self.headers));
            }
        }
        // From here the headers change. A step raises `episode_steps`, and a
        // same-step reset `episode_index`, by at most one; `fast_clean` is
        // set again below only if the step completes cleanly.
        self.fast_clean = false;
        self.counter_bound = self.counter_bound.and_then(|b| b.checked_add(1));
        #[cfg(feature = "debug-checks")]
        self.check_actions_inner()?;
        let Self {
            cfg,
            options,
            headers,
            states,
            actions: layout,
            discrete,
            continuous,
            outputs,
            authority,
            poisoned,
            penalties,
            penalized,
            sync,
            reset_source,
            #[cfg(feature = "debug-checks")]
            allocs,
            ..
        } = self;
        let ctx = StepCtx::<E> {
            cfg,
            authority,
            layout,
            policy: options.illegal_action,
            limit: options.max_episode_steps.map(core::num::NonZeroU32::get),
            auto_reset: options.auto_reset,
            archive: reset_source.as_ref(),
            #[cfg(feature = "debug-checks")]
            allocs,
        };
        let all = Chunk::<E> {
            headers,
            states,
            discrete,
            continuous,
            penalties,
            penalized,
            outputs: outputs.view_all(),
        };
        // Cleared by any chunk that ends with an instance pending reset or a
        // slot not alive and acting; read after the barrier (the gang joins
        // every worker before `par_chunks` returns).
        let clean = AtomicBool::new(true);
        if let Some(rt) = sync {
            let heads = (layout.discrete.len(), layout.continuous.len());
            let grain = (chunk_size(options), options.schedule);
            let panicked = par_chunks(&mut rt.gang, all, grain, heads, |chunk, _| {
                step_chunk::<E, M>(&ctx, chunk);
                if !may_step_cleanly::<E>() || !steps_cleanly(chunk.headers) {
                    clean.store(false, Ordering::Relaxed);
                }
            })?;
            if panicked {
                *poisoned = true;
                return Err(EngineError::Panicked);
            }
        } else {
            let mut chunk = all;
            run_kernels(poisoned, || {
                step_chunk::<E, M>(&ctx, &mut chunk);
                if !may_step_cleanly::<E>() || !steps_cleanly(chunk.headers) {
                    clean.store(false, Ordering::Relaxed);
                }
            })?;
        }
        self.check_turns()?;
        #[cfg(feature = "debug-checks")]
        {
            self.check_agent_events()?;
            self.check_allocations()?;
            self.check_nan(0..self.headers.len())?;
        }
        // Each step raises `episode_steps`, and a same-step reset
        // `episode_index`, by at most one.
        self.fast_clean = clean.load(Ordering::Relaxed);
        Ok(())
    }

    /// The turn order of a `Sequential` environment (ADR-0013): every
    /// instance whose episode goes on (not pending reset) has exactly one
    /// acting agent. A violation poisons the pool in every build, with the
    /// first instance (ADR-0028). One count per instance; nothing for
    /// `Simultaneous` and `Mixed` (where no acting agent is allowed: a step
    /// that reads no action).
    fn check_turns(&mut self) -> Result<(), EngineError> {
        if E::TURNS != TurnStructure::Sequential {
            return Ok(());
        }
        for (i, header) in self.headers.iter().enumerate() {
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

    /// `debug-checks`: an invalid agent event in the last step poisons the pool
    /// and fails the call with its first instance and slot.
    #[cfg(feature = "debug-checks")]
    fn check_agent_events(&mut self) -> Result<(), EngineError> {
        let a = self.outputs.agents;
        if let Some(k) = self
            .outputs
            .invalid_agent_event()
            .iter()
            .position(|f| f.get())
        {
            self.poisoned = true;
            return Err(EngineError::InvalidAgentEvent {
                instance: u32::try_from(k / a).unwrap_or(u32::MAX),
                agent: u32::try_from(k % a).unwrap_or(u32::MAX),
            });
        }
        Ok(())
    }

    /// `debug-checks`: an allocation counted inside a chunk step of this
    /// pool poisons it (ADR-0055).
    #[cfg(feature = "debug-checks")]
    fn check_allocations(&mut self) -> Result<(), EngineError> {
        match crate::nan_check::allocation_error(self.allocs.take()) {
            Some(e) => {
                self.poisoned = true;
                Err(e)
            }
            None => Ok(()),
        }
    }

    /// `debug-checks`: drops the allocation count left by a failed call
    /// (a panic, or a check that failed before `check_allocations`). Every
    /// path that clears the poison calls it, so a recovered pool starts
    /// from zero (ADR-0055).
    #[cfg(feature = "debug-checks")]
    pub(crate) fn discard_allocations(&self) {
        let _ = self.allocs.take();
    }

    /// `debug-checks`: the first NaN in the rewards or state of `indices`,
    /// in order, poisons the pool (ADR-0055).
    #[cfg(feature = "debug-checks")]
    fn check_nan(&mut self, indices: impl Iterator<Item = usize>) -> Result<(), EngineError> {
        let a = self.outputs.agents;
        let rewards = self.outputs.rewards();
        for i in indices {
            let Some(state) = self.states.get(i) else {
                continue;
            };
            let row = rewards.get(i * a..(i + 1) * a).unwrap_or(&[]);
            let id = u32::try_from(i).unwrap_or(u32::MAX);
            if let Some(e) = crate::nan_check::instance_nan(id, row, state) {
                self.poisoned = true;
                return Err(e);
            }
        }
        Ok(())
    }
}

/// Validates a reset request against `headers`: a seed list of the right
/// length, and an episode counter left for every instance that keeps its
/// seed. Shared by `EnvPool` and the host pool (ADR-0062).
pub(crate) fn check_reset<H: RecordLayout>(
    headers: &[H],
    indices: impl Iterator<Item = usize>,
    count: usize,
    reseed: Reseed<'_>,
) -> Result<(), EngineError> {
    if let Some(seeds) = reseed.len()
        && seeds != count
    {
        return Err(EngineError::SeedCount {
            seeds,
            instances: count,
        });
    }
    // Instances that keep their seed advance their episode index.
    for (k, i) in indices.enumerate() {
        if reseed.seed(k).is_none() {
            let engine = headers.get(i).map(RecordLayout::engine);
            if let Some(engine) = engine
                && !engine.fresh.get()
                && engine.identity.episode_index() == u32::MAX
            {
                return Err(EngineError::CounterExhausted {
                    id: u32::try_from(i).unwrap_or(u32::MAX),
                    counter: "episode_index",
                });
            }
        }
    }
    Ok(())
}

/// The native wait primitive to use for `choice`, if any.
#[cfg(not(loom))]
fn select_native(choice: WaitBackend) -> Option<crate::sys::Native> {
    let native = crate::sys::Native::detect();
    match choice {
        WaitBackend::Park => None,
        // Native by default only where it measured faster than park: Linux
        // futex (about 4% when every batch blocks). macOS os_sync showed no
        // gain, and Windows WaitOnAddress has not run in CI yet. Re-evaluate
        // on the reference hardware (ADR-0025).
        WaitBackend::Auto if !cfg!(target_os = "linux") => None,
        WaitBackend::Auto => native,
        WaitBackend::Native => {
            if native.is_none() {
                tracing::warn!("no native wait primitive on this system; workers use park");
            }
            native
        }
    }
}

/// Instances per chunk.
pub(crate) fn chunk_size(options: &PoolOptions) -> usize {
    usize::try_from(options.chunk_size.get()).unwrap_or(usize::MAX)
}

/// Whether a step of `E` can ever leave every slot acting: not when the
/// turns are `Sequential` with more than one slot (one acts per step), so
/// such pools skip the header walk of [`steps_cleanly`] (E1 review N4).
const fn may_step_cleanly<E: Environment>() -> bool {
    !(matches!(E::TURNS, TurnStructure::Sequential) && E::MAX_AGENTS > 1)
}

/// Whether every instance of `headers` will step next time: none pending
/// reset, every agent slot alive and acting (the facts of env-step plan r3
/// section 3.3, items 1 and 4).
fn steps_cleanly<H: RecordLayout>(headers: &[H]) -> bool {
    headers.iter().all(|h| {
        !h.engine().pending_reset.get()
            && h.agents().iter().all(|r| r.alive.get() && r.acting.get())
    })
}

/// The largest `episode_index` or `episode_steps` of `headers` (facts 2 and
/// 3), scanned only after a write outside the step left the bound unknown.
fn counter_max<H: RecordLayout>(headers: &[H]) -> u32 {
    headers
        .iter()
        .map(|h| {
            let e = h.engine();
            e.identity.episode_index().max(e.episode_steps)
        })
        .max()
        .unwrap_or(0)
}

/// Read-only context of a step, shared by every chunk.
pub(crate) struct StepCtx<'a, E: Environment> {
    pub(crate) cfg: &'a E::Config,
    pub(crate) authority: &'a RngAuthority,
    pub(crate) layout: &'a ActionLayout,
    pub(crate) policy: IllegalActionPolicy,
    pub(crate) limit: Option<u32>,
    pub(crate) auto_reset: AutoReset,
    /// The attached archive source of resets (ADR-0075).
    pub(crate) archive: Option<&'a ArchiveSource<E>>,
    /// The pool's allocation tally (ADR-0055).
    #[cfg(feature = "debug-checks")]
    pub(crate) allocs: &'a crate::sys::alloc_track::AllocTally,
}

/// The records, action rows and output rows of a range of instances,
/// indexed from the first instance of the range. A chunk is the unit of
/// scheduling (SPEC 5.1): kernels only ever see one chunk. `H` is the
/// record header and `S` the per-instance state: an `Environment`'s
/// `Header` and `State` ([`Chunk`]), or a host pool's header and host
/// instance (ADR-0062).
pub(crate) struct Rows<'a, H, S> {
    pub(crate) headers: &'a mut [H],
    pub(crate) states: &'a mut [S],
    pub(crate) discrete: &'a mut [i32],
    pub(crate) continuous: &'a mut [f32],
    pub(crate) penalties: &'a mut [f32],
    pub(crate) penalized: &'a mut [Bool8],
    pub(crate) outputs: OutputsView<'a>,
}

/// The rows of an `Environment`'s pool.
pub(crate) type Chunk<'a, E> = Rows<'a, <E as Environment>::Header, <E as Environment>::State>;

/// Steps every instance of `chunk` (after validation).
pub(crate) fn step_chunk<E: Environment, M: MaskProvider<E>>(
    ctx: &StepCtx<'_, E>,
    chunk: &mut Chunk<'_, E>,
) {
    // The bracket of ADR-0055 covers the whole chunk step: the illegal-action
    // pass, step, observe, the mask writes and the auto-resets below.
    #[cfg(feature = "debug-checks")]
    let _bracket = crate::sys::alloc_track::ChunkBracket::enter(ctx.allocs);
    let StepCtx {
        cfg,
        authority,
        layout,
        policy,
        limit,
        auto_reset,
        archive,
        ..
    } = *ctx;
    let Rows {
        headers,
        states,
        discrete,
        continuous,
        penalties,
        penalized,
        outputs,
    } = chunk;
    let (dh, ch) = (layout.discrete.len(), layout.continuous.len());
    // Every instance of the chunk is reset or stepped below, and each one's
    // rows are cleared before anything writes them; one fill per buffer for
    // the whole chunk replaces the per-instance fills (issue #152, G1). No
    // write to instance `i`'s rows happens between this clear and its turn.
    clear_chunk(outputs, headers.len());
    for (i, (header, state)) in headers.iter_mut().zip(states.iter_mut()).enumerate() {
        if header.engine().pending_reset.get() {
            // NextStep: reset instead of stepping (validated above).
            reset_one::<E>(cfg, authority, archive, header, state, None, None);
            write_agent_masks(outputs, i, header);
            observe_cleared::<E, M>(cfg, authority, header, state, outputs, i);
            continue;
        }
        if M::ENABLED {
            // Inside the panic catch: the author's noop may panic.
            if layout.mask.flat && layout.mask.group_heads == 0 {
                replace_illegal_one::<E, M>(
                    cfg, policy, header, state, outputs, layout, discrete, continuous, penalties,
                    penalized, i,
                );
            } else {
                replace_illegal_conditional::<E, M>(
                    cfg, policy, header, state, outputs, layout, discrete, continuous, penalties,
                    penalized, i,
                );
            }
        }
        let rows = i * E::MAX_AGENTS;
        let view = Actions::new(
            discrete
                .get(rows * dh..(rows + E::MAX_AGENTS) * dh)
                .unwrap_or(&[]),
            dh,
            continuous
                .get(rows * ch..(rows + E::MAX_AGENTS) * ch)
                .unwrap_or(&[]),
            ch,
        );
        let agents = i * E::MAX_AGENTS..(i + 1) * E::MAX_AGENTS;
        let penalty = penalties.get(agents.clone()).unwrap_or(&[]);
        let marked = penalized.get(agents).unwrap_or(&[]);
        step_one::<E>(
            cfg, authority, header, state, &view, outputs, i, limit, penalty, marked,
        );
        observe_cleared::<E, M>(cfg, authority, header, state, outputs, i);
        if auto_reset == AutoReset::SameStep && header.engine().pending_reset.get() {
            save_terminal(outputs, i);
            // Flags and stats of the ended episode stay set.
            reset_one::<E>(cfg, authority, archive, header, state, None, None);
            write_agent_masks(outputs, i, header);
            // The rows hold the terminal observation: zero them again.
            observe_instance::<E, M>(cfg, authority, header, state, outputs, i);
        }
    }
}

/// Resets every instance of `chunk`; `first` is the position of the
/// chunk's first instance among the reset instances (for `Reseed`).
pub(crate) fn reset_chunk<E: Environment, M: MaskProvider<E>>(
    cfg: &E::Config,
    authority: &RngAuthority,
    archive: Option<&ArchiveSource<E>>,
    chunk: &mut Chunk<'_, E>,
    reseed: Reseed<'_>,
    first: usize,
) {
    let outputs = &mut chunk.outputs;
    clear_chunk(outputs, chunk.headers.len());
    for (i, (header, state)) in chunk
        .headers
        .iter_mut()
        .zip(chunk.states.iter_mut())
        .enumerate()
    {
        let k = first + i;
        let seed = reseed.seed(k);
        reset_one::<E>(cfg, authority, archive, header, state, seed, None);
        write_agent_masks(outputs, i, header);
        observe_cleared::<E, M>(cfg, authority, header, state, outputs, i);
    }
}

/// Runs environment code at a chunk boundary: inside the canonical FP
/// environment, catching a panic and poisoning the pool (ADR-0028). There
/// is no rollback: instances processed before the panic keep their new
/// state. With `panic = "abort"` a panic aborts the process instead.
pub(crate) fn run_kernels(poisoned: &mut bool, f: impl FnOnce()) -> Result<(), EngineError> {
    // `catch` also disposes of the payload safely: one whose `Drop` panics
    // must not unwind out of the call.
    if crate::sys::catch(|| fp::with_canonical(f)) {
        Ok(())
    } else {
        *poisoned = true;
        Err(EngineError::Panicked)
    }
}

/// Replaces illegal-but-valid actions of instance `i` with the
/// environment's no-op, counts them and marks penalties (ADR-0028). Runs
/// before the kernel, against the masks of the current state. There is no
/// rollback: if a later kernel panics, counters already incremented stay.
/// This is the path of layouts without parents or groups, with the same
/// reads as before ADR-0051 so that its cost stays where it was (gungraun's
/// `turns` benches); the others take [`replace_illegal_conditional`].
#[expect(
    clippy::too_many_arguments,
    reason = "one instance's illegal-action pass"
)]
fn replace_illegal_one<E: Environment, M: MaskProvider<E>>(
    cfg: &E::Config,
    policy: IllegalActionPolicy,
    header: &mut E::Header,
    state: &E::State,
    outputs: &OutputsView<'_>,
    layout: &ActionLayout,
    discrete: &mut [i32],
    continuous: &mut [f32],
    penalties: &mut [f32],
    penalized: &mut [Bool8],
    i: usize,
) {
    let a = E::MAX_AGENTS;
    let (dh, ch) = (layout.discrete.len(), layout.continuous.len());
    let mask_len = outputs.mask_len;
    for (slot, row) in header.agents_mut().iter_mut().enumerate() {
        let k = i * a + slot;
        if let Some(f) = penalized.get_mut(k) {
            *f = Bool8::FALSE;
        }
        if !(row.alive.get() && row.acting.get()) {
            continue;
        }
        let mask = outputs
            .masks
            .get(k * mask_len..(k + 1) * mask_len)
            .unwrap_or(&[]);
        let heads = discrete.get(k * dh..(k + 1) * dh).unwrap_or(&[]);
        if first_illegal_flat(heads, &layout.discrete, mask).is_none() {
            continue;
        }
        // Counts saturate at u32::MAX per episode (documented).
        row.illegal_actions = row.illegal_actions.saturating_add(1);
        if let IllegalActionPolicy::Penalize(r) = policy
            && let (Some(p), Some(f)) = (penalties.get_mut(k), penalized.get_mut(k))
        {
            *p = r;
            *f = Bool8::TRUE;
        }
        let (Some(disc), Some(cont), Some(agent)) = (
            discrete.get_mut(k * dh..(k + 1) * dh),
            continuous.get_mut(k * ch..(k + 1) * ch),
            AgentSlot::new(u16::try_from(slot).unwrap_or(u16::MAX), a),
        ) else {
            continue;
        };
        disc.fill(0);
        cont.fill(0.0);
        M::noop(cfg, state, agent, &mut ActionWriter::new(disc, cont));
        debug_assert!(
            disc.iter()
                .zip(&layout.discrete)
                .all(|(&v, &n)| u32::try_from(v).is_ok_and(|v| v < n)),
            "ActionMask::noop wrote a discrete action outside its range"
        );
        debug_assert!(
            cont.iter()
                .zip(&layout.continuous)
                .all(|(&v, &(low, high))| {
                    v.is_finite() && (low..=high).contains(&f64::from(v))
                }),
            "ActionMask::noop wrote a continuous action outside its bounds"
        );
    }
}

/// Replaces illegal-but-valid actions of instance `i` with the
/// environment's no-op, counts them and marks penalties (ADR-0028). Runs
/// before the kernel, against the masks of the current state. There is no
/// rollback: if a later kernel panics, counters already incremented stay.
/// The path of layouts with conditional heads or entity groups (ADR-0051).
#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "one instance's illegal-action pass"
)]
fn replace_illegal_conditional<E: Environment, M: MaskProvider<E>>(
    cfg: &E::Config,
    policy: IllegalActionPolicy,
    header: &mut E::Header,
    state: &E::State,
    outputs: &OutputsView<'_>,
    layout: &ActionLayout,
    discrete: &mut [i32],
    continuous: &mut [f32],
    penalties: &mut [f32],
    penalized: &mut [Bool8],
    i: usize,
) {
    let a = E::MAX_AGENTS;
    let (dh, ch) = (layout.discrete.len(), layout.continuous.len());
    let mask_len = outputs.mask_len;
    for (slot, row) in header.agents_mut().iter_mut().enumerate() {
        let k = i * a + slot;
        if let Some(f) = penalized.get_mut(k) {
            *f = Bool8::FALSE;
        }
        if !(row.alive.get() && row.acting.get()) {
            continue;
        }
        let mask = outputs
            .masks
            .get(k * mask_len..(k + 1) * mask_len)
            .unwrap_or(&[]);
        let heads = discrete.get(k * dh..(k + 1) * dh).unwrap_or(&[]);
        let Some(first) = first_illegal(heads, &layout.mask, 0..dh, mask) else {
            continue;
        };
        // The penalty is once per agent on both paths (ADR-0051).
        if let IllegalActionPolicy::Penalize(r) = policy
            && let (Some(p), Some(f)) = (penalties.get_mut(k), penalized.get_mut(k))
        {
            *p = r;
            *f = Bool8::TRUE;
        }
        if layout.mask.group_heads != 0 {
            // Out of line, so that pools without groups keep the code they
            // had before ADR-0051 (gungraun's `turns` benches).
            fallback_groups::<E, M>(
                cfg,
                state,
                slot,
                discrete,
                k,
                &layout.mask,
                mask,
                first,
                row,
            );
            continue;
        }
        // Counts saturate at u32::MAX per episode (documented).
        row.illegal_actions = row.illegal_actions.saturating_add(1);
        let (Some(disc), Some(cont), Some(agent)) = (
            discrete.get_mut(k * dh..(k + 1) * dh),
            continuous.get_mut(k * ch..(k + 1) * ch),
            AgentSlot::new(u16::try_from(slot).unwrap_or(u16::MAX), a),
        ) else {
            continue;
        };
        disc.fill(0);
        cont.fill(0.0);
        M::noop(cfg, state, agent, &mut ActionWriter::new(disc, cont));
        debug_assert!(
            disc.iter()
                .zip(&layout.discrete)
                .all(|(&v, &n)| u32::try_from(v).is_ok_and(|v| v < n)),
            "ActionMask::noop wrote a discrete action outside its range"
        );
        debug_assert!(
            cont.iter()
                .zip(&layout.continuous)
                .all(|(&v, &(low, high))| {
                    v.is_finite() && (low..=high).contains(&f64::from(v))
                }),
            "ActionMask::noop wrote a continuous action outside its bounds"
        );
    }
}

/// Builds the mask layout into `actions` and returns the row length, 0
/// without masks. Refused before any allocation (SPEC 11.3).
fn mask_len_of<E: Environment, M: MaskProvider<E>>(
    cfg: &E::Config,
    actions: &mut ActionLayout,
    rows: usize,
    budget: u64,
) -> Result<usize, EngineError> {
    if !M::ENABLED {
        return Ok(0);
    }
    let parents: Vec<_> = (0..actions.discrete.len())
        .map(|head| M::head_parents(cfg, head))
        .collect();
    let groups = M::entity_group_heads(cfg).map(|g| usize::from(g.get()));
    let built = MaskLayout::build(
        &actions.discrete,
        &parents,
        groups,
        !actions.continuous.is_empty(),
    )?;
    let bytes = built
        .as_ref()
        .and_then(|m| rows.checked_mul(m.len))
        .and_then(|b| u64::try_from(b).ok());
    match (built, bytes) {
        (Some(mask), Some(bytes)) if bytes <= budget => {
            let len = mask.len;
            actions.mask = mask;
            Ok(len)
        }
        (_, bytes) => Err(EngineError::MaskBudgetExceeded {
            bytes: bytes.unwrap_or(u64::MAX),
            budget,
        }),
    }
}

/// ADR-0051: each illegal entity group, from the one holding `first` on,
/// takes its heads from one `noop`, counted per group.
#[cold]
#[inline(never)]
#[expect(clippy::too_many_arguments, reason = "the views of one agent's row")]
fn fallback_groups<E: Environment, M: MaskProvider<E>>(
    cfg: &E::Config,
    state: &E::State,
    slot: usize,
    discrete: &mut [i32],
    k: usize,
    layout: &MaskLayout,
    mask: &[Bool8],
    first: usize,
    row: &mut AgentRow,
) {
    let dh = layout.heads.len();
    let agent = AgentSlot::new(u16::try_from(slot).unwrap_or(u16::MAX), E::MAX_AGENTS);
    let (Some(heads), Some(agent)) = (discrete.get_mut(k * dh..(k + 1) * dh), agent) else {
        return;
    };
    // Zero-filled on each use: a head the author leaves unwritten is 0.
    let mut scratch = [0i32; MAX_GROUPED_HEADS];
    // `MaskLayout::build` caps grouped heads at the scratch's length, so this
    // cannot fail; the assert says so rather than letting the row run quietly.
    debug_assert!(heads.len() <= MAX_GROUPED_HEADS);
    let Some(noop) = scratch.get_mut(..heads.len()) else {
        return;
    };
    M::noop(cfg, state, agent, &mut ActionWriter::new(noop, &mut []));
    let size = layout.group_heads;
    for group in first / size..layout.group_count {
        let range = group * size..(group + 1) * size;
        if first_illegal(heads, layout, range.clone(), mask).is_none() {
            continue;
        }
        row.illegal_actions = row.illegal_actions.saturating_add(1);
        if let (Some(to), Some(from)) = (heads.get_mut(range.clone()), noop.get(range.clone())) {
            to.copy_from_slice(from);
        }
        if cfg!(any(debug_assertions, feature = "debug-checks")) {
            // The range first: `first_illegal`'s own check is off in a
            // release build with `debug-checks`, and an out-of-range no-op
            // could land on another row's legal entry.
            let in_range = heads
                .get(range.clone())
                .zip(layout.heads.get(range.clone()))
                .is_some_and(|(values, entries)| {
                    values
                        .iter()
                        .zip(entries)
                        .all(|(&v, e)| usize::try_from(v).is_ok_and(|v| v < e.size))
                });
            assert!(
                in_range && first_illegal(heads, layout, range, mask).is_none(),
                "ActionMask::noop wrote a group its mask forbids (ADR-0051)"
            );
        }
    }
}

/// The legality pass of the validation boundary (masked pools): the first
/// alive, acting slot of an instance whose discrete action the mask forbids
/// is an `IllegalAction`. `headers`, `discrete` (`[rows, A, dh]`) and
/// `masks` (`[rows, A, mask_len]`) cover the same rows, the first of which
/// is instance `first`; Sync passes the whole pool, Pipelined one half.
pub(crate) fn legality_rows<H: RecordLayout>(
    headers: &[H],
    first: usize,
    discrete: &[i32],
    layout: &MaskLayout,
    masks: &[Bool8],
    mask_len: usize,
) -> Result<(), EngineError> {
    let a = H::AGENTS;
    let dh = layout.heads.len();
    for (i, header) in headers.iter().enumerate() {
        if header.engine().pending_reset.get() {
            continue;
        }
        for (slot, row) in header.agents().iter().enumerate() {
            if !(row.alive.get() && row.acting.get()) {
                continue;
            }
            let k = i * a + slot;
            let heads = discrete.get(k * dh..(k + 1) * dh).unwrap_or(&[]);
            let mask = masks.get(k * mask_len..(k + 1) * mask_len).unwrap_or(&[]);
            if let Some(head) = first_illegal(heads, layout, 0..dh, mask) {
                return Err(EngineError::IllegalAction {
                    instance: u32::try_from(first + i).unwrap_or(u32::MAX),
                    agent: u16::try_from(slot).unwrap_or(u16::MAX),
                    head: u32::try_from(head).unwrap_or(u32::MAX),
                    group: u32::try_from(head.checked_div(layout.group_heads).unwrap_or(0))
                        .unwrap_or(u32::MAX),
                    value: heads.get(head).copied().unwrap_or(0),
                });
            }
        }
    }
    Ok(())
}

/// The first head of `range` whose value the mask forbids. A layout
/// without parents read whole takes the flat reader, so pools that declare
/// no conditional heads pay what they paid before ADR-0051 (gungraun's
/// `turns` benches); everything else takes the conditional reader.
#[inline]
pub(crate) fn first_illegal(
    heads: &[i32],
    layout: &MaskLayout,
    range: core::ops::Range<usize>,
    mask: &[Bool8],
) -> Option<usize> {
    if layout.flat && range.start == 0 && range.end == layout.sizes.len() {
        first_illegal_flat(heads, &layout.sizes, mask)
    } else {
        first_illegal_conditional(heads, layout, range, mask)
    }
}

/// The reader that predates conditional masks: heads' blocks end to end.
pub(crate) fn first_illegal_flat(heads: &[i32], sizes: &[u32], mask: &[Bool8]) -> Option<usize> {
    let mut offset = 0usize;
    for (head, (&value, &n)) in heads.iter().zip(sizes).enumerate() {
        let legal = usize::try_from(value)
            .ok()
            .and_then(|v| mask.get(offset.checked_add(v)?))
            .is_some_and(|m| m.get());
        if !legal {
            return Some(head);
        }
        offset = offset.saturating_add(usize::try_from(n).unwrap_or(0));
    }
    None
}

/// The first head of `range` whose value the mask forbids, parents first.
/// Reads each conditional head's row through its parents' values; a head
/// whose parent is illegal is never reached. Range validation has run.
/// A row shorter than `range` reads as illegal at its first missing head.
pub(crate) fn first_illegal_conditional(
    heads: &[i32],
    layout: &MaskLayout,
    range: core::ops::Range<usize>,
    mask: &[Bool8],
) -> Option<usize> {
    for head in range {
        let (Some(&value), Some(entry)) = (heads.get(head), layout.heads.get(head)) else {
            return Some(head);
        };
        let mut offset = Some(entry.offset);
        for (parent, stride) in entry.parents.iter().flatten() {
            let chosen = heads.get(*parent).and_then(|&v| usize::try_from(v).ok());
            offset = offset
                .zip(chosen)
                .and_then(|(o, c)| o.checked_add(c.checked_mul(*stride)?));
        }
        debug_assert!(
            usize::try_from(value).is_ok_and(|v| v < entry.size),
            "a head's value out of its range reached the mask reader"
        );
        let legal = usize::try_from(value)
            .ok()
            .and_then(|v| mask.get(offset?.checked_add(v)?))
            .is_some_and(|m| m.get());
        if !legal {
            return Some(head);
        }
    }
    None
}

/// Copies instance `i`'s observation rows into the terminal buffer.
pub(crate) fn save_terminal(outputs: &mut OutputsView<'_>, i: usize) {
    let a = outputs.agents;
    let OutputsView {
        observations,
        terminal_observations,
        leaf_bytes,
        ..
    } = outputs;
    for ((obs, term), &bytes) in observations
        .iter()
        .zip(terminal_observations.iter_mut())
        .zip(leaf_bytes.iter())
    {
        let range = i * a * bytes..(i + 1) * a * bytes;
        if let (Some(src), Some(dst)) = (obs.get(range.clone()), term.get_mut(range)) {
            dst.copy_from_slice(src);
        }
    }
}

pub(crate) fn too_large() -> EngineError {
    EngineError::Options {
        option: "num_envs",
        reason: "buffers too large",
    }
}

pub(crate) fn with_step<H: RecordLayout>(header: &mut H, episode: u32, step: u32) -> RngIdentity {
    let id = header.engine().identity;
    let identity =
        RngIdentity::from_parts(id.seed(), id.branch_salt(), id.logical_id(), episode, step);
    header.engine_mut().identity = identity;
    identity
}

/// Resets one instance: a fresh `Environment::reset`, or with an archive
/// source an archive reset (ADR-0075 item 3): the same header rewrite (the
/// episode index moves on, the step index is 0, the instance's own seed,
/// salt, lineage and generations), then the entry's state and alive slots
/// in place of `Environment::reset`. `entry` names the entry (`reset_from`,
/// validated at the boundary); `None` draws it (item 4).
pub(crate) fn reset_one<E: Environment>(
    cfg: &E::Config,
    authority: &RngAuthority,
    archive: Option<&ArchiveSource<E>>,
    header: &mut E::Header,
    state: &mut E::State,
    seed: Option<Seed>,
    entry: Option<usize>,
) {
    let engine = header.engine();
    let mut episode = engine.identity.episode_index();
    if let Some(seed) = seed {
        let id = engine.identity;
        header.engine_mut().identity =
            RngIdentity::from_parts(seed, id.branch_salt(), id.logical_id(), 0, 0);
        episode = 0;
    } else if !engine.fresh.get() {
        // Checked at the boundary.
        episode = episode.saturating_add(1);
    }
    let identity = with_step(header, episode, 0);
    let engine = header.engine_mut();
    engine.fresh = Bool8::FALSE;
    engine.pending_reset = Bool8::FALSE;
    engine.end_kind = 0;
    engine.episode_steps = 0;
    // Checked at construction.
    let initial = E::initial_agents(cfg);
    for (slot, row) in header.agents_mut().iter_mut().enumerate() {
        let alive = Bool8::from(slot < initial);
        row.alive = alive;
        row.acting = alive;
        row.episode_return = 0.0;
        row.illegal_actions = 0;
        row.invalid_agent_events = 0;
    }
    if let Some(source) = archive {
        let pick = entry.unwrap_or_else(|| source.select(authority, &identity));
        // `select` and `reset_from`'s boundary check keep `pick` in range.
        debug_assert!(
            pick < source.archive().len(),
            "archive entry {pick} out of range"
        );
        if let Some((saved, alive)) = source.archive().entry(pick) {
            *state = *saved;
            for (row, &on) in header.agents_mut().iter_mut().zip(alive) {
                row.alive = on;
                row.acting = on;
            }
            update_acting::<E>(cfg, header, state);
            return;
        }
    }
    *state = E::State::state_default();
    let mut rng = StepRng::new(authority, &identity, StreamId::DYNAMICS);
    E::reset(cfg, &mut rng, state);
    update_acting::<E>(cfg, header, state);
}

/// Sets the acting flag of every alive agent from the environment's turn
/// structure (ADR-0013): every alive agent for `Simultaneous`, the agents
/// `Environment::acting` names otherwise. It runs after the step's agent
/// events (deaths and spawns) are applied, so `acting` sees the step's
/// final alive set. Whether the count is valid is checked by
/// `EnvPool::check_turns` after the kernels.
fn update_acting<E: Environment>(cfg: &E::Config, header: &mut E::Header, state: &E::State) {
    if E::TURNS == TurnStructure::Simultaneous {
        return;
    }
    for (slot, row) in header.agents_mut().iter_mut().enumerate() {
        let on = row.alive.get()
            && u16::try_from(slot)
                .ok()
                .and_then(|s| AgentSlot::new(s, E::MAX_AGENTS))
                .is_some_and(|s| E::acting(cfg, state, s));
        row.acting = Bool8::from(on);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "one instance's step, split out of the loop"
)]
fn step_one<E: Environment>(
    cfg: &E::Config,
    authority: &RngAuthority,
    header: &mut E::Header,
    state: &mut E::State,
    actions: &Actions<'_, E::Action>,
    outputs: &mut OutputsView<'_>,
    i: usize,
    limit: Option<u32>,
    penalties: &[f32],
    penalized: &[Bool8],
) {
    let a = E::MAX_AGENTS;
    let range = i * a..(i + 1) * a;
    let t = header.engine().episode_steps.saturating_add(1);
    let episode = header.engine().identity.episode_index();
    let identity = with_step(header, episode, t);
    // The caller cleared instance `i`'s rows (`clear_chunk`).
    let end = {
        let OutputsView {
            rewards,
            agent_terminated,
            agent_truncated,
            died,
            spawned,
            ..
        } = outputs;
        let (Some(reward_row), Some(term_row), Some(trunc_row), Some(died_row), Some(spawned_row)) = (
            rewards.get_mut(range.clone()),
            agent_terminated.get_mut(range.clone()),
            agent_truncated.get_mut(range.clone()),
            died.get_mut(range.clone()),
            spawned.get_mut(range.clone()),
        ) else {
            return;
        };
        let mut out = StepOut::new(reward_row, term_row, trunc_row, died_row, spawned_row);
        let mut rng = StepRng::new(authority, &identity, StreamId::DYNAMICS);
        E::step(cfg, state, actions, &mut rng, &mut out);
        (out.episode_end(), out.agent_events())
    };
    let (end, events) = end;
    // TimeLimit semantics (SPEC 5.4): reaching the step limit truncates the
    // episode in addition to whatever the environment reported.
    let end = Ending {
        terminated: end == Some(EpisodeEnd::Terminated),
        truncated: end == Some(EpisodeEnd::Truncated) || limit == Some(t),
    };
    let end = (end.terminated || end.truncated).then_some(end);

    header.engine_mut().episode_steps = t;
    let mut total = 0.0f32;
    for (slot, row) in header.agents_mut().iter_mut().enumerate() {
        let k = i * a + slot;
        // Only a penalized illegal action changes the reward, so a kernel's
        // -0.0 stays -0.0 otherwise.
        if penalized.get(slot).is_some_and(|f| f.get())
            && let (Some(r), Some(&p)) = (outputs.rewards.get_mut(k), penalties.get(slot))
        {
            *r += p;
        }
        let reward = outputs.rewards.get(k).copied().unwrap_or(0.0);
        // Under `debug-checks` a NaN is reported as `NanDetected` after the
        // barrier (ADR-0055) instead of panicking here.
        debug_assert!(
            reward.is_finite() || (cfg!(feature = "debug-checks") && reward.is_nan()),
            "non-finite reward for agent slot {slot}"
        );
        row.episode_return += reward;
        total += row.episode_return;
        // The died and spawned rows were cleared before the kernel, and only
        // `StepOut::agent_died` and `agent_spawned` set them: without a call
        // they are all false, so they are not read.
        let mut died = events && outputs.died.get(k).is_some_and(|f| f.get());
        let spawned = events && outputs.spawned.get(k).is_some_and(|f| f.get());
        if died && !row.alive.get() {
            // An invalid agent event, like a spawn on an occupied slot: the
            // death of an agent that is already dead is never applied (its
            // final `terminated` flag was reported once), it is counted, and
            // `debug-checks` turns it into an error.
            died = false;
            row.invalid_agent_events = row.invalid_agent_events.saturating_add(1);
            set(outputs.invalid_agent_event, k, true);
            set(outputs.agent_terminated, k, false);
        }
        if died {
            row.alive = Bool8::FALSE;
            row.acting = Bool8::FALSE;
        }
        // Same-step slot reuse (SPEC 9.1): a spawn needs an empty slot that
        // did not die this step. Otherwise it is never applied: it is
        // counted (saturating at u16::MAX, a documented statistic bound)
        // and marked, and `debug-checks` turns it into an error.
        if spawned {
            let valid = !died && !row.alive.get();
            if !valid {
                row.invalid_agent_events = row.invalid_agent_events.saturating_add(1);
                set(outputs.invalid_agent_event, k, true);
                set(outputs.spawned, k, false);
            }
            if valid {
                row.alive = Bool8::TRUE;
                row.acting = Bool8::TRUE;
                // Generations count occupants modulo 2^32: an explicit,
                // documented wrap (unreachable in practice).
                row.generation = row.generation.wrapping_add(1);
            }
        }
    }
    // Terminal flush (SPEC 12): when the episode ends, every agent still
    // alive gets its final terminated or truncated flag in this step.
    if let Some(end) = end {
        for (slot, row) in header.agents().iter().enumerate() {
            if row.alive.get() {
                let k = i * a + slot;
                if end.terminated {
                    set(outputs.agent_terminated, k, true);
                }
                if end.truncated {
                    set(outputs.agent_truncated, k, true);
                }
            }
        }
    }
    update_acting::<E>(cfg, header, state);
    if let Some(end) = end {
        finish_episode(header, outputs, i, end, t, total);
    }
    write_agent_masks(outputs, i, header);
}

/// How an episode ended; both flags may be set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Ending {
    pub(crate) terminated: bool,
    pub(crate) truncated: bool,
}

/// Records the end of an episode: flags, pending reset and statistics.
pub(crate) fn finish_episode<H: RecordLayout>(
    header: &mut H,
    outputs: &mut OutputsView<'_>,
    i: usize,
    end: Ending,
    t: u32,
    total: f32,
) {
    let engine = header.engine_mut();
    engine.pending_reset = Bool8::TRUE;
    engine.end_kind = u8::from(end.terminated) | (u8::from(end.truncated) << 1);
    set(outputs.episode_terminated, i, end.terminated);
    set(outputs.episode_truncated, i, end.truncated);
    if let Some(r) = outputs.episode_return.get_mut(i) {
        *r = total;
    }
    let illegal: u32 = header
        .agents()
        .iter()
        .fold(0u32, |acc, row| acc.saturating_add(row.illegal_actions));
    if let Some(n) = outputs.episode_illegal_actions.get_mut(i) {
        // Statistics are f32 (SPEC 5.2); exact below 2^24.
        #[expect(clippy::cast_precision_loss, reason = "statistic, exact below 2^24")]
        let count = illegal as f32;
        *n = count;
    }
    let spawns: u32 = header.agents().iter().fold(0u32, |acc, row| {
        acc.saturating_add(u32::from(row.invalid_agent_events))
    });
    if let Some(n) = outputs.episode_invalid_agent_events.get_mut(i) {
        #[expect(clippy::cast_precision_loss, reason = "statistic, exact below 2^24")]
        let count = spawns as f32;
        *n = count;
    }
    if let Some(l) = outputs.episode_length.get_mut(i) {
        // Episode lengths are reported as f32 statistics (SPEC 5.2).
        #[expect(clippy::cast_precision_loss, reason = "statistic, exact below 2^24")]
        let length = t as f32;
        *l = length;
    }
}

pub(crate) fn set(buf: &mut [Bool8], i: usize, value: bool) {
    if let Some(f) = buf.get_mut(i) {
        *f = Bool8::new(value);
    }
}

pub(crate) fn clear_instance(outputs: &mut OutputsView<'_>, i: usize) {
    let a = outputs.agents;
    let range = i * a..(i + 1) * a;
    for buf in [
        &mut outputs.agent_terminated,
        &mut outputs.agent_truncated,
        &mut outputs.died,
        &mut outputs.spawned,
        &mut outputs.invalid_agent_event,
    ] {
        if let Some(rows) = buf.get_mut(range.clone()) {
            rows.fill(Bool8::FALSE);
        }
    }
    if let Some(rows) = outputs.rewards.get_mut(range) {
        rows.fill(0.0);
    }
    set(outputs.episode_terminated, i, false);
    set(outputs.episode_truncated, i, false);
    for buf in [
        &mut outputs.episode_return,
        &mut outputs.episode_length,
        &mut outputs.episode_illegal_actions,
        &mut outputs.episode_invalid_agent_events,
    ] {
        if let Some(v) = buf.get_mut(i) {
            *v = 0.0;
        }
    }
}

/// Clears the rows of the first `n` instances of a chunk view, as
/// `clear_instance` does for one instance, and zeroes their observation
/// rows as `observe_instance` does on entry (ADR-0058): one fill per buffer
/// instead of one per instance and buffer. Mask rows are left alone.
pub(crate) fn clear_chunk(outputs: &mut OutputsView<'_>, n: usize) {
    let rows = n.saturating_mul(outputs.agents);
    for buf in [
        &mut outputs.agent_terminated,
        &mut outputs.agent_truncated,
        &mut outputs.died,
        &mut outputs.spawned,
        &mut outputs.invalid_agent_event,
    ] {
        if let Some(r) = buf.get_mut(..rows) {
            r.fill(Bool8::FALSE);
        }
    }
    if let Some(r) = outputs.rewards.get_mut(..rows) {
        r.fill(0.0);
    }
    for buf in [
        &mut outputs.episode_terminated,
        &mut outputs.episode_truncated,
    ] {
        if let Some(r) = buf.get_mut(..n) {
            r.fill(Bool8::FALSE);
        }
    }
    for buf in [
        &mut outputs.episode_return,
        &mut outputs.episode_length,
        &mut outputs.episode_illegal_actions,
        &mut outputs.episode_invalid_agent_events,
    ] {
        if let Some(r) = buf.get_mut(..n) {
            r.fill(0.0);
        }
    }
    for (buf, &bytes) in outputs
        .observations
        .iter_mut()
        .zip(outputs.leaf_bytes.iter())
    {
        if let Some(r) = buf.get_mut(..rows.saturating_mul(bytes)) {
            r.fill(0);
        }
    }
    // Not the masks: the illegal-action pass reads each instance's masks
    // from the previous step before its turn, so they are zeroed per row
    // in `observe_rows`, as before.
}

/// Zeroes instance `i`'s terminal observation rows: they are a step
/// output, not continuation state (ADR-0039).
pub(crate) fn clear_terminal(outputs: &mut OutputsView<'_>, i: usize) {
    let a = outputs.agents;
    for (rows, &bytes) in outputs
        .terminal_observations
        .iter_mut()
        .zip(outputs.leaf_bytes)
    {
        if let Some(r) = rows.get_mut(i * a * bytes..(i + 1) * a * bytes) {
            r.fill(0);
        }
    }
}

pub(crate) fn write_agent_masks<H: RecordLayout>(
    outputs: &mut OutputsView<'_>,
    i: usize,
    header: &H,
) {
    let a = outputs.agents;
    for (slot, row) in header.agents().iter().enumerate() {
        let k = i * a + slot;
        if let Some(f) = outputs.alive.get_mut(k) {
            *f = row.alive;
        }
        if let Some(f) = outputs.acting.get_mut(k) {
            *f = row.acting;
        }
        if let Some(g) = outputs.generations.get_mut(k) {
            *g = row.generation;
        }
    }
}

pub(crate) fn observe_instance<E: Environment, M: MaskProvider<E>>(
    cfg: &E::Config,
    authority: &RngAuthority,
    header: &E::Header,
    state: &E::State,
    outputs: &mut OutputsView<'_>,
    i: usize,
) {
    observe_rows::<E, M, true>(cfg, authority, header, state, outputs, i);
}

/// `observe_instance` for rows that `clear_chunk` already zeroed and that
/// nothing wrote since: the same bytes, without the per-row fills.
pub(crate) fn observe_cleared<E: Environment, M: MaskProvider<E>>(
    cfg: &E::Config,
    authority: &RngAuthority,
    header: &E::Header,
    state: &E::State,
    outputs: &mut OutputsView<'_>,
    i: usize,
) {
    observe_rows::<E, M, false>(cfg, authority, header, state, outputs, i);
}

fn observe_rows<E: Environment, M: MaskProvider<E>, const ZERO: bool>(
    cfg: &E::Config,
    authority: &RngAuthority,
    header: &E::Header,
    state: &E::State,
    outputs: &mut OutputsView<'_>,
    i: usize,
) {
    let identity = header.engine().identity;
    let a = E::MAX_AGENTS;
    let OutputsView {
        observations,
        leaf_bytes,
        masks,
        mask_len,
        ..
    } = outputs;
    for (slot, row) in header.agents().iter().enumerate() {
        let k = i * a + slot;
        let mut views: [&mut [u8]; MAX_LEAVES] = Default::default();
        let mut used = 0;
        for ((buf, &bytes), view) in observations
            .iter_mut()
            .zip(leaf_bytes.iter())
            .zip(views.iter_mut())
        {
            if let Some(rows) = buf.get_mut(k * bytes..(k + 1) * bytes) {
                if ZERO {
                    rows.fill(0);
                }
                *view = rows;
            }
            used += 1;
        }
        // Masks describe the current state: written in the same pass as
        // the observation (ADR-0014); dead agents get an all-false mask.
        let mut mask = if M::ENABLED {
            masks.get_mut(k * *mask_len..(k + 1) * *mask_len)
        } else {
            None
        };
        if let Some(m) = mask.as_deref_mut() {
            m.fill(Bool8::FALSE);
        }
        if !row.alive.get() {
            continue;
        }
        let (Some(agent), Some(leaves)) = (
            AgentSlot::new(u16::try_from(slot).unwrap_or(u16::MAX), a),
            views.get_mut(..used),
        ) else {
            continue;
        };
        let mut writer = ObsWriter::new(leaves);
        let mut rng = ObsRng::new(authority, &identity);
        E::observe(cfg, state, agent, &mut rng, &mut writer);
        if let Some(m) = mask {
            M::write_mask(cfg, state, agent, &mut MaskWriter::new(m));
        }
    }
}
