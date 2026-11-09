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
use crate::sys::{Gang, GangOptions, available_cores};
use crate::validate::{ActionInput, check_and_copy_actions, check_ids};

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
    fn seed(self, k: usize) -> Option<Seed> {
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
    sync: Option<SyncRuntime>,
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
        self.sync.take()
    }

    /// Gives the workers back (`into_pool` of a split-phase mode).
    pub(crate) fn put_runtime(&mut self, runtime: SyncRuntime) {
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

    /// The pool-wide chunk and the discrete and continuous heads per agent,
    /// for the split-phase modes to take raw parts of (ADR-0047).
    pub(crate) fn chunk_all(&mut self) -> (Chunk<'_, E>, (usize, usize)) {
        let heads = (self.actions.discrete.len(), self.actions.continuous.len());
        let chunk = Chunk {
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
    _registration: Registration,
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
            sync: None,
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
        let workers = options.workers.get();
        let available = available_cores();
        let (registration, others) = Registration::register();
        let pin = match options.cpu_set {
            Some(cores) if cores.len() < workers => {
                return Err(EngineError::Options {
                    option: "cpu_set",
                    reason: "needs one core per worker",
                });
            }
            Some(cores) => {
                // Only ids the OS reports (and, on Windows, below 64).
                let known = |id: &usize| available.as_ref().is_some_and(|a| a.contains(id));
                if !cores.iter().all(known) {
                    return Err(EngineError::Options {
                        option: "cpu_set",
                        reason: "contains a core id the OS does not report",
                    });
                }
                Some(cores)
            }
            None if others == 0 => match available {
                Some(cores) if cores.len() >= workers => {
                    Some(cores.into_iter().take(workers).collect())
                }
                _ => {
                    tracing::warn!(
                        workers,
                        "fewer reported cores than workers; workers run unpinned"
                    );
                    None
                }
            },
            None => {
                tracing::warn!(
                    others,
                    "another engine with workers is active; workers run unpinned (set cpu_set to partition cores)"
                );
                None
            }
        };
        #[cfg(not(loom))]
        let native = select_native(options.wait);
        #[cfg(loom)]
        let native: Option<()> = None;
        let gang = Gang::new(&GangOptions {
            workers: options.workers,
            spin: options.spin,
            pin,
            #[cfg(not(loom))]
            native,
            #[cfg(loom)]
            mock_futex: false,
            #[cfg(test)]
            faults,
        })?;
        self.sync = Some(SyncRuntime {
            gang,
            workers,
            native: native.is_some(),
            _registration: registration,
        });
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
            ..
        } = self;
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
            let size = chunk_size(options);
            let panicked = par_chunks(&mut rt.gang, all, size, heads, |chunk, first| {
                reset_chunk::<E, M>(cfg, authority, chunk, reseed, first);
            })?;
            if panicked {
                *poisoned = true;
                return Err(EngineError::Panicked);
            }
        } else {
            let mut chunk = all;
            run_kernels(poisoned, || {
                reset_chunk::<E, M>(cfg, authority, &mut chunk, reseed, 0);
            })?;
        }
        self.poisoned = false;
        self.check_turns()
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
        check_ids(ids, &mut self.seen)?;
        let indices = ids.iter().filter_map(|&id| usize::try_from(id).ok());
        self.check_reset(indices.clone(), ids.len(), reseed)?;
        self.apply_reset(indices, reseed)?;
        self.check_turns()
    }

    fn check_reset(
        &self,
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
                let engine = self.headers.get(i).map(RecordLayout::engine);
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

    fn apply_reset(
        &mut self,
        indices: impl Iterator<Item = usize>,
        reseed: Reseed<'_>,
    ) -> Result<(), EngineError> {
        let Self {
            cfg,
            headers,
            states,
            outputs,
            authority,
            poisoned,
            ..
        } = self;
        run_kernels(poisoned, || {
            let outputs = &mut outputs.view_all();
            for (k, i) in indices.enumerate() {
                let (Some(header), Some(state)) = (headers.get_mut(i), states.get_mut(i)) else {
                    continue;
                };
                let seed = reseed.seed(k);
                reset_one::<E>(cfg, authority, header, state, seed);
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
        check_and_copy_actions(
            &self.actions,
            self.options.auto_reset,
            &self.headers,
            actions,
            &mut self.discrete,
            &mut self.continuous,
        )?;
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
            ..
        } = self;
        let ctx = StepCtx::<E> {
            cfg,
            authority,
            layout,
            policy: options.illegal_action,
            limit: options.max_episode_steps.map(core::num::NonZeroU32::get),
            auto_reset: options.auto_reset,
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
        if let Some(rt) = sync {
            let heads = (layout.discrete.len(), layout.continuous.len());
            let size = chunk_size(options);
            let panicked = par_chunks(&mut rt.gang, all, size, heads, |chunk, _| {
                step_chunk::<E, M>(&ctx, chunk);
            })?;
            if panicked {
                *poisoned = true;
                return Err(EngineError::Panicked);
            }
        } else {
            let mut chunk = all;
            run_kernels(poisoned, || step_chunk::<E, M>(&ctx, &mut chunk))?;
        }
        self.check_turns()?;
        #[cfg(feature = "debug-checks")]
        self.check_agent_events()?;
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

/// Read-only context of a step, shared by every chunk.
pub(crate) struct StepCtx<'a, E: Environment> {
    pub(crate) cfg: &'a E::Config,
    pub(crate) authority: &'a RngAuthority,
    pub(crate) layout: &'a ActionLayout,
    pub(crate) policy: IllegalActionPolicy,
    pub(crate) limit: Option<u32>,
    pub(crate) auto_reset: AutoReset,
}

/// The records, action rows and output rows of a range of instances,
/// indexed from the first instance of the range. A chunk is the unit of
/// scheduling (SPEC 5.1): kernels only ever see one chunk.
pub(crate) struct Chunk<'a, E: Environment> {
    pub(crate) headers: &'a mut [E::Header],
    pub(crate) states: &'a mut [E::State],
    pub(crate) discrete: &'a mut [i32],
    pub(crate) continuous: &'a mut [f32],
    pub(crate) penalties: &'a mut [f32],
    pub(crate) penalized: &'a mut [Bool8],
    pub(crate) outputs: OutputsView<'a>,
}

/// Steps every instance of `chunk` (after validation).
pub(crate) fn step_chunk<E: Environment, M: MaskProvider<E>>(
    ctx: &StepCtx<'_, E>,
    chunk: &mut Chunk<'_, E>,
) {
    let StepCtx {
        cfg,
        authority,
        layout,
        policy,
        limit,
        auto_reset,
    } = *ctx;
    let Chunk {
        headers,
        states,
        discrete,
        continuous,
        penalties,
        penalized,
        outputs,
    } = chunk;
    let (dh, ch) = (layout.discrete.len(), layout.continuous.len());
    for (i, (header, state)) in headers.iter_mut().zip(states.iter_mut()).enumerate() {
        if header.engine().pending_reset.get() {
            // NextStep: reset instead of stepping (validated above).
            reset_one::<E>(cfg, authority, header, state, None);
            clear_instance(outputs, i);
            write_agent_masks(outputs, i, header);
            observe_instance::<E, M>(cfg, authority, header, state, outputs, i);
            continue;
        }
        if M::ENABLED {
            // Inside the panic catch: the author's noop may panic.
            replace_illegal_one::<E, M>(
                cfg, policy, header, state, outputs, layout, discrete, continuous, penalties,
                penalized, i,
            );
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
        observe_instance::<E, M>(cfg, authority, header, state, outputs, i);
        if auto_reset == AutoReset::SameStep && header.engine().pending_reset.get() {
            save_terminal(outputs, i);
            // Flags and stats of the ended episode stay set.
            reset_one::<E>(cfg, authority, header, state, None);
            write_agent_masks(outputs, i, header);
            observe_instance::<E, M>(cfg, authority, header, state, outputs, i);
        }
    }
}

/// Resets every instance of `chunk`; `first` is the position of the
/// chunk's first instance among the reset instances (for `Reseed`).
pub(crate) fn reset_chunk<E: Environment, M: MaskProvider<E>>(
    cfg: &E::Config,
    authority: &RngAuthority,
    chunk: &mut Chunk<'_, E>,
    reseed: Reseed<'_>,
    first: usize,
) {
    let outputs = &mut chunk.outputs;
    for (i, (header, state)) in chunk
        .headers
        .iter_mut()
        .zip(chunk.states.iter_mut())
        .enumerate()
    {
        let k = first + i;
        let seed = reseed.seed(k);
        reset_one::<E>(cfg, authority, header, state, seed);
        clear_instance(outputs, i);
        write_agent_masks(outputs, i, header);
        observe_instance::<E, M>(cfg, authority, header, state, outputs, i);
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
        let Some(first) = first_illegal(heads, &layout.mask, 0..dh, mask) else {
            continue;
        };
        if layout.mask.group_heads != 0 {
            let agent = AgentSlot::new(u16::try_from(slot).unwrap_or(u16::MAX), a);
            let (Some(row_heads), Some(agent)) = (discrete.get_mut(k * dh..(k + 1) * dh), agent)
            else {
                continue;
            };
            fallback_groups::<E, M>(cfg, state, agent, row_heads, &layout.mask, mask, first, row);
            if let IllegalActionPolicy::Penalize(r) = policy
                && let (Some(p), Some(f)) = (penalties.get_mut(k), penalized.get_mut(k))
            {
                *p = r;
                *f = Bool8::TRUE;
            }
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
#[expect(clippy::too_many_arguments, reason = "the views of one agent's row")]
fn fallback_groups<E: Environment, M: MaskProvider<E>>(
    cfg: &E::Config,
    state: &E::State,
    agent: AgentSlot,
    heads: &mut [i32],
    layout: &MaskLayout,
    mask: &[Bool8],
    first: usize,
    row: &mut AgentRow,
) {
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

/// The first head of `range` whose value the mask forbids, parents first.
/// Reads each conditional head's row through its parents' values; a head
/// whose parent is illegal is never reached. Range validation has run.
/// A row shorter than `range` reads as illegal at its first missing head.
pub(crate) fn first_illegal(
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
fn save_terminal(outputs: &mut OutputsView<'_>, i: usize) {
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

fn too_large() -> EngineError {
    EngineError::Options {
        option: "num_envs",
        reason: "buffers too large",
    }
}

fn with_step<H: RecordLayout>(header: &mut H, episode: u32, step: u32) -> RngIdentity {
    let id = header.engine().identity;
    let identity =
        RngIdentity::from_parts(id.seed(), id.branch_salt(), id.logical_id(), episode, step);
    header.engine_mut().identity = identity;
    identity
}

fn reset_one<E: Environment>(
    cfg: &E::Config,
    authority: &RngAuthority,
    header: &mut E::Header,
    state: &mut E::State,
    seed: Option<Seed>,
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
    clear_instance(outputs, i);
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
        out.episode_end()
    };
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
        debug_assert!(
            reward.is_finite(),
            "non-finite reward for agent slot {slot}"
        );
        row.episode_return += reward;
        total += row.episode_return;
        let mut died = outputs.died.get(k).is_some_and(|f| f.get());
        let spawned = outputs.spawned.get(k).is_some_and(|f| f.get());
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
struct Ending {
    terminated: bool,
    truncated: bool,
}

/// Records the end of an episode: flags, pending reset and statistics.
fn finish_episode<H: RecordLayout>(
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

fn set(buf: &mut [Bool8], i: usize, value: bool) {
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
                rows.fill(0);
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
