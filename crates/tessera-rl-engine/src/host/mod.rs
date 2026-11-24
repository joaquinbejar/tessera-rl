// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host pool (ADR-0062): a pool of [`HostEnvironment`] instances, which
//! own their state.
//!
//! It shares the engine's output buffers, validation boundary, auto-reset
//! modes, poisoning, Sync-mode workers and panic boundary with `EnvPool`.
//! It has no snapshot, restore, fork, state hash, persistence or search:
//! those methods do not exist on the type. It runs in Inline mode or Sync
//! mode with workers; the split-phase modes are not supported.
//!
//! The engine side allocates nothing per step and uses no dynamic dispatch
//! (`HostPool<E, A>` is monomorphised). The environment's own code may
//! allocate, lock or block: under `debug-checks` its allocations are
//! counted as a statistic, never as `AllocationInChunkStep`, and they never
//! reach an `EnvPool`'s count. After each host chunk the engine puts the
//! thread back under canonical floating-point controls and counts the
//! times it had to (ADR-0062 item 4).
//!
//! Action masks are outputs only: the environment's `mask` describes the
//! next step's legal actions, but a host pool does not replace illegal
//! actions (it has no no-op); actions are validated for range only, and
//! then by the environment's own `validate` when it declares `VALIDATES`
//! (ADR-0065 item 2): a rejected call steps no instance and leaves the
//! outputs as they were.
//!
//! A host `reset` or `step` may abort its instance's episode
//! (`HostOut::abort`, ADR-0065 item 1) instead of panicking: the episode
//! ends with every live agent truncated, the instance is left pending a
//! reset, and the host-only outputs `aborted`, `abort_code` and
//! `pending_reset` report it. `pending_reset` is the exact pending state
//! of every instance, from construction on; `aborted` alone never means
//! pending. The pool is not poisoned, and [`HostPool::aborts`] counts the
//! aborts applied.
//!
//! Make-by-id (ADR-0065 item 3): code that knows its environment's type
//! builds `HostPool<E, A>` directly. Code that has only an id calls
//! [`make_host`] with a `ConfigSource`, and gets a [`DynHostPool`]: the same
//! pool behind a trait object, at one virtual call per method call. An
//! environment's factory is registered with `register_host_factory!` in a
//! crate that depends on the engine; the environment's own crate depends on
//! core only (ADR-0062).
//!
//! Experimental until 1.0 (ADR-0062 item 9).

#[cfg(test)]
mod abort_header_tests;
mod barrier;
mod dyn_pool;
mod dyn_snapshot;
mod factory;
mod kernels;
mod layout;
mod persist;
mod record;
mod snapshot;
#[cfg(feature = "testing")]
pub mod testing;

pub use dyn_pool::DynHostPool;
pub use factory::{FactoryError, HostFactory, MakeOptions, make_host};
pub use layout::{HostLayout, HostLeaf};
pub use persist::HOST_MAGIC;

pub use record::{HostRecord, RecordDecodeError};

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use tessera_rl_core::envs::HostRegistration;
use tessera_rl_core::{
    ActionReject, Bool8, EnvConfig, HostEnvironment, HostError, RecordLayout, StateInit,
    TurnStructure,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity};

use crate::buffer::par_chunks;
use crate::error::EngineError;
use crate::layout::{ActionLayout, MaskLayout, obs_leaves};
use crate::options::{AutoReset, PoolOptions, Schedule, SyncOptions};
use crate::outputs::{Outputs, OutputsView};
use crate::pool::{Reseed, Rows, SyncRuntime, check_reset, chunk_size, run_kernels, too_large};
use crate::validate::{ActionInput, check_and_copy_actions, check_ids};

use kernels::{HostCtx, HostHeader};

/// Most action heads (discrete and continuous) per agent a host pool
/// accepts. `Space::heads` returns one head per element (a `MultiBinary`
/// or box space gives one per element); the pool refuses more at
/// construction, before any buffer is sized from them.
pub const MAX_HOST_HEADS: usize = 4096;

/// Most agent slots a host environment may declare. A `SameStep`
/// auto-reset hands `reset` scratch rows on the stack, five per agent
/// (`f32` and four `Bool8`), so this bounds them to 2 KiB.
///
/// At the bound a pool builds:
///
/// ```
/// const AGENTS: usize = 256;
/// use tessera_rl_core::io::{Actions, ObsWriter};
/// use tessera_rl_core::space::Space;
/// use tessera_rl_core::{AgentSlot, EnvConfig, HostEnvironment, HostError, HostOut, HostSeed, SpaceError};
/// use tessera_rl_engine::{HostPool, PoolOptions};
/// use tessera_rl_num::identity::Seed;
///
/// struct Cfg;
/// impl EnvConfig for Cfg {}
/// struct Many;
/// impl HostEnvironment for Many {
///     const ENV_ID: &'static str = "doc/Many-v1";
///     const ENV_VERSION: u32 = 1;
///     const MAX_AGENTS: usize = AGENTS;
///     type Config = Cfg;
///     fn new(_: &Cfg) -> Result<Self, HostError> { Ok(Self) }
///     fn observation_space(_: &Cfg) -> Result<Space, SpaceError> { Ok(Space::Discrete { n: 2 }) }
///     fn action_space(_: &Cfg) -> Result<Space, SpaceError> { Ok(Space::Discrete { n: 2 }) }
///     fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {}
///     fn step(&mut self, _: &Actions<'_, ()>, _: &mut HostOut<'_>) {}
///     fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}
/// }
/// let options = PoolOptions::new(core::num::NonZeroU32::MIN, Seed::new(0));
/// let _ = HostPool::<Many, AGENTS>::new(Cfg, options);
/// ```
///
/// One past it does not compile:
///
/// ```compile_fail
/// const AGENTS: usize = 257;
/// use tessera_rl_core::io::{Actions, ObsWriter};
/// use tessera_rl_core::space::Space;
/// use tessera_rl_core::{AgentSlot, EnvConfig, HostEnvironment, HostError, HostOut, HostSeed, SpaceError};
/// use tessera_rl_engine::{HostPool, PoolOptions};
/// use tessera_rl_num::identity::Seed;
///
/// struct Cfg;
/// impl EnvConfig for Cfg {}
/// struct Many;
/// impl HostEnvironment for Many {
///     const ENV_ID: &'static str = "doc/Many-v1";
///     const ENV_VERSION: u32 = 1;
///     const MAX_AGENTS: usize = AGENTS;
///     type Config = Cfg;
///     fn new(_: &Cfg) -> Result<Self, HostError> { Ok(Self) }
///     fn observation_space(_: &Cfg) -> Result<Space, SpaceError> { Ok(Space::Discrete { n: 2 }) }
///     fn action_space(_: &Cfg) -> Result<Space, SpaceError> { Ok(Space::Discrete { n: 2 }) }
///     fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {}
///     fn step(&mut self, _: &Actions<'_, ()>, _: &mut HostOut<'_>) {}
///     fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}
/// }
/// let options = PoolOptions::new(core::num::NonZeroU32::MIN, Seed::new(0));
/// let _ = HostPool::<Many, AGENTS>::new(Cfg, options);
/// ```
pub const MAX_HOST_AGENTS: usize = 256;

/// The largest `HostSnapshot::max_snapshot_len` a host pool accepts, 1 GiB
/// (ADR-0073): a larger bound is refused with
/// [`EngineError::SnapshotBound`] before anything is reserved for it.
pub const MAX_HOST_SNAPSHOT_BYTES: usize = 1 << 30;

/// A pool's snapshot parameters, computed once (ADR-0073).
#[derive(Debug, Clone, Copy)]
struct SnapshotParams {
    identity: u64,
    max: usize,
}

/// A pool of `num_envs` instances of the host environment `E` (ADR-0062).
///
/// `A` must equal `E::MAX_AGENTS` and be at most [`MAX_HOST_AGENTS`]
/// (checked at compile time, when the pool is built); with a concrete
/// type, write `HostPool<Book, { Book::MAX_AGENTS }>`.
///
/// `E::NOTIFIES_REJECTION` needs `E::VALIDATES` (ADR-0069), also checked at
/// compile time when the pool is built. With both it builds:
///
/// ```
/// use tessera_rl_core::io::{Actions, ObsWriter};
/// use tessera_rl_core::space::Space;
/// use tessera_rl_core::{AgentSlot, EnvConfig, HostEnvironment, HostError, HostOut, HostSeed, SpaceError};
/// use tessera_rl_engine::{HostPool, PoolOptions};
/// use tessera_rl_num::identity::Seed;
///
/// struct Cfg;
/// impl EnvConfig for Cfg {}
/// struct Notified;
/// impl HostEnvironment for Notified {
///     const ENV_ID: &'static str = "doc/Notified-v1";
///     const ENV_VERSION: u32 = 1;
///     const MAX_AGENTS: usize = 1;
///     const VALIDATES: bool = true;
///     const NOTIFIES_REJECTION: bool = true;
///     type Config = Cfg;
///     fn new(_: &Cfg) -> Result<Self, HostError> { Ok(Self) }
///     fn observation_space(_: &Cfg) -> Result<Space, SpaceError> { Ok(Space::Discrete { n: 2 }) }
///     fn action_space(_: &Cfg) -> Result<Space, SpaceError> { Ok(Space::Discrete { n: 2 }) }
///     fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {}
///     fn step(&mut self, _: &Actions<'_, ()>, _: &mut HostOut<'_>) {}
///     fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}
/// }
/// let options = PoolOptions::new(core::num::NonZeroU32::MIN, Seed::new(0));
/// let _ = HostPool::<Notified, 1>::new(Cfg, options);
/// ```
///
/// With the notice and without `VALIDATES` it does not compile:
///
/// ```compile_fail,E0080
/// use tessera_rl_core::io::{Actions, ObsWriter};
/// use tessera_rl_core::space::Space;
/// use tessera_rl_core::{AgentSlot, EnvConfig, HostEnvironment, HostError, HostOut, HostSeed, SpaceError};
/// use tessera_rl_engine::{HostPool, PoolOptions};
/// use tessera_rl_num::identity::Seed;
///
/// struct Cfg;
/// impl EnvConfig for Cfg {}
/// struct Notified;
/// impl HostEnvironment for Notified {
///     const ENV_ID: &'static str = "doc/Notified-v1";
///     const ENV_VERSION: u32 = 1;
///     const MAX_AGENTS: usize = 1;
///     const NOTIFIES_REJECTION: bool = true;
///     type Config = Cfg;
///     fn new(_: &Cfg) -> Result<Self, HostError> { Ok(Self) }
///     fn observation_space(_: &Cfg) -> Result<Space, SpaceError> { Ok(Space::Discrete { n: 2 }) }
///     fn action_space(_: &Cfg) -> Result<Space, SpaceError> { Ok(Space::Discrete { n: 2 }) }
///     fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {}
///     fn step(&mut self, _: &Actions<'_, ()>, _: &mut HostOut<'_>) {}
///     fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}
/// }
/// let options = PoolOptions::new(core::num::NonZeroU32::MIN, Seed::new(0));
/// let _ = HostPool::<Notified, 1>::new(Cfg, options);
/// ```
#[derive(Debug)]
pub struct HostPool<E: HostEnvironment, const A: usize> {
    /// The configuration the pool was built with (ADR-0073 needs it after
    /// construction: `check_snapshot`, `max_snapshot_len`, the identity).
    cfg: HeldConfig<E::Config>,
    options: PoolOptions,
    /// `E::initial_agents(cfg)`, checked at construction.
    initial: usize,
    /// The fixed layout, for bindings (SPEC 14.4.3 draft).
    host_layout: HostLayout,
    headers: Vec<HostHeader<A>>,
    instances: Vec<E>,
    layout: ActionLayout,
    discrete: Vec<i32>,
    continuous: Vec<f32>,
    outputs: Outputs,
    seen: Vec<Bool8>,
    /// Episode statistics with abort counts (ADR-0081 item 8).
    episodes: crate::adaptive::EpisodeStatistics,
    poisoned: bool,
    sync: Option<SyncRuntime>,
    fp_resets: AtomicU64,
    /// Aborts applied (ADR-0065 item 1).
    aborts: AtomicU64,
    /// `aborts` when the `aborted` and `abort_code` outputs were last all
    /// clear: they need clearing before a call only if it moved since.
    abort_rows_clear: u64,
    /// Per-agent refusal bookkeeping (ADR-0070).
    refusal: Refusal,
    /// The record `fork` saves its source into, reused (ADR-0073 item 4).
    snapshot_scratch: record::HostRecord<A>,
    /// The snapshot functions for `DynHostPool`, when built with
    /// `with_dyn_snapshots` (ADR-0073).
    dyn_snapshots: Option<dyn_snapshot::DynSnapshotOps<E, A>>,
    /// `max_snapshot_len` and the identity digest, computed on first use
    /// (ADR-0073).
    snapshot_params: Option<SnapshotParams>,
    #[cfg(feature = "debug-checks")]
    allocs: crate::sys::alloc_track::AllocTally,
    #[cfg(feature = "debug-checks")]
    host_allocations: u64,
}

impl<E: HostEnvironment, const A: usize> HostPool<E, A> {
    const AGENTS_MATCH: () = {
        assert!(
            A == E::MAX_AGENTS,
            "HostPool<E, A>: A must equal HostEnvironment::MAX_AGENTS"
        );
        assert!(
            A <= MAX_HOST_AGENTS,
            "HostPool<E, A>: A must be at most MAX_HOST_AGENTS"
        );
        assert!(
            !E::NOTIFIES_REJECTION || E::VALIDATES,
            "HostEnvironment::NOTIFIES_REJECTION needs VALIDATES (ADR-0069)"
        );
    };

    /// Builds a pool: validates the config, the options and the spaces,
    /// builds `num_envs` instances with `HostEnvironment::new` (once each,
    /// in index order, on this thread, after the config and space checks;
    /// see that method) and lays out the
    /// buffers. Instances need a reset before stepping. Runs under
    /// canonical floating-point controls; a panic in `new` is
    /// [`EngineError::Panicked`].
    ///
    /// # Errors
    ///
    /// An [`EngineError`] for an invalid config, options or space, more
    /// than [`MAX_HOST_HEADS`] action heads per agent, or an environment
    /// that refuses to start.
    pub fn new(cfg: E::Config, options: PoolOptions) -> Result<Self, EngineError> {
        let () = Self::AGENTS_MATCH;
        let mut out = None;
        if crate::sys::catch(|| out = Some(crate::entry::canonical(|| Self::build(cfg, options)))) {
            out.unwrap_or(Err(EngineError::Panicked))
        } else {
            Err(EngineError::Panicked)
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one linear construction; the snapshot config field took it past the limit"
    )]
    fn build(owned: E::Config, options: PoolOptions) -> Result<Self, EngineError> {
        let cfg = &owned;
        if A == 0 {
            return Err(EngineError::Options {
                option: "MAX_AGENTS",
                reason: "a host environment needs at least one agent slot",
            });
        }
        cfg.validate()?;
        let initial = E::initial_agents(cfg);
        if !(1..=A).contains(&initial) {
            return Err(tessera_rl_core::ConfigError {
                field: "initial_agents",
                reason: "must be in 1..=MAX_AGENTS",
            }
            .into());
        }
        options.validate()?;
        let obs_space = E::observation_space(cfg)?;
        obs_space.validate()?;
        let action_space = E::action_space(cfg)?;
        action_space.validate()?;
        let mut layout = ActionLayout::from_space(&action_space)?;
        if layout
            .discrete
            .len()
            .saturating_add(layout.continuous.len())
            > MAX_HOST_HEADS
        {
            return Err(EngineError::UnsupportedSpace {
                what: "more than MAX_HOST_HEADS action heads per agent",
            });
        }
        let leaves = obs_leaves(&obs_space)?;
        let num_envs = usize::try_from(options.num_envs.get()).map_err(|_| too_large())?;
        let rows = num_envs.checked_mul(A).ok_or_else(too_large)?;
        // Without `HostEnvironment::MASKS` there is no mask buffer (SPEC 14.4).
        let mask_len = if E::MASKS {
            mask_len::<E>(cfg, &mut layout, rows, options.mask_budget_bytes)?
        } else {
            0
        };
        let host_layout = layout::host_layout::<E>(
            (num_envs, A, initial),
            obs_space,
            action_space,
            &layout,
            mask_len,
            options.auto_reset == AutoReset::SameStep,
            if E::MASKS {
                E::entity_group_heads(cfg).map_or(0, |g| usize::from(g.get()))
            } else {
                0
            },
        )?;
        let mut outputs = Outputs::new_host(
            num_envs,
            A,
            &leaves.leaves,
            options.auto_reset == AutoReset::SameStep,
            mask_len,
            E::REFUSES_AGENTS,
        )?;
        // Every instance is pending its first reset, and `outputs()` is
        // readable before any call (ADR-0065 item 1).
        outputs.view_all().pending_reset.fill(Bool8::TRUE);
        let headers = (0..num_envs)
            .map(|i| {
                let mut header = HostHeader::<A>::state_default();
                // `i < num_envs <= u32::MAX`, so the conversion never fails.
                let seed = options.seed.offset(u64::try_from(i).unwrap_or(u64::MAX));
                let engine = header.engine_mut();
                engine.identity = RngIdentity::new(seed, LogicalId::DEFAULT);
                engine.fresh = Bool8::TRUE;
                engine.pending_reset = Bool8::TRUE;
                header
            })
            .collect();
        let instances = (0..num_envs)
            .map(|_| E::new(cfg).map_err(host_error))
            .collect::<Result<Vec<E>, EngineError>>()?;
        Ok(Self {
            cfg: HeldConfig(owned),
            options,
            initial,
            host_layout,
            headers,
            instances,
            discrete: vec![
                0;
                rows.checked_mul(layout.discrete.len())
                    .ok_or_else(too_large)?
            ],
            continuous: vec![
                0.0;
                rows.checked_mul(layout.continuous.len())
                    .ok_or_else(too_large)?
            ],
            layout,
            outputs,
            seen: vec![Bool8::FALSE; num_envs],
            poisoned: false,
            sync: None,
            fp_resets: AtomicU64::new(0),
            aborts: AtomicU64::new(0),
            abort_rows_clear: 0,
            refusal: Refusal::new(),
            snapshot_scratch: record::HostRecord::new(),
            dyn_snapshots: None,
            snapshot_params: None,
            #[cfg(feature = "debug-checks")]
            allocs: crate::sys::alloc_track::AllocTally::default(),
            #[cfg(feature = "debug-checks")]
            host_allocations: 0,
            episodes: crate::adaptive::EpisodeStatistics::new(num_envs, true),
        })
    }

    /// Switches the pool to Sync mode: `step` and `reset_all` run chunks on
    /// worker threads and return when every instance is done.
    ///
    /// Pinning (ADR-0062 item 2): the workers pin only to an explicit
    /// `cpu_set`, and only then does the pool register as an active engine
    /// (ADR-0005). Without a `cpu_set` they run unpinned and the pool does
    /// not register, so an `EnvPool` started after it still pins as usual,
    /// and may pin to cores these unpinned workers also run on.
    ///
    /// # Errors
    ///
    /// [`EngineError::NestedEngine`] from an engine worker, or
    /// [`EngineError::Options`] for an invalid `cpu_set`.
    pub fn with_workers(mut self, options: SyncOptions) -> Result<Self, EngineError> {
        self.sync = Some(SyncRuntime::start(
            options,
            crate::pool::Pinning::ExplicitOnly,
            #[cfg(test)]
            crate::sys::Faults::default(),
        )?);
        Ok(self)
    }

    /// The environment's registration record, `HostRegistration::of::<E>()`
    /// (ADR-0065 item 3): its id, version, agent count, turns and
    /// information.
    #[must_use]
    pub const fn registration(&self) -> &'static HostRegistration {
        // An inline const per monomorphisation: promoted to `'static`.
        const { &HostRegistration::of::<E>() }
    }

    /// Number of worker threads; 0 in Inline mode.
    #[must_use]
    pub fn workers(&self) -> usize {
        self.sync.as_ref().map_or(0, |s| s.workers)
    }

    /// Whether the workers were pinned: only with an explicit `cpu_set`.
    #[must_use]
    pub fn pinned_workers(&self) -> bool {
        self.sync.as_ref().is_some_and(|s| s.pinned)
    }

    /// The pool options.
    #[must_use]
    pub const fn options(&self) -> &PoolOptions {
        &self.options
    }

    /// The configuration the pool was built with: for replay files
    /// (ADR-0072), the canonical config bytes and hash of a snapshot host;
    /// for host files, the asset check (ADR-0077).
    pub(crate) fn config(&self) -> &E::Config {
        &self.cfg
    }

    /// The pool's fixed layout: spaces, agents, masks (SPEC 14.4.3 draft).
    #[must_use]
    pub const fn layout(&self) -> &HostLayout {
        &self.host_layout
    }

    /// Number of discrete and continuous action heads per agent.
    #[must_use]
    pub fn action_heads(&self) -> (usize, usize) {
        (self.layout.discrete.len(), self.layout.continuous.len())
    }

    /// Whether a panic in environment code poisoned the pool (ADR-0028).
    #[must_use]
    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Times a host chunk returned with the thread's floating-point
    /// controls changed, and the engine put them back (ADR-0062 item 4).
    #[must_use]
    pub fn host_fp_resets(&self) -> u64 {
        self.fp_resets.load(Ordering::Relaxed)
    }

    /// Aborts the engine applied since construction (ADR-0065 item 1): one
    /// per aborting `step` and one per aborting `reset`, so one instance
    /// adds at most 2 in one call (an aborted step and its aborted
    /// same-call reset).
    #[must_use]
    pub fn aborts(&self) -> u64 {
        self.aborts.load(Ordering::Relaxed)
    }

    /// `debug-checks`: allocations counted in the host code of this pool's
    /// chunks since construction, with the engine's `TrackingAllocator`
    /// installed. A statistic: it never fails a call (ADR-0062 item 4).
    #[cfg(feature = "debug-checks")]
    #[must_use]
    pub const fn allocations_in_host_steps(&self) -> u64 {
        self.host_allocations
    }

    /// The outputs of the last call.
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`] on a poisoned pool.
    pub const fn outputs(&self) -> Result<&Outputs, EngineError> {
        if self.poisoned {
            return Err(EngineError::Poisoned);
        }
        Ok(&self.outputs)
    }

    /// Resets every instance. On a poisoned pool this is the recovery
    /// path: the poison is cleared only if every instance resets without
    /// panicking.
    ///
    /// # Errors
    ///
    /// A seed list of the wrong length or an exhausted episode counter
    /// (nothing is mutated then), or [`EngineError::Panicked`].
    pub fn reset_all(&mut self, reseed: Reseed<'_>) -> Result<&Outputs, EngineError> {
        crate::entry::canonical(|| self.reset_all_inner(reseed))?;
        Ok(&self.outputs)
    }

    fn reset_all_inner(&mut self, reseed: Reseed<'_>) -> Result<(), EngineError> {
        let n = self.headers.len();
        check_reset(&self.headers, 0..n, n, reseed)?;
        let grain = (chunk_size(&self.options), self.options.schedule);
        let heads = self.action_heads();
        let Self {
            headers,
            instances,
            discrete,
            continuous,
            outputs,
            poisoned,
            sync,
            layout,
            options,
            initial,
            fp_resets,
            aborts,
            abort_rows_clear,
            episodes,
            refusal:
                Refusal {
                    count: refusals,
                    clear: refusal_rows_clear,
                    misuse,
                },
            #[cfg(feature = "debug-checks")]
            allocs,
            ..
        } = self;
        let ctx = HostCtx {
            layout,
            limit: options.max_episode_steps.map(core::num::NonZeroU32::get),
            auto_reset: options.auto_reset,
            initial: *initial,
            fp_resets,
            aborts,
            refusals,
            misuse,
            #[cfg(feature = "debug-checks")]
            allocs,
        };
        clear_abort_rows(outputs, ctx.aborts, abort_rows_clear);
        clear_refusal_rows::<E>(outputs, ctx.refusals, refusal_rows_clear);
        let mut all = Rows {
            headers,
            states: instances,
            columns: (),
            discrete,
            continuous,
            penalties: &mut [],
            penalized: &mut [],
            outputs: outputs.view_all(),
        };
        (all.outputs.episode_partials, all.outputs.episode_aborts) = episodes.rows();
        if let Some(rt) = sync {
            let panicked = par_chunks(&mut rt.gang, all, grain, heads, |rows, first| {
                kernels::reset_chunk::<E, A>(&ctx, rows, reseed, first);
            })?;
            if panicked {
                *poisoned = true;
                return Err(EngineError::Panicked);
            }
        } else {
            // Borrowed, not moved: a move copies the whole view.
            run_kernels(poisoned, || {
                kernels::reset_chunk::<E, A>(&ctx, &mut all, reseed, 0);
            })?;
        }
        self.poisoned = false;
        self.after_kernels()?;
        #[cfg(feature = "debug-checks")]
        self.check_reward_nan(0..n)?;
        Ok(())
    }

    /// Resets the instances `ids`, in the order given; other instances keep
    /// their state and outputs.
    ///
    /// # Errors
    ///
    /// Unknown or duplicate ids, a seed list of the wrong length or an
    /// exhausted episode counter (nothing is mutated then);
    /// [`EngineError::Poisoned`] on a poisoned pool; or
    /// [`EngineError::Panicked`].
    pub fn reset(&mut self, ids: &[u32], reseed: Reseed<'_>) -> Result<&Outputs, EngineError> {
        if self.poisoned {
            return Err(EngineError::Poisoned);
        }
        crate::entry::canonical(|| self.reset_inner(ids, reseed))?;
        Ok(&self.outputs)
    }

    fn reset_inner(&mut self, ids: &[u32], reseed: Reseed<'_>) -> Result<(), EngineError> {
        check_ids(ids, &mut self.seen)?;
        let indices = ids.iter().filter_map(|&id| usize::try_from(id).ok());
        check_reset(&self.headers, indices.clone(), ids.len(), reseed)?;
        let Self {
            headers,
            instances,
            outputs,
            poisoned,
            layout,
            options,
            initial,
            fp_resets,
            aborts,
            refusal:
                Refusal {
                    count: refusals,
                    clear: refusal_rows_clear,
                    misuse,
                },
            #[cfg(feature = "debug-checks")]
            allocs,
            ..
        } = self;
        let ctx = HostCtx {
            layout,
            limit: options.max_episode_steps.map(core::num::NonZeroU32::get),
            auto_reset: options.auto_reset,
            initial: *initial,
            fp_resets,
            aborts,
            refusals,
            misuse,
            #[cfg(feature = "debug-checks")]
            allocs,
        };
        clear_refusal_rows::<E>(outputs, ctx.refusals, refusal_rows_clear);
        run_kernels(poisoned, || {
            #[cfg(feature = "debug-checks")]
            let _bracket = crate::sys::alloc_track::HostBracket::enter(ctx.allocs);
            let outputs = &mut outputs.view_all();
            for (k, i) in indices.clone().enumerate() {
                let (Some(header), Some(inst)) = (headers.get_mut(i), instances.get_mut(i)) else {
                    continue;
                };
                crate::pool::clear_instance(outputs, i);
                kernels::clear_abort_row(outputs, i);
                let aborted = kernels::reset_one::<E, A>(
                    &ctx,
                    header,
                    inst,
                    reseed.seed(k),
                    outputs,
                    i,
                    kernels::ResetRows::Live,
                );
                crate::pool::write_agent_masks(outputs, i, header);
                if !aborted {
                    kernels::observe_instance::<E, A>(header, inst, outputs, i);
                }
                kernels::write_pending(outputs, i, header);
            }
        })?;
        self.after_kernels()?;
        #[cfg(feature = "debug-checks")]
        self.check_reward_nan(indices)?;
        Ok(())
    }

    /// Steps every instance with `actions`.
    ///
    /// # Errors
    ///
    /// Actions of the wrong shape or out of range, an instance pending
    /// reset or an exhausted step counter (nothing is mutated then);
    /// [`EngineError::ActionRejected`] when the environment's `validate`
    /// refuses an instance's joint action (`VALIDATES`, ADR-0065; nothing
    /// is stepped then); [`EngineError::Poisoned`] on a poisoned pool; or
    /// [`EngineError::Panicked`] if environment code panics, which poisons
    /// the pool.
    pub fn step(&mut self, actions: ActionInput<'_>) -> Result<&Outputs, EngineError> {
        if self.poisoned {
            return Err(EngineError::Poisoned);
        }
        crate::entry::canonical(|| self.step_inner(actions))?;
        Ok(&self.outputs)
    }

    fn step_inner(&mut self, actions: ActionInput<'_>) -> Result<(), EngineError> {
        check_and_copy_actions(
            &self.layout,
            self.options.auto_reset,
            &self.headers,
            actions,
            &mut self.discrete,
            &mut self.continuous,
        )?;
        let grain = (chunk_size(&self.options), self.options.schedule);
        let heads = self.action_heads();
        let Self {
            headers,
            instances,
            discrete,
            continuous,
            outputs,
            poisoned,
            sync,
            layout,
            options,
            initial,
            fp_resets,
            aborts,
            abort_rows_clear,
            episodes,
            refusal:
                Refusal {
                    count: refusals,
                    clear: refusal_rows_clear,
                    misuse,
                },
            #[cfg(feature = "debug-checks")]
            allocs,
            #[cfg(feature = "debug-checks")]
            host_allocations,
            ..
        } = self;
        let ctx = HostCtx {
            layout,
            limit: options.max_episode_steps.map(core::num::NonZeroU32::get),
            auto_reset: options.auto_reset,
            initial: *initial,
            fp_resets,
            aborts,
            refusals,
            misuse,
            #[cfg(feature = "debug-checks")]
            allocs,
        };
        if E::VALIDATES {
            let validated = validate::<E, A>(
                &ctx,
                sync.as_mut(),
                poisoned,
                headers,
                instances,
                discrete,
                continuous,
                grain,
            );
            // A rejected call still charges what `validate` allocated to
            // the host tally, now (ADR-0065 item 2).
            #[cfg(feature = "debug-checks")]
            if validated.is_err() {
                *host_allocations = host_allocations.saturating_add(ctx.allocs.take());
            }
            validated?;
        }
        clear_abort_rows(outputs, ctx.aborts, abort_rows_clear);
        clear_refusal_rows::<E>(outputs, ctx.refusals, refusal_rows_clear);
        let mut all = Rows {
            headers,
            states: instances,
            columns: (),
            discrete,
            continuous,
            penalties: &mut [],
            penalized: &mut [],
            outputs: outputs.view_all(),
        };
        (all.outputs.episode_partials, all.outputs.episode_aborts) = episodes.rows();
        ctx.misuse.store(u64::MAX, Ordering::Relaxed);
        if let Some(rt) = sync {
            let panicked = par_chunks(&mut rt.gang, all, grain, heads, |rows, first| {
                kernels::step_chunk::<E, A>(&ctx, rows, first);
            })?;
            if panicked {
                *poisoned = true;
                return Err(EngineError::Panicked);
            }
        } else {
            // Borrowed, not moved: a move copies the whole view.
            run_kernels(poisoned, || kernels::step_chunk::<E, A>(&ctx, &mut all, 0))?;
        }
        self.check_refusal_misuse()?;
        self.after_kernels()?;
        #[cfg(feature = "debug-checks")]
        self.check_reward_nan(0..self.headers.len())?;
        Ok(())
    }

    /// The checks after the kernels of a successful call: the turn order of
    /// a `Sequential` environment (every build), and under `debug-checks`
    /// invalid agent events. Host allocations are added to the statistic,
    /// never turned into an error. NaN rewards are checked by the caller,
    /// for the instances it ran.
    fn after_kernels(&mut self) -> Result<(), EngineError> {
        #[cfg(feature = "debug-checks")]
        {
            self.host_allocations = self.host_allocations.saturating_add(self.allocs.take());
        }
        self.check_turns()?;
        #[cfg(feature = "debug-checks")]
        self.check_agent_events()?;
        Ok(())
    }

    /// A refusal of a slot that was not acting poisons the pool, before any
    /// other check of the call (ADR-0070 item 2).
    fn check_refusal_misuse(&mut self) -> Result<(), EngineError> {
        if !E::REFUSES_AGENTS {
            return Ok(());
        }
        let key = self.refusal.misuse.load(Ordering::Relaxed);
        if key == u64::MAX {
            return Ok(());
        }
        self.poisoned = true;
        Err(EngineError::RefusedNotActing {
            instance: u32::try_from(key >> 32).unwrap_or(u32::MAX),
            agent: u32::try_from(key & u64::from(u32::MAX)).unwrap_or(u32::MAX),
        })
    }

    /// As `EnvPool`: a `Sequential` environment names exactly one acting
    /// agent in every instance whose episode goes on; a violation poisons
    /// the pool (ADR-0013, ADR-0028).
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

    #[cfg(feature = "debug-checks")]
    fn check_agent_events(&mut self) -> Result<(), EngineError> {
        if let Some(k) = self
            .outputs
            .invalid_agent_event()
            .iter()
            .position(|f| f.get())
        {
            self.poisoned = true;
            return Err(EngineError::InvalidAgentEvent {
                instance: u32::try_from(k / A).unwrap_or(u32::MAX),
                agent: u32::try_from(k % A).unwrap_or(u32::MAX),
            });
        }
        Ok(())
    }

    /// `debug-checks`: the first NaN reward of `indices` poisons the pool.
    /// There is no state the engine can read (ADR-0062 item 4).
    #[cfg(feature = "debug-checks")]
    fn check_reward_nan(
        &mut self,
        indices: impl Iterator<Item = usize>,
    ) -> Result<(), EngineError> {
        let rewards = self.outputs.rewards();
        for i in indices {
            let row = rewards.get(i * A..(i + 1) * A).unwrap_or(&[]);
            if let Some(e) = crate::nan_check::reward_nan(u32::try_from(i).unwrap_or(u32::MAX), row)
            {
                self.poisoned = true;
                return Err(e);
            }
        }
        Ok(())
    }
}

/// Clears the `aborted` and `abort_code` outputs of every instance before
/// a call that runs them all, unless they are already clear: no abort was
/// applied since `clear` was recorded (ADR-0065 item 1). Skipping the fill
/// keeps a call without aborts as cheap as before the abort outputs.
#[inline]
fn clear_abort_rows(outputs: &mut Outputs, aborts: &AtomicU64, clear: &mut u64) {
    let now = aborts.load(Ordering::Relaxed);
    if now != *clear {
        let view = outputs.view_all();
        view.aborted.fill(Bool8::FALSE);
        view.abort_code.fill(0);
        *clear = now;
    }
}

/// A host pool's per-agent refusal bookkeeping (ADR-0070).
#[derive(Debug)]
struct Refusal {
    /// Instances whose `refuse` refused an agent.
    count: AtomicU64,
    /// `count` when the `refused` and `refuse_code` outputs were last all
    /// clear: they need clearing before a call only if it moved since.
    clear: u64,
    /// The lowest misuse key of the running call (item 2), or `u64::MAX`.
    misuse: AtomicU64,
}

impl Refusal {
    const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            clear: 0,
            misuse: AtomicU64::new(u64::MAX),
        }
    }
}

/// Clears the `refused` and `refuse_code` outputs of every instance before
/// any call that is not rejected, unless no refusal was written since
/// `clear` was recorded (ADR-0070 item 6). A no-op for a pool that does
/// not refuse: its buffers are empty.
#[inline]
fn clear_refusal_rows<E: HostEnvironment>(
    outputs: &mut Outputs,
    refusals: &AtomicU64,
    clear: &mut u64,
) {
    if !E::REFUSES_AGENTS {
        return;
    }
    let now = refusals.load(Ordering::Relaxed);
    if now != *clear {
        let view = outputs.view_all();
        view.refused.fill(Bool8::FALSE);
        view.refuse_code.fill(0);
        *clear = now;
    }
}

/// The joint-action validation pass (ADR-0065 item 2), before any instance
/// steps; it writes no output.
///
/// Inline validates serially and stops at the first rejection. Sync runs a
/// second dispatch on the existing gang, before the step's: each chunk
/// stops at its own first rejection and records its index with
/// `fetch_min`, so the lowest rejected index wins, as in Inline. Its code
/// and reason are then read by validating that one instance again on this
/// thread (`validate` is pure by contract), so nothing is allocated or
/// locked. A panic poisons the pool.
#[expect(
    clippy::too_many_arguments,
    reason = "the split borrows of the pool's fields"
)]
fn validate<E: HostEnvironment, const A: usize>(
    ctx: &HostCtx<'_>,
    sync: Option<&mut SyncRuntime>,
    poisoned: &mut bool,
    headers: &mut [HostHeader<A>],
    instances: &mut [E],
    discrete: &mut [i32],
    continuous: &mut [f32],
    grain: (usize, Schedule),
) -> Result<(), EngineError> {
    let heads = (ctx.layout.discrete.len(), ctx.layout.continuous.len());
    let mut found = None;
    if let Some(rt) = sync {
        let lowest = AtomicU32::new(u32::MAX);
        let all = Rows {
            headers: &mut *headers,
            states: &mut *instances,
            columns: (),
            discrete: &mut *discrete,
            continuous: &mut *continuous,
            penalties: &mut [],
            penalized: &mut [],
            outputs: OutputsView::empty(A),
        };
        let panicked = par_chunks(&mut rt.gang, all, grain, heads, |rows, first| {
            if let Some((i, _)) = kernels::validate_chunk::<E, A>(ctx, rows) {
                let index = u32::try_from(first.saturating_add(i)).unwrap_or(u32::MAX);
                lowest.fetch_min(index, Ordering::Relaxed);
            }
        })?;
        if panicked {
            *poisoned = true;
            return Err(EngineError::Panicked);
        }
        // The gang's join orders every worker's `fetch_min` before this
        // read.
        let lowest = lowest.load(Ordering::Relaxed);
        if lowest == u32::MAX {
            return Ok(());
        }
        let index = usize::try_from(lowest).unwrap_or(usize::MAX);
        let (dh, ch) = heads;
        let (Some(one_header), Some(one_state), Some(one_discrete), Some(one_continuous)) = (
            headers.get_mut(index..=index),
            instances.get_mut(index..=index),
            discrete.get_mut(index * A * dh..(index + 1) * A * dh),
            continuous.get_mut(index * A * ch..(index + 1) * A * ch),
        ) else {
            return Err(rejected(lowest, None));
        };
        let one = Rows {
            headers: one_header,
            states: one_state,
            columns: (),
            discrete: one_discrete,
            continuous: one_continuous,
            penalties: &mut [],
            penalized: &mut [],
            outputs: OutputsView::empty(A),
        };
        let mut again = None;
        run_kernels(poisoned, || {
            again = kernels::validate_chunk::<E, A>(ctx, &one);
        })?;
        let again = again.map(|(_, r)| r);
        if let (Some(reject), [inst]) = (again, one.states) {
            notify::<E>(ctx, poisoned, inst, &reject)?;
        }
        return Err(rejected(lowest, again));
    }
    let all = Rows {
        headers,
        states: &mut *instances,
        columns: (),
        discrete,
        continuous,
        penalties: &mut [],
        penalized: &mut [],
        outputs: OutputsView::empty(A),
    };
    run_kernels(poisoned, || {
        found = kernels::validate_chunk::<E, A>(ctx, &all);
    })?;
    match found {
        Some((i, reject)) => {
            if let Some(inst) = instances.get_mut(i) {
                notify::<E>(ctx, poisoned, inst, &reject)?;
            }
            Err(rejected(u32::try_from(i).unwrap_or(u32::MAX), Some(reject)))
        }
        None => Ok(()),
    }
}

/// The rejection notice (ADR-0069): `on_rejected` on the reported instance,
/// once, on this thread, under the panic boundary (a panic poisons). The
/// caller already runs under canonical FP controls; the kernel brackets
/// host allocations and puts the controls back. Compiled out unless
/// `E::NOTIFIES_REJECTION`.
#[inline]
fn notify<E: HostEnvironment>(
    ctx: &HostCtx<'_>,
    poisoned: &mut bool,
    inst: &mut E,
    reject: &ActionReject,
) -> Result<(), EngineError> {
    if !E::NOTIFIES_REJECTION {
        return Ok(());
    }
    run_kernels(poisoned, || {
        kernels::notify_rejected::<E>(ctx, inst, reject);
    })
}

/// The error of a rejected joint action; `None` when the instance did not
/// reject again on this thread (a `validate` that is not pure).
fn rejected(instance: u32, reject: Option<ActionReject>) -> EngineError {
    let reject = reject.unwrap_or(ActionReject {
        code: 0,
        reason: "validate was not pure: the re-run accepted an action a worker rejected",
    });
    EngineError::ActionRejected {
        instance,
        code: reject.code,
        reason: reject.reason,
    }
}

/// The mask layout of a host pool: one block per discrete head, with the
/// environment's conditional heads and entity groups (#385); 0 without
/// discrete heads. Refused past the pool's mask budget before any
/// allocation (SPEC 11.3).
fn mask_len<E: HostEnvironment>(
    cfg: &E::Config,
    layout: &mut ActionLayout,
    rows: usize,
    budget: u64,
) -> Result<usize, EngineError> {
    if layout.discrete.is_empty() {
        return Ok(0);
    }
    // Conditional heads and entity groups (#385): the same layout an
    // `EnvPool` builds from `ActionMask`.
    let parents: Vec<_> = (0..layout.discrete.len())
        .map(|head| E::head_parents(cfg, head))
        .collect();
    let groups = E::entity_group_heads(cfg).map(|g| usize::from(g.get()));
    let built = MaskLayout::build(
        &layout.discrete,
        &parents,
        groups,
        !layout.continuous.is_empty(),
    )?;
    let bytes = built
        .as_ref()
        .and_then(|m| rows.checked_mul(m.len))
        .and_then(|b| u64::try_from(b).ok());
    match (built, bytes) {
        (Some(mask), Some(bytes)) if bytes <= budget => {
            let len = mask.len;
            layout.mask = mask;
            Ok(len)
        }
        (_, bytes) => Err(EngineError::MaskBudgetExceeded {
            bytes: bytes.unwrap_or(u64::MAX),
            budget,
        }),
    }
}

/// The engine error for a host environment that could not be built.
fn host_error(e: HostError) -> EngineError {
    match e {
        HostError::Config(e) => e.into(),
        HostError::Space(e) => e.into(),
        HostError::Construction { reason } => EngineError::HostConstruction { reason },
        other => EngineError::HostConstruction {
            reason: other.to_string(),
        },
    }
}

/// The configuration a host pool keeps. `EnvConfig` does not require
/// `Debug`, so the pool's `Debug` shows it as opaque.
struct HeldConfig<C>(C);

impl<C> core::fmt::Debug for HeldConfig<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("HeldConfig(..)")
    }
}

impl<C> core::ops::Deref for HeldConfig<C> {
    type Target = C;

    fn deref(&self) -> &C {
        &self.0
    }
}
