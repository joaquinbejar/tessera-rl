// SPDX-License-Identifier: MIT OR Apache-2.0

//! How the host environment checks drive their pools (#351).

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::HostEnvironment;
use tessera_rl_num::identity::Seed;

use super::super::HostPool;
use crate::error::EngineError;
use crate::options::PoolOptions;

/// The run every check makes: the pool size and seed, the number of steps
/// after the first reset, the seed of the action sampler and the worker
/// count of the Sync-mode pool that `check_determinism` compares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CheckRun {
    /// Instances per pool.
    pub num_envs: NonZeroU32,
    /// The pool seed: instance `i` is seeded `seed + i` (ADR-0007).
    pub seed: u64,
    /// Steps after the first reset.
    pub calls: usize,
    /// The seed of the legal-action sampler.
    pub action_seed: u64,
    /// Workers of the Sync-mode pool.
    pub workers: NonZeroUsize,
    /// `PoolOptions::max_episode_steps`: `None` lets the environment end
    /// its episodes alone.
    pub max_episode_steps: Option<NonZeroU32>,
}

impl CheckRun {
    /// A run of `calls` steps over `num_envs` instances seeded `seed`, with
    /// an action seed derived from `seed`, 2 workers and no step limit.
    #[must_use]
    pub const fn new(num_envs: NonZeroU32, seed: u64, calls: usize) -> Self {
        Self {
            num_envs,
            seed,
            calls,
            action_seed: seed ^ 0xA5A5_5A5A_C3C3_3C3C,
            workers: NonZeroUsize::MIN.saturating_add(1),
            max_episode_steps: None,
        }
    }

    /// The same run with `workers` workers in the Sync-mode pool.
    #[must_use]
    pub const fn with_workers(mut self, workers: NonZeroUsize) -> Self {
        self.workers = workers;
        self
    }

    /// The same run with another action seed.
    #[must_use]
    pub const fn with_action_seed(mut self, action_seed: u64) -> Self {
        self.action_seed = action_seed;
        self
    }

    /// The same run with a step limit (`PoolOptions::max_episode_steps`).
    #[must_use]
    pub const fn with_max_episode_steps(mut self, steps: Option<NonZeroU32>) -> Self {
        self.max_episode_steps = steps;
        self
    }

    /// The pool options of this run, with the pool seed `seed`.
    #[must_use]
    pub(crate) fn options(&self, seed: u64) -> PoolOptions {
        let mut options = PoolOptions::new(self.num_envs, Seed::new(seed));
        options.max_episode_steps = self.max_episode_steps;
        options
    }
}

/// An Inline pool of `E` for `run`, seeded `seed`.
pub(crate) fn build<E: HostEnvironment, const A: usize>(
    cfg: &dyn Fn() -> E::Config,
    run: &CheckRun,
    seed: u64,
) -> Result<HostPool<E, A>, EngineError> {
    HostPool::<E, A>::new(cfg(), run.options(seed))
}
