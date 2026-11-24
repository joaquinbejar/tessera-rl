// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rebuilding a replay's pool from its header (ADR-0072 section 5, #377).
//!
//! The header holds the canonical config bytes and the pool options the
//! identity fields fix (ADR-0011): the auto-reset mode, the step limit and
//! the illegal-action policy, plus the instance count. [`Replay::build_pool`]
//! decodes the config with [`decode_canonical`] (adaptive fields keep their
//! defaults; the first keyframe's headers carry their effective values),
//! derives the options and hands both to a caller's constructor, which
//! still picks the mask provider (`EnvPool::new` or `EnvPool::with_masks`)
//! and the execution mode. The pool is then checked against the header
//! exactly as a caller-built one is.
//!
//! The seed is 0: an `EnvPool` file begins with a keyframe of every
//! instance, whose records carry each instance's RNG identity.

use core::num::NonZeroU32;

use tessera_rl_core::config::CanonicalConfig;
use tessera_rl_core::{
    AdaptiveValues, CanonicalDecode, EnvConfig, Persist, RecordLayout, Validate, decode_canonical,
};
use tessera_rl_num::identity::Seed;

use crate::error::EngineError;
use crate::masks::{IllegalActionPolicy, PersistMask};
use crate::options::{AutoReset, PoolOptions};
use crate::persist::PersistError;
use crate::pool::EnvPool;

use super::ReplayError;
use super::reader::Replay;
use super::run::ReplaySummary;

impl Replay<'_> {
    /// The pool options the header fixes: the instance count, the
    /// auto-reset mode, the step limit and the illegal-action policy; seed
    /// 0 (the first keyframe sets every RNG identity) and default chunking.
    ///
    /// # Errors
    ///
    /// [`PersistError::InvalidField`] for a header field outside its range.
    pub fn pool_options(&self) -> Result<PoolOptions, ReplayError> {
        let invalid = |field| ReplayError::Persist(PersistError::InvalidField { field });
        let num_envs =
            NonZeroU32::new(self.header.num_envs).ok_or_else(|| invalid("instance_count"))?;
        let mut options = PoolOptions::new(num_envs, Seed::new(0));
        options.auto_reset = match self.file.auto_reset {
            0 => AutoReset::NextStep,
            1 => AutoReset::SameStep,
            2 => AutoReset::Disabled,
            _ => return Err(invalid("auto_reset")),
        };
        options.max_episode_steps = NonZeroU32::new(self.file.max_episode_steps);
        options.illegal_action = match self.file.illegal_kind {
            0 => IllegalActionPolicy::Noop,
            1 => IllegalActionPolicy::Penalize(f32::from_bits(self.file.penalty_bits)),
            _ => return Err(invalid("illegal_kind")),
        };
        // #411: the recorded chunk size, when the file holds one (0: any).
        if let Some(chunk_size) = NonZeroU32::new(self.header.chunk_size) {
            options.chunk_size = chunk_size;
        }
        Ok(options)
    }

    /// Builds the recorded pool from the header: the canonical config
    /// decoded with [`decode_canonical`] and the options of
    /// [`Replay::pool_options`], handed to `build` (for example
    /// `EnvPool::with_masks`, then `with_workers` for a Sync pool), then
    /// checked against the header as [`Replay::run`] checks any pool.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Config`] if the config bytes do not decode into
    /// `E::Config`; the error of `build`; and the mismatch errors of
    /// [`Replay::run`] if the built pool does not match the header.
    pub fn build_pool<E, M>(
        &self,
        build: impl FnOnce(E::Config, PoolOptions) -> Result<EnvPool<E, M>, EngineError>,
    ) -> Result<EnvPool<E, M>, ReplayError>
    where
        E: Persist + Validate,
        E::Config: CanonicalDecode + EnvConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        if self.header.host.is_some() {
            return Err(PersistError::Incompatible {
                field: "host_protocol",
            }
            .into());
        }
        let cfg = decode_canonical::<E::Config>(self.header.config)?;
        let options = self.pool_options()?;
        let pool = build(cfg, options)?;
        self.check_pool(&pool)?;
        Ok(pool)
    }

    /// [`Replay::build_pool`] with `build`, then [`Replay::run`] into the
    /// pool built; returns the pool and the summary.
    ///
    /// The header holds an asset field's hash only (ADR-0077), so only the
    /// asset decoded from empty input (for example the Grid's open layout)
    /// is rebuilt; any other asset is
    /// [`CanonicalDecodeError::Asset`](tessera_rl_core::config_canonical::CanonicalDecodeError::Asset).
    /// Replay such a file with [`Replay::run`] into a pool the caller builds
    /// from the config, assets included.
    ///
    /// # Errors
    ///
    /// As [`Replay::build_pool`] and [`Replay::run`].
    pub fn run_rebuilt<E, M>(
        &self,
        build: impl FnOnce(E::Config, PoolOptions) -> Result<EnvPool<E, M>, EngineError>,
    ) -> Result<(EnvPool<E, M>, ReplaySummary), ReplayError>
    where
        E: Persist + Validate,
        E::Config: CanonicalDecode + EnvConfig + CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        let mut pool = self.build_pool(build)?;
        let summary = self.run(&mut pool)?;
        Ok((pool, summary))
    }
}
