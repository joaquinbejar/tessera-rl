// SPDX-License-Identifier: MIT OR Apache-2.0

//! Player-perspective roots for imperfect-information search (ADR-0035,
//! ADR-0046): a determinized record is built from the searching agent's
//! information state, a caller's sampler seed and a sample index, never
//! from the root's hidden state or RNG lineage.
//!
//! The record's author state comes from `Determinize::sample`, drawn with
//! the sampler identity; its header takes the rollout identity, copies
//! only the fields the information state determines (the step clock, the
//! per-agent `alive` and `generation` rows, the adaptive values),
//! recomputes `acting` and zeroes the rest. It then passes the engine's
//! import checks before it enters the tree.

use tessera_rl_core::encoding::encode_to_vec;
use tessera_rl_core::{
    AdaptiveValues, AgentSlot, Determinize, Environment, RecordLayout, StateInit, TurnStructure,
    Validate,
};
use tessera_rl_engine::{DecodedRecord, EngineError, EnvPool, MaskProvider, PoolOptions, Record};
use tessera_rl_num::identity::{BranchSalt, LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::key::derive_branch_salt;
use tessera_rl_num::rng::{RngAuthority, StepRng};

use crate::error::SearchError;
use crate::handle::NodeHandle;
use crate::options::SearchOptions;
use crate::pool::SearchPool;

/// Branch label of the sampler's identity (ADR-0046).
pub const SAMPLER: u64 = 0x5453_4454_5341_4D50;
/// Branch label of the rollout identity of a determinized root (ADR-0046).
pub const ROLLOUT: u64 = 0x5453_4454_524F_4C4C;

/// How the root of a pool was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RootMode {
    /// A full record of a perfect-information environment.
    Full,
    /// A determinized record (ADR-0046).
    Determinized,
    /// A full record of an imperfect-information environment: privileged
    /// search, which sees the hidden state.
    Privileged,
}

/// The salt of the sampler's identity for `sample_index`.
#[must_use]
pub const fn sampler_salt(sample_index: u64) -> BranchSalt {
    derive_branch_salt(BranchSalt::ROOT, sample_index, SAMPLER)
}

/// The salt of the rollout identity of the determinized record for
/// `sample_index`.
#[must_use]
pub const fn rollout_salt(sample_index: u64) -> BranchSalt {
    derive_branch_salt(BranchSalt::ROOT, sample_index, ROLLOUT)
}

/// The identity the sampler draws with: the sampler seed, lineage 0, the
/// sampler salt, episode 0 and step 0.
#[must_use]
pub const fn sampler_identity(sampler_seed: Seed, sample_index: u64) -> RngIdentity {
    RngIdentity::from_parts(
        sampler_seed,
        sampler_salt(sample_index),
        LogicalId::DEFAULT,
        0,
        0,
    )
}

/// The rollout identity of a determinized record at `step_index`.
#[must_use]
pub const fn rollout_identity(
    sampler_seed: Seed,
    sample_index: u64,
    step_index: u32,
) -> RngIdentity {
    RngIdentity::from_parts(
        sampler_seed,
        rollout_salt(sample_index),
        LogicalId::DEFAULT,
        0,
        step_index,
    )
}

/// The inputs of one determinized root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Determinization {
    /// The searching agent, whose information state is sampled from.
    pub agent: AgentSlot,
    /// The sampler seed, chosen by the caller. It must not be derived from
    /// the root's RNG identity (seed, logical id, branch salt): a derived
    /// seed reopens the lineage channel ADR-0046 closes, and no test of
    /// the search alone can detect it.
    pub sampler_seed: Seed,
    /// The sample index: one determinization per index.
    pub sample_index: u64,
}

impl<E, M> SearchPool<E, M>
where
    E: Environment<
            Config: Sync,
            State: Send,
            Header: Send + RecordLayout<Adaptive: AdaptiveValues<E::Config>>,
        >,
    M: MaskProvider<E>,
{
    /// A pool for player-perspective search: its root is the determinized
    /// record of `root` for `how` (ADR-0046), and every later root comes
    /// from [`Self::set_root_determinized`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`] other than the information check, and
    /// those of [`Self::set_root_determinized`].
    pub fn new_determinized(
        cfg: E::Config,
        options: SearchOptions,
        root: &Record<E>,
        how: Determinization,
        build: impl FnOnce(E::Config, PoolOptions) -> Result<EnvPool<E, M>, EngineError>,
    ) -> Result<Self, SearchError>
    where
        E: Determinize + Validate,
    {
        let mut pool = Self::build_pool(cfg, options, root, build)?;
        let record = pool.determinize(root, how)?;
        // No slot or buffer keeps a copy of the full root.
        pool.records.fill(record);
        pool.child_records.fill(record);
        pool.load_records.fill(record);
        pool.replace_root(&record)?;
        pool.mode = RootMode::Determinized;
        Ok(pool)
    }

    /// A pool for privileged search on an imperfect-information
    /// environment: the root is the full record, hidden state included.
    /// Never the default; [`Self::is_privileged`] labels it.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`] other than the information check.
    pub fn new_privileged(
        cfg: E::Config,
        options: SearchOptions,
        root: &Record<E>,
        build: impl FnOnce(E::Config, PoolOptions) -> Result<EnvPool<E, M>, EngineError>,
    ) -> Result<Self, SearchError> {
        let mut pool = Self::build_pool(cfg, options, root, build)?;
        if !E::PERFECT_INFORMATION {
            pool.mode = RootMode::Privileged;
        }
        Ok(pool)
    }

    /// Whether the pool searches full records of an imperfect-information
    /// environment (made by [`Self::new_privileged`]).
    #[must_use]
    pub fn is_privileged(&self) -> bool {
        self.mode == RootMode::Privileged
    }

    /// Whether the pool searches determinized roots (made by
    /// [`Self::new_determinized`]).
    #[must_use]
    pub fn is_determinized(&self) -> bool {
        self.mode == RootMode::Determinized
    }

    /// Replaces the whole tree by the determinized record of `root` for
    /// `how` (ADR-0046). The retained tree belongs to the old sample, so a
    /// new real move clears it this way, never by `reroot`.
    ///
    /// # Errors
    ///
    /// [`SearchError::NotDeterminized`] on a pool not made by
    /// [`Self::new_determinized`]; the errors of the determinization
    /// (a root waiting for its reset, a root that fails the readiness
    /// checks, an inconsistent sample); those of [`Self::set_root`]. On
    /// error the tree is unchanged.
    pub fn set_root_determinized(
        &mut self,
        root: &Record<E>,
        how: Determinization,
    ) -> Result<NodeHandle, SearchError>
    where
        E: Determinize + Validate,
    {
        if self.mode != RootMode::Determinized {
            return Err(SearchError::NotDeterminized);
        }
        let record = self.determinize(root, how)?;
        self.replace_root(&record)
    }

    /// Builds the determinized record of `root`. The tree is never
    /// touched; the scratch pool is, and every expansion reloads it.
    fn determinize(
        &mut self,
        root: &Record<E>,
        how: Determinization,
    ) -> Result<Record<E>, SearchError>
    where
        E: Determinize + Validate,
    {
        let engine = root.header().engine();
        if engine.pending_reset.get() {
            return Err(SearchError::PendingResetRoot);
        }
        let row = [0_u32];
        // Readiness: the root passes the import checks of an initialized
        // record. `restore_all` first recovers a scratch pool an earlier
        // failed step poisoned.
        self.load_records.fill(*root);
        self.scratch.restore_all(&self.load_records)?;
        self.scratch
            .import(&row, &[DecodedRecord::from(*root)])
            .map_err(|source| SearchError::RootNotReady { source })?;

        let cfg = self.scratch.config();
        let info = E::info_state(cfg, root.state(), how.agent);
        let auth = RngAuthority::engine_only();
        let mut rng = StepRng::new(
            &auth,
            &sampler_identity(how.sampler_seed, how.sample_index),
            StreamId::DYNAMICS,
        );
        let mut state = <E::State as StateInit>::state_default();
        E::sample(cfg, &info, &mut rng, &mut state);
        if rng.exhausted() {
            return Err(SearchError::InconsistentSample {
                reason: "the sampler exhausted its RNG counters",
            });
        }
        if encode_to_vec(&E::info_state(cfg, &state, how.agent)) != encode_to_vec(&info) {
            return Err(SearchError::InconsistentSample {
                reason: "the sampled state has a different information state",
            });
        }

        let mut source = *root.header();
        let mut header = <E::Header as StateInit>::state_default();
        *header.adaptive_mut() = *source.adaptive_mut();
        {
            let out = header.engine_mut();
            out.identity = rollout_identity(
                how.sampler_seed,
                how.sample_index,
                engine.identity.step_index(),
            );
            out.episode_steps = engine.episode_steps;
        }
        let mut acting = 0_usize;
        for (slot, (out, from)) in header
            .agents_mut()
            .iter_mut()
            .zip(source.agents())
            .enumerate()
        {
            out.generation = from.generation;
            out.alive = from.alive;
            let acts = from.alive.get()
                && (E::TURNS == TurnStructure::Simultaneous
                    || u16::try_from(slot)
                        .ok()
                        .and_then(|s| AgentSlot::new(s, E::MAX_AGENTS))
                        .is_some_and(|s| E::acting(cfg, &state, s)));
            out.acting = acts.into();
            acting += usize::from(acts);
        }
        if E::TURNS == TurnStructure::Sequential && acting != 1 {
            return Err(SearchError::SampleNotSingleActor { acting });
        }
        let decoded = DecodedRecord::from_parts(header, state).map_err(|_| {
            SearchError::InconsistentSample {
                reason: "the sampled state has an invalid field",
            }
        })?;
        self.scratch.import(&row, &[decoded])?;
        Ok(self.scratch.snapshot(0)?)
    }
}

#[cfg(test)]
mod tests {
    use super::{ROLLOUT, SAMPLER, rollout_salt, sampler_salt};

    #[test]
    fn test_determinize_labels_match_the_adr_values() {
        assert_eq!(SAMPLER, 0x5453_4454_5341_4D50);
        assert_eq!(ROLLOUT, 0x5453_4454_524F_4C4C);
    }

    #[test]
    fn test_determinize_salts_match_the_reference_vectors() {
        assert_eq!(sampler_salt(0).get(), 0x6B5D_1A48_306D_A7B8);
        assert_eq!(sampler_salt(1).get(), 0x6766_544A_FFA0_CC49);
        assert_eq!(rollout_salt(0).get(), 0x058A_FC2A_50F7_395F);
        assert_eq!(rollout_salt(1).get(), 0x0D27_A07C_566A_AF22);
    }
}
