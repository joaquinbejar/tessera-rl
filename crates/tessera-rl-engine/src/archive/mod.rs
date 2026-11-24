// SPDX-License-Identifier: MIT OR Apache-2.0

//! Archive resets (ADR-0075, SPEC 9.3).
//!
//! An [`Archive`] is an immutable set of saved states of one environment
//! and config, built once and fully validated, then shared by any number of
//! pools (an `Arc` inside; cloning is cheap). An archive reset copies an
//! entry's author state and its alive slots into an instance; everything
//! else follows the instance's own reset rules (item 3): the episode index
//! moves on and the step index is 0, the seed, branch salt and lineage are
//! the instance's own, and generations are kept (ADR-0067 item 1).
//!
//! Which entry a reset uses is drawn from the reserved `archive_select`
//! stream at the reset counter domain, weighted by `u32` weights through a
//! `u64` prefix table, in integers only (item 4); `reset_from` names the
//! entries explicitly (item 5).

mod attach;
mod context;
mod source;

pub use context::ResetSourceContext;
pub(crate) use source::ArchiveSource;
pub use source::ResetSource;

use std::sync::Arc;

use tessera_rl_core::config::{CanonicalConfig, config_hash};
use tessera_rl_core::encoding::StateHasher;
use tessera_rl_core::{Bool8, Environment, HeaderView, Persist, RecordLayout};

use crate::error::EngineError;
use crate::record::Record;

/// The largest number of entries an archive may hold (ADR-0075 item 1).
pub const MAX_ARCHIVE_ENTRIES: usize = 1 << 24;

/// An immutable, validated set of saved states (ADR-0075 item 1).
pub struct Archive<E: Environment> {
    inner: Arc<Inner<E>>,
}

struct Inner<E: Environment> {
    /// XXH3-64 over the environment id, version, config hash, the entry
    /// count, then each entry's state and alive slots.
    digest: u64,
    /// The config hash of the environment and config the entries belong to.
    config_hash: u64,
    states: Vec<E::State>,
    /// `MAX_AGENTS` alive flags per entry, in entry order.
    alive: Vec<Bool8>,
}

impl<E: Environment> Clone for Archive<E> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<E: Environment> core::fmt::Debug for Archive<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Archive")
            .field("len", &self.len())
            .field("digest", &format_args!("{:016x}", self.digest()))
            .finish()
    }
}

impl<E> Archive<E>
where
    E: Persist,
    E::Config: CanonicalConfig,
{
    /// Builds an archive from `records` of an environment with config
    /// `cfg`. Every record's state is checked with the environment's
    /// invariants (`Validate::validate_record`), as an import checks it.
    ///
    /// # Errors
    ///
    /// [`EngineError::Archive`] for no records or more than
    /// [`MAX_ARCHIVE_ENTRIES`]; [`EngineError::Invariant`] (with the
    /// record's position) for a state that breaks an invariant;
    /// [`EngineError::Config`] if the config hash cannot be formed.
    pub fn new(cfg: &E::Config, records: &[Record<E>]) -> Result<Self, EngineError> {
        if records.is_empty() {
            return Err(EngineError::Archive {
                reason: "an archive needs at least one entry",
            });
        }
        if records.len() > MAX_ARCHIVE_ENTRIES {
            return Err(EngineError::Archive {
                reason: "an archive holds at most 2^24 entries",
            });
        }
        for (index, record) in records.iter().enumerate() {
            let view = HeaderView::of(&record.header);
            E::validate_record(cfg, &view, &record.state)
                .map_err(|violation| EngineError::Invariant { index, violation })?;
        }
        let config_hash = config_hash(E::ENV_ID, E::ENV_VERSION, cfg)?;
        let states: Vec<E::State> = records.iter().map(|r| r.state).collect();
        let mut alive = Vec::with_capacity(records.len().saturating_mul(E::MAX_AGENTS));
        for record in records {
            alive.extend(record.header.agents().iter().map(|row| row.alive));
        }
        let mut hasher = StateHasher::new();
        hasher.add_bytes(E::ENV_ID.as_bytes());
        hasher.add(&E::ENV_VERSION);
        hasher.add(&config_hash);
        hasher.add(&u64::try_from(records.len()).unwrap_or(u64::MAX));
        for (i, state) in states.iter().enumerate() {
            hasher.add(state);
            for flag in alive
                .get(i * E::MAX_AGENTS..(i + 1) * E::MAX_AGENTS)
                .unwrap_or(&[])
            {
                hasher.add(flag);
            }
        }
        Ok(Self {
            inner: Arc::new(Inner {
                digest: hasher.finish().get(),
                config_hash,
                states,
                alive,
            }),
        })
    }
}

impl<E> Archive<E>
where
    E: Persist,
    E::Config: CanonicalConfig,
{
    /// [`Archive::new`] from untrusted records ([`DecodedRecord`](crate::DecodedRecord), from
    /// bytes or a file: their canonical encoding is already checked). The
    /// same `Validate` checks run on every entry.
    ///
    /// # Errors
    ///
    /// As [`Archive::new`].
    pub fn from_decoded(
        cfg: &E::Config,
        records: &[crate::record::DecodedRecord<E>],
    ) -> Result<Self, EngineError> {
        let trusted: Vec<Record<E>> = records.iter().map(|r| r.trust()).collect();
        Self::new(cfg, &trusted)
    }
}

impl<E: Environment> Archive<E> {
    /// The archive's identity (ADR-0075 item 1): not authentication, a
    /// check against a wrong or changed archive.
    #[must_use]
    pub fn digest(&self) -> u64 {
        self.inner.digest
    }

    /// The number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.states.len()
    }

    /// Always `false`: an archive has at least one entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.states.is_empty()
    }

    /// The config hash the entries were checked against.
    #[must_use]
    pub(crate) fn config_hash(&self) -> u64 {
        self.inner.config_hash
    }

    /// Entry `i`'s state and alive flags (`MAX_AGENTS` of them).
    #[must_use]
    pub(crate) fn entry(&self, i: usize) -> Option<(&E::State, &[Bool8])> {
        let state = self.inner.states.get(i)?;
        let alive = self
            .inner
            .alive
            .get(i * E::MAX_AGENTS..(i + 1) * E::MAX_AGENTS)?;
        Some((state, alive))
    }
}
