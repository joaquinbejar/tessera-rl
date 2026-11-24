// SPDX-License-Identifier: MIT OR Apache-2.0

//! Attaching a reset source to a pool (ADR-0075 item 2).

use tessera_rl_core::config::{CanonicalConfig, config_hash};
use tessera_rl_core::{AdaptiveValues, Persist, RecordLayout};

use super::{ArchiveSource, ResetSource};
use crate::error::EngineError;
use crate::masks::MaskProvider;
use crate::pool::{EnvPool, Poison};

impl<E, M> EnvPool<E, M>
where
    E: Persist,
    E::Config: CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    M: MaskProvider<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    /// Sets what this pool's resets start from (ADR-0075 item 2): fresh
    /// `Environment::reset`, or archive resets. It applies to every reset
    /// after the call, auto-resets and explicit resets alike. A Sync pool
    /// is between steps whenever this can be called, so the barrier of
    /// item 2 holds by construction.
    ///
    /// # Errors
    ///
    /// [`EngineError::ArchiveMismatch`] for an archive of another
    /// environment, version or config (its config hash differs);
    /// [`EngineError::Archive`] for weights that do not fit the archive or
    /// are all zero; nothing is mutated then. Rejected with
    /// [`EngineError::Poisoned`] on a poisoned pool.
    pub fn set_reset_source(&mut self, source: ResetSource<E>) -> Result<(), EngineError> {
        self.guarded(Poison::Reject, |pool| {
            let attached = match source {
                ResetSource::Fresh => None,
                ResetSource::Archive { archive, weights } => {
                    let pool_hash = config_hash(E::ENV_ID, E::ENV_VERSION, &pool.cfg)?;
                    if archive.config_hash() != pool_hash {
                        return Err(EngineError::ArchiveMismatch {
                            expected: pool_hash,
                            found: archive.config_hash(),
                        });
                    }
                    Some(ArchiveSource::new(archive, weights)?)
                }
            };
            pool.reset_source = attached;
            Ok(())
        })
    }
}
