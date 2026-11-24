// SPDX-License-Identifier: MIT OR Apache-2.0

//! The epoch barrier on a pool and the freeze setters (ADR-0081 items 1.3
//! and 2).

use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout};

use super::{EpochReport, Subsystem};
use crate::error::EngineError;
use crate::masks::MaskProvider;
use crate::pool::EnvPool;

impl<E, M> EnvPool<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    /// The epoch barrier (ADR-0081 item 1.3): runs every adaptive subsystem
    /// in order, episode statistics, then normalization (ADR-0078 item 5.3:
    /// the chunk accumulators merge into the pool statistics in chunk
    /// order), and reports. A frozen subsystem changes nothing. Called by
    /// the trainer at its own epoch boundary; never automatic.
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`], or [`EngineError::StatisticsOverflow`]
    /// (nothing is changed then).
    pub fn epoch_barrier(&mut self) -> Result<EpochReport, EngineError> {
        // `guarded` runs the whole barrier in the canonical FP environment
        // (`entry::canonical`, ADR-0037): the caller's rounding mode or
        // flush-to-zero cannot change the report's bits.
        self.guarded(crate::pool::Poison::Reject, |pool| {
            let mut report = EpochReport::default();
            pool.episodes.barrier(&mut report)?;
            if let Some(t) = pool.transform.as_mut() {
                t.barrier();
            }
            // The league, the fourth subsystem (ADR-0088 section 7): on
            // error its view and rings stay, and the pool is not poisoned.
            if let Some(league) = pool.league.as_mut() {
                report.league = league.barrier()?;
            }
            Ok(report)
        })
    }

    /// Freezes (or unfreezes) every adaptive subsystem (ADR-0081 item 2.2).
    pub fn freeze_adaptive(&mut self, frozen: bool) {
        self.freeze_episode_stats(frozen);
        self.freeze_normalization(frozen);
        self.freeze_league(frozen);
    }

    /// Freezes (or unfreezes) the league (ADR-0088 item 4.6): while frozen
    /// no result is kept and the barrier changes nothing; assignments are
    /// still drawn from the current view.
    pub fn freeze_league(&mut self, frozen: bool) {
        if let Some(league) = self.league.as_mut() {
            league.frozen = frozen;
        }
    }

    /// Freezes (or unfreezes) episode statistics: while frozen no episode
    /// is counted and the barrier reports none.
    pub fn freeze_episode_stats(&mut self, frozen: bool) {
        self.episodes.set_frozen(frozen);
    }

    /// Whether every adaptive subsystem is frozen (normalization counts as
    /// frozen without a `Normalize`): the Async barrier's condition.
    #[must_use]
    pub fn adaptive_frozen(&self) -> bool {
        self.episodes.frozen()
            && !self.normalization_unfrozen()
            && self.league.as_ref().is_none_or(|l| l.frozen)
    }
}
