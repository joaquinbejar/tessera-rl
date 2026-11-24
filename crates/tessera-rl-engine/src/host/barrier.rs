// SPDX-License-Identifier: MIT OR Apache-2.0

//! The epoch barrier of a host pool (ADR-0081 item 8): episode statistics
//! only, with the aborted-episode count; normalization and curricula do
//! not exist for host pools.

use super::HostPool;
use crate::adaptive::{EpochReport, Subsystem};
use crate::error::EngineError;
use tessera_rl_core::HostEnvironment;

impl<E: HostEnvironment, const A: usize> HostPool<E, A> {
    /// The epoch barrier (ADR-0081 items 1.3 and 8): episode statistics,
    /// aborted episodes counted in `episodes.count` and in
    /// `episodes.aborted`.
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`] on a poisoned pool, or
    /// [`EngineError::StatisticsOverflow`] (nothing is changed then).
    pub fn epoch_barrier(&mut self) -> Result<EpochReport, EngineError> {
        if self.poisoned {
            return Err(EngineError::Poisoned);
        }
        // The f64 reduction runs in the canonical FP environment, whatever
        // the caller's controls (ADR-0037).
        crate::entry::canonical(|| {
            let mut report = EpochReport::default();
            self.episodes.barrier(&mut report)?;
            Ok(report)
        })
    }

    /// Freezes (or unfreezes) episode statistics, the host pool's only
    /// adaptive subsystem.
    pub fn freeze_adaptive(&mut self, frozen: bool) {
        self.episodes.set_frozen(frozen);
    }
}
