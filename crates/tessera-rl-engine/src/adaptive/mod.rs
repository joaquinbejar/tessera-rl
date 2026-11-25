// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pool-level adaptive state and the one epoch barrier (ADR-0081): episode
//! statistics, normalization (ADR-0078) and, later, curricula, each a
//! subsystem with an accumulation on the workers and a merge in a fixed
//! order at the barrier, on the calling thread.

mod barrier;
mod curriculum;
mod episodes;
mod report;
mod targets_api;

pub use curriculum::{AdaptiveTargets, Curriculum, CurriculumChanges, CurriculumContext};
pub(crate) use curriculum::{CurriculumSlot, Targets, check_curriculum};
pub(crate) use episodes::{EpisodePartial, EpisodeStatistics};
pub use report::{AdaptiveContext, EpisodeStats, EpochReport};

use crate::error::EngineError;

/// A pool-level adaptive subsystem (ADR-0081 item 1.1). Accumulation runs
/// on the workers into the subsystem's own accumulators; the barrier runs
/// on the calling thread, merges them in a fixed order, updates the
/// pool-level state and clears them.
pub(crate) trait Subsystem {
    /// Whether the subsystem accumulates nothing and its barrier changes
    /// nothing.
    fn frozen(&self) -> bool;
    /// Sets the frozen flag.
    fn set_frozen(&mut self, frozen: bool);
    /// The barrier: merge, update, clear, and report into `report`.
    fn barrier(&mut self, report: &mut EpochReport) -> Result<(), EngineError>;
}
