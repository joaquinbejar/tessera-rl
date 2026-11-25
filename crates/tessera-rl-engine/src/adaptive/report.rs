// SPDX-License-Identifier: MIT OR Apache-2.0

//! What the epoch barrier reports, and the adaptive part of the pool
//! context (ADR-0081 items 1.3 and 6).

/// Episode statistics since the last barrier (ADR-0081 item 3): sums
/// reduced pairwise over instance order, counts exact.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[non_exhaustive]
pub struct EpisodeStats {
    /// Episodes ended.
    pub count: u64,
    /// Sum of their returns.
    pub sum_return: f64,
    /// Sum of their lengths.
    pub sum_length: f64,
    /// Mean of their returns (Chan's pairwise merge of per-instance
    /// Welford means, in instance order).
    pub return_mean: f64,
    /// Sum of squared deviations of their returns from `return_mean`
    /// (Chan's merge): the population variance is `return_m2 / count`.
    pub return_m2: f64,
    /// Aborted episodes of a host pool (ADR-0065), counted in `count` too;
    /// `0` for other pools.
    pub aborted: u64,
}

impl EpisodeStats {
    /// The mean return, `None` without episodes.
    #[must_use]
    pub fn mean_return(&self) -> Option<f64> {
        (self.count > 0).then_some(self.return_mean)
    }

    /// The mean length, `None` without episodes.
    #[must_use]
    pub fn mean_length(&self) -> Option<f64> {
        self.mean(self.sum_length)
    }

    /// The population variance of the returns, `return_m2 / count`, `None`
    /// without episodes. Chan's merge keeps it exact to rounding when
    /// returns are large and vary little. It runs in the caller's FP
    /// environment, like any arithmetic on the report.
    #[must_use]
    pub fn var_return(&self) -> Option<f64> {
        self.mean(self.return_m2)
    }

    fn mean(&self, sum: f64) -> Option<f64> {
        #[expect(clippy::cast_precision_loss, reason = "a statistic")]
        let n = self.count as f64;
        (self.count > 0).then(|| sum / n)
    }
}

/// What one epoch barrier reports (ADR-0081 item 1.3).
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct EpochReport {
    /// Episode statistics; zero when the subsystem is frozen.
    pub episodes: EpisodeStats,
    /// The league's barrier (ADR-0088 section 7); `None` without a league
    /// or when it is frozen.
    pub league: Option<crate::league::LeagueReport>,
    /// The slots the curriculum changed; `None` without a curriculum or
    /// when it is frozen.
    pub curriculum: Option<super::CurriculumChanges>,
}

/// The adaptive subsystems' context in the pool context (ADR-0081 item 6),
/// in subsystem order. Accumulators and partials are not context.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct AdaptiveContext {
    /// Normalization (ADR-0078): the transform identity, the statistics and
    /// the frozen flag.
    pub normalization: crate::transform::TransformContext,
    /// The league (ADR-0088 item 7.3): its view and the assignment array;
    /// `None` without a league. Result rings are accumulators, not context.
    pub league: Option<Box<crate::league::LeagueContext>>,
    /// The curriculum (ADR-0081 item 5): identity, state and targets.
    pub curriculum: super::CurriculumContext,
}
