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
    /// Sum of their squared returns.
    pub sum_sq_return: f64,
    /// Aborted episodes of a host pool (ADR-0065), counted in `count` too;
    /// `0` for other pools.
    pub aborted: u64,
}

impl EpisodeStats {
    /// The mean return, `None` without episodes.
    #[must_use]
    pub fn mean_return(&self) -> Option<f64> {
        self.mean(self.sum_return)
    }

    /// The mean length, `None` without episodes.
    #[must_use]
    pub fn mean_length(&self) -> Option<f64> {
        self.mean(self.sum_length)
    }

    /// The population variance of the returns, `None` without episodes:
    /// `E[r^2] - E[r]^2` from the sums, clamped at zero. This naive form
    /// loses precision when the mean is large against the spread (ADR-0081
    /// item 3 fixes the partials as sums); compute in `f64` from
    /// `sum_sq_return` and `count` yourself for another estimator. It runs
    /// in the caller's FP environment, like any arithmetic on the report.
    #[must_use]
    pub fn var_return(&self) -> Option<f64> {
        let mean = self.mean_return()?;
        let sq = self.mean(self.sum_sq_return)?;
        Some((sq - mean * mean).max(0.0))
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
}

/// The adaptive subsystems' context in the pool context (ADR-0081 item 6),
/// in subsystem order. Accumulators and partials are not context.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct AdaptiveContext {
    /// Normalization (ADR-0078): the transform identity, the statistics and
    /// the frozen flag.
    pub normalization: crate::transform::TransformContext,
}
