// SPDX-License-Identifier: MIT OR Apache-2.0

//! Episode statistics (ADR-0081 item 3): per-instance partials added by the
//! instance's own worker at each episode end, reduced at the barrier by a
//! pairwise tree over instance order (SPEC 6.3), independent of
//! `chunk_size`, worker count and schedule.

use super::{EpisodeStats, EpochReport, Subsystem};
use crate::error::EngineError;

/// One instance's episodes since the last barrier: 40 bytes. Returns are
/// kept as a count, mean and M2 (Welford), merged at the barrier by Chan's
/// formula, so the variance stays exact to rounding when returns are large
/// and vary little.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[repr(C)]
pub(crate) struct EpisodePartial {
    /// Episodes ended.
    pub(crate) count: u64,
    /// Sum of their returns.
    pub(crate) sum_return: f64,
    /// Sum of their lengths.
    pub(crate) sum_length: f64,
    /// Mean of their returns.
    pub(crate) mean: f64,
    /// Sum of squared deviations of their returns from `mean`.
    pub(crate) m2: f64,
}

impl EpisodePartial {
    /// Adds one ended episode (Welford). Runs in the step kernel, which is
    /// infallible: the count saturates at `u64::MAX`, a documented kernel
    /// bound. It cannot be reached (an episode takes at least one step, and
    /// 2^64 steps of one instance between two barriers would take
    /// centuries); the barrier's merge checks every sum exactly.
    #[inline]
    pub(crate) fn add(&mut self, ret: f32, length: u32) {
        let r = f64::from(ret);
        self.count = self.count.saturating_add(1);
        self.sum_return += r;
        self.sum_length += f64::from(length);
        #[expect(clippy::cast_precision_loss, reason = "episode counts far below 2^53")]
        let n = self.count as f64;
        let delta = r - self.mean;
        self.mean += delta / n;
        self.m2 += delta * (r - self.mean);
    }
}

/// The merged statistics of a partial range.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Sums {
    count: u64,
    ret: f64,
    length: f64,
    mean: f64,
    m2: f64,
}

impl Sums {
    fn of(p: &EpisodePartial) -> Self {
        Self {
            count: p.count,
            ret: p.sum_return,
            length: p.sum_length,
            mean: p.mean,
            m2: p.m2,
        }
    }

    /// Chan's merge of `self` (left) and `other` (right): `n = na + nb`,
    /// `delta = mean_b - mean_a`, `mean = mean_a + delta * nb / n`,
    /// `m2 = m2_a + m2_b + delta^2 * na * nb / n`. An empty side leaves the
    /// other unchanged. `None` when the count overflows `u64`.
    fn merge(self, other: Self) -> Option<Self> {
        if other.count == 0 {
            return Some(Self {
                ret: self.ret + other.ret,
                length: self.length + other.length,
                ..self
            });
        }
        if self.count == 0 {
            return Some(Self {
                ret: self.ret + other.ret,
                length: self.length + other.length,
                ..other
            });
        }
        let count = self.count.checked_add(other.count)?;
        #[expect(clippy::cast_precision_loss, reason = "episode counts far below 2^53")]
        let (na, nb, n) = (self.count as f64, other.count as f64, count as f64);
        let delta = other.mean - self.mean;
        Some(Self {
            count,
            ret: self.ret + other.ret,
            length: self.length + other.length,
            mean: self.mean + delta * nb / n,
            m2: self.m2 + other.m2 + delta * delta * na * nb / n,
        })
    }
}

/// The pairwise tree over instance order: a range of one element is that
/// element; a longer one splits at `lo + (hi - lo) / 2` and merges the two
/// halves, left then right. Depth is `ceil(log2(N))` (at most 32). `None`
/// when the episode count overflows `u64`.
fn pairwise(parts: &[EpisodePartial]) -> Option<Sums> {
    match parts {
        [] => Some(Sums::default()),
        [p] => Some(Sums::of(p)),
        _ => {
            let (left, right) = parts.split_at(parts.len() / 2);
            pairwise(left)?.merge(pairwise(right)?)
        }
    }
}

/// The episode-statistics subsystem: partials per instance, and per
/// instance the aborted episodes of a host pool (ADR-0081 item 8; empty
/// otherwise).
#[derive(Debug, Clone, Default)]
pub(crate) struct EpisodeStatistics {
    pub(crate) partials: Vec<EpisodePartial>,
    pub(crate) aborts: Vec<u64>,
    pub(crate) frozen: bool,
}

impl EpisodeStatistics {
    /// Statistics for `instances` instances; `host` adds the abort counts.
    pub(crate) fn new(instances: usize, host: bool) -> Self {
        Self {
            partials: vec![EpisodePartial::default(); instances],
            aborts: if host { vec![0; instances] } else { Vec::new() },
            frozen: false,
        }
    }

    /// The partials a step pass adds to: empty while frozen, so the kernels
    /// add nothing.
    pub(crate) fn rows(&mut self) -> (&mut [EpisodePartial], &mut [u64]) {
        if self.frozen {
            (&mut [], &mut [])
        } else {
            (&mut self.partials, &mut self.aborts)
        }
    }

    /// Clears every partial (a restore: partials are not context).
    pub(crate) fn clear(&mut self) {
        self.partials.fill(EpisodePartial::default());
        self.aborts.fill(0);
    }

    /// The reduced statistics, in instance order.
    fn reduce(&self) -> Result<EpisodeStats, EngineError> {
        let aborted = self
            .aborts
            .iter()
            .try_fold(0u64, |acc, &a| acc.checked_add(a))
            .ok_or(EngineError::StatisticsOverflow {
                counter: "aborted episode count",
            })?;
        let sums = pairwise(&self.partials).ok_or(EngineError::StatisticsOverflow {
            counter: "episode count",
        })?;
        Ok(EpisodeStats {
            count: sums.count,
            sum_return: sums.ret,
            sum_length: sums.length,
            return_mean: sums.mean,
            return_m2: sums.m2,
            aborted,
        })
    }
}

impl Subsystem for EpisodeStatistics {
    fn frozen(&self) -> bool {
        self.frozen
    }

    fn set_frozen(&mut self, frozen: bool) {
        self.frozen = frozen;
    }

    fn barrier(&mut self, report: &mut EpochReport) -> Result<(), EngineError> {
        if self.frozen {
            return Ok(());
        }
        report.episodes = self.reduce()?;
        self.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with(returns: &[f32]) -> EpisodePartial {
        let mut p = EpisodePartial::default();
        for &r in returns {
            p.add(r, 1);
        }
        p
    }

    #[test]
    fn test_pairwise_splits_left_half_first_and_is_not_sequential() {
        // ((a + b) + (c + d)) differs from ((a + b) + c) + d here.
        let parts = [with(&[1e16]), with(&[1.0]), with(&[-1e16]), with(&[1.0])];
        let s = pairwise(&parts).unwrap();
        assert_eq!(
            s.ret,
            (f64::from(1e16f32) + 1.0) + (-f64::from(1e16f32) + 1.0)
        );
        let sequential = ((f64::from(1e16f32) + 1.0) + -f64::from(1e16f32)) + 1.0;
        assert_ne!(s.ret.to_bits(), f64::to_bits(sequential));
    }

    #[test]
    fn test_pairwise_odd_length_splits_at_floor_half() {
        let parts = [with(&[1.0]), with(&[2.0]), with(&[3.0])];
        // [1] + ([2] + [3]).
        assert_eq!(pairwise(&parts).unwrap().ret, 1.0 + (2.0 + 3.0));
        assert_eq!(pairwise(&parts).unwrap().count, 3);
    }

    #[test]
    fn test_chan_merge_keeps_the_variance_of_large_returns_that_vary_little() {
        // Returns near 1.3e12 that differ by multiples of 2^18 (exact in
        // f32, whose spacing there is 2^17). The exact population variance
        // is 280_095_621_120 (computed with rationals); the naive
        // E[r^2] - E[r]^2 is off by about 4e-4 relative, Chan's merge by
        // about 7e-11.
        const BASE: f32 = 1.3e12;
        const STEP: f32 = 262_144.0;
        const VAR: f64 = 280_095_621_120.0;
        let values: Vec<f32> = (0u16..64)
            .map(|k| BASE + f32::from((k * 37) % 7) * STEP - 3.0 * STEP)
            .collect();
        let parts: Vec<EpisodePartial> = values.chunks(3).map(with).collect();
        let s = pairwise(&parts).unwrap();
        assert_eq!(s.count, 64);
        #[expect(clippy::cast_precision_loss, reason = "test count")]
        let var = s.m2 / s.count as f64;
        assert!((var - VAR).abs() <= VAR * 1e-9, "variance {var}");
        // The naive formula over the same values is far off.
        let n = 64.0;
        let sum: f64 = values.iter().map(|&v| f64::from(v)).sum();
        let sq: f64 = values.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let naive = sq / n - (sum / n) * (sum / n);
        assert!((naive - VAR).abs() > VAR * 1e-4, "naive {naive}");
    }

    #[test]
    fn test_chan_merge_with_empty_sides_is_the_other_side() {
        let a = Sums::of(&with(&[2.0, 4.0]));
        assert_eq!(a.merge(Sums::default()), Some(a));
        assert_eq!(Sums::default().merge(a), Some(a));
    }

    #[test]
    fn test_chan_merge_count_overflow_is_none_and_the_barrier_refuses() {
        let near = EpisodePartial {
            count: u64::MAX,
            ..EpisodePartial::default()
        };
        assert_eq!(Sums::of(&near).merge(Sums::of(&with(&[1.0]))), None);
        let mut s = EpisodeStatistics::new(2, false);
        s.partials[0] = near;
        s.partials[1].add(1.0, 1);
        let mut report = EpochReport::default();
        assert!(matches!(
            s.barrier(&mut report),
            Err(EngineError::StatisticsOverflow { .. })
        ));
        assert_eq!(
            s.partials[0].count,
            u64::MAX,
            "nothing cleared on the error"
        );
    }

    #[test]
    fn test_episode_statistics_barrier_reports_and_clears_and_frozen_reports_nothing() {
        let mut s = EpisodeStatistics::new(3, false);
        s.partials[0].add(2.0, 5);
        s.partials[2].add(-1.0, 3);
        let mut report = EpochReport::default();
        s.barrier(&mut report).unwrap();
        assert_eq!(report.episodes.count, 2);
        assert_eq!(report.episodes.sum_return, 1.0);
        assert_eq!(report.episodes.sum_length, 8.0);
        assert_eq!(report.episodes.mean_return(), Some(0.5));
        assert_eq!(report.episodes.var_return(), Some(2.25));
        assert!(s.partials.iter().all(|p| *p == EpisodePartial::default()));
        s.partials[1].add(1.0, 1);
        s.set_frozen(true);
        let mut frozen = EpochReport::default();
        s.barrier(&mut frozen).unwrap();
        assert_eq!(frozen.episodes.count, 0);
        assert_eq!(s.partials[1].count, 1, "a frozen barrier clears nothing");
        assert!(s.rows().0.is_empty(), "a frozen pass adds nothing");
    }
}
