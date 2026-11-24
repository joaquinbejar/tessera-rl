// SPDX-License-Identifier: MIT OR Apache-2.0

//! Episode statistics (ADR-0081 item 3): per-instance partials added by the
//! instance's own worker at each episode end, reduced at the barrier by a
//! pairwise tree over instance order (SPEC 6.3), independent of
//! `chunk_size`, worker count and schedule.

use super::{EpisodeStats, EpochReport, Subsystem};
use crate::error::EngineError;

/// One instance's episodes since the last barrier: 32 bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[repr(C)]
pub(crate) struct EpisodePartial {
    /// Episodes ended.
    pub(crate) count: u64,
    /// Sum of their returns.
    pub(crate) sum_return: f64,
    /// Sum of their lengths.
    pub(crate) sum_length: f64,
    /// Sum of their squared returns.
    pub(crate) sum_sq_return: f64,
}

impl EpisodePartial {
    /// Adds one ended episode. Runs in the step kernel: infallible; the
    /// count saturates at `u64::MAX`, which 2^64 episodes cannot reach.
    #[inline]
    pub(crate) fn add(&mut self, ret: f32, length: u32) {
        let r = f64::from(ret);
        self.count = self.count.saturating_add(1);
        self.sum_return += r;
        self.sum_length += f64::from(length);
        self.sum_sq_return += r * r;
    }
}

/// The three `f64` sums of a partial range.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Sums {
    ret: f64,
    length: f64,
    sq: f64,
}

/// The pairwise tree over instance order: a range of one element is that
/// element; a longer one splits at `lo + (hi - lo) / 2` and adds the two
/// halves' sums, left then right. Depth is `ceil(log2(N))` (at most 32).
fn pairwise(parts: &[EpisodePartial]) -> Sums {
    match parts {
        [] => Sums::default(),
        [p] => Sums {
            ret: p.sum_return,
            length: p.sum_length,
            sq: p.sum_sq_return,
        },
        _ => {
            let (left, right) = parts.split_at(parts.len() / 2);
            let (l, r) = (pairwise(left), pairwise(right));
            Sums {
                ret: l.ret + r.ret,
                length: l.length + r.length,
                sq: l.sq + r.sq,
            }
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
        let count = self
            .partials
            .iter()
            .try_fold(0u64, |acc, p| acc.checked_add(p.count))
            .ok_or(EngineError::StatisticsOverflow {
                counter: "episode count",
            })?;
        let aborted = self
            .aborts
            .iter()
            .try_fold(0u64, |acc, &a| acc.checked_add(a))
            .ok_or(EngineError::StatisticsOverflow {
                counter: "aborted episode count",
            })?;
        let sums = pairwise(&self.partials);
        Ok(EpisodeStats {
            count,
            sum_return: sums.ret,
            sum_length: sums.length,
            sum_sq_return: sums.sq,
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

    fn part(count: u64, r: f64) -> EpisodePartial {
        EpisodePartial {
            count,
            sum_return: r,
            sum_length: r * 2.0,
            sum_sq_return: r * r,
        }
    }

    #[test]
    fn test_pairwise_splits_left_half_first_and_is_not_sequential() {
        // ((a + b) + (c + d)) differs from ((a + b) + c) + d here.
        let parts = [part(1, 1e16), part(1, 1.0), part(1, -1e16), part(1, 1.0)];
        let s = pairwise(&parts);
        assert_eq!(s.ret, (1e16 + 1.0) + (-1e16 + 1.0));
        let sequential = ((1e16 + 1.0) + -1e16) + 1.0;
        assert_ne!(s.ret.to_bits(), f64::to_bits(sequential));
    }

    #[test]
    fn test_pairwise_odd_length_splits_at_floor_half() {
        let parts = [part(1, 1.0), part(1, 2.0), part(1, 3.0)];
        // [1] + ([2] + [3]).
        assert_eq!(pairwise(&parts).ret, 1.0 + (2.0 + 3.0));
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
        assert_eq!(report.episodes.sum_sq_return, 5.0);
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
