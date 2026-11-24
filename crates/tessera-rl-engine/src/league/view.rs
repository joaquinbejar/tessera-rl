// SPDX-License-Identifier: MIT OR Apache-2.0

//! The league's per-step data in the engine (ADR-0088 sections 2 and 3,
//! erratum 1): plain values the reset draw reads, so the step path never
//! calls into `tessera-rl-league`.

use thiserror::Error;

/// How the engine plays an instance's opponent (ADR-0088 item 2.3). ONNX
/// opponents are head C.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpponentKind {
    /// The environment's heuristic (ADR-0079 section 4): computed and
    /// validated on the worker, as a heuristic slot.
    Heuristic,
    /// A trainer-side checkpoint named by `tag`: the caller submits the
    /// opponent rows, validated at the boundary.
    External {
        /// The trainer's tag for the checkpoint.
        tag: u64,
    },
}

/// Why a league view is refused (ADR-0088 erratum 1, condition 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LeagueViewError {
    /// No policy.
    #[error("a league view needs at least one policy")]
    Empty,
    /// Another number of thresholds than of policies.
    #[error("{thresholds} thresholds for {kinds} policies")]
    Length {
        /// Thresholds given.
        thresholds: usize,
        /// Policies given.
        kinds: usize,
    },
    /// A threshold below the one before it.
    #[error("threshold {index} is below the one before it")]
    NotMonotone {
        /// The first decreasing threshold.
        index: usize,
    },
    /// No drawable entry, or a last entry out of range or whose threshold
    /// is not the maximum (a zero total).
    #[error("the view has no drawable policy (last {last:?})")]
    NoDrawable {
        /// The declared last drawable entry.
        last: Option<u32>,
    },
}

/// The league data one pool reads (ADR-0088 erratum 1): each policy's
/// kind, the cumulative `u64` draw thresholds of the matchmaking table
/// (one per policy, last drawable at `u64::MAX`), the last drawable entry
/// and the population digest. Head E's shared league hands each pool this
/// same type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeagueView {
    kinds: Vec<OpponentKind>,
    thresholds: Vec<u64>,
    last: u32,
    population_digest: u64,
}

impl LeagueView {
    /// A checked view.
    ///
    /// # Errors
    ///
    /// [`LeagueViewError`] for no policy, thresholds of another length or
    /// not monotone, or a `last` that is not a drawable entry whose
    /// threshold is `u64::MAX`.
    pub fn new(
        kinds: Vec<OpponentKind>,
        thresholds: Vec<u64>,
        last: Option<u32>,
        population_digest: u64,
    ) -> Result<Self, LeagueViewError> {
        if kinds.is_empty() {
            return Err(LeagueViewError::Empty);
        }
        if thresholds.len() != kinds.len() {
            return Err(LeagueViewError::Length {
                thresholds: thresholds.len(),
                kinds: kinds.len(),
            });
        }
        if let Some(index) = thresholds
            .windows(2)
            .position(|w| matches!(w, [a, b] if b < a))
        {
            return Err(LeagueViewError::NotMonotone { index: index + 1 });
        }
        let drawable = last.filter(|&l| {
            usize::try_from(l)
                .ok()
                .and_then(|i| thresholds.get(i))
                .is_some_and(|&t| t == u64::MAX)
        });
        let Some(last) = drawable else {
            return Err(LeagueViewError::NoDrawable { last });
        };
        Ok(Self {
            kinds,
            thresholds,
            last,
            population_digest,
        })
    }

    /// The policy a uniform `u64` draw `u` selects: the first entry with
    /// `u < threshold`, else the last drawable entry (the rule of
    /// `tessera_rl_league::Table::draw`).
    #[must_use]
    pub fn draw(&self, u: u64) -> u32 {
        let i = self.thresholds.partition_point(|&t| t <= u);
        if i < self.thresholds.len() {
            u32::try_from(i).unwrap_or(self.last)
        } else {
            self.last
        }
    }

    /// The kind of policy `id`, or `None` past the population.
    #[must_use]
    pub fn kind(&self, id: u32) -> Option<OpponentKind> {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.kinds.get(i))
            .copied()
    }

    /// The policies' kinds, in id order.
    #[must_use]
    pub fn kinds(&self) -> &[OpponentKind] {
        &self.kinds
    }

    /// The draw thresholds, in id order.
    #[must_use]
    pub fn thresholds(&self) -> &[u64] {
        &self.thresholds
    }

    /// The last drawable entry.
    #[must_use]
    pub const fn last(&self) -> u32 {
        self.last
    }

    /// The population digest (ADR-0088 item 3.2).
    #[must_use]
    pub const fn population_digest(&self) -> u64 {
        self.population_digest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: OpponentKind = OpponentKind::Heuristic;

    #[test]
    fn test_league_view_draw_follows_the_thresholds() {
        let v = LeagueView::new(vec![H, H, H], vec![10, 10, u64::MAX], Some(2), 7).unwrap();
        assert_eq!(v.draw(0), 0);
        assert_eq!(v.draw(9), 0);
        // Entry 1 has zero weight (same threshold as entry 0): never drawn.
        assert_eq!(v.draw(10), 2);
        assert_eq!(v.draw(u64::MAX), 2);
    }

    #[test]
    fn test_league_view_refuses_invalid_data() {
        assert_eq!(
            LeagueView::new(vec![], vec![], None, 0),
            Err(LeagueViewError::Empty)
        );
        assert_eq!(
            LeagueView::new(vec![H], vec![1, 2], Some(0), 0),
            Err(LeagueViewError::Length {
                thresholds: 2,
                kinds: 1
            })
        );
        assert_eq!(
            LeagueView::new(vec![H, H], vec![5, 3], Some(1), 0),
            Err(LeagueViewError::NotMonotone { index: 1 })
        );
        assert_eq!(
            LeagueView::new(vec![H, H], vec![5, 9], Some(1), 0),
            Err(LeagueViewError::NoDrawable { last: Some(1) })
        );
        assert_eq!(
            LeagueView::new(vec![H], vec![u64::MAX], Some(3), 0),
            Err(LeagueViewError::NoDrawable { last: Some(3) })
        );
        assert_eq!(
            LeagueView::new(vec![H], vec![u64::MAX], None, 0),
            Err(LeagueViewError::NoDrawable { last: None })
        );
    }
}
