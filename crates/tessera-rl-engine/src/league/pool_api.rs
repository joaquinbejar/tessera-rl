// SPDX-License-Identifier: MIT OR Apache-2.0

//! The exact path for records with their league assignments (ADR-0088
//! item 3.2): `snapshot_league` and `restore_with_league`, as ADR-0078's
//! `snapshot_transforms` and `restore_with_transforms` for frame stacks.

use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout};

use crate::error::EngineError;
use crate::masks::MaskProvider;
use crate::pool::{EnvPool, Poison};
use crate::record::Record;
use crate::transform::kernels::StackMode;
use crate::validate::check_ids;

/// The league assignments of some instances, with the population they
/// refer to (ADR-0088 item 3.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeagueSnapshot {
    /// One policy per instance, in the order of the ids.
    pub assignments: Vec<u32>,
    /// The digest of the population the policies belong to.
    pub population_digest: u64,
}

impl<E, M> EnvPool<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    /// The league assignments of `ids` with the population digest.
    ///
    /// # Errors
    ///
    /// [`EngineError::League`] without a league; unknown or repeated ids.
    pub fn snapshot_league(&mut self, ids: &[u32]) -> Result<LeagueSnapshot, EngineError> {
        check_ids(ids, &mut self.seen)?;
        let league = self.league.as_ref().ok_or(EngineError::League {
            reason: "the pool has no league",
        })?;
        let assignments = ids
            .iter()
            .filter_map(|&id| usize::try_from(id).ok())
            .map(|i| league.assignment.get(i).copied().unwrap_or(0))
            .collect();
        Ok(LeagueSnapshot {
            assignments,
            population_digest: league.view.population_digest(),
        })
    }

    /// Restores `records` into `ids` with their league assignments: the
    /// exact continuation, opponent included (ADR-0088 item 3.2). Every
    /// check runs before anything is written.
    ///
    /// # Errors
    ///
    /// [`EngineError::League`] without a league or with assignments of
    /// another length or past the population; [`EngineError::LeagueMismatch`]
    /// for another population digest; unknown or repeated ids, a record
    /// count other than the ids'; [`EngineError::Panicked`] if
    /// re-observing panics. Nothing is mutated on a check error.
    pub fn restore_with_league(
        &mut self,
        ids: &[u32],
        records: &[Record<E>],
        league: &LeagueSnapshot,
    ) -> Result<(), EngineError> {
        let result = self.guarded(Poison::Reject, |pool| {
            check_ids(ids, &mut pool.seen)?;
            if records.len() != ids.len() || league.assignments.len() != ids.len() {
                return Err(EngineError::RecordCount {
                    records: records.len(),
                    instances: ids.len(),
                });
            }
            let state = pool.league.as_ref().ok_or(EngineError::League {
                reason: "the pool has no league",
            })?;
            let expected = state.view.population_digest();
            if league.population_digest != expected {
                return Err(EngineError::LeagueMismatch {
                    expected,
                    found: league.population_digest,
                });
            }
            let population = state.view.kinds().len();
            if league
                .assignments
                .iter()
                .any(|&p| usize::try_from(p).map_or(true, |p| p >= population))
            {
                return Err(EngineError::League {
                    reason: "an assignment past the population",
                });
            }
            let indices: Vec<usize> = ids
                .iter()
                .filter_map(|&id| usize::try_from(id).ok())
                .collect();
            for (&i, record) in indices.iter().zip(records) {
                pool.write_record(i, record);
            }
            for (&i, &policy) in indices.iter().zip(&league.assignments) {
                let identity = pool.headers.get(i).map(|h| h.engine().identity);
                if let (Some(state), Some(identity)) = (pool.league.as_mut(), identity) {
                    state.set(i, Some(policy), &identity);
                }
            }
            pool.reobserve(indices.iter().copied(), StackMode::Fresh)?;
            pool.refresh_league();
            Ok(())
        });
        self.watch_boundary(result.is_ok());
        result
    }
}
