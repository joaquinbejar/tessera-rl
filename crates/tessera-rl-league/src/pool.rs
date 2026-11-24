// SPDX-License-Identifier: MIT OR Apache-2.0

//! A league attached to one engine pool (ADR-0088 sections 2 and 7,
//! erratum 1): [`PoolLeague`] implements the engine's league subsystem and
//! builds the plain [`LeagueView`] the pool draws assignments from.

use tessera_rl_engine::league::{
    LeagueBarrierError, LeagueReport as EngineReport, LeagueResults, LeagueSubsystem, LeagueView,
    MatchOutcome, OpponentKind, sealed,
};

use crate::{League, LeagueError, Outcome, PolicyId, PolicyKind, Results};

/// A [`League`] serving one pool: hand [`PoolLeague::view`] and the boxed
/// league to `EnvPool::with_league` (or `with_info_league`).
#[derive(Debug)]
pub struct PoolLeague {
    league: League,
}

impl PoolLeague {
    /// The league of one pool.
    ///
    /// # Errors
    ///
    /// [`LeagueError::OnnxUnsupported`] for a population with an ONNX
    /// policy (head C).
    pub fn new(league: League) -> Result<Self, LeagueError> {
        view_of(&league)?;
        Ok(Self { league })
    }

    /// The view of the league's current table, for `with_league`.
    ///
    /// # Errors
    ///
    /// [`LeagueError::OnnxUnsupported`] for an ONNX policy;
    /// [`LeagueError::View`] if the table does not make a valid view.
    pub fn view(&self) -> Result<LeagueView, LeagueError> {
        view_of(&self.league)
    }

    /// The league.
    #[must_use]
    pub const fn league(&self) -> &League {
        &self.league
    }
}

/// The engine view of `league`'s population and table (ADR-0088 erratum 1).
fn view_of(league: &League) -> Result<LeagueView, LeagueError> {
    let kinds = league
        .population()
        .entries()
        .iter()
        .enumerate()
        .map(|(i, p)| match p.kind {
            PolicyKind::Heuristic => Ok(OpponentKind::Heuristic),
            PolicyKind::External { tag } => Ok(OpponentKind::External { tag }),
            PolicyKind::Onnx { .. } => Err(LeagueError::OnnxUnsupported(
                u32::try_from(i).unwrap_or(u32::MAX),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let table = league.table();
    Ok(LeagueView::new(
        kinds,
        table.thresholds().to_vec(),
        table.last(),
        league.population().digest(),
    )?)
}

impl sealed::Sealed for PoolLeague {}

impl LeagueSubsystem for PoolLeague {
    fn barrier(
        &mut self,
        results: LeagueResults<'_>,
    ) -> Result<(LeagueView, EngineReport), LeagueBarrierError> {
        let fail = |e: LeagueError| LeagueBarrierError {
            reason: e.to_string(),
        };
        // The engine's results, already in instance then oldest-first
        // order, into one ring per instance in that order.
        let instances = results
            .results
            .iter()
            .map(|r| r.instance as usize + 1)
            .max()
            .unwrap_or(1);
        let mut per = vec![0usize; instances];
        for r in results.results {
            if let Some(n) = per.get_mut(r.instance as usize) {
                *n += 1;
            }
        }
        let capacity = per.iter().copied().max().unwrap_or(0).max(1);
        let mut rings = Results::new(instances, capacity).map_err(fail)?;
        for r in results.results {
            let outcome = match r.outcome {
                MatchOutcome::Win => Outcome::Win,
                MatchOutcome::Draw => Outcome::Draw,
                MatchOutcome::Loss => Outcome::Loss,
            };
            rings
                .record(r.instance as usize, PolicyId(r.policy), outcome)
                .map_err(fail)?;
        }
        let report = self.league.barrier(&mut rings).map_err(fail)?;
        let view = view_of(&self.league).map_err(fail)?;
        Ok((
            view,
            EngineReport::new(
                report.applied,
                report.dropped.saturating_add(results.dropped),
                report.uniform_fallback,
            ),
        ))
    }
}
