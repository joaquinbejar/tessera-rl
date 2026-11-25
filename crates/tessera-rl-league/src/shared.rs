// SPDX-License-Identifier: MIT OR Apache-2.0

//! A league shared by several pools (ADR-0088 section 6; #384 head E1,
//! plan r2, option A): one population, one learner per role, and a
//! two-phase barrier that does not depend on thread timing.
//!
//! Every pool registers with [`SharedLeague::member`], which gives it the
//! next registration index and a [`RoleHandle`], the pool's league
//! subsystem. A pool's barrier in round `r` puts its results into its slot
//! and returns the view committed at round `r - 1` (the initial view in
//! round 0). When the last registered slot of the round arrives, from any
//! thread, one commit merges the slots in registration order (each in the
//! engine's instance then oldest-first order), applies the snapshot rules
//! and recomputes every role's table. Every pool installs that view at its
//! next barrier, inside its own `epoch_barrier`, so a recorder writes it
//! as a `League` event there (head D). A shared league therefore lags its
//! pools' results by one round, whatever the number of pools, one
//! included; the unshared [`crate::PoolLeague`] does not lag.
//!
//! A barrier of a pool that already submitted the current round is
//! refused with a typed error naming the pools still to submit; the pool
//! keeps its view and rings, unpoisoned, and nothing ever blocks. A pool
//! frozen with `freeze_league` submits an empty contribution at each
//! barrier (`LeagueSubsystem::frozen_barrier`) and keeps its view, so it
//! never stalls the other pools' rounds. The lock
//! is taken only at barriers, never on the step path.

use std::sync::{Arc, Mutex};

use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout};
use tessera_rl_engine::league::{
    LeagueBarrierError, LeagueReport as EngineReport, LeagueResults, LeagueSubsystem, LeagueView,
    MatchOutcome, OpponentKind, sealed,
};
use tessera_rl_engine::{EngineError, EnvPool, EpochReport, MaskProvider};
use tessera_rl_num::fp::with_canonical;

use crate::roles::{RoleConfig, SnapshotRule, serves};
use crate::trueskill::{self, Params, Rating};
use crate::{
    LeagueConfig, LeagueError, Matchmaking, Outcome, PairCounts, PolicyId, PolicyKind, Population,
    Table, elo,
};

/// A shared league's parameters, fixed for its life.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedConfig {
    /// The Elo update factor.
    pub elo_k: f64,
    /// The `TrueSkill` parameters.
    pub trueskill: Params,
    /// Whether population entries' ratings move too, not only the
    /// learners'.
    pub rate_population: bool,
    /// The roles, at most 32; a role's index is its role bit.
    pub roles: Vec<RoleConfig>,
}

/// A snapshot a commit added: the trainer saves the role's learner under
/// `tag` (ADR-0018: the engine never trains).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Snapshot {
    /// The role.
    pub role: usize,
    /// The new entry.
    pub id: PolicyId,
    /// Its `External` tag.
    pub tag: u64,
}

/// What one commit did.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CommitReport {
    /// The round committed.
    pub round: u64,
    /// Results applied, every pool.
    pub applied: u64,
    /// Results the pools' rings dropped.
    pub dropped: u64,
    /// Snapshots added, in role order.
    pub snapshots: Vec<Snapshot>,
}

/// One role's learner and table.
#[derive(Debug)]
struct Role {
    cfg: RoleConfig,
    elo: f64,
    trueskill: Rating,
    counts: Vec<PairCounts>,
    self_play: Option<PolicyId>,
    rounds: u64,
    snapshots: u64,
    since: PairCounts,
    view: LeagueView,
    fallback: bool,
}

#[derive(Debug)]
struct Inner {
    elo_k: f64,
    trueskill: Params,
    rate_population: bool,
    frozen: bool,
    population: Population,
    roles: Vec<Role>,
    /// Per registered pool: its role.
    members: Vec<usize>,
    round: u64,
    /// Per registered pool: its results this round, once submitted.
    slots: Vec<Option<Vec<(PolicyId, Outcome)>>>,
    dropped: u64,
    last: Option<CommitReport>,
}

/// A league shared by several pools.
#[derive(Debug, Clone)]
pub struct SharedLeague {
    inner: Arc<Mutex<Inner>>,
}

/// One pool's league subsystem in a [`SharedLeague`]: hand it, with
/// [`SharedLeague::view`] of its role, to `EnvPool::with_league`.
#[derive(Debug)]
pub struct RoleHandle {
    inner: Arc<Mutex<Inner>>,
    index: usize,
    role: usize,
}

const fn kind_of(kind: PolicyKind) -> OpponentKind {
    match kind {
        PolicyKind::Heuristic => OpponentKind::Heuristic,
        PolicyKind::External { tag } => OpponentKind::External { tag },
        PolicyKind::Onnx { content_hash } => OpponentKind::Onnx { content_hash },
    }
}

impl SharedLeague {
    /// A shared league over `population`. Each role with self-play adds its
    /// `External` self-play entry, with that role's bit only, in role order.
    ///
    /// # Errors
    ///
    /// [`LeagueError::InvalidConfig`] for no role, more than 32, or an
    /// invalid rating or role parameter; [`LeagueError::NoActivePolicy`]
    /// when a role has nothing to draw; [`LeagueError::View`].
    pub fn new(config: SharedConfig, mut population: Population) -> Result<Self, LeagueError> {
        let SharedConfig {
            elo_k,
            trueskill,
            rate_population,
            roles: configs,
        } = config;
        if configs.is_empty() || configs.len() > 32 {
            return Err(LeagueError::InvalidConfig(
                "a shared league has 1 to 32 roles",
            ));
        }
        for role in &configs {
            role.check()?;
            LeagueConfig {
                matchmaking: role.matchmaking,
                elo_k,
                trueskill,
                rate_population,
            }
            .check()?;
        }
        let mut self_play = Vec::with_capacity(configs.len());
        for (r, role) in configs.iter().enumerate() {
            self_play.push(match role.self_play {
                Some(s) => Some(population.add(
                    PolicyKind::External { tag: s.tag },
                    elo::INITIAL,
                    Rating::initial(),
                    1u32 << r,
                )?),
                None => None,
            });
        }
        let mut roles = Vec::with_capacity(configs.len());
        for (cfg, self_play) in configs.iter().zip(self_play) {
            let mut role = Role {
                cfg: *cfg,
                elo: elo::INITIAL,
                trueskill: Rating::initial(),
                counts: vec![PairCounts::default(); population.len()],
                self_play,
                rounds: 0,
                snapshots: 0,
                since: PairCounts::default(),
                view: placeholder()?,
                fallback: false,
            };
            role_view(&mut role, &population, roles.len())?;
            roles.push(role);
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                elo_k,
                trueskill,
                rate_population,
                frozen: false,
                population,
                roles,
                members: Vec::new(),
                round: 0,
                slots: Vec::new(),
                dropped: 0,
                last: None,
            })),
        })
    }

    /// Registers a pool of `role`: the next registration index. Every pool
    /// registers before the first barrier.
    ///
    /// # Errors
    ///
    /// [`LeagueError::InvalidConfig`] for an unknown role, or after the
    /// first barrier.
    pub fn member(&self, role: usize) -> Result<RoleHandle, LeagueError> {
        let mut inner = self.lock()?;
        if role >= inner.roles.len() {
            return Err(LeagueError::InvalidConfig("unknown role"));
        }
        if inner.round != 0 || inner.slots.iter().any(Option::is_some) {
            return Err(LeagueError::InvalidConfig(
                "pools register before the first barrier",
            ));
        }
        let index = inner.members.len();
        inner.members.push(role);
        inner.slots.push(None);
        Ok(RoleHandle {
            inner: Arc::clone(&self.inner),
            index,
            role,
        })
    }

    /// The current view of `role`, for `with_league`.
    ///
    /// # Errors
    ///
    /// [`LeagueError::InvalidConfig`] for an unknown role.
    pub fn view(&self, role: usize) -> Result<LeagueView, LeagueError> {
        let inner = self.lock()?;
        inner
            .roles
            .get(role)
            .map(|r| r.view.clone())
            .ok_or(LeagueError::InvalidConfig("unknown role"))
    }

    /// The population.
    ///
    /// # Errors
    ///
    /// [`LeagueError::InvalidConfig`] if a thread panicked holding the
    /// league.
    pub fn population(&self) -> Result<Population, LeagueError> {
        Ok(self.lock()?.population.clone())
    }

    /// `role`'s learner Elo and `TrueSkill` ratings, its counts against
    /// each entry, and its table.
    ///
    /// # Errors
    ///
    /// [`LeagueError::InvalidConfig`] for an unknown role.
    pub fn role(
        &self,
        role: usize,
    ) -> Result<(f64, Rating, Vec<PairCounts>, LeagueView), LeagueError> {
        let inner = self.lock()?;
        let r = inner
            .roles
            .get(role)
            .ok_or(LeagueError::InvalidConfig("unknown role"))?;
        Ok((r.elo, r.trueskill, r.counts.clone(), r.view.clone()))
    }

    /// The last commit's report.
    ///
    /// # Errors
    ///
    /// As [`SharedLeague::population`].
    pub fn last_commit(&self) -> Result<Option<CommitReport>, LeagueError> {
        Ok(self.lock()?.last.clone())
    }

    /// The round the pools' next barriers submit to.
    ///
    /// # Errors
    ///
    /// As [`SharedLeague::population`].
    pub fn round(&self) -> Result<u64, LeagueError> {
        Ok(self.lock()?.round)
    }

    /// Freezes or unfreezes the league: frozen, a commit applies nothing
    /// and changes no view. (A pool frozen alone with `freeze_league`
    /// submits empty rounds and keeps its view; the others go on.)
    ///
    /// # Errors
    ///
    /// As [`SharedLeague::population`].
    pub fn set_frozen(&self, frozen: bool) -> Result<(), LeagueError> {
        self.lock()?.frozen = frozen;
        Ok(())
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Inner>, LeagueError> {
        self.inner
            .lock()
            .map_err(|_| LeagueError::InvalidConfig("a thread panicked holding the league"))
    }
}

/// A pool that takes part in a [`SharedLeague`]'s barrier: an `EnvPool`
/// whose league subsystem is a [`RoleHandle`].
pub trait LeagueMember {
    /// The pool's epoch barrier.
    ///
    /// # Errors
    ///
    /// The pool's barrier error.
    fn epoch_barrier(&mut self) -> Result<EpochReport, EngineError>;
}

impl<E, M> LeagueMember for EnvPool<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    fn epoch_barrier(&mut self) -> Result<EpochReport, EngineError> {
        Self::epoch_barrier(self)
    }
}

impl SharedLeague {
    /// Runs every member's epoch barrier, in the order given (ADR-0088
    /// item 6.2): any order gives the same result, since the commit merges
    /// in registration order. The members must be every registered pool
    /// for the round to commit.
    #[must_use]
    pub fn barrier(members: &mut [&mut dyn LeagueMember]) -> Vec<Result<EpochReport, EngineError>> {
        members.iter_mut().map(|m| m.epoch_barrier()).collect()
    }
}

/// A one-entry view, replaced before use.
fn placeholder() -> Result<LeagueView, LeagueError> {
    Ok(LeagueView::new(
        vec![OpponentKind::Heuristic],
        vec![u64::MAX],
        Some(0),
        0,
    )?)
}

/// Recomputes `role`'s table and view over `population`.
fn role_view(role: &mut Role, population: &Population, index: usize) -> Result<(), LeagueError> {
    // The weights and the table in the canonical FP scope (ADR-0088 4.5).
    let mut table = with_canonical(|| {
        let weights: Vec<Option<f64>> = population
            .entries()
            .iter()
            .enumerate()
            .zip(
                role.counts
                    .iter()
                    .chain(core::iter::repeat(&PairCounts::default())),
            )
            .map(|((j, p), c)| {
                let own = role.self_play.is_some_and(|s| s.index() == j);
                (!p.retired && !own && serves(p.roles, index))
                    .then(|| weight(role.cfg.matchmaking, c, role.elo, p.elo))
            })
            .collect();
        Table::from_weights(&weights)
    })?;
    if let (Some(id), Some(s)) = (role.self_play, role.cfg.self_play) {
        table = table.with_reserved_share(id, s.share)?;
    }
    role.fallback = table.uniform_fallback();
    role.view = LeagueView::new(
        population
            .entries()
            .iter()
            .map(|p| kind_of(p.kind))
            .collect(),
        table.thresholds().to_vec(),
        table.last(),
        population.digest(),
    )?;
    Ok(())
}

fn weight(m: Matchmaking, c: &PairCounts, learner: f64, opponent: f64) -> f64 {
    m.weight(c.win_rate(), learner, opponent)
}

impl Inner {
    fn pending(&self) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.is_none().then_some(i))
            .collect()
    }

    /// Merges every slot in registration order and recomputes the views.
    fn commit(&mut self) -> Result<(), LeagueError> {
        let round = self.round;
        let slots: Vec<_> = self.slots.iter_mut().map(Option::take).collect();
        let dropped = core::mem::take(&mut self.dropped);
        self.round = round.saturating_add(1);
        if self.frozen {
            self.last = Some(CommitReport {
                round,
                dropped,
                ..CommitReport::default()
            });
            return Ok(());
        }
        let mut applied = 0u64;
        let mut snapshots = Vec::new();
        with_canonical(|| -> Result<(), LeagueError> {
            for (member, slot) in slots.iter().enumerate() {
                let role = self.members.get(member).copied().unwrap_or(0);
                for &(id, outcome) in slot.iter().flatten() {
                    self.apply(role, id, outcome)?;
                    applied = applied.saturating_add(1);
                }
            }
            for r in 0..self.roles.len() {
                if let Some(s) = self.snapshot(r)? {
                    snapshots.push(s);
                }
            }
            Ok(())
        })?;
        for r in 0..self.roles.len() {
            let len = self.population.len();
            if let Some(role) = self.roles.get_mut(r) {
                role.counts.resize(len, PairCounts::default());
                role_view(role, &self.population, r)?;
            }
        }
        self.last = Some(CommitReport {
            round,
            applied,
            dropped,
            snapshots,
        });
        Ok(())
    }

    fn apply(&mut self, role: usize, id: PolicyId, outcome: Outcome) -> Result<(), LeagueError> {
        let opp = *self.population.get(id)?;
        let (k, ts, rate) = (self.elo_k, self.trueskill, self.rate_population);
        let r = self
            .roles
            .get_mut(role)
            .ok_or(LeagueError::InvalidConfig("unknown role"))?;
        if let Some(c) = r.counts.get_mut(id.index()) {
            c.add(outcome);
        }
        // Self-play counts in its pair counts only, never in the ratings.
        if r.self_play == Some(id) {
            return Ok(());
        }
        r.since.add(outcome);
        let (le, oe) = elo::update(r.elo, opp.elo, outcome, k);
        let (lt, ot) = trueskill::update(r.trueskill, opp.trueskill, outcome, ts);
        r.elo = le;
        r.trueskill = lt;
        if rate {
            let entry = self.population.get_mut(id)?;
            entry.elo = oe;
            entry.trueskill = ot;
        }
        Ok(())
    }

    /// `role`'s snapshot rule, after its results.
    #[expect(
        clippy::cast_precision_loss,
        reason = "game counts below 2^53 are exact; larger ones round, deterministically"
    )]
    fn snapshot(&mut self, role: usize) -> Result<Option<Snapshot>, LeagueError> {
        let Some(r) = self.roles.get_mut(role) else {
            return Ok(None);
        };
        r.rounds = r.rounds.saturating_add(1);
        let fire = match r.cfg.snapshot {
            None => false,
            Some(SnapshotRule::EveryRounds(n)) => r.rounds.is_multiple_of(u64::from(n)),
            Some(SnapshotRule::WinRate {
                threshold,
                min_games,
            }) => {
                let s = r.since;
                let games = s.wins.saturating_add(s.draws).saturating_add(s.losses);
                games >= min_games.max(1)
                    && (s.wins as f64 + 0.5 * s.draws as f64) / games as f64 >= threshold
            }
        };
        if !fire {
            return Ok(None);
        }
        let tag = r.cfg.snapshot_tags.wrapping_add(r.snapshots);
        let (elo, rating) = (r.elo, r.trueskill);
        r.snapshots = r.snapshots.saturating_add(1);
        r.since = PairCounts::default();
        let id = self
            .population
            .add(PolicyKind::External { tag }, elo, rating, 0)?;
        Ok(Some(Snapshot { role, id, tag }))
    }
}

impl sealed::Sealed for RoleHandle {}

impl RoleHandle {
    /// Puts `slot` into this pool's slot of the current round and commits
    /// the round if it was the last: the refusal of an early barrier first.
    fn submit(
        inner: &mut Inner,
        index: usize,
        slot: Vec<(PolicyId, Outcome)>,
        dropped: u64,
    ) -> Result<(), LeagueBarrierError> {
        let fail = |reason: String| LeagueBarrierError { reason };
        if inner.slots.get(index).is_some_and(Option::is_some) {
            return Err(fail(format!(
                "pool {index} already submitted round {}: waiting for pools {:?}",
                inner.round,
                inner.pending()
            )));
        }
        if let Some(s) = inner.slots.get_mut(index) {
            *s = Some(slot);
        }
        inner.dropped = inner.dropped.saturating_add(dropped);
        if inner.pending().is_empty() {
            inner.commit().map_err(|e| fail(e.to_string()))?;
        }
        Ok(())
    }
}

impl LeagueSubsystem for RoleHandle {
    /// A frozen pool's round is an empty contribution: the other pools'
    /// round still commits, and the frozen pool keeps its view.
    fn frozen_barrier(&mut self) -> Result<(), LeagueBarrierError> {
        let mut inner = self.inner.lock().map_err(|_| LeagueBarrierError {
            reason: "a thread panicked holding the league".to_owned(),
        })?;
        Self::submit(&mut inner, self.index, Vec::new(), 0)
    }

    fn barrier(
        &mut self,
        results: LeagueResults<'_>,
    ) -> Result<(LeagueView, EngineReport), LeagueBarrierError> {
        let fail = |reason: String| LeagueBarrierError { reason };
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| fail("a thread panicked holding the league".to_owned()))?;
        // Checked before anything is kept, so a bad id mutates nothing.
        let population = inner.population.len();
        let mut slot = Vec::with_capacity(results.results.len());
        for r in results.results {
            if usize::try_from(r.policy).map_or(true, |p| p >= population) {
                return Err(fail(format!("unknown policy id {}", r.policy)));
            }
            let outcome = match r.outcome {
                MatchOutcome::Win => Outcome::Win,
                MatchOutcome::Draw => Outcome::Draw,
                MatchOutcome::Loss => Outcome::Loss,
            };
            slot.push((PolicyId(r.policy), outcome));
        }
        let submitted = u64::try_from(slot.len()).unwrap_or(u64::MAX);
        // The view committed at the previous round (one-round lag).
        let role = inner
            .roles
            .get(self.role)
            .ok_or_else(|| fail("unknown role".to_owned()))?;
        let (view, fallback) = (role.view.clone(), role.fallback);
        Self::submit(&mut inner, self.index, slot, results.dropped)?;
        Ok((
            view,
            EngineReport::new(submitted, results.dropped, fallback),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roles::SelfPlay;

    /// Self-play results count in the self-play entry's pair counts only;
    /// a result against another entry moves the learner's ratings.
    #[test]
    fn test_shared_self_play_results_stay_out_of_the_ratings() {
        let mut population = Population::new();
        let other = population
            .add(PolicyKind::Heuristic, elo::INITIAL, Rating::initial(), 0)
            .unwrap();
        let role = RoleConfig {
            self_play: Some(SelfPlay {
                share: 0.5,
                tag: 77,
            }),
            ..RoleConfig::new(Matchmaking::Uniform)
        };
        let league = SharedLeague::new(
            SharedConfig {
                elo_k: 16.0,
                trueskill: Params::default(),
                rate_population: true,
                roles: vec![role],
            },
            population,
        )
        .unwrap();
        let mut inner = league.lock().unwrap();
        let own = inner.roles[0].self_play.unwrap();
        inner.apply(0, own, Outcome::Win).unwrap();
        let r = &inner.roles[0];
        assert_eq!(r.elo.to_bits(), elo::INITIAL.to_bits());
        assert_eq!(r.trueskill, Rating::initial());
        assert_eq!(r.counts[own.index()].wins, 1);
        assert_eq!(r.since, PairCounts::default(), "not in the snapshot tally");
        inner.apply(0, other, Outcome::Win).unwrap();
        assert!(inner.roles[0].elo > elo::INITIAL);
    }

    fn two_roles() -> SharedLeague {
        let mut population = Population::new();
        population
            .add(PolicyKind::Heuristic, elo::INITIAL, Rating::initial(), 0)
            .unwrap();
        SharedLeague::new(
            SharedConfig {
                elo_k: 16.0,
                trueskill: Params::default(),
                rate_population: true,
                roles: vec![RoleConfig::new(Matchmaking::Uniform); 2],
            },
            population,
        )
        .unwrap()
    }

    /// The commit merges slots in registration order: pool 0's win, then
    /// pool 1's loss, against the shared entry, exactly as sequential Elo
    /// updates in that order (the population entry's rating carries over).
    #[test]
    fn test_shared_commit_merges_in_registration_order() {
        let league = two_roles();
        let _a = league.member(0).unwrap();
        let _b = league.member(1).unwrap();
        let mut inner = league.lock().unwrap();
        inner.slots[0] = Some(vec![(PolicyId(0), Outcome::Win)]);
        inner.slots[1] = Some(vec![(PolicyId(0), Outcome::Loss)]);
        inner.commit().unwrap();
        let (l0, o0) = elo::update(elo::INITIAL, elo::INITIAL, Outcome::Win, 16.0);
        let (l1, o1) = elo::update(elo::INITIAL, o0, Outcome::Loss, 16.0);
        assert_eq!(inner.roles[0].elo.to_bits(), l0.to_bits());
        assert_eq!(inner.roles[1].elo.to_bits(), l1.to_bits());
        assert_eq!(inner.population.entries()[0].elo.to_bits(), o1.to_bits());
        assert_eq!(inner.round, 1);
    }

    /// Frozen, a commit applies nothing and changes no view.
    #[test]
    fn test_shared_frozen_commit_changes_nothing() {
        let league = two_roles();
        let _a = league.member(0).unwrap();
        league.set_frozen(true).unwrap();
        let before = league.view(0).unwrap();
        let mut inner = league.lock().unwrap();
        inner.slots[0] = Some(vec![(PolicyId(0), Outcome::Win)]);
        inner.commit().unwrap();
        assert_eq!(inner.roles[0].elo.to_bits(), elo::INITIAL.to_bits());
        assert_eq!(inner.roles[0].counts[0], PairCounts::default());
        assert_eq!(inner.roles[0].view, before);
        assert_eq!(inner.round, 1);
    }

    /// A role's self-play entry is drawn by that role only.
    #[test]
    fn test_shared_self_play_entry_serves_its_role_only() {
        let mut population = Population::new();
        population
            .add(PolicyKind::Heuristic, elo::INITIAL, Rating::initial(), 0)
            .unwrap();
        let with_self = RoleConfig {
            self_play: Some(SelfPlay {
                share: 0.5,
                tag: 77,
            }),
            ..RoleConfig::new(Matchmaking::Uniform)
        };
        let league = SharedLeague::new(
            SharedConfig {
                elo_k: 16.0,
                trueskill: Params::default(),
                rate_population: false,
                roles: vec![with_self, RoleConfig::new(Matchmaking::Uniform)],
            },
            population,
        )
        .unwrap();
        let other = league.view(1).unwrap();
        // Entry 1 is role 0's self-play entry: zero width for role 1.
        assert_eq!(other.thresholds()[1], other.thresholds()[0]);
        assert_eq!(other.last(), 0);
    }
}
