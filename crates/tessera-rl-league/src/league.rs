// SPDX-License-Identifier: MIT OR Apache-2.0

//! The league state a barrier updates (ADR-0088 items 4 to 6).

use tessera_rl_num::fp::with_canonical;

use crate::trueskill::{self, Params, Rating};
use crate::{LeagueError, Matchmaking, Outcome, PolicyId, Population, Results, Table, elo};

/// League parameters, fixed for the league's life.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeagueConfig {
    /// How opponents are weighted.
    pub matchmaking: Matchmaking,
    /// The Elo update factor.
    pub elo_k: f64,
    /// The `TrueSkill` parameters.
    pub trueskill: Params,
    /// Whether population entries' ratings move too, not only the learner's.
    pub rate_population: bool,
}

impl Default for LeagueConfig {
    fn default() -> Self {
        Self {
            matchmaking: Matchmaking::Pfsp(crate::PfspWeighting::Hard),
            elo_k: 16.0,
            trueskill: Params::default(),
            rate_population: false,
        }
    }
}

impl LeagueConfig {
    fn check(&self) -> Result<(), LeagueError> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if !positive(self.elo_k) {
            return Err(LeagueError::InvalidConfig("elo_k must be positive"));
        }
        let p = self.trueskill;
        let tau_ok = p.tau.is_finite() && p.tau >= 0.0;
        if !positive(p.beta) || !tau_ok {
            return Err(LeagueError::InvalidConfig(
                "trueskill beta must be positive and tau non-negative",
            ));
        }
        if !(p.draw_margin.is_finite() && p.draw_margin >= 0.0) {
            return Err(LeagueError::InvalidConfig(
                "trueskill draw_margin must be non-negative",
            ));
        }
        if let Matchmaking::Elo { temperature } = self.matchmaking
            && !positive(temperature)
        {
            return Err(LeagueError::InvalidConfig(
                "elo temperature must be positive",
            ));
        }
        Ok(())
    }
}

/// The learner's exact win, draw and loss counts against one opponent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PairCounts {
    /// Learner wins.
    pub wins: u64,
    /// Draws.
    pub draws: u64,
    /// Learner losses.
    pub losses: u64,
}

impl PairCounts {
    /// The learner's win rate with a (1 win, 1 loss) prior, draws counting
    /// half: `(wins + draws / 2 + 1) / (games + 2)`.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "counts below 2^53 are exact; larger ones round, deterministically"
    )]
    pub fn win_rate(&self) -> f64 {
        let games = self
            .wins
            .saturating_add(self.draws)
            .saturating_add(self.losses);
        (self.wins as f64 + 0.5 * self.draws as f64 + 1.0) / (games as f64 + 2.0)
    }

    fn add(&mut self, outcome: Outcome) {
        let c = match outcome {
            Outcome::Win => &mut self.wins,
            Outcome::Draw => &mut self.draws,
            Outcome::Loss => &mut self.losses,
        };
        *c = c.saturating_add(1);
    }
}

/// What one barrier did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LeagueReport {
    /// Results applied.
    pub applied: u64,
    /// Results dropped by full rings since the previous barrier.
    pub dropped: u64,
    /// Whether the new table fell back to uniform.
    pub uniform_fallback: bool,
}

/// The league state: the population, the learner's ratings, the per-pair
/// counts and the draw table.
#[derive(Debug, Clone, PartialEq)]
pub struct League {
    config: LeagueConfig,
    population: Population,
    learner_elo: f64,
    learner_trueskill: Rating,
    counts: Vec<PairCounts>,
    table: Table,
    frozen: bool,
}

impl League {
    /// A league over `population`, with the learner at the initial ratings.
    ///
    /// # Errors
    ///
    /// [`LeagueError::InvalidConfig`] for an invalid configuration and
    /// [`LeagueError::NoActivePolicy`] when no entry is active.
    pub fn new(config: LeagueConfig, population: Population) -> Result<Self, LeagueError> {
        config.check()?;
        let mut league = Self {
            config,
            counts: vec![PairCounts::default(); population.len()],
            population,
            learner_elo: elo::INITIAL,
            learner_trueskill: Rating::initial(),
            table: Table::default(),
            frozen: false,
        };
        league.table = league.compute_table()?;
        Ok(league)
    }

    /// The population.
    #[must_use]
    pub const fn population(&self) -> &Population {
        &self.population
    }

    /// The learner's Elo rating.
    #[must_use]
    pub const fn learner_elo(&self) -> f64 {
        self.learner_elo
    }

    /// The learner's `TrueSkill` rating.
    #[must_use]
    pub const fn learner_trueskill(&self) -> Rating {
        self.learner_trueskill
    }

    /// The learner's counts against each entry, in id order.
    #[must_use]
    pub fn counts(&self) -> &[PairCounts] {
        &self.counts
    }

    /// The current draw table.
    #[must_use]
    pub const fn table(&self) -> &Table {
        &self.table
    }

    /// Freezes or unfreezes the league: frozen, a barrier changes nothing.
    pub const fn set_frozen(&mut self, frozen: bool) {
        self.frozen = frozen;
    }

    /// Whether the league is frozen.
    #[must_use]
    pub const fn frozen(&self) -> bool {
        self.frozen
    }

    /// Adds a policy at a barrier, with the given initial ratings.
    ///
    /// # Errors
    ///
    /// As [`Population::add`], and as [`League::barrier`]'s table rebuild.
    pub fn add_policy(
        &mut self,
        kind: crate::PolicyKind,
        roles: u32,
    ) -> Result<PolicyId, LeagueError> {
        let id = self
            .population
            .add(kind, elo::INITIAL, Rating::initial(), roles)?;
        self.counts.push(PairCounts::default());
        self.table = self.compute_table()?;
        Ok(id)
    }

    /// Retires a policy at a barrier.
    ///
    /// # Errors
    ///
    /// [`LeagueError::UnknownPolicy`], and [`LeagueError::NoActivePolicy`]
    /// when it was the last active one (the league is then unchanged).
    pub fn retire(&mut self, id: PolicyId) -> Result<(), LeagueError> {
        let mut next = self.population.clone();
        next.retire(id)?;
        let table = self.table_for(&next)?;
        self.population = next;
        self.table = table;
        Ok(())
    }

    /// The barrier: applies `results` in instance order, then oldest-first
    /// within an instance, inside the canonical floating-point scope;
    /// recomputes the table; and clears the rings. Frozen, it only clears.
    ///
    /// # Errors
    ///
    /// [`LeagueError::UnknownPolicy`] for a result naming no entry; the
    /// league is then unchanged and the rings are kept.
    pub fn barrier(&mut self, results: &mut Results) -> Result<LeagueReport, LeagueError> {
        let dropped = results.dropped();
        if self.frozen {
            results.clear();
            return Ok(LeagueReport {
                dropped,
                uniform_fallback: self.table.uniform_fallback(),
                ..LeagueReport::default()
            });
        }
        // Validate first, so a bad id mutates nothing.
        for i in 0..results.instances() {
            for (id, _) in results.of(i) {
                self.population.get(id)?;
            }
        }
        let mut applied = 0u64;
        with_canonical(|| -> Result<(), LeagueError> {
            for i in 0..results.instances() {
                for (id, outcome) in results.of(i) {
                    self.apply(id, outcome)?;
                    applied = applied.saturating_add(1);
                }
            }
            self.table = self.compute_table()?;
            Ok(())
        })?;
        results.clear();
        Ok(LeagueReport {
            applied,
            dropped,
            uniform_fallback: self.table.uniform_fallback(),
        })
    }

    fn apply(&mut self, id: PolicyId, outcome: Outcome) -> Result<(), LeagueError> {
        let cfg = self.config;
        let opp = *self.population.get(id)?;
        let (le, oe) = elo::update(self.learner_elo, opp.elo, outcome, cfg.elo_k);
        let (lt, ot) = trueskill::update(
            self.learner_trueskill,
            opp.trueskill,
            outcome,
            cfg.trueskill,
        );
        self.learner_elo = le;
        self.learner_trueskill = lt;
        if cfg.rate_population {
            let entry = self.population.get_mut(id)?;
            entry.elo = oe;
            entry.trueskill = ot;
        }
        if let Some(c) = self.counts.get_mut(id.index()) {
            c.add(outcome);
        }
        Ok(())
    }

    fn compute_table(&self) -> Result<Table, LeagueError> {
        self.table_for(&self.population)
    }

    fn table_for(&self, population: &Population) -> Result<Table, LeagueError> {
        with_canonical(|| {
            let weights: Vec<Option<f64>> = population
                .entries()
                .iter()
                .zip(
                    self.counts
                        .iter()
                        .chain(core::iter::repeat(&PairCounts::default())),
                )
                .map(|(p, c)| {
                    (!p.retired).then(|| {
                        self.config
                            .matchmaking
                            .weight(c.win_rate(), self.learner_elo, p.elo)
                    })
                })
                .collect();
            Table::from_weights(&weights)
        })
    }
}
