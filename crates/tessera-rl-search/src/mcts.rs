// SPDX-License-Identifier: MIT OR Apache-2.0

//! The batched PUCT helper (ADR-0043 sections 2 to 4) over a
//! [`SearchPool`].
//!
//! One [`Mcts::run`] call:
//! 1. evaluates the root if it has no priors yet, and mixes root noise if
//!    configured (into a per-call copy of the root priors, never into the
//!    tree);
//! 2. repeats, until the simulation budget is spent: collect up to
//!    `batch_size` leaves by repeated selection from the root, applying
//!    virtual loss along each path; step the new children in one batch;
//!    evaluate the non-terminal ones in batch order; commit the children;
//!    back every leaf up in batch order, removing its virtual loss.
//!
//! Every write of the call is journaled: a failed call (an evaluator
//! error, an exhausted counter, a full pool) restores the tree and the
//! call counter exactly.

use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout};
use tessera_rl_engine::{MaskProvider, Record};
use tessera_rl_num::identity::{RngIdentity, StreamId};
use tessera_rl_num::math;
use tessera_rl_num::rng::{RngAuthority, StepRng};

use crate::error::SearchError;
use crate::evaluator::{Evaluator, LeafBatch, validate_leaf};
use crate::gumbel::{Gumbel, RootEdge, RootSelection};
use crate::handle::NodeHandle;
use crate::noise::dirichlet;
use crate::pool::{NONE, SearchPool};

/// The PUCT score of an edge (ADR-0043 section 2), with `N` and `VL` the
/// edge's integer visit and virtual-loss counts and `W` its total value
/// from the view of the player to move at the parent:
///
/// ```text
/// Q = fpu                      if N + VL = 0
///   = (W - VL) / (N + VL)      otherwise (each virtual loss is a loss, -1)
/// U = c_puct * P * sqrt(N(s)) / (1 + N + VL),  N(s) = sum over edges of (N + VL)
/// score = Q + U;  ties go to the lowest action.
/// ```
///
/// `sqrt` is the IEEE square root. Removing a virtual loss restores `N`
/// and `W` exactly, because both counts are integers and `W` never
/// absorbs them.
pub const PUCT_FORMULA: &str =
    "Q + c_puct * P * sqrt(N(s)) / (1 + N + VL), Q = (W - VL) / (N + VL)";

/// Root noise: `(1 - epsilon) * P + epsilon * Dirichlet(alpha)` over the
/// legal root actions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootNoise {
    /// Dirichlet concentration, `> 0`.
    pub alpha: f64,
    /// Mixing weight, in `[0, 1]`.
    pub epsilon: f64,
}

/// PUCT configuration (ADR-0043).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PuctConfig {
    /// Exploration constant (default 1.25).
    pub c_puct: f64,
    /// First-play urgency: `Q` of an edge with no visits (default 0).
    pub fpu: f64,
    /// Virtual losses applied per selection (default 1).
    pub virtual_loss: u32,
    /// Leaves per batch, at most the pool's batch (default 8).
    pub batch_size: u32,
    /// Simulations per call (default 64).
    pub simulations: u32,
    /// Root noise, off by default.
    pub noise: Option<RootNoise>,
    /// Root action selection: PUCT (the default) or the Gumbel root,
    /// which excludes `noise`.
    pub root: RootSelection,
}

impl Default for PuctConfig {
    fn default() -> Self {
        Self {
            c_puct: 1.25,
            fpu: 0.0,
            virtual_loss: 1,
            batch_size: 8,
            simulations: 64,
            noise: None,
            root: RootSelection::Puct,
        }
    }
}

/// One collected leaf.
#[derive(Debug, Clone, Copy)]
enum Leaf {
    /// An unexpanded edge: its parent slot, action and staging row.
    Edge {
        parent: u32,
        action: u32,
        row: usize,
    },
    /// A terminal node reached by selection.
    Terminal { slot: u32 },
}

/// The counter and search stream of a call in progress.
#[derive(Debug, Clone)]
pub(crate) struct CallStart {
    pub(crate) calls: u64,
    pub(crate) rng: StepRng,
}

/// The gathered leaves of one tree, for a group batch.
pub(crate) struct Gathered<'a, E: Environment> {
    pub(crate) records: &'a [Record<E>],
    pub(crate) picks: &'a [(usize, usize)],
    pub(crate) legal: &'a [bool],
    pub(crate) source: crate::pool::ObsSource<'a>,
}

/// A PUCT search over a [`SearchPool`].
pub struct Mcts<E: Environment, M: MaskProvider<E>> {
    pool: SearchPool<E, M>,
    cfg: PuctConfig,
    /// Search calls so far: the key of each call's search stream.
    calls: u64,
    rng: Option<StepRng>,
    leaves: Vec<Leaf>,
    requests: Vec<(NodeHandle, u32)>,
    children: Vec<NodeHandle>,
    eval_records: Vec<Record<E>>,
    eval_rows: Vec<usize>,
    /// Each evaluated leaf's scratch row and player to move.
    eval_picks: Vec<(usize, usize)>,
    raw_priors: Vec<f64>,
    values: Vec<f64>,
    priors: Vec<f64>,
    root_priors: Vec<f64>,
    noise: Vec<f64>,
    legal: Vec<bool>,
    /// Bytes of one leaf's observation.
    obs_bytes: usize,
    /// The batch's observations, then legal actions, for the evaluator.
    obs_buf: Vec<u8>,
    leaf_legal: Vec<bool>,
    /// Whether the gathered batch is the root's (its legal actions are in
    /// `legal`, not `leaf_legal`).
    root_batch: bool,
    /// The current root's evaluator value, keyed by the root node's
    /// allocation number (current root only), and its copy at the start
    /// of the call in progress (restored on rollback).
    v_root: Option<(u64, f64)>,
    saved_v_root: Option<(u64, f64)>,
    /// The Gumbel root's state, under [`RootSelection::Gumbel`] only.
    gumbel: Option<Gumbel>,
}

fn idx(v: u32) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}

fn count(n: u32) -> f64 {
    f64::from(n)
}

/// The root edge statistics of action index `a`.
fn root_edge<E, M>(pool: &SearchPool<E, M>, a: usize) -> RootEdge
where
    E: Environment<
            Config: Sync,
            State: Send,
            Header: Send + RecordLayout<Adaptive: AdaptiveValues<E::Config>>,
        >,
    M: MaskProvider<E>,
{
    let e = u32::try_from(a).ok().and_then(|a| pool.edge(pool.root, a));
    e.map_or(
        RootEdge {
            visits: 0,
            value: 0.0,
            virtual_loss: 0,
        },
        |e| RootEdge {
            visits: e.visits,
            value: e.value,
            virtual_loss: e.virtual_loss,
        },
    )
}

fn config_error(option: &'static str, reason: &'static str) -> SearchError {
    SearchError::Options { option, reason }
}

impl<E, M> Mcts<E, M>
where
    E: Environment<
            Config: Sync,
            State: Send,
            Header: Send + RecordLayout<Adaptive: AdaptiveValues<E::Config>>,
        >,
    M: MaskProvider<E>,
{
    /// A search over `pool` with `cfg`.
    ///
    /// # Errors
    ///
    /// An invalid configuration: a non-finite `c_puct` or `fpu`, a batch
    /// of 0 or above the pool's, no simulations, or invalid noise; under
    /// the Gumbel root, `m = 0`, a negative or non-finite `c_visit` or
    /// `c_scale`, or root noise (the two are refused together).
    pub fn new(pool: SearchPool<E, M>, cfg: PuctConfig) -> Result<Self, SearchError> {
        if !cfg.c_puct.is_finite() || cfg.c_puct < 0.0 {
            return Err(config_error("c_puct", "must be finite and non-negative"));
        }
        if !cfg.fpu.is_finite() {
            return Err(config_error("fpu", "must be finite"));
        }
        if cfg.batch_size == 0 || cfg.batch_size > pool.batch() {
            return Err(config_error(
                "batch_size",
                "must be in 1..=the pool's batch",
            ));
        }
        if cfg.simulations == 0 {
            return Err(config_error("simulations", "must be at least 1"));
        }
        if let Some(n) = cfg.noise
            && !(n.alpha.is_finite() && n.alpha > 0.0 && (0.0..=1.0).contains(&n.epsilon))
        {
            return Err(config_error("noise", "alpha > 0 and epsilon in [0, 1]"));
        }
        let gumbel = match cfg.root {
            RootSelection::Gumbel(g) => {
                if cfg.noise.is_some() {
                    return Err(config_error(
                        "root",
                        "the Gumbel root excludes Dirichlet root noise",
                    ));
                }
                if g.m == 0 {
                    return Err(config_error("gumbel.m", "must be at least 1"));
                }
                if !(g.c_visit.is_finite() && g.c_visit >= 0.0) {
                    return Err(config_error(
                        "gumbel.c_visit",
                        "must be finite and non-negative",
                    ));
                }
                if !(g.c_scale.is_finite() && g.c_scale >= 0.0) {
                    return Err(config_error(
                        "gumbel.c_scale",
                        "must be finite and non-negative",
                    ));
                }
                Some(Gumbel::new(g, idx(pool.actions()), cfg.simulations))
            }
            RootSelection::Puct => None,
        };
        let (b, a) = (idx(cfg.batch_size), idx(pool.actions()));
        let obs_bytes = pool.obs_row_bytes();
        Ok(Self {
            obs_bytes,
            obs_buf: vec![0; b.max(1) * obs_bytes],
            leaf_legal: vec![false; b.max(1) * a],
            root_batch: false,
            cfg,
            calls: 0,
            rng: None,
            leaves: Vec::with_capacity(b),
            requests: Vec::with_capacity(b),
            children: vec![pool.root(); b],
            eval_records: Vec::with_capacity(b.max(1)),
            eval_rows: Vec::with_capacity(b.max(1)),
            eval_picks: Vec::with_capacity(b.max(1)),
            raw_priors: vec![0.0; b.max(1) * a],
            values: vec![0.0; b.max(1)],
            priors: vec![0.0; a],
            root_priors: vec![0.0; a],
            noise: vec![0.0; a],
            legal: vec![false; a],
            v_root: None,
            saved_v_root: None,
            gumbel,
            pool,
        })
    }

    /// The search pool.
    #[must_use]
    pub const fn pool(&self) -> &SearchPool<E, M> {
        &self.pool
    }

    /// The search pool, to re-root or clear it between calls.
    pub const fn pool_mut(&mut self) -> &mut SearchPool<E, M> {
        &mut self.pool
    }

    /// The configuration.
    #[must_use]
    pub const fn config(&self) -> &PuctConfig {
        &self.cfg
    }

    /// Bytes of one leaf's observation in the evaluator batches.
    #[must_use]
    pub const fn obs_bytes(&self) -> usize {
        self.obs_bytes
    }

    /// Whether the search runs on full records of an imperfect-information
    /// environment (`SearchPool::new_privileged`): its results see hidden
    /// state and must be labelled as such.
    #[must_use]
    pub fn is_privileged(&self) -> bool {
        self.pool.is_privileged()
    }

    /// Search calls completed (the persistent call counter).
    #[must_use]
    pub const fn calls(&self) -> u64 {
        self.calls
    }

    /// The visit count of every root action.
    pub fn root_visits(&self, out: &mut [u32]) {
        let root = self.pool.root;
        for (a, o) in (0u32..).zip(out.iter_mut()) {
            *o = self.pool.edge(root, a).map_or(0, |e| e.visits);
        }
    }

    /// The improved policy `pi'` of the last Gumbel call (one entry per
    /// action, `0` for illegal ones), or `None` under PUCT, before any
    /// call, or once the root changed.
    #[must_use]
    pub fn improved_policy(&self) -> Option<&[f64]> {
        self.gumbel.as_ref()?.improved_policy(self.root_key()?)
    }

    /// The actions `S_0` the last Gumbel call considered, in `g + l`
    /// order, or `None` as for [`Self::improved_policy`].
    #[must_use]
    pub fn considered(&self) -> Option<&[u32]> {
        self.gumbel.as_ref()?.considered(self.root_key()?)
    }

    /// The allocation number of the root node: the key of `v_root`.
    fn root_key(&self) -> Option<u64> {
        self.pool.node(self.pool.root).map(|n| n.alloc)
    }

    /// The Gumbel state's phase snapshot, for tests.
    #[cfg(test)]
    pub(crate) fn gumbel_snapshot(&self) -> Option<(u32, u32, u32, u32, bool, Vec<u32>)> {
        self.gumbel.as_ref().map(crate::gumbel::Gumbel::snapshot)
    }

    /// The current root's evaluator value, if known.
    fn root_value(&self) -> Option<f64> {
        let key = self.root_key()?;
        self.v_root.filter(|&(k, _)| k == key).map(|(_, v)| v)
    }

    /// The player to move at `slot` (its single acting agent).
    fn mover(&self, slot: u32) -> Result<usize, SearchError> {
        let Some(record) = self.pool.records.get(idx(slot)) else {
            return Err(SearchError::TerminalNode);
        };
        SearchPool::<E, M>::single_actor(record)
    }

    /// Whether `slot` has priors (it was evaluated).
    fn evaluated(&self, slot: u32) -> bool {
        (0..self.pool.actions).any(|a| self.pool.edge(slot, a).is_some_and(|e| e.legal))
    }

    /// The search stream of call `calls`: the root instance's identity
    /// with the call counter as its step index.
    fn search_rng(&self) -> Result<StepRng, SearchError> {
        let step = u32::try_from(self.calls).map_err(|_| SearchError::CounterExhausted)?;
        let id = self
            .pool
            .records
            .get(idx(self.pool.root))
            .map_or_else(RngIdentity::default, |r| r.header().engine().identity);
        let id = RngIdentity::from_parts(
            id.seed(),
            id.branch_salt(),
            id.logical_id(),
            id.episode_index(),
            step,
        );
        Ok(StepRng::new(
            &RngAuthority::engine_only(),
            &id,
            StreamId::SEARCH,
        ))
    }

    /// Runs one search call of `simulations` simulations. All search
    /// arithmetic (selection, backup, noise) runs under the canonical
    /// floating-point controls (ADR-0037, ADR-0043 section 2), whatever the
    /// caller's.
    ///
    /// # Errors
    ///
    /// A terminal root, an evaluator error or invalid response, a full
    /// pool, an exhausted counter, or a scratch pool error. The tree and
    /// the call counter are then exactly as before the call.
    pub fn run(&mut self, evaluator: &mut impl Evaluator<E>) -> Result<(), SearchError> {
        tessera_rl_num::fp::with_canonical(|| self.run_canonical(evaluator))
    }

    fn run_canonical(&mut self, evaluator: &mut impl Evaluator<E>) -> Result<(), SearchError> {
        let start = self.call_begin()?;
        match self.run_inner(evaluator, &start) {
            Ok(rng) => {
                self.call_commit(start.calls, rng);
                Ok(())
            }
            Err(e) => {
                self.call_rollback();
                Err(e)
            }
        }
    }

    /// The first step of a call, before any change to the tree: the next
    /// call counter and its search stream, then the journal is opened.
    pub(crate) fn call_begin(&mut self) -> Result<CallStart, SearchError> {
        let Some(calls) = self.calls.checked_add(1) else {
            return Err(SearchError::CounterExhausted);
        };
        let rng = self.search_rng()?;
        self.pool.begin();
        self.saved_v_root = self.v_root;
        Ok(CallStart { calls, rng })
    }

    /// Closes the journal and keeps the call's counter and stream.
    pub(crate) fn call_commit(&mut self, calls: u64, rng: StepRng) {
        self.pool.finish();
        self.calls = calls;
        self.rng = Some(rng);
        if let (Some(key), Some(g)) = (self.root_key(), self.gumbel.as_mut()) {
            g.publish(key);
        }
    }

    /// Restores the tree to its state at [`Self::call_begin`].
    pub(crate) fn call_rollback(&mut self) {
        self.pool.rollback();
        self.v_root = self.saved_v_root;
    }

    /// Whether the root is terminal (a call on it is an error).
    pub(crate) fn root_terminal(&self) -> bool {
        self.pool.node(self.pool.root).is_some_and(|n| n.terminal)
    }

    /// Whether the root has no priors yet.
    pub(crate) fn root_unevaluated(&self) -> bool {
        !self.evaluated(self.pool.root)
    }

    /// Whether the call must evaluate the root: it has no priors, or,
    /// under the Gumbel root only, its value is unknown (after a reroot).
    pub(crate) fn root_needs_eval(&self) -> bool {
        self.root_unevaluated() || (self.gumbel.is_some() && self.root_value().is_none())
    }

    fn run_inner(
        &mut self,
        evaluator: &mut impl Evaluator<E>,
        start: &CallStart,
    ) -> Result<StepRng, SearchError> {
        if self.root_terminal() {
            return Err(SearchError::TerminalNode);
        }
        evaluator.check()?;
        if self.root_needs_eval() {
            self.evaluate_root(evaluator)?;
        }
        let mut rng = start.rng.clone();
        self.prepare_root_priors(&mut rng);
        let mut done = 0u32;
        while let Some(want) = self.want(done) {
            // Every batch, even one of terminal leaves only, which calls
            // no evaluator.
            evaluator.check()?;
            let n = self.collect_and_stage(want)?;
            let k = self.eval_len();
            if k > 0 {
                self.evaluate_staged(evaluator, k)?;
                self.validate_staged(k)?;
            }
            self.commit_and_backup(k)?;
            self.advance_phase()?;
            done = done.saturating_add(n.max(1));
        }
        self.finish_root()?;
        Ok(rng)
    }

    /// The leaves to collect next after `done` simulations, or `None`
    /// when the budget is spent. Under the Gumbel root, at most the
    /// visits left in the current phase: a batch never crosses a phase.
    /// Pure: asking twice changes nothing.
    pub(crate) fn want(&self, done: u32) -> Option<u32> {
        (done < self.cfg.simulations).then(|| {
            let w = (self.cfg.simulations - done).min(self.cfg.batch_size);
            self.gumbel.as_ref().map_or(w, |g| {
                let left = g.phase_left();
                debug_assert!(left > 0, "a Gumbel phase end without advance_phase");
                w.min(left)
            })
        })
    }

    /// The Gumbel phase step, called once after each batch's
    /// [`Self::commit_and_backup`] by every driver: when the current
    /// phase's last visit is backed up, halves the survivors and starts
    /// the next phase. At any other time, and under PUCT, it changes
    /// nothing.
    ///
    /// # Errors
    ///
    /// [`SearchError::GumbelInvariant`] for a halving with pending
    /// virtual loss or no root visit (unreachable; the call rolls back).
    pub(crate) fn advance_phase(&mut self) -> Result<(), SearchError> {
        let v_root = self.root_value();
        let pool = &self.pool;
        self.gumbel
            .as_mut()
            .map_or(Ok(()), |g| g.advance(|a| root_edge(pool, a), v_root))
    }

    /// The Gumbel call's end, after its last backup: the final action and
    /// `pi'` (published at commit). Nothing under PUCT.
    ///
    /// # Errors
    ///
    /// As [`Self::advance_phase`].
    pub(crate) fn finish_root(&mut self) -> Result<(), SearchError> {
        let v_root = self.root_value();
        let pool = &self.pool;
        self.gumbel
            .as_mut()
            .map_or(Ok(()), |g| g.finish(|a| root_edge(pool, a), v_root))
    }

    /// Loads the root into the scratch pool and the evaluation lists, as a
    /// one-leaf batch: its record, its mover and its legal actions (in
    /// `self.legal`).
    pub(crate) fn load_root(&mut self) -> Result<(), SearchError> {
        let root = self.pool.root;
        let Some(&record) = self.pool.records.get(idx(root)) else {
            return Err(SearchError::TerminalNode);
        };
        let agent = SearchPool::<E, M>::single_actor(&record)?;
        // Every scratch row gets the root, by `restore_all`, which also
        // recovers a scratch pool poisoned by an earlier failed call (C3).
        self.pool.load_records.fill(record);
        self.pool.scratch.restore_all(&self.pool.load_records)?;
        let width = idx(self.pool.actions);
        let masks = self.pool.scratch.outputs()?.masks();
        for (a, l) in self.legal.iter_mut().enumerate() {
            *l = masks.is_empty() || masks.get(agent * width + a).is_some_and(|m| m.get());
        }
        self.eval_records.clear();
        self.eval_records.push(record);
        self.eval_picks.clear();
        self.eval_picks.push((0, agent));
        self.root_batch = true;
        Ok(())
    }

    /// Validates the root's response (in `raw_priors` and `values`, row 0),
    /// keeps its value as `v_root` and writes its priors if it has none
    /// (a rerooted root re-evaluated for its value keeps its priors).
    pub(crate) fn store_root(&mut self) -> Result<(), SearchError> {
        let width = idx(self.pool.actions);
        let value = self.values.first().copied().unwrap_or(f64::NAN);
        validate_leaf(
            0,
            self.raw_priors.get(..width).unwrap_or(&[]),
            value,
            &self.legal,
            &mut self.priors,
        )?;
        if let Some(key) = self.root_key() {
            self.v_root = Some((key, value));
        }
        if self.root_unevaluated() {
            self.write_priors(self.pool.root);
        }
        Ok(())
    }

    /// Evaluates the root alone and sets its priors.
    fn evaluate_root(&mut self, evaluator: &mut impl Evaluator<E>) -> Result<(), SearchError> {
        self.load_root()?;
        let width = idx(self.pool.actions);
        let ob = self.obs_bytes;
        let mut batch = LeafBatch::new(
            &self.eval_records,
            self.pool.obs_source()?,
            &self.eval_picks,
            self.obs_buf.get_mut(..ob).unwrap_or(&mut []),
            ob,
            &self.legal,
            width,
        );
        let (raw, values) = (
            self.raw_priors.get_mut(..width).unwrap_or(&mut []),
            self.values.get_mut(..1).unwrap_or(&mut []),
        );
        evaluator.evaluate_batch(&mut batch, raw, values)?;
        self.store_root()
    }

    /// Writes `self.priors` and `self.legal` into `slot`'s edges.
    fn write_priors(&mut self, slot: u32) {
        for a in 0..self.pool.actions {
            let (p, l) = (
                self.priors.get(idx(a)).copied().unwrap_or(0.0),
                self.legal.get(idx(a)).copied().unwrap_or(false),
            );
            if let Some(e) = self.pool.edge_mut(slot, a) {
                e.prior = p;
                e.legal = l;
            }
        }
    }

    /// The root priors of this call: the tree's, with noise mixed in over
    /// the legal actions when configured.
    fn prepare_root_priors(&mut self, rng: &mut StepRng) {
        let root = self.pool.root;
        for (a, p) in (0u32..).zip(self.root_priors.iter_mut()) {
            *p = self.pool.edge(root, a).map_or(0.0, |e| e.prior);
        }
        if let Some(g) = self.gumbel.as_mut() {
            let pool = &self.pool;
            g.begin(rng, |a| {
                u32::try_from(a)
                    .ok()
                    .and_then(|a| pool.edge(root, a))
                    .map_or((0.0, false), |e| (e.prior, e.legal))
            });
        }
        let Some(noise) = self.cfg.noise else {
            return;
        };
        let legal = (0..self.pool.actions)
            .filter(|&a| self.pool.edge(root, a).is_some_and(|e| e.legal))
            .count();
        let Some(sample) = self.noise.get_mut(..legal) else {
            return;
        };
        dirichlet(noise.alpha, rng, sample);
        let mut k = 0;
        for a in 0..self.pool.actions {
            if !self.pool.edge(root, a).is_some_and(|e| e.legal) {
                continue;
            }
            let eta = self.noise.get(k).copied().unwrap_or(0.0);
            k += 1;
            if let Some(p) = self.root_priors.get_mut(idx(a)) {
                *p = (1.0 - noise.epsilon) * *p + noise.epsilon * eta;
            }
        }
    }

    /// The selected action at `slot`, or `None` if no edge is legal.
    fn select(&self, slot: u32) -> Option<u32> {
        let root = slot == self.pool.root;
        let mut total = 0u64;
        for a in 0..self.pool.actions {
            if let Some(e) = self.pool.edge(slot, a) {
                total += u64::from(e.visits) + u64::from(e.virtual_loss);
            }
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "visit counts are far below 2^53"
        )]
        let sqrt_n = (total as f64).sqrt();
        let mut best: Option<(f64, u32)> = None;
        for a in 0..self.pool.actions {
            let Some(e) = self.pool.edge(slot, a).filter(|e| e.legal) else {
                continue;
            };
            let n = count(e.visits) + count(e.virtual_loss);
            let q = if n == 0.0 {
                self.cfg.fpu
            } else {
                (e.value - count(e.virtual_loss)) / n
            };
            let p = if root {
                self.root_priors.get(idx(a)).copied().unwrap_or(0.0)
            } else {
                e.prior
            };
            let score = q + self.cfg.c_puct * p * sqrt_n / (1.0 + n);
            if best.is_none_or(|(s, _)| score > s) {
                best = Some((score, a));
            }
        }
        best.map(|(_, a)| a)
    }

    /// Adds the configured virtual loss to edge `(slot, action)` only:
    /// selection calls it once per edge of the path.
    fn add_virtual_loss(&mut self, slot: u32, action: u32) -> Result<(), SearchError> {
        let vl = self.cfg.virtual_loss;
        let Some(e) = self.pool.edge_mut(slot, action) else {
            return Err(SearchError::VisitsExhausted);
        };
        let Some(next) = e.virtual_loss.checked_add(vl) else {
            return Err(SearchError::VisitsExhausted);
        };
        e.virtual_loss = next;
        Ok(())
    }

    /// Removes the configured virtual loss from every edge of the path
    /// that ends with edge `(slot, action)`, up to the root.
    fn remove_path_virtual_loss(&mut self, slot: u32, action: u32) -> Result<(), SearchError> {
        let vl = self.cfg.virtual_loss;
        let (mut s, mut a) = (slot, action);
        loop {
            let Some(e) = self.pool.edge_mut(s, a) else {
                return Err(SearchError::VisitsExhausted);
            };
            let Some(next) = e.virtual_loss.checked_sub(vl) else {
                return Err(SearchError::VisitsExhausted);
            };
            e.virtual_loss = next;
            let Some(node) = self.pool.node(s) else {
                break;
            };
            if node.parent == NONE {
                break;
            }
            (s, a) = (node.parent, node.action);
        }
        Ok(())
    }

    /// Collects up to `want` leaves by repeated selection from the root.
    fn collect(&mut self, want: u32) -> Result<(), SearchError> {
        self.leaves.clear();
        self.requests.clear();
        while u32::try_from(self.leaves.len()).unwrap_or(u32::MAX) < want {
            let mut s = self.pool.root;
            let forced = self.gumbel.as_ref().map(|g| g.action(self.leaves.len()));
            let leaf = loop {
                let pick = match forced {
                    Some(f) if s == self.pool.root => f,
                    _ => self.select(s),
                };
                let Some(a) = pick else {
                    return Err(SearchError::TerminalNode);
                };
                self.add_virtual_loss(s, a)?;
                let child = self.pool.edge(s, a).map_or(NONE, |e| e.child);
                if child == NONE {
                    break Leaf::Edge {
                        parent: s,
                        action: a,
                        row: self.requests.len(),
                    };
                }
                if self.pool.node(child).is_some_and(|n| n.terminal) {
                    break Leaf::Terminal { slot: child };
                }
                s = child;
            };
            let duplicate = match leaf {
                Leaf::Edge { parent, action, .. } => self
                    .requests
                    .iter()
                    .any(|&(h, a)| h.slot == parent && a == action),
                Leaf::Terminal { slot } => self
                    .leaves
                    .iter()
                    .any(|l| matches!(l, Leaf::Terminal { slot: t } if *t == slot)),
            };
            if duplicate {
                // Undo this path's virtual loss and stop collecting.
                match leaf {
                    Leaf::Edge { parent, action, .. } => {
                        self.remove_path_virtual_loss(parent, action)?;
                    }
                    Leaf::Terminal { slot } => {
                        let (p, a) = self
                            .pool
                            .node(slot)
                            .map_or((NONE, 0), |n| (n.parent, n.action));
                        self.remove_path_virtual_loss(p, a)?;
                    }
                }
                break;
            }
            if let Leaf::Edge { parent, action, .. } = leaf {
                self.requests.push((self.pool.handle_of(parent), action));
            }
            if let Leaf::Terminal { slot } = leaf {
                // Pending terminal leaves are held so expansion never
                // evicts them or their path.
                self.pool.hold(self.pool.handle_of(slot))?;
            }
            self.leaves.push(leaf);
        }
        Ok(())
    }

    /// Collects up to `want` leaves, steps the new children in the
    /// scratch pool, and gathers the non-terminal ones for evaluation
    /// (their records, movers and legal actions). Returns the leaves
    /// collected.
    pub(crate) fn collect_and_stage(&mut self, want: u32) -> Result<u32, SearchError> {
        self.root_batch = false;
        self.collect(want)?;
        let n = u32::try_from(self.leaves.len()).unwrap_or(u32::MAX);
        let requests = core::mem::take(&mut self.requests);
        let staged = self.pool.stage(&requests);
        self.requests = requests;
        staged?;
        let width = idx(self.pool.actions);
        self.eval_records.clear();
        self.eval_rows.clear();
        for row in 0..self.requests.len() {
            if !self.pool.child_terminal.get(row).copied().unwrap_or(true)
                && let Some(&r) = self.pool.child_records.get(row)
            {
                self.eval_records.push(r);
                self.eval_rows.push(row);
            }
        }
        self.eval_picks.clear();
        for leaf in 0..self.eval_records.len() {
            let row = self.eval_rows.get(leaf).copied().unwrap_or(0);
            let mover = self
                .eval_records
                .get(leaf)
                .map_or_else(|| Ok(0), SearchPool::<E, M>::single_actor)?;
            self.eval_picks.push((row, mover));
            if let (Some(dst), Some(src)) = (
                self.leaf_legal.get_mut(leaf * width..leaf * width + width),
                self.pool.child_masks.get(row * width..row * width + width),
            ) {
                dst.copy_from_slice(src);
            }
        }
        Ok(n)
    }

    /// Leaves gathered for evaluation by the last collection.
    pub(crate) fn eval_len(&self) -> usize {
        self.eval_records.len()
    }

    /// Evaluates the `k` gathered leaves into `raw_priors` and `values`.
    fn evaluate_staged(
        &mut self,
        evaluator: &mut impl Evaluator<E>,
        k: usize,
    ) -> Result<(), SearchError> {
        let width = idx(self.pool.actions);
        let ob = self.obs_bytes;
        let mut batch = LeafBatch::new(
            &self.eval_records,
            self.pool.obs_source()?,
            &self.eval_picks,
            self.obs_buf.get_mut(..k * ob).unwrap_or(&mut []),
            ob,
            self.leaf_legal.get(..k * width).unwrap_or(&[]),
            width,
        );
        let (raw, values) = (
            self.raw_priors.get_mut(..k * width).unwrap_or(&mut []),
            self.values.get_mut(..k).unwrap_or(&mut []),
        );
        evaluator.evaluate_batch(&mut batch, raw, values)?;
        Ok(())
    }

    /// First pass: validates every gathered leaf's response before
    /// anything is committed, so an invalid response fails the call with
    /// no child written. The renormalized priors are discarded here: one
    /// scratch row serves every leaf.
    pub(crate) fn validate_staged(&mut self, k: usize) -> Result<(), SearchError> {
        let width = idx(self.pool.actions);
        for leaf in 0..k {
            let row = self.eval_rows.get(leaf).copied().unwrap_or(0);
            let legal = self
                .pool
                .child_masks
                .get(row * width..row * width + width)
                .unwrap_or(&[]);
            let raw = self
                .raw_priors
                .get(leaf * width..leaf * width + width)
                .unwrap_or(&[]);
            let value = self.values.get(leaf).copied().unwrap_or(f64::NAN);
            validate_leaf(leaf, raw, value, legal, &mut self.priors)?;
        }
        Ok(())
    }

    /// Commits the staged children with the (validated) priors of the `k`
    /// evaluated ones, then backs every collected leaf up in batch order.
    pub(crate) fn commit_and_backup(&mut self, k: usize) -> Result<(), SearchError> {
        let width = idx(self.pool.actions);
        for t in 0..self.leaves.len() {
            if let Some(Leaf::Terminal { slot }) = self.leaves.get(t).copied() {
                self.pool.release(self.pool.handle_of(slot))?;
            }
        }
        let mut children = core::mem::take(&mut self.children);
        self.pool
            .commit_staged(children.get_mut(..self.requests.len()).unwrap_or(&mut []));
        self.children = children;
        for leaf in 0..k {
            let row = self.eval_rows.get(leaf).copied().unwrap_or(0);
            let Some(child) = self.children.get(row).map(|h| h.slot) else {
                continue;
            };
            let legal = self
                .pool
                .child_masks
                .get(row * width..row * width + width)
                .unwrap_or(&[]);
            let raw = self
                .raw_priors
                .get(leaf * width..leaf * width + width)
                .unwrap_or(&[]);
            let value = self.values.get(leaf).copied().unwrap_or(f64::NAN);
            self.legal.clear();
            self.legal.extend_from_slice(legal);
            let mut priors = core::mem::take(&mut self.priors);
            // Second pass, after the commit: the same (already valid)
            // response is renormalized again into the scratch row, then
            // written to the child. It cannot fail here.
            validate_leaf(leaf, raw, value, &self.legal, &mut priors)?;
            self.priors = priors;
            self.write_priors(child);
        }
        self.backup_all()?;
        if let Some(g) = self.gumbel.as_mut() {
            g.backed_up(self.leaves.len());
        }
        Ok(())
    }

    /// The gathered leaves for a group batch: their records, their scratch
    /// rows and movers, their legal actions (`eval_len() * actions`), and
    /// the scratch pool's observation source.
    pub(crate) fn gathered(&self) -> Result<Gathered<'_, E>, SearchError> {
        let k = self.eval_records.len();
        let width = idx(self.pool.actions);
        let legal: &[bool] = if self.root_batch {
            // The root batch: its legal actions are in `self.legal`.
            &self.legal
        } else {
            self.leaf_legal.get(..k * width).unwrap_or(&[])
        };
        Ok(Gathered {
            records: &self.eval_records,
            picks: &self.eval_picks,
            legal,
            source: self.pool.obs_source()?,
        })
    }

    /// Writes a group response for the gathered leaves into this tree's
    /// response rows: `raw` is `eval_len() * actions` priors, `values`
    /// `eval_len()` values.
    pub(crate) fn put_response(&mut self, raw: &[f64], values: &[f64]) {
        if let Some(dst) = self.raw_priors.get_mut(..raw.len()) {
            dst.copy_from_slice(raw);
        }
        if let Some(dst) = self.values.get_mut(..values.len()) {
            dst.copy_from_slice(values);
        }
    }

    /// Mixes this call's root noise into the root priors.
    /// Under the Gumbel root, also draws `g` and builds `S_0` and the first
    /// phase.
    pub(crate) fn prepare_noise(&mut self, rng: &mut StepRng) {
        self.prepare_root_priors(rng);
    }

    /// The value of a leaf from the view of the player to move at its
    /// parent, and the parent edge to start the backup from.
    fn leaf_value(&self, leaf: Leaf) -> Result<(u32, u32, f64), SearchError> {
        let (node, parent, action) = match leaf {
            Leaf::Edge {
                parent,
                action,
                row,
            } => {
                let child = self.children.get(row).map_or(NONE, |h| h.slot);
                (child, parent, action)
            }
            Leaf::Terminal { slot } => {
                let Some(n) = self.pool.node(slot) else {
                    return Err(SearchError::TerminalNode);
                };
                (slot, n.parent, n.action)
            }
        };
        let mover = self.mover(parent)?;
        if self.pool.node(node).is_some_and(|n| n.terminal) {
            let r = self
                .pool
                .rewards
                .get(idx(node) * self.pool.agents + mover)
                .copied()
                .unwrap_or(f32::NAN);
            let v = f64::from(r);
            if !(v.is_finite() && (-1.0..=1.0).contains(&v)) {
                return Err(SearchError::TerminalValue);
            }
            return Ok((parent, action, v));
        }
        let row = match leaf {
            Leaf::Edge { row, .. } => row,
            Leaf::Terminal { .. } => 0,
        };
        let Some(k) = self.eval_rows.iter().position(|&r| r == row) else {
            return Err(SearchError::TerminalNode);
        };
        let v = self.values.get(k).copied().unwrap_or(0.0);
        let child_mover = self.mover(node)?;
        Ok((parent, action, if child_mover == mover { v } else { -v }))
    }

    /// Backs every collected leaf up in batch order.
    fn backup_all(&mut self) -> Result<(), SearchError> {
        for i in 0..self.leaves.len() {
            let Some(leaf) = self.leaves.get(i).copied() else {
                continue;
            };
            let (mut s, mut a, mut v) = self.leaf_value(leaf)?;
            loop {
                let vl = self.cfg.virtual_loss;
                let Some(e) = self.pool.edge_mut(s, a) else {
                    return Err(SearchError::VisitsExhausted);
                };
                let Some(next) = e.virtual_loss.checked_sub(vl) else {
                    return Err(SearchError::VisitsExhausted);
                };
                e.virtual_loss = next;
                let Some(next) = e.visits.checked_add(1) else {
                    return Err(SearchError::VisitsExhausted);
                };
                e.visits = next;
                e.value += v;
                let Some(&node) = self.pool.node(s) else {
                    break;
                };
                if node.parent == NONE {
                    break;
                }
                let (here, up) = (self.mover(s)?, self.mover(node.parent)?);
                if here != up {
                    v = -v;
                }
                (s, a) = (node.parent, node.action);
            }
        }
        Ok(())
    }

    /// The action to play after a call: the most visited root action
    /// (ties to the lowest) when `temperature` is 0, else a draw
    /// proportional to `N^(1/T)`, computed as `exp((ln(N) - ln(N_max)) / T)`
    /// so the largest weight is 1 and none overflows (`N = 0` weighs 0),
    /// under the canonical controls.
    ///
    /// Each sampled call consumes one draw of the last search call's
    /// stream, which advances: repeated calls after one search give a
    /// reproducible sequence, not the same action.
    ///
    /// # Errors
    ///
    /// No call has completed, a negative or non-finite temperature, or no
    /// visited action. Under the Gumbel root, `temperature` must be 0 and
    /// the action is the last call's halving winner (its root must still
    /// be the root).
    pub fn select_action(&mut self, temperature: f64) -> Result<u32, SearchError> {
        tessera_rl_num::fp::with_canonical(|| self.select_canonical(temperature))
    }

    fn select_canonical(&mut self, temperature: f64) -> Result<u32, SearchError> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(config_error(
                "temperature",
                "must be finite and non-negative",
            ));
        }
        if let Some(g) = self.gumbel.as_ref() {
            if temperature != 0.0 {
                return Err(config_error(
                    "temperature",
                    "must be 0 under the Gumbel root (pi' is the target)",
                ));
            }
            return self
                .root_key()
                .and_then(|k| g.winner(k))
                .ok_or(config_error("select_action", "run a search first"));
        }
        let root = self.pool.root;
        let visits = |a: u32| self.pool.edge(root, a).map_or(0, |e| e.visits);
        if temperature == 0.0 {
            let mut best: Option<(u32, u32)> = None;
            for a in 0..self.pool.actions {
                let n = visits(a);
                if n > 0 && best.is_none_or(|(m, _)| n > m) {
                    best = Some((n, a));
                }
            }
            return best.map(|(_, a)| a).ok_or(SearchError::TerminalNode);
        }
        // Normalized by the most visited action, so no weight overflows
        // for a small T: the largest weight is exactly 1 (C2).
        let most = (0..self.pool.actions).map(visits).max().unwrap_or(0);
        if most == 0 {
            return Err(SearchError::TerminalNode);
        }
        let ln_most = math::ln(count(most));
        let weights = (0..self.pool.actions).map(|a| {
            let n = visits(a);
            if n == 0 {
                0.0
            } else {
                math::exp((math::ln(count(n)) - ln_most) / temperature)
            }
        });
        let total: f64 = weights.clone().sum();
        let rng = self
            .rng
            .as_mut()
            .ok_or(config_error("select_action", "run a search first"))?;
        // Never a silent fallback: a total that is not a positive finite
        // number (impossible after the normalization) is an error.
        if !(total.is_finite() && total > 0.0) {
            return Err(SearchError::TerminalNode);
        }
        let u = rng.unit_f64() * total;
        let mut acc = 0.0;
        let mut last = None;
        for (a, w) in (0u32..).zip(weights) {
            if w > 0.0 {
                acc += w;
                last = Some(a);
                if u < acc {
                    return Ok(a);
                }
            }
        }
        last.ok_or(SearchError::TerminalNode)
    }
}

#[cfg(test)]
#[path = "mcts_gumbel_tests.rs"]
mod gumbel_tests;
