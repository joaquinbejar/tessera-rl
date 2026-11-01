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
}

fn idx(v: u32) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}

fn count(n: u32) -> f64 {
    f64::from(n)
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
    /// of 0 or above the pool's, no simulations, or invalid noise.
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
        let (b, a) = (idx(cfg.batch_size), idx(pool.actions()));
        let obs_bytes = pool.obs_row_bytes();
        Ok(Self {
            obs_bytes,
            obs_buf: vec![0; b.max(1) * obs_bytes],
            leaf_legal: vec![false; b.max(1) * a],
            cfg,
            calls: 0,
            rng: None,
            leaves: Vec::with_capacity(b),
            requests: Vec::with_capacity(b),
            children: vec![pool.root(); b],
            eval_records: Vec::with_capacity(b.max(1)),
            eval_rows: Vec::with_capacity(b.max(1)),
            raw_priors: vec![0.0; b.max(1) * a],
            values: vec![0.0; b.max(1)],
            priors: vec![0.0; a],
            root_priors: vec![0.0; a],
            noise: vec![0.0; a],
            legal: vec![false; a],
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

    /// The player to move at `slot` (its single acting agent).
    fn mover(&self, slot: u32) -> Result<usize, SearchError> {
        let record = self
            .pool
            .records
            .get(idx(slot))
            .ok_or(SearchError::TerminalNode)?;
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
        let calls = self
            .calls
            .checked_add(1)
            .ok_or(SearchError::CounterExhausted)?;
        let mut rng = self.search_rng()?;
        self.pool.begin();
        match self.run_inner(evaluator, &mut rng) {
            Ok(()) => {
                self.pool.finish();
                self.calls = calls;
                self.rng = Some(rng);
                Ok(())
            }
            Err(e) => {
                self.pool.rollback();
                Err(e)
            }
        }
    }

    fn run_inner(
        &mut self,
        evaluator: &mut impl Evaluator<E>,
        rng: &mut StepRng,
    ) -> Result<(), SearchError> {
        let root = self.pool.root;
        if self.pool.node(root).is_some_and(|n| n.terminal) {
            return Err(SearchError::TerminalNode);
        }
        if !self.evaluated(root) {
            self.evaluate_root(evaluator)?;
        }
        self.prepare_root_priors(rng);
        let mut done = 0u32;
        while done < self.cfg.simulations {
            let want = (self.cfg.simulations - done).min(self.cfg.batch_size);
            self.collect(want)?;
            let n = u32::try_from(self.leaves.len()).unwrap_or(u32::MAX);
            self.expand_and_evaluate(evaluator)?;
            self.backup_all()?;
            done = done.saturating_add(n.max(1));
        }
        Ok(())
    }

    /// Evaluates the root alone and sets its priors.
    fn evaluate_root(&mut self, evaluator: &mut impl Evaluator<E>) -> Result<(), SearchError> {
        let root = self.pool.root;
        let record = *self
            .pool
            .records
            .get(idx(root))
            .ok_or(SearchError::TerminalNode)?;
        let agent = SearchPool::<E, M>::single_actor(&record)?;
        // Every scratch row gets the root, by `restore_all`, which also
        // recovers a scratch pool poisoned by an earlier failed call (C3).
        self.pool.load_records.fill(record);
        self.pool.scratch.restore_all(&self.pool.load_records)?;
        let width = idx(self.pool.actions);
        let masks = self.pool.scratch.outputs().masks();
        for (a, l) in self.legal.iter_mut().enumerate() {
            *l = masks.is_empty() || masks.get(agent * width + a).is_some_and(|m| m.get());
        }
        self.eval_records.clear();
        self.eval_records.push(record);
        let ob = self.obs_bytes;
        self.pool
            .copy_observation(0, agent, self.obs_buf.get_mut(..ob).unwrap_or(&mut []));
        let batch = LeafBatch::new(
            &self.eval_records,
            self.obs_buf.get(..ob).unwrap_or(&[]),
            ob,
            &self.legal,
            width,
        );
        let (raw, values) = (
            self.raw_priors.get_mut(..width).unwrap_or(&mut []),
            self.values.get_mut(..1).unwrap_or(&mut []),
        );
        evaluator.evaluate_batch(&batch, raw, values)?;
        let value = self.values.first().copied().unwrap_or(f64::NAN);
        validate_leaf(
            0,
            self.raw_priors.get(..width).unwrap_or(&[]),
            value,
            &self.legal,
            &mut self.priors,
        )?;
        self.write_priors(root);
        Ok(())
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
        let e = self
            .pool
            .edge_mut(slot, action)
            .ok_or(SearchError::VisitsExhausted)?;
        e.virtual_loss = e
            .virtual_loss
            .checked_add(vl)
            .ok_or(SearchError::VisitsExhausted)?;
        Ok(())
    }

    /// Removes the configured virtual loss from every edge of the path
    /// that ends with edge `(slot, action)`, up to the root.
    fn remove_path_virtual_loss(&mut self, slot: u32, action: u32) -> Result<(), SearchError> {
        let vl = self.cfg.virtual_loss;
        let (mut s, mut a) = (slot, action);
        loop {
            let e = self
                .pool
                .edge_mut(s, a)
                .ok_or(SearchError::VisitsExhausted)?;
            e.virtual_loss = e
                .virtual_loss
                .checked_sub(vl)
                .ok_or(SearchError::VisitsExhausted)?;
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
            let leaf = loop {
                let Some(a) = self.select(s) else {
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

    /// Steps the new children, evaluates the non-terminal ones, and
    /// commits them with their priors.
    fn expand_and_evaluate(
        &mut self,
        evaluator: &mut impl Evaluator<E>,
    ) -> Result<(), SearchError> {
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
        let k = self.eval_records.len();
        if k > 0 {
            let ob = self.obs_bytes;
            for leaf in 0..k {
                let row = self.eval_rows.get(leaf).copied().unwrap_or(0);
                let mover = self
                    .eval_records
                    .get(leaf)
                    .map_or(Ok(0), SearchPool::<E, M>::single_actor)?;
                self.pool.copy_observation(
                    row,
                    mover,
                    self.obs_buf
                        .get_mut(leaf * ob..leaf * ob + ob)
                        .unwrap_or(&mut []),
                );
                if let (Some(dst), Some(src)) = (
                    self.leaf_legal.get_mut(leaf * width..leaf * width + width),
                    self.pool.child_masks.get(row * width..row * width + width),
                ) {
                    dst.copy_from_slice(src);
                }
            }
            let batch = LeafBatch::new(
                &self.eval_records,
                self.obs_buf.get(..k * ob).unwrap_or(&[]),
                ob,
                self.leaf_legal.get(..k * width).unwrap_or(&[]),
                width,
            );
            let (raw, values) = (
                self.raw_priors.get_mut(..k * width).unwrap_or(&mut []),
                self.values.get_mut(..k).unwrap_or(&mut []),
            );
            evaluator.evaluate_batch(&batch, raw, values)?;
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
                // First pass: validate every leaf before anything is
                // committed, so an invalid response fails the call with no
                // child written. The renormalized priors are discarded
                // here: one scratch row serves every leaf.
                validate_leaf(leaf, raw, value, legal, &mut self.priors)?;
            }
        }
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
        Ok(())
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
                let n = self.pool.node(slot).ok_or(SearchError::TerminalNode)?;
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
        let k = self
            .eval_rows
            .iter()
            .position(|&r| r == row)
            .ok_or(SearchError::TerminalNode)?;
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
                let e = self
                    .pool
                    .edge_mut(s, a)
                    .ok_or(SearchError::VisitsExhausted)?;
                e.virtual_loss = e
                    .virtual_loss
                    .checked_sub(vl)
                    .ok_or(SearchError::VisitsExhausted)?;
                e.visits = e
                    .visits
                    .checked_add(1)
                    .ok_or(SearchError::VisitsExhausted)?;
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
    /// visited action.
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
