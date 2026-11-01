// SPDX-License-Identifier: MIT OR Apache-2.0

//! The bounded search pool (ADR-0043 section 1): a fixed arena of node
//! records and edges, expanded in batch through a scratch engine pool.
//!
//! Expansion is plan, step, commit: the plan validates every request and
//! computes the evictions without touching the tree; the scratch pool then
//! steps the children; only then is the tree changed. An error at any
//! point before the commit leaves the tree and its counters unchanged.

use core::mem::size_of;
use core::num::NonZeroU32;

use tessera_rl_core::encoding::{Encoder, HashSink};
use tessera_rl_core::space::Space;
use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout};
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, EnvPool, MaskProvider, PoolOptions, Record, Reseed,
};
use tessera_rl_num::identity::Seed;

use crate::error::SearchError;
use crate::handle::{EdgeStats, NodeHandle};
use crate::options::{Eviction, SearchOptions};

/// No node: an unexpanded edge, or the root's parent.
pub(crate) const NONE: u32 = u32::MAX;

/// One node slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Node {
    /// The parent slot, `NONE` for the root.
    pub(crate) parent: u32,
    /// The parent's action that leads here.
    pub(crate) action: u32,
    /// Bumped when the slot is freed; `u32::MAX` retires the slot.
    pub(crate) generation: u32,
    /// Allocation number: the eviction order.
    pub(crate) alloc: u64,
    /// Handles held on the node (pending leaves included).
    pub(crate) holds: u32,
    pub(crate) live: bool,
    /// The step that made the node ended the episode.
    pub(crate) terminal: bool,
}

impl Node {
    const FREE: Self = Self {
        parent: NONE,
        action: 0,
        generation: 0,
        alloc: 0,
        holds: 0,
        live: false,
        terminal: false,
    };
}

/// One edge: a parent's action.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Edge {
    pub(crate) child: u32,
    /// The parent's mask allowed the action when its priors were set.
    pub(crate) legal: bool,
    pub(crate) visits: u32,
    pub(crate) virtual_loss: u32,
    pub(crate) value: f64,
    pub(crate) prior: f64,
}

impl Edge {
    pub(crate) const EMPTY: Self = Self {
        child: NONE,
        legal: false,
        visits: 0,
        virtual_loss: 0,
        value: 0.0,
        prior: 0.0,
    };
}

/// A planned child: its parent slot, action and acting agent.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Planned {
    pub(crate) parent: u32,
    pub(crate) action: u32,
    pub(crate) agent: usize,
}

/// A bounded tree of environment records for search (ADR-0043).
///
/// Every node holds the full record of its instance; its children are
/// made by restoring it into a scratch pool and stepping one action. The
/// scratch pool runs with [`AutoReset::Disabled`], so a child whose step
/// ends the episode keeps its terminal state: it is marked terminal and
/// cannot be expanded.
///
/// A failed scratch step (a kernel panic the engine contains, or a turn
/// order violation) poisons the scratch pool. The expansion fails and the
/// tree is unchanged; the next expansion reloads every scratch row with
/// `restore_all`, which recovers the scratch pool.
pub struct SearchPool<E: Environment, M: MaskProvider<E>> {
    pub(crate) records: Vec<Record<E>>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) edges: Vec<Edge>,
    /// Rewards of the step that made each node, `[capacity, agents]`.
    pub(crate) rewards: Vec<f32>,
    /// Free slots; the next allocation pops the last.
    pub(crate) free: Vec<u32>,
    pub(crate) root: u32,
    pub(crate) next_alloc: u64,
    pub(crate) actions: u32,
    pub(crate) agents: usize,
    pub(crate) capacity: u32,
    pub(crate) batch: u32,
    pub(crate) eviction: Eviction,
    pub(crate) scratch: EnvPool<E, M>,
    // Per-call working storage, allocated at construction.
    pub(crate) protected: Vec<bool>,
    pub(crate) doomed: Vec<bool>,
    pub(crate) stack: Vec<u32>,
    pub(crate) evict_roots: Vec<u32>,
    pub(crate) planned: Vec<Planned>,
    pub(crate) act_buf: Vec<i32>,
    /// The stepped children's records, taken before the commit.
    pub(crate) child_records: Vec<Record<E>>,
    /// The parents loaded into the scratch pool, one per row.
    pub(crate) load_records: Vec<Record<E>>,
    /// Per child row: terminal, then the rewards and the mask of its
    /// acting agent are read from the scratch outputs before the commit.
    pub(crate) child_terminal: Vec<bool>,
    pub(crate) child_rewards: Vec<f32>,
    pub(crate) child_masks: Vec<bool>,
    pub(crate) journal: crate::journal::Journal<E>,
}

fn options_error(option: &'static str, reason: &'static str) -> SearchError {
    SearchError::Options { option, reason }
}

fn idx(v: u32) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}

impl<E, M> SearchPool<E, M>
where
    E: Environment<
            Config: Sync,
            State: Send,
            Header: Send + RecordLayout<Adaptive: AdaptiveValues<E::Config>>,
        >,
    M: MaskProvider<E>,
{
    /// Bytes of node storage per slot: the record, the node, one edge per
    /// action and one reward per agent, twice (the tree and its rollback
    /// journal, which holds at most one copy of each), plus the per-call
    /// flags.
    #[must_use]
    pub fn bytes_per_node(actions: u32) -> u64 {
        let once = size_of::<Record<E>>()
            + size_of::<Node>()
            + size_of::<u32>()
            + idx(actions).saturating_mul(size_of::<Edge>() + size_of::<usize>())
            + E::MAX_AGENTS.saturating_mul(size_of::<f32>());
        let flags = 3 + idx(actions);
        u64::try_from(once.saturating_mul(2).saturating_add(flags)).unwrap_or(u64::MAX)
    }

    /// A pool rooted at `root`, a record of an instance with config `cfg`.
    /// `build` makes the scratch pool from the options it is given (for
    /// example `EnvPool::with_masks`).
    ///
    /// # Errors
    ///
    /// [`SearchError::ImperfectInformation`] when the environment does
    /// not have perfect information (ADR-0035); a non-discrete action
    /// space; a budget overrun (the error names the bytes per node); a
    /// scratch pool error.
    pub fn new(
        cfg: E::Config,
        options: SearchOptions,
        root: &Record<E>,
        build: impl FnOnce(E::Config, PoolOptions) -> Result<EnvPool<E, M>, EngineError>,
    ) -> Result<Self, SearchError> {
        if !E::PERFECT_INFORMATION {
            return Err(SearchError::ImperfectInformation);
        }
        let Ok(Space::Discrete { n: actions }) = E::action_space(&cfg) else {
            return Err(SearchError::UnsupportedActions);
        };
        if options.capacity.get() == NONE {
            return Err(options_error("capacity", "must be below u32::MAX"));
        }
        let capacity = options.capacity.get();
        let batch = options.batch.get();
        if batch > capacity {
            return Err(options_error("batch", "must not exceed the capacity"));
        }
        let per_node = Self::bytes_per_node(actions);
        let bytes = per_node.saturating_mul(u64::from(capacity));
        if bytes > options.budget_bytes {
            return Err(SearchError::Budget {
                bytes,
                per_node,
                budget: options.budget_bytes,
            });
        }
        let pool_options = PoolOptions {
            auto_reset: AutoReset::Disabled,
            ..PoolOptions::new(
                NonZeroU32::new(batch).unwrap_or(NonZeroU32::MIN),
                Seed::new(0),
            )
        };
        let mut scratch = build(cfg, pool_options)?;
        scratch.reset_all(Reseed::Keep)?;
        let cap = idx(capacity);
        let agents = E::MAX_AGENTS;
        let mut pool = Self {
            records: vec![*root; cap],
            nodes: vec![Node::FREE; cap],
            edges: vec![Edge::EMPTY; cap.saturating_mul(idx(actions))],
            rewards: vec![0.0; cap.saturating_mul(agents)],
            free: (0..capacity).rev().collect(),
            root: NONE,
            next_alloc: 0,
            actions,
            agents,
            capacity,
            batch,
            eviction: options.eviction,
            scratch,
            protected: vec![false; cap],
            doomed: vec![false; cap],
            stack: Vec::with_capacity(cap),
            evict_roots: Vec::with_capacity(cap),
            planned: Vec::with_capacity(idx(batch)),
            act_buf: vec![0; idx(batch).saturating_mul(agents)],
            child_records: vec![*root; idx(batch)],
            load_records: vec![*root; idx(batch)],
            child_terminal: vec![false; idx(batch)],
            child_rewards: vec![0.0; idx(batch).saturating_mul(agents)],
            child_masks: vec![true; idx(batch).saturating_mul(idx(actions))],
            journal: crate::journal::Journal::new(cap, cap.saturating_mul(idx(actions)), agents),
        };
        pool.set_root(root)?;
        Ok(pool)
    }

    /// Node slots, the root included.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Children stepped per expansion call.
    #[must_use]
    pub const fn batch(&self) -> u32 {
        self.batch
    }

    /// Actions per node.
    #[must_use]
    pub const fn actions(&self) -> u32 {
        self.actions
    }

    /// Live nodes, the root included. It scans every slot (`O(capacity)`):
    /// for tests and reports, not for the hot path.
    #[must_use]
    pub fn live_nodes(&self) -> usize {
        self.nodes.iter().filter(|n| n.live).count()
    }

    /// The root.
    #[must_use]
    pub fn root(&self) -> NodeHandle {
        NodeHandle {
            slot: self.root,
            generation: self.node(self.root).map_or(0, |n| n.generation),
        }
    }

    pub(crate) fn node(&self, slot: u32) -> Option<&Node> {
        self.nodes.get(idx(slot))
    }

    fn node_mut(&mut self, slot: u32) -> Option<&mut Node> {
        self.touch_node(slot);
        self.nodes.get_mut(idx(slot))
    }

    pub(crate) fn edge_index(&self, slot: u32, action: u32) -> usize {
        idx(slot)
            .saturating_mul(idx(self.actions))
            .saturating_add(idx(action))
    }

    pub(crate) fn edge(&self, slot: u32, action: u32) -> Option<&Edge> {
        self.edges.get(self.edge_index(slot, action))
    }

    pub(crate) fn edge_mut(&mut self, slot: u32, action: u32) -> Option<&mut Edge> {
        let i = self.edge_index(slot, action);
        self.touch_edge(i);
        self.edges.get_mut(i)
    }

    /// Resets every edge of `slot`.
    fn clear_edges(&mut self, slot: u32) {
        for a in 0..self.actions {
            if let Some(e) = self.edge_mut(slot, a) {
                *e = Edge::EMPTY;
            }
        }
    }

    /// The live slot of `h`.
    ///
    /// # Errors
    ///
    /// [`SearchError::StaleHandle`] if the slot was freed or reused.
    pub fn resolve(&self, h: NodeHandle) -> Result<u32, SearchError> {
        match self.node(h.slot) {
            Some(n) if n.live && n.generation == h.generation => Ok(h.slot),
            _ => Err(SearchError::StaleHandle {
                slot: h.slot,
                generation: h.generation,
            }),
        }
    }

    fn check_action(&self, action: u32) -> Result<(), SearchError> {
        if action >= self.actions {
            return Err(SearchError::ActionOutOfRange {
                action,
                actions: self.actions,
            });
        }
        Ok(())
    }

    pub(crate) fn handle_of(&self, slot: u32) -> NodeHandle {
        NodeHandle {
            slot,
            generation: self.node(slot).map_or(0, |n| n.generation),
        }
    }

    /// The record of node `h`.
    ///
    /// # Errors
    ///
    /// A stale handle.
    pub fn record(&self, h: NodeHandle) -> Result<&Record<E>, SearchError> {
        let slot = self.resolve(h)?;
        self.records.get(idx(slot)).ok_or(SearchError::StaleHandle {
            slot: h.slot,
            generation: h.generation,
        })
    }

    /// Whether the step that made `h` ended the episode (a terminal node
    /// has no children).
    ///
    /// # Errors
    ///
    /// A stale handle.
    pub fn is_terminal(&self, h: NodeHandle) -> Result<bool, SearchError> {
        let slot = self.resolve(h)?;
        Ok(self.node(slot).is_some_and(|n| n.terminal))
    }

    /// The rewards of the step that made `h`, one per agent (zero for the
    /// root).
    ///
    /// # Errors
    ///
    /// A stale handle.
    pub fn rewards(&self, h: NodeHandle) -> Result<&[f32], SearchError> {
        let slot = self.resolve(h)?;
        let from = idx(slot).saturating_mul(self.agents);
        Ok(self
            .rewards
            .get(from..from.saturating_add(self.agents))
            .unwrap_or(&[]))
    }

    /// The parent of `h` and the action that leads to `h`; `None` for the
    /// root.
    ///
    /// # Errors
    ///
    /// A stale handle.
    pub fn parent(&self, h: NodeHandle) -> Result<Option<(NodeHandle, u32)>, SearchError> {
        let slot = self.resolve(h)?;
        Ok(self
            .node(slot)
            .filter(|n| n.parent != NONE)
            .map(|n| (self.handle_of(n.parent), n.action)))
    }

    /// The child of `h` through `action`, if expanded.
    ///
    /// # Errors
    ///
    /// A stale handle or an action out of range.
    pub fn child(&self, h: NodeHandle, action: u32) -> Result<Option<NodeHandle>, SearchError> {
        let slot = self.resolve(h)?;
        self.check_action(action)?;
        Ok(self
            .edge(slot, action)
            .filter(|e| e.child != NONE)
            .map(|e| self.handle_of(e.child)))
    }

    /// The statistics of edge `(h, action)`.
    ///
    /// # Errors
    ///
    /// A stale handle or an action out of range.
    pub fn edge_stats(&self, h: NodeHandle, action: u32) -> Result<EdgeStats, SearchError> {
        let slot = self.resolve(h)?;
        self.check_action(action)?;
        Ok(self
            .edge(slot, action)
            .map(|e| EdgeStats {
                visits: e.visits,
                value: e.value,
                virtual_loss: e.virtual_loss,
                prior: e.prior,
                expanded: e.child != NONE,
            })
            .unwrap_or_default())
    }

    /// Adds `visits` and `value` to edge `(h, action)`: the backup
    /// primitive of the search helper. The `f64` addition runs under the
    /// canonical floating-point controls (ADR-0037, ADR-0043 section 2),
    /// whatever the caller's.
    ///
    /// # Errors
    ///
    /// A stale handle, an action out of range, a value that is not finite,
    /// or a visit count that would overflow (nothing is changed).
    pub fn add_visits(
        &mut self,
        h: NodeHandle,
        action: u32,
        visits: u32,
        value: f64,
    ) -> Result<(), SearchError> {
        tessera_rl_num::fp::with_canonical(|| self.add_visits_canonical(h, action, visits, value))
    }

    fn add_visits_canonical(
        &mut self,
        h: NodeHandle,
        action: u32,
        visits: u32,
        value: f64,
    ) -> Result<(), SearchError> {
        let slot = self.resolve(h)?;
        self.check_action(action)?;
        if !value.is_finite() {
            return Err(SearchError::NonFiniteValue);
        }
        let e = self
            .edge_mut(slot, action)
            .ok_or(SearchError::VisitsExhausted)?;
        e.visits = e
            .visits
            .checked_add(visits)
            .ok_or(SearchError::VisitsExhausted)?;
        e.value += value;
        Ok(())
    }

    /// Sets the priors of `h`'s edges, one per action.
    ///
    /// A primitive: it stores the values as given, without validation. The
    /// evaluator contract (no NaN or negative priors, a sum of 1 over all
    /// actions, renormalization over the legal actions, a legal mass above
    /// 0) is enforced by the search helper before it calls this (ADR-0043
    /// section 3).
    ///
    /// # Errors
    ///
    /// A stale handle, or a slice that is not one prior per action.
    pub fn set_priors(&mut self, h: NodeHandle, priors: &[f64]) -> Result<(), SearchError> {
        let slot = self.resolve(h)?;
        if priors.len() != idx(self.actions) {
            return Err(SearchError::OutputLength {
                expected: idx(self.actions),
                got: priors.len(),
            });
        }
        for (a, &p) in (0u32..).zip(priors) {
            if let Some(e) = self.edge_mut(slot, a) {
                e.prior = p;
            }
        }
        Ok(())
    }

    /// Holds `h`: a held node, its subtree's holders and its ancestors are
    /// never evicted. Pending leaves are held.
    ///
    /// # Errors
    ///
    /// A stale handle, or a hold count that would overflow.
    pub fn hold(&mut self, h: NodeHandle) -> Result<(), SearchError> {
        let slot = self.resolve(h)?;
        let n = self.node_mut(slot).ok_or(SearchError::HoldsExhausted)?;
        n.holds = n.holds.checked_add(1).ok_or(SearchError::HoldsExhausted)?;
        Ok(())
    }

    /// Releases one hold of `h`.
    ///
    /// # Errors
    ///
    /// A stale handle, or a node that is not held.
    pub fn release(&mut self, h: NodeHandle) -> Result<(), SearchError> {
        let slot = self.resolve(h)?;
        let n = self.node_mut(slot).ok_or(SearchError::NotHeld)?;
        n.holds = n.holds.checked_sub(1).ok_or(SearchError::NotHeld)?;
        Ok(())
    }

    /// Frees one slot: its edges are cleared, its generation bumped, and
    /// it returns to the free list unless the generation is exhausted,
    /// which retires it.
    pub(crate) fn free_slot(&mut self, slot: u32) {
        self.clear_edges(slot);
        let Some(n) = self.node_mut(slot) else {
            return;
        };
        let retired = n.generation == u32::MAX;
        *n = Node {
            generation: n.generation.saturating_add(1),
            ..Node::FREE
        };
        if !retired {
            self.free.push(slot);
        }
    }

    /// Frees the subtree under `top` (with `top`) and reverts its parent
    /// edge to unexpanded, keeping the edge's statistics.
    pub(crate) fn free_subtree(&mut self, top: u32) {
        if let Some(&Node { parent, action, .. }) = self.node(top)
            && parent != NONE
            && let Some(e) = self.edge_mut(parent, action)
        {
            e.child = NONE;
        }
        self.stack.clear();
        self.stack.push(top);
        while let Some(slot) = self.stack.pop() {
            for a in 0..self.actions {
                if let Some(&Edge { child, .. }) = self.edge(slot, a)
                    && child != NONE
                {
                    self.stack.push(child);
                }
            }
            self.free_slot(slot);
        }
    }

    /// Marks `slot` and its ancestors protected.
    pub(crate) fn protect_path(&mut self, slot: u32) {
        let mut s = slot;
        while let Some(flag) = self.protected.get_mut(idx(s)) {
            if *flag {
                break;
            }
            *flag = true;
            s = self.nodes.get(idx(s)).map_or(NONE, |n| n.parent);
        }
    }

    /// The first held live node outside the protected set, if any.
    fn held_outside(&self) -> Option<u32> {
        self.nodes
            .iter()
            .zip(&self.protected)
            .position(|(n, &keep)| n.live && n.holds > 0 && !keep)
            .and_then(|i| u32::try_from(i).ok())
    }

    /// Frees every live node outside the protected set, in slot order.
    fn free_unprotected(&mut self) {
        for slot in 0..self.capacity {
            let live = self.node(slot).is_some_and(|n| n.live);
            let keep = self.protected.get(idx(slot)).copied().unwrap_or(true);
            if live && !keep {
                self.free_slot(slot);
            }
        }
    }

    fn mark_subtree(&mut self, top: u32) {
        self.protected.fill(false);
        self.stack.clear();
        self.stack.push(top);
        while let Some(slot) = self.stack.pop() {
            if let Some(flag) = self.protected.get_mut(idx(slot)) {
                *flag = true;
            }
            for a in 0..self.actions {
                if let Some(&Edge { child, .. }) = self.edge(slot, a)
                    && child != NONE
                {
                    self.stack.push(child);
                }
            }
        }
    }

    /// Makes `h` the root, keeping its subtree (subtree reuse) and freeing
    /// every other node.
    ///
    /// # Errors
    ///
    /// A stale handle, or a held node outside the subtree (nothing is
    /// freed).
    pub fn reroot(&mut self, h: NodeHandle) -> Result<(), SearchError> {
        let slot = self.resolve(h)?;
        self.mark_subtree(slot);
        if let Some(held) = self.held_outside() {
            return Err(SearchError::Held { slot: held });
        }
        self.free_unprotected();
        if let Some(n) = self.node_mut(slot) {
            n.parent = NONE;
            n.action = 0;
        }
        self.root = slot;
        Ok(())
    }

    /// Frees every node but the root and clears the root's edges.
    ///
    /// # Errors
    ///
    /// A held node other than the root (nothing is freed).
    pub fn clear(&mut self) -> Result<(), SearchError> {
        self.protected.fill(false);
        if let Some(flag) = self.protected.get_mut(idx(self.root)) {
            *flag = true;
        }
        if let Some(held) = self.held_outside() {
            return Err(SearchError::Held { slot: held });
        }
        self.free_unprotected();
        self.clear_edges(self.root);
        Ok(())
    }

    /// Replaces the whole tree by a new root.
    ///
    /// # Errors
    ///
    /// A held node, an exhausted allocation counter, or no usable slot
    /// (nothing is changed).
    pub fn set_root(&mut self, record: &Record<E>) -> Result<NodeHandle, SearchError> {
        self.protected.fill(false);
        if let Some(held) = self.held_outside() {
            return Err(SearchError::Held { slot: held });
        }
        let alloc = self.next_alloc;
        let next = alloc.checked_add(1).ok_or(SearchError::CounterExhausted)?;
        // Plan before freeing anything: after the frees, the free list
        // gains every live slot whose generation is not exhausted (those
        // retire). With none, nothing is changed (R38-02).
        let reusable = self
            .nodes
            .iter()
            .filter(|n| n.live && n.generation != u32::MAX)
            .count();
        if self.free.len() + reusable == 0 {
            return Err(SearchError::CapacityExceeded {
                capacity: self.capacity,
            });
        }
        self.free_unprotected();
        let Some(slot) = self.free.pop() else {
            return Err(SearchError::CapacityExceeded {
                capacity: self.capacity,
            });
        };
        self.next_alloc = next;
        let terminal = record.header().engine().pending_reset.get();
        self.clear_edges(slot);
        if let Some(n) = self.node_mut(slot) {
            *n = Node {
                alloc,
                live: true,
                terminal,
                ..Node {
                    generation: n.generation,
                    ..Node::FREE
                }
            };
        }
        if let Some(r) = self.records.get_mut(idx(slot)) {
            *r = *record;
        }
        let from = idx(slot).saturating_mul(self.agents);
        if let Some(rw) = self.rewards.get_mut(from..from.saturating_add(self.agents)) {
            rw.fill(0.0);
        }
        self.root = slot;
        Ok(self.handle_of(slot))
    }

    /// The acting agent of a parent record: exactly one is required.
    pub(crate) fn single_actor(record: &Record<E>) -> Result<usize, SearchError> {
        let rows = record.header().agents();
        let acting = rows.iter().filter(|r| r.acting.get()).count();
        if acting != 1 {
            return Err(SearchError::NotSingleActor { acting });
        }
        Ok(rows.iter().position(|r| r.acting.get()).unwrap_or(0))
    }

    /// Validates the requests into `self.planned`.
    fn plan_requests(&mut self, requests: &[(NodeHandle, u32)]) -> Result<(), SearchError> {
        self.planned.clear();
        for (index, &(h, action)) in requests.iter().enumerate() {
            let parent = self.resolve(h)?;
            self.check_action(action)?;
            if self.node(parent).is_some_and(|n| n.terminal) {
                return Err(SearchError::TerminalNode);
            }
            if self.edge(parent, action).is_some_and(|e| e.child != NONE) {
                return Err(SearchError::AlreadyExpanded { action });
            }
            if self
                .planned
                .iter()
                .any(|p| p.parent == parent && p.action == action)
            {
                return Err(SearchError::DuplicateRequest { index });
            }
            let record = self
                .records
                .get(idx(parent))
                .ok_or(SearchError::TerminalNode)?;
            let agent = Self::single_actor(record)?;
            self.planned.push(Planned {
                parent,
                action,
                agent,
            });
        }
        Ok(())
    }

    /// Restores the planned parents into the scratch pool and checks each
    /// action against the parent's mask. Unused rows get the first
    /// parent's record; their outputs are discarded.
    fn load_scratch(&mut self) -> Result<(), SearchError> {
        let Some(first) = self.planned.first().map(|p| p.parent) else {
            return Ok(());
        };
        for (row, slot) in self.load_records.iter_mut().enumerate() {
            let parent = self.planned.get(row).map_or(first, |p| p.parent);
            if let Some(&record) = self.records.get(idx(parent)) {
                *slot = record;
            }
        }
        // `restore_all` recovers a scratch pool poisoned by an earlier
        // failed step (a contained kernel panic): every row is rewritten,
        // so nothing of the failed step survives.
        self.scratch.restore_all(&self.load_records)?;
        let masks = self.scratch.outputs().masks();
        if !masks.is_empty() {
            let width = idx(self.actions);
            for (row, p) in self.planned.iter().enumerate() {
                let at = (row * self.agents + p.agent) * width + idx(p.action);
                if !masks.get(at).is_some_and(|m| m.get()) {
                    return Err(SearchError::IllegalAction { action: p.action });
                }
            }
        }
        // Unused rows repeat the first request (a legal action at the
        // parent they were loaded with), so strict checks accept them.
        self.act_buf.fill(0);
        let Some(&lead) = self.planned.first() else {
            return Ok(());
        };
        for row in 0..idx(self.batch) {
            let p = self.planned.get(row).copied().unwrap_or(lead);
            if let Some(a) = self.act_buf.get_mut(row * self.agents + p.agent) {
                *a = i32::try_from(p.action).unwrap_or(0);
            }
        }
        Ok(())
    }

    /// Expands one child per request `(parent, action)`, stepping them in
    /// one batch, and writes their handles into `out`.
    ///
    /// # Errors
    ///
    /// Any invalid request (stale handle, action out of range or illegal,
    /// terminal parent, edge already expanded, repeated edge, not one
    /// acting agent), too many requests, no free slot (strict) or nothing
    /// evictable, an exhausted counter, or a scratch pool error. On error
    /// the tree and its counters are unchanged.
    pub fn expand(
        &mut self,
        requests: &[(NodeHandle, u32)],
        out: &mut [NodeHandle],
    ) -> Result<(), SearchError> {
        if out.len() != requests.len() {
            return Err(SearchError::OutputLength {
                expected: requests.len(),
                got: out.len(),
            });
        }
        self.stage(requests)?;
        self.commit_staged(out);
        Ok(())
    }

    /// Plans and steps the children of `requests` without changing the
    /// tree: their records, terminal flags, rewards and masks wait in the
    /// staging buffers for [`Self::commit_staged`].
    pub(crate) fn stage(&mut self, requests: &[(NodeHandle, u32)]) -> Result<(), SearchError> {
        if requests.len() > idx(self.batch) {
            return Err(SearchError::BatchTooLarge {
                requests: requests.len(),
                batch: self.batch,
            });
        }
        // Every stage starts from an empty plan, so a commit never
        // replays an earlier call's children or evictions (C1).
        self.planned.clear();
        self.evict_roots.clear();
        if requests.is_empty() {
            return Ok(());
        }
        self.plan_requests(requests)?;
        let n = u64::try_from(requests.len()).unwrap_or(u64::MAX);
        self.next_alloc
            .checked_add(n)
            .ok_or(SearchError::CounterExhausted)?;
        self.plan_evictions(requests.len())?;
        self.load_scratch()?;
        self.scratch.step(ActionInput {
            discrete: &self.act_buf,
            continuous: &[],
        })?;
        let width = idx(self.actions);
        for row in 0..self.planned.len() {
            let record = self.scratch.snapshot(u32::try_from(row).unwrap_or(0))?;
            if let Some(r) = self.child_records.get_mut(row) {
                *r = record;
            }
            let outputs = self.scratch.outputs();
            let terminal = outputs
                .episode_terminated()
                .get(row)
                .is_some_and(|f| f.get())
                || outputs
                    .episode_truncated()
                    .get(row)
                    .is_some_and(|f| f.get());
            if let Some(t) = self.child_terminal.get_mut(row) {
                *t = terminal;
            }
            let from = row * self.agents;
            if let (Some(src), Some(dst)) = (
                outputs.rewards().get(from..from + self.agents),
                self.child_rewards.get_mut(from..from + self.agents),
            ) {
                dst.copy_from_slice(src);
            }
            // The child's legal actions: the mask of its acting agent (all
            // true without masks or without a single actor).
            let agent = Self::single_actor(&record).ok();
            let masks = outputs.masks();
            for a in 0..width {
                let legal = match agent {
                    Some(k) if !masks.is_empty() => masks
                        .get((row * self.agents + k) * width + a)
                        .is_some_and(|m| m.get()),
                    _ => true,
                };
                if let Some(m) = self.child_masks.get_mut(row * width + a) {
                    *m = legal;
                }
            }
        }
        Ok(())
    }

    /// Commits the staged children: the planned evictions, then one node
    /// per child, written to `out` in request order. Infallible.
    pub(crate) fn commit_staged(&mut self, out: &mut [NodeHandle]) {
        self.commit_evictions();
        let base = self.next_alloc;
        let n = u64::try_from(self.planned.len()).unwrap_or(0);
        self.next_alloc = base.saturating_add(n);
        for row in 0..self.planned.len() {
            let Some(p) = self.planned.get(row).copied() else {
                continue;
            };
            let Some(slot) = self.free.pop() else {
                continue;
            };
            self.touch_node(slot);
            let record = self.child_records.get(row).copied();
            if let (Some(r), Some(record)) = (self.records.get_mut(idx(slot)), record) {
                *r = record;
            }
            let (from, to) = (row * self.agents, slot_rewards(slot, self.agents));
            if let (Some(src), Some(dst)) = (
                self.child_rewards.get(from..from + self.agents),
                self.rewards.get_mut(to..to + self.agents),
            ) {
                dst.copy_from_slice(src);
            }
            let terminal = self.child_terminal.get(row).copied().unwrap_or(false);
            if let Some(node) = self.nodes.get_mut(idx(slot)) {
                *node = Node {
                    parent: p.parent,
                    action: p.action,
                    generation: node.generation,
                    alloc: base + u64::try_from(row).unwrap_or(0),
                    holds: 0,
                    live: true,
                    terminal,
                };
            }
            if let Some(e) = self.edge_mut(p.parent, p.action) {
                e.child = slot;
            }
            if let Some(h) = out.get_mut(row) {
                *h = self.handle_of(slot);
            }
        }
    }

    /// Bytes of one agent's observation (every observation leaf, in order).
    pub(crate) fn obs_row_bytes(&self) -> usize {
        let rows = idx(self.batch).saturating_mul(self.agents).max(1);
        (0..tessera_rl_engine::MAX_LEAVES)
            .map_while(|l| self.scratch.outputs().observation(l).map(<[u8]>::len))
            .map(|len| len / rows)
            .sum()
    }

    /// Copies the observation of `agent` at scratch row `row` into `out`
    /// (every observation leaf, in order).
    pub(crate) fn copy_observation(&self, row: usize, agent: usize, out: &mut [u8]) {
        let rows = idx(self.batch).saturating_mul(self.agents).max(1);
        let mut at = 0;
        for l in 0..tessera_rl_engine::MAX_LEAVES {
            let Some(bytes) = self.scratch.outputs().observation(l) else {
                break;
            };
            let per = bytes.len() / rows;
            let from = (row * self.agents + agent) * per;
            if let (Some(src), Some(dst)) = (bytes.get(from..from + per), out.get_mut(at..at + per))
            {
                dst.copy_from_slice(src);
            }
            at += per;
        }
    }

    /// A digest of the whole tree (records, nodes, edges, rewards, the free
    /// list, the root and the counter), for tests of atomicity and of
    /// determinism.
    #[must_use]
    pub fn tree_hash(&self) -> u64 {
        let mut out = Encoder::new(HashSink::new());
        for r in &self.records {
            out.write_bytes(&r.hash().get().to_le_bytes());
        }
        for n in &self.nodes {
            for v in [n.parent, n.action, n.generation, n.holds] {
                out.write_bytes(&v.to_le_bytes());
            }
            out.write_bytes(&n.alloc.to_le_bytes());
            out.write_bytes(&[u8::from(n.live), u8::from(n.terminal)]);
        }
        for e in &self.edges {
            for v in [e.child, e.visits, e.virtual_loss] {
                out.write_bytes(&v.to_le_bytes());
            }
            out.write_bytes(&e.value.to_bits().to_le_bytes());
            out.write_bytes(&e.prior.to_bits().to_le_bytes());
            out.write_bytes(&[u8::from(e.legal)]);
        }
        for r in &self.rewards {
            out.write_bytes(&r.to_bits().to_le_bytes());
        }
        for f in &self.free {
            out.write_bytes(&f.to_le_bytes());
        }
        out.write_bytes(&self.root.to_le_bytes());
        out.write_bytes(&self.next_alloc.to_le_bytes());
        out.sink().digest()
    }
}

fn slot_rewards(slot: u32, agents: usize) -> usize {
    idx(slot).saturating_mul(agents)
}
