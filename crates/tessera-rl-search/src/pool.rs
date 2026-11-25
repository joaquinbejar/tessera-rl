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
use tessera_rl_core::space::{JointError, JointHeads, Space};
use tessera_rl_core::{AdaptiveValues, Bool8, Environment, PolicyGroup, RecordLayout};
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, EnvPool, MaskProvider, Outputs, PoolOptions, Record,
    Reseed,
};
use tessera_rl_num::identity::Seed;

use crate::error::SearchError;
use crate::handle::{EdgeStats, NodeHandle};
use crate::options::{Eviction, SearchOptions};

/// No node: an unexpanded edge, or the root's parent.
pub(crate) const NONE: u32 = u32::MAX;

/// Most joint actions of a `MultiDiscrete` action space (ADR-0086): the
/// product of its head sizes. A `Discrete` space is not capped.
pub const MAX_JOINT_ACTIONS: u32 = 4096;

/// The joint index of the action space of `cfg` (ADR-0086): a `Discrete`
/// head, or the heads of a `MultiDiscrete` space with their parents and
/// activity from `M`.
pub(crate) fn joint_heads<E: Environment, M: MaskProvider<E>>(
    cfg: &E::Config,
) -> Result<JointHeads, SearchError> {
    let (sizes, max) = match E::action_space(cfg) {
        Ok(Space::Discrete { n }) => (vec![n], u32::MAX),
        Ok(Space::MultiDiscrete { nvec }) => (nvec, MAX_JOINT_ACTIONS),
        _ => return Err(SearchError::UnsupportedActions),
    };
    if M::entity_group_heads(cfg).is_some() {
        return Err(SearchError::UnsupportedActions);
    }
    let parents: Vec<[Option<u16>; 2]> =
        (0..sizes.len()).map(|h| M::head_parents(cfg, h)).collect();
    JointHeads::new(&sizes, &parents, max, |h, p| M::head_active(cfg, h, p)).map_err(|e| match e {
        JointError::TooMany { joint, max } => SearchError::TooManyJointActions { joint, max },
        JointError::Invalid { .. } => SearchError::UnsupportedActions,
    })
}

/// Writes the legal joints of one agent into `out` (ADR-0086): from its
/// mask row starting at `row` in `masks`, or every canonical joint when
/// there is no row (no masks, or no single acting agent).
pub(crate) fn fill_legal(
    joint: &JointHeads,
    masks: &[Bool8],
    row: Option<usize>,
    out: &mut [bool],
) {
    for (j, legal) in (0..joint.joint()).zip(out.iter_mut()) {
        *legal = match row {
            Some(base) if !masks.is_empty() => joint.is_legal(j, |at| {
                base.checked_add(at)
                    .and_then(|i| masks.get(i))
                    .is_some_and(|m| m.get())
            }),
            _ => joint.is_canonical(j),
        };
    }
}

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
    /// The agent whose action the step takes; `None` at a node where the
    /// searcher is not acting (ADR-0093 decision 4: one edge, joint 0).
    pub(crate) agent: Option<usize>,
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
    /// The joint index of the action heads (ADR-0086).
    pub(crate) joint: JointHeads,
    /// The searcher slot of a pool with heuristic slots (ADR-0093); `None`
    /// for the default policy map (ADR-0043's single mover).
    pub(crate) searcher: Option<usize>,
    /// The scratch pool's policy groups, read at construction (ADR-0093).
    pub(crate) groups: Vec<PolicyGroup>,
    /// Whether `PrivilegedHeuristic` opponents are allowed (privileged
    /// search).
    pub(crate) privileged_opponents: bool,
    pub(crate) agents: usize,
    pub(crate) capacity: u32,
    pub(crate) batch: u32,
    pub(crate) eviction: Eviction,
    pub(crate) scratch: EnvPool<E, M>,
    /// Bytes of one agent's observation, a shape fixed at construction.
    obs_row_bytes: usize,
    /// The exported leaves of an observation row, and its sets (#490).
    obs_leaves: Vec<crate::obs_layout::LeafLayout>,
    obs_sets: Vec<crate::obs_layout::SetLayout>,
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
    /// How the root was made (ADR-0035, ADR-0046).
    pub(crate) mode: crate::determinize::RootMode,
    /// The frame-stack bytes of every node (ADR-0078 item 7).
    pub(crate) stacks: crate::stacks::Stacks,
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
        Self::bytes_per_node_with(actions, 0)
    }

    /// [`Self::bytes_per_node`] for a scratch pool with `stack_bytes` of
    /// frame stacks per instance (ADR-0078 item 7.5): each node also keeps
    /// them, twice (the tree and the journal).
    #[must_use]
    pub fn bytes_per_node_with(actions: u32, stack_bytes: usize) -> u64 {
        let once = stack_bytes
            + size_of::<Record<E>>()
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
    /// scratch pool error. `Set` observations are accepted (ADR-0082 point 7
    /// as amended, #490): their item and mask leaves are ordinary leaves of
    /// the evaluator's rows, see [`Self::observation_sets`].
    pub fn new(
        cfg: E::Config,
        options: SearchOptions,
        root: &Record<E>,
        build: impl FnOnce(E::Config, PoolOptions) -> Result<EnvPool<E, M>, EngineError>,
    ) -> Result<Self, SearchError> {
        if !E::PERFECT_INFORMATION {
            return Err(SearchError::ImperfectInformation);
        }
        let mut pool = Self::build_pool(cfg, options, root, &[], build)?;
        pool.resolve_searcher(root, options.searcher, None, false)?;
        Ok(pool)
    }

    /// [`Self::new`] on a scratch pool with frame stacks: `build` sets the
    /// live pool's transforms (`EnvPool::with_transforms`) and `root_bytes`
    /// are the root's frame stacks (`EnvPool::snapshot_transforms`),
    /// ADR-0078 item 7.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], and [`SearchError::StackedRoot`] for root bytes
    /// of the wrong length.
    pub fn new_with_transforms(
        cfg: E::Config,
        options: SearchOptions,
        root: &Record<E>,
        root_bytes: &[u8],
        build: impl FnOnce(E::Config, PoolOptions) -> Result<EnvPool<E, M>, EngineError>,
    ) -> Result<Self, SearchError> {
        if !E::PERFECT_INFORMATION {
            return Err(SearchError::ImperfectInformation);
        }
        let mut pool = Self::build_pool(cfg, options, root, root_bytes, build)?;
        pool.resolve_searcher(root, options.searcher, None, false)?;
        Ok(pool)
    }

    /// [`Self::new`] without the information check: the shared body of
    /// every constructor.
    pub(crate) fn build_pool(
        cfg: E::Config,
        options: SearchOptions,
        root: &Record<E>,
        root_bytes: &[u8],
        build: impl FnOnce(E::Config, PoolOptions) -> Result<EnvPool<E, M>, EngineError>,
    ) -> Result<Self, SearchError> {
        let joint = joint_heads::<E, M>(&cfg)?;
        let actions = joint.joint();
        let heads = joint.heads();
        if options.capacity.get() == NONE {
            return Err(options_error("capacity", "must be below u32::MAX"));
        }
        let capacity = options.capacity.get();
        let batch = options.batch.get();
        if batch > capacity {
            return Err(options_error("batch", "must not exceed the capacity"));
        }
        let pool_options = PoolOptions {
            auto_reset: AutoReset::Disabled,
            ..PoolOptions::new(
                NonZeroU32::new(batch).unwrap_or(NonZeroU32::MIN),
                Seed::new(0),
            )
        };
        let mut scratch = build(cfg, pool_options)?;
        refuse_privileged(&scratch)?;
        // The joint's mask offsets are the scratch pool's (SPEC 11.3).
        if M::ENABLED
            && (0..heads)
                .any(|h| scratch.mask_head_shape(h).map(|m| m.offset) != joint.mask_offset(h))
        {
            return Err(SearchError::UnsupportedActions);
        }
        // The scratch pool never learns statistics (ADR-0078 item 7.2).
        scratch.freeze_normalization(true);
        let stack = scratch.transform_state_bytes();
        if root_bytes.len() != stack {
            return Err(SearchError::StackedRoot {
                expected: stack,
                found: root_bytes.len(),
            });
        }
        let per_node = Self::bytes_per_node_with(actions, stack);
        let bytes = per_node.saturating_mul(u64::from(capacity));
        if bytes > options.budget_bytes {
            return Err(SearchError::Budget {
                bytes,
                per_node,
                budget: options.budget_bytes,
            });
        }
        scratch.reset_all(Reseed::Keep)?;
        let cap = idx(capacity);
        let agents = E::MAX_AGENTS;
        let rows = idx(batch).saturating_mul(agents).max(1);
        let outputs = scratch.outputs()?;
        let obs_row_bytes = (0..tessera_rl_engine::MAX_LEAVES)
            .map_while(|l| outputs.observation(l).map(<[u8]>::len))
            .map(|len| len / rows)
            .sum();
        let (obs_leaves, obs_sets) = scratch
            .observation_space()
            .ok()
            .and_then(|s| crate::obs_layout::observation_layout(&s))
            .unwrap_or_default();
        let mut pool = Self {
            obs_row_bytes,
            obs_leaves,
            obs_sets,
            records: vec![*root; cap],
            nodes: vec![Node::FREE; cap],
            edges: vec![Edge::EMPTY; cap.saturating_mul(idx(actions))],
            rewards: vec![0.0; cap.saturating_mul(agents)],
            free: (0..capacity).rev().collect(),
            root: NONE,
            next_alloc: 0,
            actions,
            joint,
            searcher: None,
            groups: Vec::new(),
            privileged_opponents: false,
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
            act_buf: vec![0; idx(batch).saturating_mul(agents).saturating_mul(heads)],
            child_records: vec![*root; idx(batch)],
            load_records: vec![*root; idx(batch)],
            child_terminal: vec![false; idx(batch)],
            child_rewards: vec![0.0; idx(batch).saturating_mul(agents)],
            child_masks: vec![true; idx(batch).saturating_mul(idx(actions))],
            journal: crate::journal::Journal::new(cap, cap.saturating_mul(idx(actions)), agents),
            mode: crate::determinize::RootMode::Full,
            stacks: crate::stacks::Stacks::new(stack, cap, idx(batch)),
        };
        pool.journal.reserve_stacks(stack.saturating_mul(cap));
        pool.replace_root(root)?;
        pool.stacks.set_slot(idx(pool.root), root_bytes);
        Ok(pool)
    }

    /// The exported leaves of one agent's evaluator row, in row order, with
    /// their byte offsets (#490); empty for an observation space that is
    /// not boxes, dicts and sets.
    #[must_use]
    pub fn observation_leaves(&self) -> &[crate::obs_layout::LeafLayout] {
        &self.obs_leaves
    }

    /// The `Set`s of the observation (ADR-0082 point 6): each set's item
    /// leaves and mask leaf, as indices into [`Self::observation_leaves`].
    #[must_use]
    pub fn observation_sets(&self) -> &[crate::obs_layout::SetLayout] {
        &self.obs_sets
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
        if self.mode == crate::determinize::RootMode::Determinized {
            return Err(SearchError::DeterminizedReroot);
        }
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
    /// [`SearchError::DeterminizedPool`] on a pool made by
    /// `new_determinized`, whose roots come only from
    /// `set_root_determinized`; a held node, an exhausted allocation
    /// counter, or no usable slot (nothing is changed).
    pub fn set_root(&mut self, record: &Record<E>) -> Result<NodeHandle, SearchError> {
        if self.mode == crate::determinize::RootMode::Determinized {
            return Err(SearchError::DeterminizedPool);
        }
        // A stacked scratch pool needs the root's frame stacks (ADR-0078
        // item 7.4): `set_root_with_transforms`.
        self.check_root_bytes(0)?;
        self.replace_root(record)
    }

    /// [`Self::set_root`] without the mode check.
    pub(crate) fn replace_root(&mut self, record: &Record<E>) -> Result<NodeHandle, SearchError> {
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

    /// Sets the searcher of a pool whose scratch pool has heuristic slots
    /// and checks the policy groups (ADR-0093 decisions 1 to 3): `option`
    /// is `SearchOptions::searcher`, `determinized` the determinization
    /// agent, `privileged` whether `PrivilegedHeuristic` opponents are
    /// allowed. Without heuristic slots nothing changes.
    ///
    /// The groups are checked for the slots alive in `root` (a slot that is
    /// never alive, such as the unused seats of a 16-seat environment
    /// configured for 4, needs no group); a slot that becomes alive and
    /// acting later is checked at expansion ([`Self::actor`]).
    pub(crate) fn resolve_searcher(
        &mut self,
        root: &Record<E>,
        option: Option<usize>,
        determinized: Option<usize>,
        privileged: bool,
    ) -> Result<(), SearchError> {
        let groups = self.scratch.policy_groups().to_vec();
        let heuristic = groups
            .iter()
            .any(|g| matches!(g, PolicyGroup::Heuristic | PolicyGroup::PrivilegedHeuristic));
        if !heuristic {
            self.searcher = None;
            return Ok(());
        }
        let searcher = match (option, determinized) {
            (Some(s), Some(d)) if s != d => {
                return Err(SearchError::SearcherMismatch {
                    searcher: s,
                    determinized: d,
                });
            }
            (Some(s), _) | (None, Some(s)) => s,
            (None, None) => return Err(SearchError::SearcherRequired),
        };
        if searcher >= self.agents {
            return Err(options_error("searcher", "must be an agent slot"));
        }
        let group = groups.get(searcher).copied().unwrap_or_default();
        if group != PolicyGroup::Learner {
            return Err(SearchError::SearcherNotLearner {
                slot: searcher,
                group,
            });
        }
        let alive = root.header().agents();
        for (slot, &group) in groups.iter().enumerate() {
            let live = alive.get(slot).is_some_and(|r| r.alive.get());
            if slot == searcher || !live {
                continue;
            }
            Self::check_opponent(slot, group, privileged)?;
        }
        self.searcher = Some(searcher);
        self.groups = groups;
        self.privileged_opponents = privileged;
        Ok(())
    }

    /// Whether `group` may play slot `slot` against the searcher (ADR-0093
    /// decision 3).
    fn check_opponent(
        slot: usize,
        group: PolicyGroup,
        privileged: bool,
    ) -> Result<(), SearchError> {
        match group {
            PolicyGroup::Heuristic => Ok(()),
            PolicyGroup::PrivilegedHeuristic if privileged => Ok(()),
            _ => Err(SearchError::NotSearchable { slot, group }),
        }
    }

    /// The searcher slot (ADR-0093), or `None` for a pool without
    /// heuristic slots.
    #[must_use]
    pub const fn searcher(&self) -> Option<usize> {
        self.searcher
    }

    /// The agent whose action an expansion of `record` takes: its single
    /// acting agent for the default policy map (ADR-0043), else the
    /// searcher when it is acting and `None` when it is not (ADR-0093
    /// decision 4).
    pub(crate) fn actor(&self, record: &Record<E>) -> Result<Option<usize>, SearchError> {
        match self.searcher {
            None => Self::single_actor(record).map(Some),
            Some(s) => {
                let rows = record.header().agents();
                for (slot, r) in rows.iter().enumerate() {
                    if slot != s && r.acting.get() {
                        let group = self.groups.get(slot).copied().unwrap_or_default();
                        Self::check_opponent(slot, group, self.privileged_opponents)?;
                    }
                }
                Ok(rows.get(s).filter(|r| r.acting.get()).map(|_| s))
            }
        }
    }

    /// The player whose view values take at `record`: the searcher at
    /// every node of a pool with heuristic slots (ADR-0093 decision 5),
    /// else the single acting agent.
    pub(crate) fn mover_of(&self, record: &Record<E>) -> Result<usize, SearchError> {
        match self.searcher {
            Some(s) => Ok(s),
            None => Self::single_actor(record),
        }
    }

    /// Writes the legal joints at `record` (scratch row `row`) into `out`:
    /// from the actor's mask row, or joint 0 alone when the searcher is
    /// not acting (ADR-0093 decision 4).
    pub(crate) fn legal_at(
        joint: &JointHeads,
        actor: &Result<Option<usize>, SearchError>,
        masks: &[Bool8],
        base_of: impl Fn(usize) -> usize,
        out: &mut [bool],
    ) {
        match actor {
            Ok(Some(k)) => fill_legal(joint, masks, Some(base_of(*k)), out),
            Ok(None) => {
                out.fill(false);
                if let Some(first) = out.first_mut() {
                    *first = true;
                }
            }
            Err(_) => fill_legal(joint, masks, None, out),
        }
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
            let agent = self.actor(record)?;
            if agent.is_none() && action != 0 {
                return Err(SearchError::IllegalAction { action });
            }
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
            self.stacks.load_from(row, idx(parent));
        }
        // `restore_all` recovers a scratch pool poisoned by an earlier
        // failed step (a contained kernel panic): every row is rewritten,
        // so nothing of the failed step survives.
        self.restore_rows()?;
        let masks = self.scratch.outputs()?.masks();
        let width = self.joint.mask_len();
        for (row, p) in self.planned.iter().enumerate() {
            // A searcher that is not acting has the single edge 0, checked
            // in the plan; its row is not read (ADR-0093 decision 4).
            let Some(agent) = p.agent else {
                continue;
            };
            let base = (row * self.agents + agent) * width;
            let legal = if masks.is_empty() {
                self.joint.is_canonical(p.action)
            } else {
                self.joint
                    .is_legal(p.action, |at| masks.get(base + at).is_some_and(|m| m.get()))
            };
            if !legal {
                return Err(SearchError::IllegalAction { action: p.action });
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
            let Some(agent) = p.agent else {
                continue;
            };
            let heads = self.joint.heads();
            let at = (row * self.agents + agent) * heads;
            if let Some(a) = self.act_buf.get_mut(at..at + heads) {
                self.joint.decode(p.action, a);
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
        let mask_len = self.joint.mask_len();
        for row in 0..self.planned.len() {
            let id = u32::try_from(row).unwrap_or(0);
            let record = self.scratch.snapshot(id)?;
            if let Some(r) = self.child_records.get_mut(row) {
                *r = record;
            }
            if self.stacks.per > 0 {
                self.scratch
                    .snapshot_transforms(id, self.stacks.child_mut(row))?;
            }
            let outputs = self.scratch.outputs()?;
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
            // The child's legal joints: from the mask of its acting agent
            // (every canonical joint without masks or without a single
            // actor), ADR-0086.
            let actor = self.actor(&record);
            let agents = self.agents;
            if let Some(out) = self.child_masks.get_mut(row * width..(row + 1) * width) {
                Self::legal_at(
                    &self.joint,
                    &actor,
                    outputs.masks(),
                    |k| (row * agents + k) * mask_len,
                    out,
                );
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
            self.stacks.commit(row, idx(slot));
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
    pub(crate) const fn obs_row_bytes(&self) -> usize {
        self.obs_row_bytes
    }

    /// Where the scratch rows' observations are read from.
    pub(crate) fn obs_source(&self) -> Result<ObsSource<'_>, SearchError> {
        Ok(ObsSource {
            outputs: self.scratch.outputs()?,
            rows: idx(self.batch).saturating_mul(self.agents).max(1),
            agents: self.agents,
        })
    }
}

/// The observations of the scratch pool's rows, read on demand.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ObsSource<'a> {
    outputs: &'a Outputs,
    /// Agent rows of the scratch outputs.
    rows: usize,
    agents: usize,
}

impl ObsSource<'_> {
    /// Copies the observation of `agent` at scratch row `row` into `out`
    /// (every observation leaf, in order).
    pub(crate) fn copy(&self, row: usize, agent: usize, out: &mut [u8]) {
        let mut at = 0;
        for l in 0..tessera_rl_engine::MAX_LEAVES {
            let Some(bytes) = self.outputs.observation(l) else {
                break;
            };
            let per = bytes.len() / self.rows;
            let from = (row * self.agents + agent) * per;
            if let (Some(src), Some(dst)) = (bytes.get(from..from + per), out.get_mut(at..at + per))
            {
                dst.copy_from_slice(src);
            }
            at += per;
        }
    }
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
    /// A digest of the whole tree (records, nodes, edges, rewards, the free
    /// list, the root and the counter, and the privileged label when
    /// set), for tests of atomicity and of determinism.
    #[must_use]
    pub fn tree_hash(&self) -> u64 {
        let mut out = Encoder::new(HashSink::new());
        for r in &self.records {
            out.write_bytes(&r.hash().get().to_le_bytes());
        }
        // The frame stacks of every node (ADR-0078 item 7); empty without.
        out.write_bytes(&self.stacks.tree);
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
        // A privileged tree never hashes equal to a non-privileged one.
        // The label is folded in only when set, so every other hash,
        // golden values included, is unchanged.
        if self.mode == crate::determinize::RootMode::Privileged {
            out.write_bytes(b"privileged");
        }
        out.sink().digest()
    }
}

fn slot_rewards(slot: u32, agents: usize) -> usize {
    idx(slot).saturating_mul(agents)
}

/// Search never builds or reads the privileged state buffer (ADR-0079 item
/// 7.3): a scratch pool that has it is refused.
fn refuse_privileged<E: Environment, M: MaskProvider<E>>(
    scratch: &EnvPool<E, M>,
) -> Result<(), SearchError> {
    if scratch.has_privileged_state() {
        return Err(SearchError::Engine(EngineError::PrivilegedState {
            reason: "a search scratch pool never has the privileged state buffer",
        }));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/support/fragile.rs"]
mod fragile;

#[cfg(test)]
mod poison_tests {
    use super::fragile::{CountConfig, Fragile};
    use super::*;
    use core::num::NonZeroU32;
    use tessera_rl_engine::{NoMasks, PoolOptions};
    use tessera_rl_num::identity::Seed;

    /// A failed expansion poisons the scratch pool: the observation source
    /// refuses rather than read rows, and the cached size stays right.
    #[test]
    fn test_obs_source_refuses_a_poisoned_scratch_pool() {
        let mut p = EnvPool::<Fragile>::new(
            CountConfig {},
            PoolOptions::new(NonZeroU32::MIN, Seed::new(1)),
        )
        .unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        let mut s = SearchPool::<Fragile, NoMasks>::new(
            CountConfig {},
            SearchOptions::new(NonZeroU32::new(8).unwrap(), NonZeroU32::new(2).unwrap()),
            &p.snapshot(0).unwrap(),
            EnvPool::new,
        )
        .unwrap();
        let bytes = s.obs_row_bytes();
        assert!(bytes > 0);
        assert!(s.obs_source().is_ok());
        let root = s.root();
        let mut out = [root; 2];
        s.expand(&[(root, 1), (root, 2)], &mut out).unwrap_err();
        assert!(matches!(
            s.obs_source(),
            Err(SearchError::Engine(EngineError::Poisoned))
        ));
        assert_eq!(s.obs_row_bytes(), bytes);
    }
}
