// SPDX-License-Identifier: MIT OR Apache-2.0

//! The rollback journal of a search call (ADR-0043 section 3b): a failed
//! call restores the tree exactly as it was before the call.
//!
//! While a call runs, the first write to a node slot saves the node, its
//! record and its rewards, and the first write to an edge saves the edge
//! (copy on write); the free list, the root and the allocation counter are
//! saved when the call starts. The journal holds at most one copy of each
//! slot and edge, so its storage is bounded by the tree's own and is
//! allocated at construction (it is counted in the memory budget).

use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout};
use tessera_rl_engine::{MaskProvider, Record};

use crate::pool::{Edge, Node, SearchPool};

/// Saved state of one call.
pub(crate) struct Journal<E: Environment> {
    pub(crate) on: bool,
    pub(crate) node_saved: Vec<bool>,
    pub(crate) nodes: Vec<(u32, Node, Record<E>)>,
    /// Rewards of the saved nodes, `agents` per entry, in `nodes` order.
    pub(crate) rewards: Vec<f32>,
    pub(crate) edge_saved: Vec<bool>,
    pub(crate) edges: Vec<(usize, Edge)>,
    pub(crate) free: Vec<u32>,
    pub(crate) root: u32,
    pub(crate) next_alloc: u64,
}

impl<E: Environment> Journal<E> {
    pub(crate) fn new(capacity: usize, edges: usize, agents: usize) -> Self {
        Self {
            on: false,
            node_saved: vec![false; capacity],
            nodes: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity.saturating_mul(agents)),
            edge_saved: vec![false; edges],
            edges: Vec::with_capacity(edges),
            free: Vec::with_capacity(capacity),
            root: 0,
            next_alloc: 0,
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
    /// Starts journaling a call.
    pub(crate) fn begin(&mut self) {
        let j = &mut self.journal;
        j.on = true;
        j.nodes.clear();
        j.rewards.clear();
        j.edges.clear();
        j.free.clear();
        j.free.extend_from_slice(&self.free);
        j.root = self.root;
        j.next_alloc = self.next_alloc;
    }

    /// Ends a successful call: the changes stay.
    pub(crate) fn finish(&mut self) {
        let j = &mut self.journal;
        for &(slot, ..) in &j.nodes {
            if let Some(f) = j.node_saved.get_mut(slot as usize) {
                *f = false;
            }
        }
        for &(i, _) in &j.edges {
            if let Some(f) = j.edge_saved.get_mut(i) {
                *f = false;
            }
        }
        j.on = false;
    }

    /// Ends a failed call: everything written since `begin` is restored.
    pub(crate) fn rollback(&mut self) {
        let agents = self.agents;
        let j = &mut self.journal;
        for (k, &(slot, node, record)) in j.nodes.iter().enumerate() {
            let s = slot as usize;
            if let Some(n) = self.nodes.get_mut(s) {
                *n = node;
            }
            if let Some(r) = self.records.get_mut(s) {
                *r = record;
            }
            let (from, to) = (s * agents, s * agents + agents);
            if let (Some(dst), Some(src)) = (
                self.rewards.get_mut(from..to),
                j.rewards.get(k * agents..k * agents + agents),
            ) {
                dst.copy_from_slice(src);
            }
        }
        for &(i, edge) in &j.edges {
            if let Some(e) = self.edges.get_mut(i) {
                *e = edge;
            }
        }
        self.free.clear();
        self.free.extend_from_slice(&j.free);
        self.root = j.root;
        self.next_alloc = j.next_alloc;
        self.finish();
    }

    /// Saves slot `slot` before its first write in a call.
    pub(crate) fn touch_node(&mut self, slot: u32) {
        let j = &mut self.journal;
        let s = slot as usize;
        if !j.on || j.node_saved.get(s).copied().unwrap_or(true) {
            return;
        }
        if let (Some(&node), Some(&record)) = (self.nodes.get(s), self.records.get(s)) {
            j.nodes.push((slot, node, record));
            if let Some(row) = self
                .rewards
                .get(s * self.agents..s * self.agents + self.agents)
            {
                j.rewards.extend_from_slice(row);
            }
            if let Some(f) = j.node_saved.get_mut(s) {
                *f = true;
            }
        }
    }

    /// Saves edge `i` before its first write in a call.
    pub(crate) fn touch_edge(&mut self, i: usize) {
        let j = &mut self.journal;
        if !j.on || j.edge_saved.get(i).copied().unwrap_or(true) {
            return;
        }
        if let Some(&edge) = self.edges.get(i) {
            j.edges.push((i, edge));
            if let Some(f) = j.edge_saved.get_mut(i) {
                *f = true;
            }
        }
    }
}
