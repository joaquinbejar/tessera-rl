// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic eviction on ring exhaustion (ADR-0043 section 1).
//!
//! Never evictable: the root, every held node (pending leaves are held),
//! every parent of the current requests, and every ancestor of those.
//! Among the rest, the node with the lowest allocation number goes first,
//! with its whole subtree; a subtree of an evictable node holds no
//! protected node, because protection climbs to every ancestor. Planning
//! only marks scratch flags; the tree changes at the commit.

use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout};
use tessera_rl_engine::MaskProvider;

use crate::error::SearchError;
use crate::options::Eviction;
use crate::pool::{Edge, NONE, SearchPool};

impl<E, M> SearchPool<E, M>
where
    E: Environment<
            Config: Sync,
            State: Send,
            Header: Send + RecordLayout<Adaptive: AdaptiveValues<E::Config>>,
        >,
    M: MaskProvider<E>,
{
    /// Plans enough evictions for `need` new nodes into `evict_roots`.
    ///
    /// # Errors
    ///
    /// [`SearchError::CapacityExceeded`] in strict mode, or
    /// [`SearchError::Exhausted`] when too few nodes are evictable.
    pub(crate) fn plan_evictions(&mut self, need: usize) -> Result<(), SearchError> {
        self.evict_roots.clear();
        let available = self.free.len();
        if available >= need {
            return Ok(());
        }
        if self.eviction == Eviction::Strict {
            return Err(SearchError::CapacityExceeded {
                capacity: self.capacity,
            });
        }
        self.protected.fill(false);
        self.doomed.fill(false);
        self.protect_path(self.root);
        for slot in 0..self.capacity {
            if self.node(slot).is_some_and(|n| n.live && n.holds > 0) {
                self.protect_path(slot);
            }
        }
        for i in 0..self.planned.len() {
            if let Some(p) = self.planned.get(i).map(|p| p.parent) {
                self.protect_path(p);
            }
        }
        let mut freed = available;
        while freed < need {
            let Some(victim) = self.oldest_evictable() else {
                return Err(SearchError::Exhausted);
            };
            freed += self.doom_subtree(victim);
            self.evict_roots.push(victim);
        }
        Ok(())
    }

    /// The live, unprotected, not yet doomed node with the lowest
    /// allocation number.
    fn oldest_evictable(&self) -> Option<u32> {
        let mut best: Option<(u64, u32)> = None;
        for (slot, n) in (0u32..).zip(&self.nodes) {
            let blocked = self.protected.get(slot as usize).copied().unwrap_or(true)
                || self.doomed.get(slot as usize).copied().unwrap_or(true);
            if n.live && !blocked && best.is_none_or(|(a, _)| n.alloc < a) {
                best = Some((n.alloc, slot));
            }
        }
        best.map(|(_, slot)| slot)
    }

    /// Marks the subtree under `top` doomed; returns how many of its slots
    /// return to the free list (retiring slots do not).
    fn doom_subtree(&mut self, top: u32) -> usize {
        let mut usable = 0;
        self.stack.clear();
        self.stack.push(top);
        while let Some(slot) = self.stack.pop() {
            if let Some(flag) = self.doomed.get_mut(slot as usize) {
                *flag = true;
            }
            if self.node(slot).is_some_and(|n| n.generation != u32::MAX) {
                usable += 1;
            }
            for a in 0..self.actions {
                if let Some(&Edge { child, .. }) = self.edge(slot, a)
                    && child != NONE
                {
                    self.stack.push(child);
                }
            }
        }
        usable
    }

    /// Frees the planned subtrees, in plan order.
    pub(crate) fn commit_evictions(&mut self) {
        for i in 0..self.evict_roots.len() {
            if let Some(&top) = self.evict_roots.get(i) {
                self.free_subtree(top);
            }
        }
    }
}
