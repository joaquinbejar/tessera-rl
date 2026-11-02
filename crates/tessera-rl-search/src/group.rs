// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`SearchGroup`]: independent [`Mcts`] trees whose leaves share one
//! evaluator batch per round (ADR-0045).
//!
//! A group call runs one ADR-0043 call on every active slot, interleaved
//! in slot order: one root batch, then each slot's root noise, then rounds
//! of collection, one evaluator call on the concatenated batch, validation
//! of every slot's rows, and each slot's commit and backup. The call is all
//! or nothing over its participants: any failure rolls every one of them
//! back through its journal.

use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout};
use tessera_rl_engine::{MaskProvider, Record};
use tessera_rl_num::rng::StepRng;
use thiserror::Error;

use crate::error::SearchError;
use crate::evaluator::{Evaluator, LeafBatch};
use crate::mcts::{CallStart, Mcts};
use crate::pool::SearchPool;

/// The most trees a group may hold.
pub const MAX_TREES: usize = 4_096;

/// Default bound on a group's aggregate memory: 16 GiB.
pub const DEFAULT_GROUP_BUDGET: u64 = 16 << 30;

/// A failed group call: the error, and the slot it belongs to (`None`
/// for an evaluator failure or an interruption, which belong to no slot).
#[derive(Debug, Clone, PartialEq, Error)]
#[error("{}{error}", slot.map_or_else(String::new, |s| format!("slot {s}: ")))]
pub struct GroupError {
    /// The slot that failed, if any.
    pub slot: Option<usize>,
    /// What failed.
    pub error: SearchError,
}

/// The record of the last failed call, frozen when it failed and cleared
/// at the start of every call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupFailure {
    /// The slot that failed, if any.
    pub slot: Option<usize>,
    /// The failure's message.
    pub message: String,
}

fn at(slot: usize) -> impl Fn(SearchError) -> GroupError {
    move |error| GroupError {
        slot: Some(slot),
        error,
    }
}

fn unslotted(error: SearchError) -> GroupError {
    GroupError { slot: None, error }
}

/// Independent PUCT trees sharing evaluator batches (ADR-0045).
pub struct SearchGroup<E: Environment, M: MaskProvider<E>> {
    slots: Vec<Mcts<E, M>>,
    active: Vec<bool>,
    actions: usize,
    obs_bytes: usize,
    /// Group batch: records, picks (unused once filled), observations,
    /// legal actions, responses, and each slot's row range.
    records: Vec<Record<E>>,
    picks: Vec<(usize, usize)>,
    obs: Vec<u8>,
    legal: Vec<bool>,
    raw: Vec<f64>,
    values: Vec<f64>,
    ranges: Vec<(usize, usize)>,
    /// Per call: the participants, their starts, streams, leaves of the
    /// round and simulations done.
    participants: Vec<usize>,
    starts: Vec<Option<CallStart>>,
    rngs: Vec<Option<StepRng>>,
    collected: Vec<u32>,
    done: Vec<u32>,
    last_failure: Option<GroupFailure>,
}

impl<E, M> SearchGroup<E, M>
where
    E: Environment<
            Config: Sync,
            State: Send,
            Header: Send + RecordLayout<Adaptive: AdaptiveValues<E::Config>>,
        >,
    M: MaskProvider<E>,
{
    /// The aggregate memory of `slots`, in bytes: every tree's node
    /// storage plus the group batch, or `None` on overflow.
    fn aggregate_bytes(
        slots: &[Mcts<E, M>],
        rows: usize,
        actions: usize,
        obs_bytes: usize,
    ) -> Option<u64> {
        let mut bytes = 0u64;
        for m in slots {
            let pool = m.pool();
            let per = SearchPool::<E, M>::bytes_per_node(pool.actions());
            bytes = bytes.checked_add(per.checked_mul(u64::from(pool.capacity()))?)?;
        }
        let per_row = obs_bytes
            .checked_add(actions.checked_mul(size_of::<bool>() + size_of::<f64>())?)?
            .checked_add(size_of::<f64>() + size_of::<Record<E>>() + size_of::<(usize, usize)>())?;
        bytes.checked_add(u64::try_from(rows.checked_mul(per_row)?).ok()?)
    }

    /// A group of `slots`, every slot active once it has a root (a slot
    /// is built with one). `budget_bytes` bounds the aggregate memory.
    ///
    /// # Errors
    ///
    /// [`SearchError::Options`] for 0 or more than [`MAX_TREES`] slots, or
    /// slots of different action counts or observation sizes;
    /// [`SearchError::Budget`] when the aggregate memory (computed with
    /// checked arithmetic) exceeds `budget_bytes`.
    pub fn new(slots: Vec<Mcts<E, M>>, budget_bytes: u64) -> Result<Self, SearchError> {
        let t = slots.len();
        if t == 0 || t > MAX_TREES {
            return Err(SearchError::Options {
                option: "trees",
                reason: "must be in 1..=4096",
            });
        }
        let first = slots.first().ok_or(SearchError::Options {
            option: "trees",
            reason: "must be at least 1",
        })?;
        let actions = usize::try_from(first.pool().actions()).unwrap_or(usize::MAX);
        let obs_bytes = first.obs_bytes();
        if slots.iter().any(|m| {
            usize::try_from(m.pool().actions()).ok() != Some(actions) || m.obs_bytes() != obs_bytes
        }) {
            return Err(SearchError::Options {
                option: "trees",
                reason: "every slot needs the same actions and observation size",
            });
        }
        // Rows: every slot's largest batch, at least one (its root).
        let rows = slots.iter().try_fold(0usize, |acc, m| {
            acc.checked_add(usize::try_from(m.config().batch_size).ok()?.max(1))
        });
        let bytes = rows.and_then(|r| Self::aggregate_bytes(&slots, r, actions, obs_bytes));
        let (Some(rows), Some(bytes)) = (rows, bytes) else {
            return Err(SearchError::Budget {
                bytes: u64::MAX,
                per_node: 0,
                budget: budget_bytes,
            });
        };
        if bytes > budget_bytes {
            let per_node = slots.first().map_or(0, |m| {
                SearchPool::<E, M>::bytes_per_node(m.pool().actions())
            });
            return Err(SearchError::Budget {
                bytes,
                per_node,
                budget: budget_bytes,
            });
        }
        Ok(Self {
            active: vec![true; t],
            actions,
            obs_bytes,
            records: Vec::with_capacity(rows),
            picks: Vec::with_capacity(rows),
            obs: vec![0; rows.saturating_mul(obs_bytes)],
            legal: vec![false; rows.saturating_mul(actions)],
            raw: vec![0.0; rows.saturating_mul(actions)],
            values: vec![0.0; rows],
            ranges: vec![(0, 0); t],
            participants: Vec::with_capacity(t),
            starts: vec![None; t],
            rngs: vec![None; t],
            collected: vec![0; t],
            done: vec![0; t],
            last_failure: None,
            slots,
        })
    }

    /// The aggregate memory bound of this group, in bytes (what
    /// construction checked against the budget).
    #[must_use]
    pub fn memory_bytes(&self) -> u64 {
        Self::aggregate_bytes(&self.slots, self.values.len(), self.actions, self.obs_bytes)
            .unwrap_or(u64::MAX)
    }

    /// Slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the group has no slot (never: construction needs one).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Slot `i`'s tree.
    #[must_use]
    pub fn slot(&self, i: usize) -> Option<&Mcts<E, M>> {
        self.slots.get(i)
    }

    /// Slot `i`'s tree, to re-root it or select an action between calls.
    pub fn slot_mut(&mut self, i: usize) -> Option<&mut Mcts<E, M>> {
        self.slots.get_mut(i)
    }

    /// Whether slot `i` takes part in the next call.
    #[must_use]
    pub fn active(&self, i: usize) -> Option<bool> {
        self.active.get(i).copied()
    }

    /// Sets whether slot `i` takes part in the next call; `false` if `i`
    /// is out of range.
    pub fn set_active(&mut self, i: usize, flag: bool) -> bool {
        self.active.get_mut(i).map(|a| *a = flag).is_some()
    }

    /// The last failed call, cleared at the start of every call.
    #[must_use]
    pub const fn last_failure(&self) -> Option<&GroupFailure> {
        self.last_failure.as_ref()
    }

    /// One group call (ADR-0045 section 2), under the canonical
    /// floating-point controls.
    ///
    /// # Errors
    ///
    /// The first failure, with its slot (see [`GroupError`]); every
    /// participant is then exactly as before the call.
    pub fn run(&mut self, evaluator: &mut impl Evaluator<E>) -> Result<(), GroupError> {
        self.last_failure = None;
        let result = tessera_rl_num::fp::with_canonical(|| self.run_canonical(evaluator));
        if let Err(e) = &result {
            self.last_failure = Some(GroupFailure {
                slot: e.slot,
                message: e.error.to_string(),
            });
        }
        result
    }

    fn run_canonical(&mut self, evaluator: &mut impl Evaluator<E>) -> Result<(), GroupError> {
        self.participants.clear();
        self.participants.extend(
            (0..self.slots.len()).filter(|&i| self.active.get(i).copied().unwrap_or(false)),
        );
        if self.participants.is_empty() {
            return Ok(());
        }
        // Checks that change nothing, in slot order.
        for &i in &self.participants {
            if self.slots.get(i).is_some_and(Mcts::root_terminal) {
                return Err(at(i)(SearchError::TerminalNode));
            }
        }
        // Every journal is opened before any mutation of the trees.
        for p in 0..self.participants.len() {
            let i = self.participants.get(p).copied().unwrap_or(0);
            let begun = self.slots.get_mut(i).map(Mcts::call_begin);
            match begun {
                Some(Ok(start)) => {
                    if let Some(s) = self.starts.get_mut(i) {
                        *s = Some(start);
                    }
                }
                Some(Err(e)) => {
                    self.rollback(p);
                    return Err(at(i)(e));
                }
                None => {
                    self.rollback(p);
                    return Err(at(i)(SearchError::TerminalNode));
                }
            }
        }
        match self.call(evaluator) {
            Ok(()) => {
                for &i in &self.participants {
                    let start = self.starts.get_mut(i).and_then(Option::take);
                    let rng = self.rngs.get_mut(i).and_then(Option::take);
                    if let (Some(m), Some(start), Some(rng)) = (self.slots.get_mut(i), start, rng) {
                        m.call_commit(start.calls, rng);
                    }
                }
                Ok(())
            }
            Err(e) => {
                self.rollback(self.participants.len());
                Err(e)
            }
        }
    }

    /// Rolls back the first `begun` participants.
    fn rollback(&mut self, begun: usize) {
        for p in 0..begun {
            let i = self.participants.get(p).copied().unwrap_or(0);
            if let Some(m) = self.slots.get_mut(i) {
                m.call_rollback();
            }
            if let Some(s) = self.starts.get_mut(i) {
                *s = None;
            }
            if let Some(r) = self.rngs.get_mut(i) {
                *r = None;
            }
        }
    }

    fn call(&mut self, evaluator: &mut impl Evaluator<E>) -> Result<(), GroupError> {
        self.prologue(evaluator)?;
        while self.round(evaluator)? {}
        Ok(())
    }

    /// The poll, the root batch and each slot's root noise.
    fn prologue(&mut self, evaluator: &mut impl Evaluator<E>) -> Result<(), GroupError> {
        evaluator.check().map_err(|e| unslotted(e.into()))?;
        // One root batch, in slot order.
        let mut any_root = false;
        for p in 0..self.participants.len() {
            let i = self.participants.get(p).copied().unwrap_or(0);
            let Some(m) = self.slots.get_mut(i) else {
                continue;
            };
            let unevaluated = m.root_unevaluated();
            if unevaluated {
                m.load_root().map_err(at(i))?;
                any_root = true;
            }
            if let Some(r) = self.ranges.get_mut(i) {
                *r = (0, usize::from(unevaluated));
            }
        }
        if any_root {
            self.evaluate_group(evaluator)?;
            for p in 0..self.participants.len() {
                let i = self.participants.get(p).copied().unwrap_or(0);
                let (from, k) = self.ranges.get(i).copied().unwrap_or((0, 0));
                if k == 0 {
                    continue;
                }
                self.route(i, from, k);
                if let Some(m) = self.slots.get_mut(i) {
                    m.store_root().map_err(at(i))?;
                }
            }
        }
        // Each slot's root noise, from its own stream, in slot order.
        for p in 0..self.participants.len() {
            let i = self.participants.get(p).copied().unwrap_or(0);
            let rng = self
                .starts
                .get(i)
                .and_then(|s| s.as_ref())
                .map(|s| s.rng.clone());
            if let (Some(m), Some(mut rng)) = (self.slots.get_mut(i), rng) {
                m.prepare_noise(&mut rng);
                if let Some(r) = self.rngs.get_mut(i) {
                    *r = Some(rng);
                }
            }
            if let Some(d) = self.done.get_mut(i) {
                *d = 0;
            }
        }
        Ok(())
    }

    /// One round; `false` when every participant's budget is spent.
    fn round(&mut self, evaluator: &mut impl Evaluator<E>) -> Result<bool, GroupError> {
        let pending = self.participants.iter().any(|&i| {
            let done = self.done.get(i).copied().unwrap_or(u32::MAX);
            self.slots.get(i).and_then(|m| m.want(done)).is_some()
        });
        if !pending {
            return Ok(false);
        }
        evaluator.check().map_err(|e| unslotted(e.into()))?;
        let mut rows = 0usize;
        for p in 0..self.participants.len() {
            let i = self.participants.get(p).copied().unwrap_or(0);
            let done = self.done.get(i).copied().unwrap_or(u32::MAX);
            let Some(m) = self.slots.get_mut(i) else {
                continue;
            };
            let (n, k) = match m.want(done) {
                Some(want) => (m.collect_and_stage(want).map_err(at(i))?, m.eval_len()),
                None => (0, 0),
            };
            if let Some(c) = self.collected.get_mut(i) {
                *c = n;
            }
            if let Some(r) = self.ranges.get_mut(i) {
                *r = (0, k);
            }
            rows = rows.saturating_add(k);
        }
        if rows > 0 {
            self.evaluate_group(evaluator)?;
        }
        // Every slot's rows are validated before any slot commits.
        for p in 0..self.participants.len() {
            let i = self.participants.get(p).copied().unwrap_or(0);
            let (from, k) = self.ranges.get(i).copied().unwrap_or((0, 0));
            if k == 0 {
                continue;
            }
            self.route(i, from, k);
            if let Some(m) = self.slots.get_mut(i) {
                m.validate_staged(k).map_err(at(i))?;
            }
        }
        for p in 0..self.participants.len() {
            let i = self.participants.get(p).copied().unwrap_or(0);
            let done = self.done.get(i).copied().unwrap_or(u32::MAX);
            let Some(m) = self.slots.get_mut(i) else {
                continue;
            };
            if m.want(done).is_none() {
                continue;
            }
            let k = self.ranges.get(i).map_or(0, |r| r.1);
            m.commit_and_backup(k).map_err(at(i))?;
            let n = self.collected.get(i).copied().unwrap_or(0);
            if let Some(d) = self.done.get_mut(i) {
                *d = d.saturating_add(n.max(1));
            }
        }
        Ok(true)
    }

    /// Composes the group batch from each participant's gathered leaves
    /// (their `ranges` hold the counts; this sets the offsets), in slot
    /// order, and evaluates it in one call.
    fn evaluate_group(&mut self, evaluator: &mut impl Evaluator<E>) -> Result<(), GroupError> {
        let (ob, width) = (self.obs_bytes, self.actions);
        self.records.clear();
        self.picks.clear();
        let mut off = 0usize;
        let mut source = None;
        for p in 0..self.participants.len() {
            let i = self.participants.get(p).copied().unwrap_or(0);
            let k = self.ranges.get(i).map_or(0, |r| r.1);
            if let Some(r) = self.ranges.get_mut(i) {
                *r = (off, k);
            }
            if k == 0 {
                continue;
            }
            let Some(m) = self.slots.get(i) else { continue };
            let g = m.gathered();
            source.get_or_insert(g.source);
            self.records.extend_from_slice(g.records);
            self.picks.extend_from_slice(g.picks);
            if let (Some(dst), Some(src)) = (
                self.legal.get_mut(off * width..(off + k) * width),
                g.legal.get(..k * width),
            ) {
                dst.copy_from_slice(src);
            }
            // The slot copies its own observations into its rows.
            let mut slot_batch = LeafBatch::new(
                g.records,
                g.source,
                g.picks,
                self.obs
                    .get_mut(off * ob..(off + k) * ob)
                    .unwrap_or(&mut []),
                ob,
                g.legal,
                width,
            );
            let _ = slot_batch.observations();
            off += k;
        }
        let Some(source) = source else {
            return Ok(());
        };
        let mut batch = LeafBatch::prefilled(
            &self.records,
            source,
            &self.picks,
            self.obs.get_mut(..off * ob).unwrap_or(&mut []),
            ob,
            self.legal.get(..off * width).unwrap_or(&[]),
            width,
        );
        let (raw, values) = (
            self.raw.get_mut(..off * width).unwrap_or(&mut []),
            self.values.get_mut(..off).unwrap_or(&mut []),
        );
        evaluator
            .evaluate_batch(&mut batch, raw, values)
            .map_err(|e| unslotted(e.into()))
    }

    /// Copies rows `from..from + k` of the group response into slot `i`.
    fn route(&mut self, i: usize, from: usize, k: usize) {
        let width = self.actions;
        let raw = self
            .raw
            .get(from * width..(from + k) * width)
            .unwrap_or(&[]);
        let values = self.values.get(from..from + k).unwrap_or(&[]);
        if let Some(m) = self.slots.get_mut(i) {
            m.put_response(raw, values);
        }
    }
}
