// SPDX-License-Identifier: MIT OR Apache-2.0

//! Snapshot, restore, import and fork (ADR-0010, ADR-0007, ADR-0029,
//! ADR-0039).
//!
//! Records are copied whole, then the destination's per-step outputs
//! (flags, rewards, terminal observation rows) are cleared and it is
//! re-observed in the canonical FP environment, so observations and masks
//! always describe the copied state and the result is deterministic.
//! Pending resets travel with the record. Single-instance operations never
//! allocate.
//!
//! Every operation goes through the pool's guarded path. On a poisoned
//! pool everything here is rejected except `restore_all` and `import_all`,
//! which are recovery paths (ADR-0028).
//!
//! # Restoring into another pool
//!
//! A record continues identically in another pool when both pools have the
//! same environment config (static fields and assets), the same spaces, the
//! same step limit, auto-reset mode, mask provider and illegal-action
//! policy. The seed, slot and pool size may differ. Checking a pool
//! fingerprint is deferred to the persistent format (ADR-0011, phase 2).

use tessera_rl_core::{
    AdaptiveValues, AgentSlot, Environment, RecordLayout, StateHash, TurnStructure, Validate,
};

use crate::error::EngineError;
use crate::masks::MaskProvider;
use crate::options::AutoReset;
use crate::pool::{
    EnvPool, Poison, clear_instance, clear_terminal, observe_instance, run_kernels,
    write_agent_masks,
};
use crate::record::{DecodedRecord, PoolContext, Record};
use crate::validate::check_ids;

impl<E, M> EnvPool<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    fn index(&self, id: u32) -> Result<usize, EngineError> {
        let num_envs = self.options.num_envs.get();
        usize::try_from(id)
            .ok()
            .filter(|&i| i < self.headers.len())
            .ok_or(EngineError::InstanceOutOfRange { id, num_envs })
    }

    fn record(&self, i: usize) -> Option<Record<E>> {
        match (self.headers.get(i), self.states.get(i)) {
            (Some(&header), Some(&state)) => Some(Record { header, state }),
            _ => None,
        }
    }

    /// The continuation record of `id`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Poisoned`] on a poisoned pool, or
    /// [`EngineError::InstanceOutOfRange`] for an unknown id.
    pub fn snapshot(&self, id: u32) -> Result<Record<E>, EngineError> {
        self.guarded_ref(|pool| {
            let i = pool.index(id)?;
            pool.record(i).ok_or(EngineError::InstanceOutOfRange {
                id,
                num_envs: pool.options.num_envs.get(),
            })
        })
    }

    /// Writes the records of every instance into `out`, which must have one
    /// slot per instance. Never allocates.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Poisoned`] on a poisoned pool, or
    /// [`EngineError::RecordCount`] if `out` has the wrong length.
    pub fn snapshot_all(&self, out: &mut [Record<E>]) -> Result<(), EngineError> {
        self.guarded_ref(|pool| {
            pool.check_count(out.len())?;
            for (i, slot) in out.iter_mut().enumerate() {
                if let Some(record) = pool.record(i) {
                    *slot = record;
                }
            }
            Ok(())
        })
    }

    /// The `StateHash` of instance `id` (header then state).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Poisoned`] on a poisoned pool, or
    /// [`EngineError::InstanceOutOfRange`] for an unknown id.
    pub fn state_hash(&self, id: u32) -> Result<StateHash, EngineError> {
        self.snapshot(id).map(|r| r.hash())
    }

    /// Restores a trusted `record` (from [`EnvPool::snapshot`]) into slot
    /// `id` of this or another compatible pool (see the module docs). The
    /// instance then continues exactly as the original would have.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Poisoned`] on a poisoned pool (use
    /// [`EnvPool::restore_all`]), [`EngineError::InstanceOutOfRange`] for an
    /// unknown id, or [`EngineError::Panicked`] if re-observing panics.
    pub fn restore(&mut self, id: u32, record: &Record<E>) -> Result<(), EngineError> {
        self.guarded(Poison::Reject, |pool| {
            let i = pool.index(id)?;
            pool.write_record(i, record);
            pool.reobserve(core::iter::once(i))
        })
    }

    /// Restores every instance from trusted `records`, one per instance. On
    /// a poisoned pool this is a recovery path: the poison is cleared only
    /// if it completes.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::RecordCount`] if `records` has the wrong
    /// length (nothing is mutated), or [`EngineError::Panicked`] if
    /// re-observing panics.
    pub fn restore_all(&mut self, records: &[Record<E>]) -> Result<(), EngineError> {
        self.guarded(Poison::Recover, |pool| {
            pool.check_count(records.len())?;
            for (i, record) in records.iter().enumerate() {
                pool.write_record(i, record);
            }
            pool.reobserve(0..records.len())?;
            pool.poisoned = false;
            Ok(())
        })
    }

    /// Copies instance `src` into every instance of `dst`, identity
    /// included: given the same actions, each destination has exactly the
    /// future of the source.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Poisoned`], unknown or duplicate ids, or
    /// [`EngineError::Panicked`] if re-observing panics; on an id error
    /// nothing is mutated.
    pub fn fork(&mut self, src: u32, dst: &[u32]) -> Result<(), EngineError> {
        self.guarded(Poison::Reject, |pool| {
            let record = pool.fork_source(src, dst)?;
            for i in valid_indices(dst) {
                pool.write_record(i, &record);
            }
            pool.reobserve(valid_indices(dst))
        })
    }

    /// Like [`EnvPool::fork`], but each destination gets a distinct branch
    /// salt, so its randomness diverges (ADR-0007):
    /// `derive_branch_salt(parent salt, branch_counter, label)`. Labels are
    /// `labels[k]`, or the destination's position `k` in `dst` when `None`;
    /// they must be unique. The pool's branch counter is incremented once
    /// per call.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::DuplicateLabel`], [`EngineError::LabelCount`],
    /// [`EngineError::BranchCounterExhausted`] and the errors of
    /// [`EnvPool::fork`]; nothing is mutated on these errors.
    pub fn fork_branched(
        &mut self,
        src: u32,
        dst: &[u32],
        labels: Option<&[u64]>,
    ) -> Result<(), EngineError> {
        self.guarded(Poison::Reject, |pool| {
            pool.fork_branched_inner(src, dst, labels)
        })
    }

    fn fork_branched_inner(
        &mut self,
        src: u32,
        dst: &[u32],
        labels: Option<&[u64]>,
    ) -> Result<(), EngineError> {
        use tessera_rl_num::identity::RngIdentity;
        use tessera_rl_num::key::derive_branch_salt;

        let record = self.fork_source(src, dst)?;
        if let Some(labels) = labels {
            if labels.len() != dst.len() {
                return Err(EngineError::LabelCount {
                    labels: labels.len(),
                    destinations: dst.len(),
                });
            }
            for (k, label) in labels.iter().enumerate() {
                if labels.iter().take(k).any(|l| l == label) {
                    return Err(EngineError::DuplicateLabel { label: *label });
                }
            }
        }
        let counter = self.branch_counter;
        self.branch_counter = counter
            .checked_add(1)
            .ok_or(EngineError::BranchCounterExhausted)?;
        let parent = record.header.engine().identity;
        for (k, i) in valid_indices(dst).enumerate() {
            // Labels were checked to have one entry per destination.
            let label = match labels {
                Some(labels) => match labels.get(k) {
                    Some(&label) => label,
                    None => continue,
                },
                None => u64::try_from(k).unwrap_or(u64::MAX),
            };
            let salt = derive_branch_salt(parent.branch_salt(), counter, label);
            let mut branched = record;
            branched.header.engine_mut().identity = RngIdentity::from_parts(
                parent.seed(),
                salt,
                parent.logical_id(),
                parent.episode_index(),
                parent.step_index(),
            );
            self.write_record(i, &branched);
        }
        self.reobserve(valid_indices(dst))
    }

    /// The pool-level context, for pool snapshots.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Poisoned`] on a poisoned pool.
    pub fn pool_context(&self) -> Result<PoolContext, EngineError> {
        self.guarded_ref(|pool| {
            Ok(PoolContext {
                branch_counter: pool.branch_counter,
            })
        })
    }

    /// Restores the pool-level context.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Poisoned`] on a poisoned pool.
    pub fn restore_pool_context(&mut self, context: PoolContext) -> Result<(), EngineError> {
        self.guarded(Poison::Reject, |pool| {
            pool.branch_counter = context.branch_counter;
            Ok(())
        })
    }

    fn check_count(&self, records: usize) -> Result<(), EngineError> {
        if records == self.headers.len() {
            Ok(())
        } else {
            Err(EngineError::RecordCount {
                records,
                instances: self.headers.len(),
            })
        }
    }

    fn fork_source(&mut self, src: u32, dst: &[u32]) -> Result<Record<E>, EngineError> {
        let i = self.index(src)?;
        let record = self.record(i).ok_or(EngineError::InstanceOutOfRange {
            id: src,
            num_envs: self.options.num_envs.get(),
        })?;
        check_ids(dst, &mut self.seen)?;
        Ok(record)
    }

    fn write_record(&mut self, i: usize, record: &Record<E>) {
        if let (Some(header), Some(state)) = (self.headers.get_mut(i), self.states.get_mut(i)) {
            *header = record.header;
            *state = record.state;
        }
    }

    /// Recomputes the outputs of the given instances from their records:
    /// per-step outputs and terminal rows cleared (ADR-0039), alive and
    /// acting masks, observations and action masks for the current state.
    fn reobserve(&mut self, indices: impl Iterator<Item = usize>) -> Result<(), EngineError> {
        let Self {
            cfg,
            headers,
            states,
            outputs,
            authority,
            poisoned,
            ..
        } = self;
        run_kernels(poisoned, || {
            let outputs = &mut outputs.view_all();
            for i in indices {
                let (Some(header), Some(state)) = (headers.get(i), states.get(i)) else {
                    continue;
                };
                clear_instance(outputs, i);
                clear_terminal(outputs, i);
                write_agent_masks(outputs, i, header);
                observe_instance::<E, M>(cfg, authority, header, state, outputs, i);
            }
        })
    }

    fn check_import(&self, index: usize, record: &DecodedRecord<E>) -> Result<(), EngineError>
    where
        E: Validate,
    {
        check_record::<E>(&self.cfg, &self.options, index, record)
    }

    /// Restores untrusted records (decoded from bytes) into `ids`, in
    /// order. Every record is checked first (engine header consistency and
    /// `Validate::validate_record`, ADR-0040); then all are written. Atomic: on any error nothing is
    /// mutated.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Poisoned`], id and count errors,
    /// [`EngineError::InvalidRecord`], [`EngineError::Invariant`], or
    /// [`EngineError::Panicked`] if re-observing panics.
    pub fn import(&mut self, ids: &[u32], records: &[DecodedRecord<E>]) -> Result<(), EngineError>
    where
        E: Validate,
    {
        self.guarded(Poison::Reject, |pool| {
            check_ids(ids, &mut pool.seen)?;
            if ids.len() != records.len() {
                return Err(EngineError::RecordCount {
                    records: records.len(),
                    instances: ids.len(),
                });
            }
            for (index, record) in records.iter().enumerate() {
                pool.check_import(index, record)?;
            }
            for (i, record) in valid_indices(ids).zip(records) {
                pool.write_record(i, &record.trust());
            }
            pool.reobserve(valid_indices(ids))
        })
    }

    /// [`EnvPool::import`] for every instance. A recovery path: on a
    /// poisoned pool the poison is cleared only if every record is valid and
    /// the import completes; a failed check leaves the pool (and its
    /// poison) unchanged.
    ///
    /// # Errors
    ///
    /// As [`EnvPool::import`], without the poison rejection.
    pub fn import_all(&mut self, records: &[DecodedRecord<E>]) -> Result<(), EngineError>
    where
        E: Validate,
    {
        self.guarded(Poison::Recover, |pool| {
            pool.check_count(records.len())?;
            for (index, record) in records.iter().enumerate() {
                pool.check_import(index, record)?;
            }
            for (i, record) in records.iter().enumerate() {
                pool.write_record(i, &record.trust());
            }
            pool.reobserve(0..records.len())?;
            pool.poisoned = false;
            Ok(())
        })
    }
}

/// Checks an untrusted record: the engine header against this pool,
/// then the environment's invariants.
pub(crate) fn check_record<E: Environment + Validate>(
    cfg: &E::Config,
    options: &crate::options::PoolOptions,
    index: usize,
    record: &DecodedRecord<E>,
) -> Result<(), EngineError> {
    let invalid = |reason| Err(EngineError::InvalidRecord { index, reason });
    let header = record.header();
    let engine = header.engine();
    let pending = engine.pending_reset.get();
    if !pending && engine.end_kind != 0 {
        return invalid("an episode end without a pending reset");
    }
    if pending && !engine.fresh.get() && engine.end_kind == 0 {
        return invalid("a pending reset without an episode end");
    }
    // The engine sets `fresh` only on a never-reset record: pending
    // reset, no episode end, no steps, episode and step index 0. Any
    // other `fresh` record would make the next reset reuse an episode
    // index, and with it RNG counters (ADR-0007).
    if engine.fresh.get()
        && !(pending
            && engine.end_kind == 0
            && engine.episode_steps == 0
            && engine.identity.episode_index() == 0
            && engine.identity.step_index() == 0)
    {
        return invalid("a fresh record that is not a never-reset record");
    }
    if engine.identity.step_index() != engine.episode_steps {
        return invalid("the RNG step index differs from the episode steps");
    }
    if let Some(limit) = options.max_episode_steps
        && engine.episode_steps > limit.get()
    {
        return invalid("an episode longer than the step limit");
    }
    if pending && !engine.fresh.get() && options.auto_reset == AutoReset::SameStep {
        return invalid("a pending reset in same_step mode");
    }
    for row in header.agents() {
        if row.acting.get() && !row.alive.get() {
            return invalid("an acting agent that is not alive");
        }
        if row.illegal_actions > engine.episode_steps {
            return invalid("more illegal actions than steps");
        }
        if u32::from(row.invalid_agent_events) > engine.episode_steps {
            return invalid("more invalid agent events than steps");
        }
        if !row.episode_return.is_finite() {
            return invalid("a non-finite episode return");
        }
    }
    let view = tessera_rl_core::HeaderView::of(header);
    E::validate_record(cfg, &view, record.state())
        .map_err(|violation| EngineError::Invariant { index, violation })?;
    // Turn order (ADR-0013), checked only after the record passed
    // `validate_record`, so `acting` never sees an invalid state:
    // outside a pending reset, the acting flags
    // of a non-simultaneous environment are exactly the ones its
    // `acting` names for this state, so a record cannot make a waiting
    // agent act.
    if E::TURNS != TurnStructure::Simultaneous && !pending {
        for (slot, row) in header.agents().iter().enumerate() {
            let expected = row.alive.get()
                && u16::try_from(slot)
                    .ok()
                    .and_then(|s| AgentSlot::new(s, E::MAX_AGENTS))
                    .is_some_and(|s| E::acting(cfg, record.state(), s));
            if row.acting.get() != expected {
                return invalid("acting flags that differ from the environment's turn order");
            }
        }
        // A Sequential game that goes on has exactly one (alive) agent
        // to move; a record without one would let the next step run on
        // an ignored action. Mixed may have none.
        if E::TURNS == TurnStructure::Sequential
            && header.agents().iter().filter(|r| r.acting.get()).count() != 1
        {
            return invalid("a Sequential record that goes on without exactly one acting agent");
        }
    }
    Ok(())
}

/// The ids as indices; ids were validated before this is called.
fn valid_indices(ids: &[u32]) -> impl Iterator<Item = usize> + '_ {
    ids.iter().filter_map(|&id| usize::try_from(id).ok())
}
