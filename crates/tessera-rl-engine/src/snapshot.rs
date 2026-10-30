// SPDX-License-Identifier: MIT OR Apache-2.0

//! Snapshot, restore and fork (ADR-0010, ADR-0007).
//!
//! Records are copied whole, then the instance is re-observed in the
//! canonical FP environment, so observations and masks always describe the
//! copied state. Pending resets travel with the record. Single-instance
//! operations never allocate.

use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout, StateHash};
use tessera_rl_num::identity::RngIdentity;
use tessera_rl_num::key::derive_branch_salt;

use crate::error::EngineError;
use crate::masks::MaskProvider;
use crate::pool::{EnvPool, clear_instance, observe_instance, run_kernels, write_agent_masks};
use crate::record::{PoolContext, Record};
use crate::validate::check_ids;

impl<E, M> EnvPool<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    fn index(&self, id: u32) -> Result<usize, EngineError> {
        let num_envs = self.options.num_envs.get();
        usize::try_from(id)
            .ok()
            .filter(|&i| i < self.headers.len())
            .ok_or(EngineError::InstanceOutOfRange { id, num_envs })
    }

    /// The continuation record of `id`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InstanceOutOfRange`] for an unknown id.
    pub fn snapshot(&self, id: u32) -> Result<Record<E>, EngineError> {
        let i = self.index(id)?;
        match (self.headers.get(i), self.states.get(i)) {
            (Some(&header), Some(&state)) => Ok(Record { header, state }),
            _ => Err(EngineError::InstanceOutOfRange {
                id,
                num_envs: self.options.num_envs.get(),
            }),
        }
    }

    /// Writes the records of every instance into `out`, which must have one
    /// slot per instance. Never allocates.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::RecordCount`] if `out` has the wrong length.
    pub fn snapshot_all(&self, out: &mut [Record<E>]) -> Result<(), EngineError> {
        if out.len() != self.headers.len() {
            return Err(EngineError::RecordCount {
                records: out.len(),
                instances: self.headers.len(),
            });
        }
        for ((slot, &header), &state) in out.iter_mut().zip(&self.headers).zip(&self.states) {
            *slot = Record { header, state };
        }
        Ok(())
    }

    /// The `StateHash` of instance `id` (header then state).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InstanceOutOfRange`] for an unknown id.
    pub fn state_hash(&self, id: u32) -> Result<StateHash, EngineError> {
        self.snapshot(id).map(|r| r.hash())
    }

    /// Restores `record` into slot `id`, from this or any other pool of the
    /// same environment. The instance then continues exactly as the
    /// original would have.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Poisoned`] on a poisoned pool (use
    /// [`EnvPool::restore_all`]), [`EngineError::InstanceOutOfRange`] for an
    /// unknown id, or [`EngineError::Panicked`] if re-observing panics.
    pub fn restore(&mut self, id: u32, record: &Record<E>) -> Result<(), EngineError> {
        if self.poisoned {
            return Err(EngineError::Poisoned);
        }
        let i = self.index(id)?;
        self.write_record(i, record);
        self.reobserve(core::iter::once(i))
    }

    /// Restores every instance from `records`, one per instance. On a
    /// poisoned pool this is a recovery path: the poison is cleared only if
    /// it completes.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::RecordCount`] if `records` has the wrong
    /// length (nothing is mutated), or [`EngineError::Panicked`] if
    /// re-observing panics.
    pub fn restore_all(&mut self, records: &[Record<E>]) -> Result<(), EngineError> {
        if records.len() != self.headers.len() {
            return Err(EngineError::RecordCount {
                records: records.len(),
                instances: self.headers.len(),
            });
        }
        for (i, record) in records.iter().enumerate() {
            self.write_record(i, record);
        }
        self.reobserve(0..records.len())?;
        self.poisoned = false;
        Ok(())
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
        let record = self.checked_fork_source(src, dst)?;
        for &id in dst {
            let i = self.index(id)?;
            self.write_record(i, &record);
        }
        self.reobserve(dst.iter().filter_map(|&id| usize::try_from(id).ok()))
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
        let record = self.checked_fork_source(src, dst)?;
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
        for (k, &id) in dst.iter().enumerate() {
            let label = match labels {
                Some(labels) => labels.get(k).copied().unwrap_or(0),
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
            let i = self.index(id)?;
            self.write_record(i, &branched);
        }
        self.reobserve(dst.iter().filter_map(|&id| usize::try_from(id).ok()))
    }

    /// The pool-level context, for pool snapshots.
    #[must_use]
    pub const fn pool_context(&self) -> PoolContext {
        PoolContext {
            branch_counter: self.branch_counter,
        }
    }

    /// Restores the pool-level context.
    pub const fn restore_pool_context(&mut self, context: PoolContext) {
        self.branch_counter = context.branch_counter;
    }

    fn checked_fork_source(&mut self, src: u32, dst: &[u32]) -> Result<Record<E>, EngineError> {
        if self.poisoned {
            return Err(EngineError::Poisoned);
        }
        let record = self.snapshot(src)?;
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
    /// flags cleared, alive and acting masks, observations and action
    /// masks for the current state.
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
            for i in indices {
                let (Some(header), Some(state)) = (headers.get(i), states.get(i)) else {
                    continue;
                };
                clear_instance(outputs, i);
                write_agent_masks(outputs, i, header);
                observe_instance::<E, M>(cfg, authority, header, state, outputs, i);
            }
        })
    }
}
