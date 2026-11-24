// SPDX-License-Identifier: MIT OR Apache-2.0

//! Whole-pool snapshots with their context (ADR-0075 item 7, ADR-0078
//! item 6): every record, every frame stack and the full [`PoolContext`]
//! (branch counter, reset source, transform identity and statistics), and
//! `restore_pool`, which checks all of it before any write.

use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout};

use crate::archive::Archive;
use crate::error::EngineError;
use crate::masks::MaskProvider;
use crate::pool::{EnvPool, Poison};
use crate::record::{PoolContext, Record};
use crate::transform::kernels::StackMode;

/// A whole-pool snapshot: what [`EnvPool::restore_pool`] needs to continue
/// exactly, archive resets and transforms included. In memory only; the
/// archive bytes are not in it (ADR-0075 item 7).
pub struct PoolState<E: Environment> {
    pub(crate) records: Vec<Record<E>>,
    pub(crate) transforms: Vec<u8>,
    pub(crate) context: PoolContext,
}

impl<E: Environment> Clone for PoolState<E> {
    fn clone(&self) -> Self {
        Self {
            records: self.records.clone(),
            transforms: self.transforms.clone(),
            context: self.context.clone(),
        }
    }
}

impl<E: Environment> PoolState<E> {
    /// The pool context it restores.
    #[must_use]
    pub fn context(&self) -> &PoolContext {
        &self.context
    }

    /// Every instance's frame stacks, in instance order (empty without a
    /// `FrameStack`).
    #[must_use]
    pub fn transforms(&self) -> &[u8] {
        &self.transforms
    }

    /// The instance records, in instance order.
    #[must_use]
    pub fn records(&self) -> &[Record<E>] {
        &self.records
    }
}

impl<E, M> EnvPool<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    /// A whole-pool snapshot: records, frame stacks and context. Allocates
    /// (not a step-path call).
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`] on a poisoned pool.
    pub fn snapshot_pool(&self) -> Result<PoolState<E>, EngineError> {
        let context = self.pool_context()?;
        // Instance 0 sizes the buffer; `snapshot_all` overwrites every slot.
        let template = self.snapshot(0)?;
        let mut records = vec![template; self.headers.len()];
        self.snapshot_all(&mut records)?;
        let mut transforms = vec![0; self.state_bytes_for(records.len())?];
        self.snapshot_transforms_all(&mut transforms)?;
        Ok(PoolState {
            records,
            transforms,
            context,
        })
    }

    /// Restores a [`PoolState`] (ADR-0075 item 7): records, frame stacks and
    /// context, as an exact continuation. Every check comes before any
    /// write, so an error mutates nothing: the record count, the frame-stack
    /// length, the transform identity (ADR-0078 item 2.3), and the archive,
    /// which must be given and match when the context names one and must be
    /// absent for fresh resets. On a poisoned pool this is a recovery path,
    /// as [`EnvPool::restore_all`] is.
    ///
    /// # Errors
    ///
    /// [`EngineError::RecordCount`], [`EngineError::ArchiveMismatch`],
    /// [`EngineError::Archive`], a transform error, or
    /// [`EngineError::Panicked`] if re-observing panics.
    pub fn restore_pool(
        &mut self,
        state: &PoolState<E>,
        archive: Option<&Archive<E>>,
    ) -> Result<(), EngineError> {
        let result = self.guarded(Poison::Recover, |pool| {
            pool.check_count(state.records.len())?;
            pool.check_state_len(state.transforms.len(), state.records.len())?;
            let (branch_counter, source, transforms) =
                pool.check_context(state.context.clone(), archive)?;
            for (i, record) in state.records.iter().enumerate() {
                pool.write_record(i, record);
            }
            pool.branch_counter = branch_counter;
            pool.reset_source = source;
            if let Some(t) = pool.transform.as_mut() {
                t.arena.copy_from_slice(&state.transforms);
                t.restore_context(&transforms);
            }
            // Partials are not context: they restart empty (ADR-0081 item 6).
            pool.episodes.clear();
            pool.reobserve(0..state.records.len(), StackMode::Keep)?;
            pool.poisoned = false;
            #[cfg(feature = "debug-checks")]
            pool.discard_allocations();
            Ok(())
        });
        self.watch_boundary(result.is_ok());
        result
    }
}
