// SPDX-License-Identifier: MIT OR Apache-2.0

//! Frame stacks in search trees (ADR-0078 item 7): every node stores the
//! `S` frame-stack bytes of its instance beside its record, so the scratch
//! pool continues exactly the observations the live pool would produce.
//! With `S = 0` (no `FrameStack`) every buffer is empty and nothing changes.

use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout};
use tessera_rl_engine::{MaskProvider, Record, TransformContext, Transforms};

use crate::error::SearchError;
use crate::handle::NodeHandle;
use crate::pool::SearchPool;

/// The frame-stack bytes of a tree, sized at construction.
#[derive(Debug, Default)]
pub(crate) struct Stacks {
    /// `S`, bytes per instance.
    pub(crate) per: usize,
    /// `capacity * S`, by node slot.
    pub(crate) tree: Vec<u8>,
    /// `batch * S`: the stepped children's bytes, taken before the commit.
    pub(crate) child: Vec<u8>,
    /// `batch * S`: the parents loaded into the scratch pool.
    pub(crate) load: Vec<u8>,
}

impl Stacks {
    pub(crate) fn new(per: usize, capacity: usize, batch: usize) -> Self {
        Self {
            per,
            tree: vec![0; per.saturating_mul(capacity)],
            child: vec![0; per.saturating_mul(batch)],
            load: vec![0; per.saturating_mul(batch)],
        }
    }

    fn range(&self, i: usize) -> core::ops::Range<usize> {
        let start = i.saturating_mul(self.per);
        start..start.saturating_add(self.per)
    }

    /// Node slot `i`'s bytes.
    pub(crate) fn slot(&self, i: usize) -> &[u8] {
        self.tree.get(self.range(i)).unwrap_or(&[])
    }

    /// Copies `bytes` into node slot `i`.
    pub(crate) fn set_slot(&mut self, i: usize, bytes: &[u8]) {
        let r = self.range(i);
        if let Some(dst) = self.tree.get_mut(r) {
            dst.copy_from_slice(bytes);
        }
    }

    /// Copies node slot `i` into load row `row`.
    pub(crate) fn load_from(&mut self, row: usize, i: usize) {
        let (src, dst) = (self.range(i), self.range(row));
        if let (Some(s), Some(d)) = (self.tree.get(src), self.load.get_mut(dst)) {
            d.copy_from_slice(s);
        }
    }

    /// Copies child row `row` into node slot `i`.
    pub(crate) fn commit(&mut self, row: usize, i: usize) {
        let (src, dst) = (self.range(row), self.range(i));
        if let (Some(s), Some(d)) = (self.child.get(src), self.tree.get_mut(dst)) {
            d.copy_from_slice(s);
        }
    }

    /// Child row `row`, mutable.
    pub(crate) fn child_mut(&mut self, row: usize) -> &mut [u8] {
        let r = self.range(row);
        self.child.get_mut(r).unwrap_or(&mut [])
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
    /// The scratch pool's transforms (ADR-0078 item 7.1).
    #[must_use]
    pub fn transforms(&self) -> Transforms {
        self.scratch.transforms()
    }

    /// The scratch pool's transform identity; `0` without transforms.
    #[must_use]
    pub fn transform_identity(&self) -> u64 {
        self.scratch.transform_identity()
    }

    /// `S`, the frame-stack bytes a root needs; `0` without a `FrameStack`.
    #[must_use]
    pub fn transform_state_bytes(&self) -> usize {
        self.stacks.per
    }

    /// The scratch pool's transform context: its identity, the statistics
    /// it applies and its freeze flag (always frozen: a search never
    /// learns statistics).
    ///
    /// # Errors
    ///
    /// An engine error on a poisoned scratch pool.
    pub fn transform_context(&self) -> Result<TransformContext, SearchError> {
        Ok(self.scratch.pool_context()?.transforms)
    }

    /// The frame-stack bytes of node `h` (empty without frame stacks): with
    /// its record, an exact continuation of the node in a live pool
    /// (`EnvPool::restore_with_transforms`).
    ///
    /// # Errors
    ///
    /// A stale handle.
    pub fn transform_bytes(&self, h: NodeHandle) -> Result<&[u8], SearchError> {
        let slot = self.resolve(h)?;
        Ok(self.stacks.slot(slot as usize))
    }

    /// Gives the scratch pool the live pool's transform context
    /// (ADR-0078 item 7.2): its normalization statistics, frozen for the
    /// search's life (the scratch pool never accumulates).
    ///
    /// # Errors
    ///
    /// [`SearchError::TransformIdentity`] for a context of another
    /// transform list (nothing is changed); an engine error.
    pub fn set_transform_context(&mut self, context: TransformContext) -> Result<(), SearchError> {
        let expected = self.scratch.transform_identity();
        if context.identity != expected {
            return Err(SearchError::TransformIdentity {
                expected,
                found: context.identity,
            });
        }
        let mut pool_context = self.scratch.pool_context()?;
        pool_context.transforms = context;
        self.scratch.restore_pool_context(pool_context, None)?;
        self.scratch.freeze_normalization(true);
        Ok(())
    }

    /// [`Self::set_root`] with the root's frame-stack bytes (from
    /// `EnvPool::snapshot_transforms`): required on a scratch pool with
    /// frame stacks (ADR-0078 item 7.3).
    ///
    /// # Errors
    ///
    /// [`SearchError::StackedRoot`] for bytes of the wrong length (nothing
    /// is changed), and the errors of [`Self::set_root`].
    pub fn set_root_with_transforms(
        &mut self,
        record: &Record<E>,
        bytes: &[u8],
    ) -> Result<NodeHandle, SearchError> {
        if self.mode == crate::determinize::RootMode::Determinized {
            return Err(SearchError::DeterminizedPool);
        }
        self.check_root_bytes(bytes.len())?;
        let handle = self.replace_root(record)?;
        self.stacks.set_slot(self.root as usize, bytes);
        Ok(handle)
    }

    pub(crate) fn check_root_bytes(&self, found: usize) -> Result<(), SearchError> {
        if found == self.stacks.per {
            Ok(())
        } else {
            Err(SearchError::StackedRoot {
                expected: self.stacks.per,
                found,
            })
        }
    }

    /// Restores the load rows into the scratch pool, with their frame
    /// stacks when the pool has any. `restore_all` also recovers a scratch
    /// pool poisoned by an earlier failed step.
    pub(crate) fn restore_rows(&mut self) -> Result<(), SearchError> {
        if self.stacks.per == 0 {
            self.scratch.restore_all(&self.load_records)?;
        } else {
            self.scratch
                .restore_all_with_transforms(&self.load_records, &self.stacks.load)?;
        }
        Ok(())
    }
}
