// SPDX-License-Identifier: MIT OR Apache-2.0

//! The pool operations of ADR-0078: the exported observation space, the
//! frame-stack continuation (items 4.4 and 4.5), the epoch barrier and
//! freeze (item 5).

use tessera_rl_core::space::Space;
use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout};

use super::error::TransformError;
use super::kernels::StackMode;
use crate::error::EngineError;
use crate::masks::MaskProvider;
use crate::pool::EnvPool;
use crate::record::Record;

impl<E: Environment, M: MaskProvider<E>> EnvPool<E, M> {
    /// The observation the pool exports: the environment's, transformed by
    /// the pool's transforms (ADR-0078 item 2.2). Adapters read this.
    ///
    /// # Errors
    ///
    /// The environment's error for its observation space.
    pub fn observation_space(&self) -> Result<Space, EngineError> {
        match &self.transform {
            Some(t) => Ok(t.space().clone()),
            None => Ok(E::observation_space(&self.cfg)?),
        }
    }

    /// The transform list (empty without transforms).
    #[must_use]
    pub fn transforms(&self) -> crate::Transforms {
        self.transform
            .as_ref()
            .map_or_else(crate::Transforms::new, |t| t.transforms.clone())
    }

    /// The transform identity (ADR-0078 item 2.3); `0` without transforms.
    #[must_use]
    pub fn transform_identity(&self) -> u64 {
        self.transform.as_ref().map_or(0, |t| t.identity)
    }

    /// `S`, the frame-stack bytes of one instance (ADR-0078 item 4.1 and
    /// erratum 1): for each agent slot and each `FrameStack` of a leaf of
    /// `b` bytes per frame, `8 + k * b`; `0` without a `FrameStack`. No
    /// padding: the arena is read and written bytewise (header words as
    /// little-endian `u32`, ADR-0011; frames with byte copies), never through typed
    /// pointers, so a frame size that is not a multiple of 4 aligns
    /// nothing wrongly.
    #[must_use]
    pub fn transform_state_bytes(&self) -> usize {
        self.transform
            .as_ref()
            .map_or(0, |t| t.shared.instance_bytes)
    }

    /// Freezes (or unfreezes) normalization (ADR-0078 item 5.6): while
    /// frozen nothing accumulates.
    pub fn freeze_normalization(&mut self, frozen: bool) {
        if let Some(t) = self.transform.as_mut() {
            t.frozen = frozen;
        }
    }

    /// Whether the pool has a `Normalize` that accumulates (not frozen).
    pub(crate) fn normalization_unfrozen(&self) -> bool {
        self.transform
            .as_ref()
            .is_some_and(|t| t.shared.plan.normalized > 0 && !t.frozen)
    }

    /// Whether normalization is frozen.
    #[must_use]
    pub fn normalization_frozen(&self) -> bool {
        self.transform.as_ref().is_some_and(|t| t.frozen)
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
    /// Copies instance `id`'s frame stacks into `out` (exactly
    /// [`Self::transform_state_bytes`] bytes).
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`], [`EngineError::InstanceOutOfRange`], or
    /// [`TransformError::StateLength`].
    pub fn snapshot_transforms(&self, id: u32, out: &mut [u8]) -> Result<(), EngineError> {
        self.guarded_ref(|pool| {
            let i = pool.index(id)?;
            let need = pool.transform_state_bytes();
            if out.len() != need {
                return Err(TransformError::StateLength {
                    expected: need,
                    found: out.len(),
                }
                .into());
            }
            if let Some(src) = pool.transform.as_ref().and_then(|t| t.slice(i)) {
                out.copy_from_slice(src);
            }
            Ok(())
        })
    }

    /// Restores `record` and its frame stacks `bytes` (from
    /// [`Self::snapshot_transforms`]) into slot `id`: an exact continuation
    /// (ADR-0078 item 4.5).
    ///
    /// # Errors
    ///
    /// As [`EnvPool::restore`], and [`TransformError::StateLength`] (nothing
    /// is mutated).
    pub fn restore_with_transforms(
        &mut self,
        id: u32,
        record: &Record<E>,
        bytes: &[u8],
    ) -> Result<(), EngineError> {
        let result = self.guarded(crate::pool::Poison::Reject, |pool| {
            let i = pool.index(id)?;
            pool.check_state_len(bytes.len(), 1)?;
            pool.write_record(i, record);
            if let Some(dst) = pool.transform.as_mut().and_then(|t| t.slice_mut(i)) {
                dst.copy_from_slice(bytes);
            }
            pool.reobserve(core::iter::once(i), StackMode::Keep)
        });
        // A batch boundary of the live watch, as `restore` (ADR-0083).
        self.watch_boundary(result.is_ok());
        result
    }

    /// Copies every instance's frame stacks into `out` (exactly
    /// `num_envs * transform_state_bytes()` bytes, instance order).
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`] or [`TransformError::StateLength`].
    pub fn snapshot_transforms_all(&self, out: &mut [u8]) -> Result<(), EngineError> {
        self.guarded_ref(|pool| {
            pool.check_state_len(out.len(), pool.headers.len())?;
            if let Some(t) = pool.transform.as_ref() {
                out.copy_from_slice(&t.arena);
            }
            Ok(())
        })
    }

    /// [`EnvPool::restore_all`] with every instance's frame stacks: an exact
    /// continuation, and a recovery path as `restore_all` is.
    ///
    /// # Errors
    ///
    /// As [`EnvPool::restore_all`], and [`TransformError::StateLength`]
    /// (nothing is mutated).
    pub fn restore_all_with_transforms(
        &mut self,
        records: &[Record<E>],
        bytes: &[u8],
    ) -> Result<(), EngineError> {
        let result = self.guarded(crate::pool::Poison::Recover, |pool| {
            pool.check_count(records.len())?;
            pool.check_state_len(bytes.len(), records.len())?;
            for (i, record) in records.iter().enumerate() {
                pool.write_record(i, record);
            }
            if let Some(t) = pool.transform.as_mut() {
                t.arena.copy_from_slice(bytes);
            }
            pool.reobserve(0..records.len(), StackMode::Keep)?;
            pool.poisoned = false;
            #[cfg(feature = "debug-checks")]
            pool.discard_allocations();
            Ok(())
        });
        self.watch_boundary(result.is_ok());
        result
    }

    /// The epoch barrier (ADR-0078 item 5.3): merges the chunk accumulators
    /// into the pool statistics in chunk order and clears them; later
    /// observations use the new statistics. A no-op without a `Normalize`.
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`].
    pub fn epoch_barrier(&mut self) -> Result<(), EngineError> {
        self.guarded(crate::pool::Poison::Reject, |pool| {
            if let Some(t) = pool.transform.as_mut() {
                t.barrier();
            }
            Ok(())
        })
    }

    /// The frame-stack bytes of `instances` instances.
    ///
    /// # Errors
    ///
    /// [`TransformError::StateTooLarge`] if the product overflows `usize`.
    pub(crate) fn state_bytes_for(&self, instances: usize) -> Result<usize, EngineError> {
        let per_instance = self.transform_state_bytes();
        per_instance.checked_mul(instances).ok_or_else(|| {
            TransformError::StateTooLarge {
                per_instance,
                instances,
            }
            .into()
        })
    }

    /// `len` bytes of frame stacks for `instances` instances, or an error.
    pub(crate) fn check_state_len(&self, len: usize, instances: usize) -> Result<(), EngineError> {
        let expected = self.state_bytes_for(instances)?;
        if len == expected {
            Ok(())
        } else {
            Err(TransformError::StateLength {
                expected,
                found: len,
            }
            .into())
        }
    }

    /// A plain restore re-initializes frame stacks (ADR-0078 item 4.5):
    /// logged once per pool.
    pub(crate) fn note_plain_restore(&mut self) {
        if let Some(t) = self.transform.as_mut()
            && t.shared.plan.stateful()
            && !t.warned_plain_restore
        {
            t.warned_plain_restore = true;
            tracing::warn!(
                "restore without frame stacks on a pool with FrameStack transforms: \
                 the stacks restart from the restored observation (not an exact \
                 continuation); use restore_with_transforms"
            );
        }
    }

    /// Copies instance `src`'s frame stacks to every destination (fork).
    pub(crate) fn copy_transform_state(&mut self, src: u32, dst: &[u32]) {
        let Some(t) = self.transform.as_mut() else {
            return;
        };
        let n = t.shared.instance_bytes;
        let Ok(s) = usize::try_from(src) else {
            return;
        };
        for &d in dst {
            let Ok(d) = usize::try_from(d) else {
                continue;
            };
            let len = t.arena.len();
            let fits = |i: usize| {
                i.checked_add(1)
                    .and_then(|e| e.checked_mul(n))
                    .is_some_and(|e| e <= len)
            };
            if d != s && fits(s) && fits(d) {
                t.arena.copy_within(s * n..(s + 1) * n, d * n);
            }
        }
    }
}
