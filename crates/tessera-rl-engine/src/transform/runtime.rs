// SPDX-License-Identifier: MIT OR Apache-2.0

//! A pool's transform state (ADR-0078 items 4 to 6): the shared plan and
//! statistics, the frame-stack arena, the chunk accumulators, the scratch
//! rows and the freeze flag.

use tessera_rl_core::space::Space;

use super::error::TransformError;
use super::kernels::StackMode;
use super::plan::Plan;
use super::rows::{Shared, TransformRows};
use super::spec::{MAX_NORMALIZE_ACCUMULATOR_BYTES, Transforms};
use super::stats::{Applied, MOMENT_BYTES, Moments};
use crate::layout::ObsLayout;

/// The transform part of a pool context (ADR-0078 item 6): carried by
/// `pool_context` and `restore_pool_context`, and by pool snapshots.
/// Equality compares the statistics bit for bit.
#[derive(Debug, Clone, Default)]
pub struct TransformContext {
    /// The transform identity (item 2.3); `0` without transforms.
    pub identity: u64,
    /// The pool statistics of every normalized element, in leaf order.
    pub normalization: Vec<Moments>,
    /// Whether accumulation is stopped.
    pub frozen: bool,
}

impl PartialEq for TransformContext {
    fn eq(&self, other: &Self) -> bool {
        let bits = |m: &Moments| (m.count.to_bits(), m.mean.to_bits(), m.m2.to_bits());
        self.identity == other.identity
            && self.frozen == other.frozen
            && self.normalization.len() == other.normalization.len()
            && self
                .normalization
                .iter()
                .zip(&other.normalization)
                .all(|(a, b)| bits(a) == bits(b))
    }
}

impl Eq for TransformContext {}

/// The transform state of a pool with a non-empty transform list.
#[derive(Debug)]
pub(crate) struct Runtime {
    pub(crate) transforms: Transforms,
    pub(crate) identity: u64,
    pub(crate) shared: Shared,
    /// The pool statistics, merged at barriers.
    pub(crate) moments: Vec<Moments>,
    /// `ceil(N / chunk_size)` accumulators of `normalized` moments.
    pub(crate) accum: Vec<Moments>,
    /// `N * instance_bytes`.
    pub(crate) arena: Vec<u8>,
    /// One scratch per worker.
    pub(crate) worker_scratch: Vec<u8>,
    /// The calling thread's scratch.
    pub(crate) caller_scratch: Vec<u8>,
    pub(crate) frozen: bool,
    /// Whether a plain `restore` already logged its warning (item 4.5).
    pub(crate) warned_plain_restore: bool,
}

impl Runtime {
    /// The state of `transforms` on a pool of `instances` instances of
    /// `max_agents` agent slots with observation `space` (raw `layout`).
    pub(crate) fn new(
        transforms: Transforms,
        space: &Space,
        layout: &ObsLayout,
        instances: usize,
        max_agents: usize,
        chunk_size: usize,
    ) -> Result<Self, TransformError> {
        let plan = Plan::new(&transforms, space, layout)?;
        let chunks = instances.div_ceil(chunk_size.max(1));
        let bound = chunks
            .checked_mul(plan.normalized)
            .and_then(|b| b.checked_mul(MOMENT_BYTES))
            .unwrap_or(usize::MAX);
        if bound > MAX_NORMALIZE_ACCUMULATOR_BYTES {
            return Err(TransformError::AccumulatorTooLarge { bytes: bound });
        }
        let too_large = || TransformError::RowTooLarge { bytes: usize::MAX };
        let instance_bytes = plan
            .agent_state
            .checked_mul(max_agents)
            .ok_or_else(too_large)?;
        let arena = instances
            .checked_mul(instance_bytes)
            .ok_or_else(too_large)?;
        let scratch_bytes = plan
            .widest
            .checked_mul(2)
            .and_then(|w| w.checked_add(plan.raw_bytes))
            .ok_or_else(too_large)?;
        let chunk_size = u32::try_from(chunk_size).unwrap_or(u32::MAX);
        let identity = transforms.identity(chunk_size);
        let normalized = plan.normalized;
        Ok(Self {
            transforms,
            identity,
            shared: Shared {
                applied: vec![Applied::default(); normalized],
                agent_bytes: plan.agent_state,
                instance_bytes,
                scratch_bytes,
                chunk_size: usize::try_from(chunk_size).unwrap_or(usize::MAX),
                plan,
            },
            moments: vec![Moments::default(); normalized],
            accum: vec![Moments::default(); chunks.saturating_mul(normalized)],
            arena: vec![0; arena],
            worker_scratch: Vec::new(),
            caller_scratch: vec![0; scratch_bytes],
            frozen: false,
            warned_plain_restore: false,
        })
    }

    /// Sizes the worker scratch for `workers` workers (construction only).
    pub(crate) fn ensure_workers(&mut self, workers: usize) {
        let want = workers.saturating_mul(self.shared.scratch_bytes);
        if self.worker_scratch.len() < want {
            self.worker_scratch.resize(want, 0);
        }
    }

    /// The exported observation space.
    pub(crate) fn space(&self) -> &Space {
        &self.shared.plan.space
    }

    /// The rows of a whole-pool pass run by workers, with stack `mode`;
    /// accumulating unless frozen or `accumulate` is false.
    pub(crate) fn worker_rows(&mut self, mode: StackMode, accumulate: bool) -> TransformRows<'_> {
        let accum: &mut [Moments] = if accumulate && !self.frozen {
            &mut self.accum
        } else {
            &mut []
        };
        TransformRows {
            shared: Some(&self.shared),
            arena: &mut self.arena,
            accum,
            base: 0,
            scratch: &mut self.worker_scratch,
            mode,
        }
    }

    /// The rows of a whole-pool pass on the calling thread.
    pub(crate) fn caller_rows(&mut self, mode: StackMode, accumulate: bool) -> TransformRows<'_> {
        let accum: &mut [Moments] = if accumulate && !self.frozen {
            &mut self.accum
        } else {
            &mut []
        };
        TransformRows {
            shared: Some(&self.shared),
            arena: &mut self.arena,
            accum,
            base: 0,
            scratch: &mut self.caller_scratch,
            mode,
        }
    }

    /// The epoch barrier (item 5.3): merges the chunk accumulators into the
    /// pool statistics in chunk order, clears them and fixes the applied
    /// statistics for the next epoch.
    pub(crate) fn barrier(&mut self) {
        let per = self.shared.plan.normalized;
        if per == 0 {
            return;
        }
        for chunk in self.accum.chunks_exact_mut(per) {
            for (total, part) in self.moments.iter_mut().zip(chunk.iter_mut()) {
                total.merge(part);
                *part = Moments::default();
            }
        }
        self.apply_moments();
    }

    fn apply_moments(&mut self) {
        for (applied, m) in self.shared.applied.iter_mut().zip(&self.moments) {
            *applied = Applied::from(m);
        }
    }

    /// The transform context.
    pub(crate) fn context(&self) -> TransformContext {
        TransformContext {
            identity: self.identity,
            normalization: self.moments.clone(),
            frozen: self.frozen,
        }
    }

    /// Checks a context against this pool (validate first).
    pub(crate) fn check_context(&self, context: &TransformContext) -> Result<(), TransformError> {
        if context.identity != self.identity || context.normalization.len() != self.moments.len() {
            return Err(TransformError::IdentityMismatch {
                expected: self.identity,
                found: context.identity,
            });
        }
        Ok(())
    }

    /// Restores a checked context; accumulators restart empty (item 5.7).
    pub(crate) fn restore_context(&mut self, context: &TransformContext) {
        self.moments.clone_from(&context.normalization);
        self.frozen = context.frozen;
        self.accum.fill(Moments::default());
        self.apply_moments();
    }

    /// The arena slice of instance `id`.
    pub(crate) fn slice(&self, id: usize) -> Option<&[u8]> {
        let n = self.shared.instance_bytes;
        self.arena
            .get(id.checked_mul(n)?..id.checked_mul(n)?.checked_add(n)?)
    }

    /// The arena slice of instance `id`, mutable.
    pub(crate) fn slice_mut(&mut self, id: usize) -> Option<&mut [u8]> {
        let n = self.shared.instance_bytes;
        self.arena
            .get_mut(id.checked_mul(n)?..id.checked_mul(n)?.checked_add(n)?)
    }
}

#[cfg(test)]
mod tests {
    use super::Runtime;
    use crate::layout::obs_leaves;
    use crate::transform::Transforms;
    use crate::transform::stats::Moments;
    use tessera_rl_core::space::{BoxBounds, DType, Space};

    fn moments(xs: &[f64]) -> Moments {
        let mut m = Moments::default();
        for &x in xs {
            m.add(x);
        }
        m
    }

    /// The barrier merges chunks in chunk order (ADR-0078 item 5.3): the
    /// result is bit for bit the forward merge, which here differs from the
    /// reverse one.
    #[test]
    fn test_runtime_barrier_merges_in_chunk_order() {
        let space = Space::Box {
            shape: vec![1],
            dtype: DType::F64,
            bounds: BoxBounds::Float {
                low: vec![-1e12],
                high: vec![1e12],
            },
        };
        let layout = obs_leaves(&space).unwrap();
        let t = Transforms::new().normalize("", 5.0);
        let mut rt = Runtime::new(t, &space, &layout, 3, 1, 1).unwrap();
        let parts = [
            moments(&[0.1, 0.7]),
            moments(&[1e8, 3.0, 5.5]),
            moments(&[3.3, -2.0, 1e-3, 9.0]),
        ];
        rt.accum.copy_from_slice(&parts);
        rt.barrier();
        let mut forward = Moments::default();
        for p in &parts {
            forward.merge(p);
        }
        let mut reverse = Moments::default();
        for p in parts.iter().rev() {
            reverse.merge(p);
        }
        let bits = |m: &Moments| (m.count.to_bits(), m.mean.to_bits(), m.m2.to_bits());
        assert_ne!(
            bits(&forward),
            bits(&reverse),
            "the case distinguishes orders"
        );
        assert_eq!(bits(&rt.moments[0]), bits(&forward));
        assert!(rt.accum.iter().all(|m| *m == Moments::default()), "cleared");
    }
}
