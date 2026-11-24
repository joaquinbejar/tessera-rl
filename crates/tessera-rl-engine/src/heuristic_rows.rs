// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `heuristic_actions` rows of a split-phase pool (ADR-0079 item 3.5,
//! #347 stage B1).
//!
//! While a pool runs in Pipelined or Async mode it is shared with its
//! workers, so the caller cannot write the pool's own buffers. The wrapper
//! takes them out of the pool when it is built ([`HeuristicRows::take`]),
//! fills the rows of each collected unit on the calling thread (a returned
//! half, or the ids of a returned batch) and puts them back in `into_pool`.
//! Both buffers are sized once; a copy allocates nothing.

use tessera_rl_core::Environment;

use crate::masks::MaskProvider;
use crate::pool::EnvPool;

/// The last heuristic actions of every instance, `[N, A, heads]`: the
/// pool's buffers while the split-phase wrapper holds them.
#[derive(Debug, Default)]
pub(crate) struct HeuristicRows {
    discrete: Vec<i32>,
    continuous: Vec<f32>,
}

impl HeuristicRows {
    /// Takes the pool's buffers (empty without a heuristic slot).
    pub(crate) fn take<E: Environment, M: MaskProvider<E>>(pool: &mut EnvPool<E, M>) -> Self {
        Self {
            discrete: core::mem::take(&mut pool.heuristic_discrete),
            continuous: core::mem::take(&mut pool.heuristic_continuous),
        }
    }

    /// Gives the buffers back to the pool (`into_pool`).
    pub(crate) fn put_back<E: Environment, M: MaskProvider<E>>(self, pool: &mut EnvPool<E, M>) {
        pool.heuristic_discrete = self.discrete;
        pool.heuristic_continuous = self.continuous;
    }

    /// The discrete rows, `[N, A, discrete heads]`.
    #[must_use]
    pub(crate) fn discrete(&self) -> &[i32] {
        &self.discrete
    }

    /// The continuous rows, `[N, A, continuous heads]`.
    #[must_use]
    pub(crate) fn continuous(&self) -> &[f32] {
        &self.continuous
    }

    /// Copies the heuristic slots' rows of `count` consecutive instances
    /// from pool instance `first` on, out of the engine's action copy of
    /// those instances (`rows`, discrete and continuous, start at instance
    /// `first`). `agents` is the slot count and `heads` the discrete and
    /// continuous head counts. Rows out of range are skipped.
    pub(crate) fn copy(
        &mut self,
        first: usize,
        count: usize,
        agents: usize,
        slots: &[u16],
        heads: (usize, usize),
        rows: (&[i32], &[f32]),
    ) {
        let (dh, ch) = heads;
        let (discrete, continuous) = rows;
        for i in 0..count {
            for &slot in slots {
                let s = usize::from(slot);
                let Some(local) = i.checked_mul(agents).and_then(|k| k.checked_add(s)) else {
                    continue;
                };
                let Some(global) = first
                    .checked_add(i)
                    .and_then(|g| g.checked_mul(agents))
                    .and_then(|k| k.checked_add(s))
                else {
                    continue;
                };
                copy_row(&mut self.discrete, discrete, global, local, dh);
                copy_row(&mut self.continuous, continuous, global, local, ch);
            }
        }
    }
}

/// Copies row `local` of `src` into row `global` of `dst`, rows of `per`
/// values; nothing if either is out of range.
fn copy_row<T: Copy>(dst: &mut [T], src: &[T], global: usize, local: usize, per: usize) {
    let range = |row: usize| {
        let start = row.checked_mul(per)?;
        Some(start..start.checked_add(per)?)
    };
    let (Some(to), Some(from)) = (range(global), range(local)) else {
        return;
    };
    if let (Some(dst), Some(src)) = (dst.get_mut(to), src.get(from)) {
        dst.copy_from_slice(src);
    }
}

#[cfg(test)]
mod tests {
    use super::HeuristicRows;

    #[test]
    fn test_heuristic_rows_copy_writes_only_the_slots_of_the_unit() {
        // 4 instances, 2 agents, 1 discrete head, no continuous head.
        let mut rows = HeuristicRows {
            discrete: vec![-1; 8],
            continuous: Vec::new(),
        };
        // Instances 2 and 3, slot 1 heuristic.
        rows.copy(2, 2, 2, &[1], (1, 0), (&[10, 11, 12, 13], &[]));
        assert_eq!(rows.discrete(), &[-1, -1, -1, -1, -1, 11, -1, 13]);
    }
}
