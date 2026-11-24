// SPDX-License-Identifier: MIT OR Apache-2.0

//! The transform parts of a view of rows: what `observe_rows` needs to run
//! the pipeline for the instances of the view (ADR-0078 item 1).

use super::kernels::{StackMode, run_leaf};
use super::plan::Plan;
use super::stats::{Applied, Moments};

/// The read-only parts shared by every view of a pool.
#[derive(Debug)]
pub(crate) struct Shared {
    pub(crate) plan: Plan,
    /// The statistics applied in this epoch, one per normalized element.
    pub(crate) applied: Vec<Applied>,
    /// Arena bytes per agent and per instance (`MAX_AGENTS` agents).
    pub(crate) agent_bytes: usize,
    pub(crate) instance_bytes: usize,
    /// Scratch bytes per worker: the raw rows of the transformed leaves,
    /// then two intermediate rows.
    pub(crate) scratch_bytes: usize,
    /// The pool's chunk size: accumulators are per chunk (item 5.2).
    pub(crate) chunk_size: usize,
}

/// The transform parts of a view: empty for a pool without transforms.
#[derive(Debug)]
pub(crate) struct TransformRows<'a> {
    pub(crate) shared: Option<&'a Shared>,
    /// The arena slices of the view's instances, `instance_bytes` each.
    pub(crate) arena: &'a mut [u8],
    /// The accumulators of the chunks the view's instances belong to, from
    /// the chunk of its first instance, when the pass accumulates; empty
    /// otherwise.
    pub(crate) accum: &'a mut [Moments],
    /// The position of the view's first instance in its chunk.
    pub(crate) base: usize,
    /// The scratch of the thread running the view.
    pub(crate) scratch: &'a mut [u8],
    /// What observations of this pass do to frame stacks.
    pub(crate) mode: StackMode,
}

impl Default for TransformRows<'_> {
    fn default() -> Self {
        Self {
            shared: None,
            arena: &mut [],
            accum: &mut [],
            base: 0,
            scratch: &mut [],
            mode: StackMode::Push,
        }
    }
}

impl TransformRows<'_> {
    /// Clears the arena slice of agent `slot` of instance `i` (a dead
    /// agent: its frames go with its row, ADR-0078 item 4.3).
    pub(crate) fn clear_agent(&mut self, i: usize, slot: usize) {
        if let Some(s) = self.shared {
            let at = i * s.instance_bytes + slot * s.agent_bytes;
            if let Some(bytes) = self.arena.get_mut(at..at + s.agent_bytes) {
                bytes.fill(0);
            }
        }
    }

    /// Runs the plan for agent `slot` of instance `i`: `raw` holds, by
    /// leaf index, the transformed leaves as `E::observe` wrote them, `out` is
    /// the agent's leaf rows (the exported leaves), `work` the intermediate
    /// scratch.
    pub(crate) fn run(
        &mut self,
        i: usize,
        slot: usize,
        generation: u32,
        out: &mut [&mut [u8]],
        raw: &[&mut [u8]],
        work: &mut [u8],
    ) {
        let Some(s) = self.shared else {
            return;
        };
        let at = i * s.instance_bytes + slot * s.agent_bytes;
        let state = self
            .arena
            .get_mut(at..at + s.agent_bytes)
            .unwrap_or(&mut []);
        for leaf in &s.plan.leaves {
            let input: &[u8] = raw.get(leaf.leaf).map_or(&[], |r| &**r);
            let per = s.plan.normalized;
            let chunk = (self.base + i) / s.chunk_size.max(1);
            let accum = if per == 0 {
                None
            } else {
                self.accum.get_mut(chunk * per..(chunk + 1) * per)
            };
            if let Some(dst) = out.get_mut(leaf.leaf) {
                run_leaf(
                    leaf, input, dst, work, state, generation, self.mode, &s.applied, accum,
                );
            }
        }
    }
}
