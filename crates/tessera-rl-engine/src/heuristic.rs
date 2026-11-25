// SPDX-License-Identifier: MIT OR Apache-2.0

//! Heuristic slots on the worker (ADR-0079 items 3 and 4): per chunk,
//! before any instance of it steps, the action of every alive, acting
//! heuristic slot is computed into the engine's action copy and validated
//! there with the boundary's rules.
//!
//! The routine is chosen once, at construction, as one `fn` pointer per
//! pool ([`HeuristicFn`]): a monomorphic instance of [`run`] for the
//! environment's `Heuristic::act`, its `InfoHeuristic::act_info` through
//! the information state, or the privileged `act` on an
//! imperfect-information environment. The step calls it once per chunk;
//! inside, every call is static. There is no per-instance or per-slot
//! dynamic dispatch.

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Bool8, Environment, Heuristic, InfoHeuristic, RecordLayout};
use tessera_rl_num::identity::RngIdentity;
use tessera_rl_num::rng::StepRng;

use crate::layout::ActionLayout;
use crate::pool::{Chunk, StepCtx, first_illegal};
use crate::sys::sync::{AtomicU64, Ordering};

/// The heuristic pass of one chunk: `(ctx, chunk, first)`, where `first`
/// is the pool index of the chunk's first instance.
pub(crate) type HeuristicFn<E> = fn(&StepCtx<'_, E>, &mut Chunk<'_, E>, usize);

/// What the pool's heuristic slots run, chosen at construction.
pub(crate) struct HeuristicRoute<E: Environment> {
    /// The pass, called once per chunk.
    pub(crate) run: HeuristicFn<E>,
    /// The heuristic slots, in slot order.
    pub(crate) slots: Vec<u16>,
}

impl<E: Environment> core::fmt::Debug for HeuristicRoute<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HeuristicRoute")
            .field("slots", &self.slots)
            .finish_non_exhaustive()
    }
}

impl<E: Environment> Clone for HeuristicRoute<E> {
    fn clone(&self) -> Self {
        Self {
            run: self.run,
            slots: self.slots.clone(),
        }
    }
}

/// No fault recorded in this step.
pub(crate) const NO_FAULT: u64 = u64::MAX;

/// The fault code of `(instance, slot)`: ordered by instance, then slot,
/// so `fetch_min` keeps the first in instance order whatever the workers'
/// order.
#[must_use]
#[inline]
pub(crate) fn fault_code(instance: usize, slot: u16) -> u64 {
    let instance = u64::try_from(instance).unwrap_or(u64::from(u32::MAX));
    (instance.min(u64::from(u32::MAX)) << 16) | u64::from(slot)
}

/// The `(instance, slot)` of a fault code; `None` for [`NO_FAULT`].
#[must_use]
pub(crate) fn fault_of(code: u64) -> Option<(u32, u16)> {
    if code == NO_FAULT {
        return None;
    }
    let instance = u32::try_from(code >> 16).unwrap_or(u32::MAX);
    let slot = u16::try_from(code & 0xFFFF).unwrap_or(u16::MAX);
    Some((instance, slot))
}

/// The pass for a perfect-information environment: `Heuristic::act` on
/// the state (ADR-0079 item 4.1).
pub(crate) fn run_act<E: Heuristic>(ctx: &StepCtx<'_, E>, chunk: &mut Chunk<'_, E>, first: usize) {
    run::<E, _>(ctx, chunk, first, E::act);
}

/// The pass for an imperfect-information environment: `act_info` on the
/// agent's information state only (ADR-0079 item 4.2).
pub(crate) fn run_info<E: InfoHeuristic>(
    ctx: &StepCtx<'_, E>,
    chunk: &mut Chunk<'_, E>,
    first: usize,
) {
    run::<E, _>(ctx, chunk, first, |cfg, state, agent, rng, out| {
        let info = E::info_state(cfg, state, agent);
        E::act_info(cfg, &info, agent, rng, out);
    });
}

/// The heuristic pass over one chunk with `act`, a static function or a
/// closure monomorphized into this instance. Runs before any instance of
/// the chunk steps, so the masks read are those of the current states.
fn run<E, F>(ctx: &StepCtx<'_, E>, chunk: &mut Chunk<'_, E>, first: usize, act: F)
where
    E: Environment,
    F: Fn(&E::Config, &E::State, AgentSlot, &mut StepRng, &mut ActionWriter<'_>),
{
    let Some(faults) = ctx.heuristic_fault else {
        return;
    };
    let layout = ctx.layout;
    let a = E::MAX_AGENTS;
    let (dh, ch) = (layout.discrete.len(), layout.continuous.len());
    let mask_len = chunk.outputs.mask_len;
    for (i, (header, state)) in chunk.headers.iter().zip(chunk.states.iter()).enumerate() {
        let engine = header.engine();
        if engine.pending_reset.get() {
            // It resets instead of stepping: no action is read.
            continue;
        }
        // The identity of the transition about to run (`step_one` sets the
        // same step index), so the draws are a function of the state's
        // instance, episode and step only.
        let id = engine.identity;
        let identity = RngIdentity::from_parts(
            id.seed(),
            id.branch_salt(),
            id.logical_id(),
            id.episode_index(),
            engine.episode_steps.saturating_add(1),
        );
        for &slot in ctx.heuristic_slots {
            let s = usize::from(slot);
            // An opponent slot whose instance plays an `External` policy
            // keeps the caller's validated row (ADR-0088 item 2.4); an
            // ONNX policy's row, written on the calling thread, is only
            // validated here (head C).
            let mode = layout.opponent_mode(first + i, s);
            if mode == Some(crate::layout::OPPONENT_EXTERNAL) {
                continue;
            }
            let onnx = mode == Some(crate::layout::OPPONENT_ONNX);
            let k = i * a + s;
            let (Some(disc), Some(cont)) = (
                chunk.discrete.get_mut(k * dh..(k + 1) * dh),
                chunk.continuous.get_mut(k * ch..(k + 1) * ch),
            ) else {
                continue;
            };
            let acting = header
                .agents()
                .get(s)
                .is_some_and(|r| r.alive.get() && r.acting.get());
            if !onnx {
                // Overwrite the engine's copy whatever was submitted (item
                // 3.1); a slot that does not act keeps zeros.
                disc.fill(0);
                cont.fill(0.0);
                let Some(agent) = AgentSlot::new(slot, a).filter(|_| acting) else {
                    continue;
                };
                let mut rng = StepRng::heuristic(ctx.authority, &identity, slot);
                act(
                    ctx.cfg,
                    state,
                    agent,
                    &mut rng,
                    &mut ActionWriter::new(disc, cont),
                );
            } else if !acting {
                continue;
            }
            let mask = chunk
                .outputs
                .masks
                .get(k * mask_len..(k + 1) * mask_len)
                .unwrap_or(&[]);
            if !row_valid(layout, disc, cont, mask) {
                let instance = first + i;
                // One cell for the whole unit, or one per pool instance.
                let cell = match faults {
                    [one] => Some(one),
                    all => all.get(instance),
                };
                if let Some(cell) = cell {
                    cell.fetch_min(fault_code(instance, slot), Ordering::Relaxed);
                }
                // The pool is poisoned after the barrier; meanwhile the
                // kernel reads an in-range row.
                safe_row(layout, disc, cont);
            }
        }
    }
}

/// Whether a computed row passes the boundary's rules: every discrete
/// value in range, every continuous value finite and in bounds, and with
/// masks (`mask` non-empty) legal.
#[must_use]
pub(crate) fn row_valid(layout: &ActionLayout, disc: &[i32], cont: &[f32], mask: &[Bool8]) -> bool {
    let discrete = disc
        .iter()
        .zip(&layout.discrete)
        .all(|(&v, &n)| u32::try_from(v).is_ok_and(|v| v < n));
    let continuous = cont
        .iter()
        .zip(&layout.continuous)
        .all(|(&v, &(low, high))| {
            let wide = f64::from(v);
            v.is_finite() && wide >= low && wide <= high
        });
    discrete
        && continuous
        && (mask.is_empty() || first_illegal(disc, &layout.mask, 0..disc.len(), mask).is_none())
}

/// Overwrites an invalid row with in-range values: zero discrete heads and
/// each continuous head's value nearest zero.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a bound of an f32 head, narrowed back to f32"
)]
fn safe_row(layout: &ActionLayout, disc: &mut [i32], cont: &mut [f32]) {
    disc.fill(0);
    for (v, &(low, high)) in cont.iter_mut().zip(&layout.continuous) {
        // No `clamp`: it panics on unordered bounds.
        let nearest = if low > 0.0 {
            low
        } else if high < 0.0 {
            high
        } else {
            0.0
        };
        *v = nearest as f32;
    }
}

/// A fault cell holding no fault: one per Sync call, per Pipelined half or
/// per Async instance.
pub(crate) fn new_fault() -> AtomicU64 {
    AtomicU64::new(NO_FAULT)
}

#[cfg(test)]
mod tests {
    use super::{NO_FAULT, fault_code, fault_of};

    #[test]
    fn test_fault_code_orders_by_instance_then_slot() {
        assert!(fault_code(1, 7) < fault_code(2, 0));
        assert!(fault_code(3, 0) < fault_code(3, 1));
        assert_eq!(fault_of(fault_code(5, 3)), Some((5, 3)));
        assert_eq!(fault_of(NO_FAULT), None);
    }
}
