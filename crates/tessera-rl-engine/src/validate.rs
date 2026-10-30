// SPDX-License-Identifier: MIT OR Apache-2.0

//! The validation boundary (ADR-0028): every check runs before any
//! instance is mutated.

use tessera_rl_core::{Bool8, RecordLayout};

use crate::error::EngineError;
use crate::layout::ActionLayout;

/// Caller-provided actions for every instance, `[N, A, heads]`.
#[derive(Debug, Clone, Copy)]
pub struct ActionInput<'a> {
    /// Discrete heads, row-major `[N, A, discrete heads]`.
    pub discrete: &'a [i32],
    /// Continuous heads, row-major `[N, A, continuous heads]`.
    pub continuous: &'a [f32],
}

/// Marks instance ids and rejects duplicates and out-of-range ids, using a
/// preallocated scratch array so it never allocates.
pub(crate) fn check_ids(ids: &[u32], seen: &mut [Bool8]) -> Result<(), EngineError> {
    let num_envs = u32::try_from(seen.len()).unwrap_or(u32::MAX);
    let mut result = Ok(());
    for &id in ids {
        match usize::try_from(id).ok().and_then(|i| seen.get_mut(i)) {
            None => {
                result = Err(EngineError::InstanceOutOfRange { id, num_envs });
                break;
            }
            Some(mark) if mark.get() => {
                result = Err(EngineError::DuplicateInstance { id });
                break;
            }
            Some(mark) => *mark = Bool8::TRUE,
        }
    }
    for &id in ids {
        if let Some(mark) = usize::try_from(id).ok().and_then(|i| seen.get_mut(i)) {
            *mark = Bool8::FALSE;
        }
    }
    result
}

/// Validates a step for every instance and copies the required actions
/// into the internal buffers; values of slots that are not required are
/// stored as zero and never validated.
pub(crate) fn check_and_copy_actions<H: RecordLayout>(
    layout: &ActionLayout,
    headers: &[H],
    input: ActionInput<'_>,
    discrete_out: &mut [i32],
    continuous_out: &mut [f32],
) -> Result<(), EngineError> {
    let agents = H::AGENTS;
    let (dh, ch) = (layout.discrete.len(), layout.continuous.len());
    check_len("discrete", discrete_out.len(), input.discrete.len())?;
    check_len("continuous", continuous_out.len(), input.continuous.len())?;
    for (index, header) in headers.iter().enumerate() {
        let instance = u32::try_from(index).unwrap_or(u32::MAX);
        let engine = header.engine();
        if engine.pending_reset.get() {
            return Err(EngineError::PendingReset { id: instance });
        }
        if engine.episode_steps == u32::MAX {
            return Err(EngineError::CounterExhausted {
                id: instance,
                counter: "step_index",
            });
        }
        for (slot, row) in header.agents().iter().enumerate() {
            if !(row.alive.get() && row.acting.get()) {
                continue;
            }
            let agent = u16::try_from(slot).unwrap_or(u16::MAX);
            let base = index * agents + slot;
            for (head, &n) in layout.discrete.iter().enumerate() {
                let value = input.discrete.get(base * dh + head).copied().unwrap_or(-1);
                if u32::try_from(value).map_or(true, |v| v >= n) {
                    return Err(EngineError::DiscreteOutOfRange {
                        instance,
                        agent,
                        head: u32::try_from(head).unwrap_or(u32::MAX),
                        value,
                        n,
                    });
                }
            }
            for (head, &(low, high)) in layout.continuous.iter().enumerate() {
                let value = input
                    .continuous
                    .get(base * ch + head)
                    .copied()
                    .unwrap_or(f32::NAN);
                let wide = f64::from(value);
                if !value.is_finite() || wide < low || wide > high {
                    return Err(EngineError::ContinuousOutOfRange {
                        instance,
                        agent,
                        head: u32::try_from(head).unwrap_or(u32::MAX),
                        value,
                    });
                }
            }
        }
    }
    // Validated: copy, zeroing slots whose actions are not required.
    for (index, header) in headers.iter().enumerate() {
        for (slot, row) in header.agents().iter().enumerate() {
            let required = row.alive.get() && row.acting.get();
            let base = index * agents + slot;
            copy_row(required, input.discrete, discrete_out, base * dh, dh, 0);
            copy_row(
                required,
                input.continuous,
                continuous_out,
                base * ch,
                ch,
                0.0,
            );
        }
    }
    Ok(())
}

fn check_len(buffer: &'static str, expected: usize, actual: usize) -> Result<(), EngineError> {
    if expected == actual {
        Ok(())
    } else {
        Err(EngineError::ActionLength {
            buffer,
            expected,
            actual,
        })
    }
}

fn copy_row<T: Copy>(required: bool, src: &[T], dst: &mut [T], start: usize, len: usize, zero: T) {
    let Some(end) = start.checked_add(len) else {
        return;
    };
    let Some(dst) = dst.get_mut(start..end) else {
        return;
    };
    match src.get(start..end) {
        Some(src) if required => dst.copy_from_slice(src),
        _ => dst.fill(zero),
    }
}
