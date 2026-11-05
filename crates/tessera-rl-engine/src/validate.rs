// SPDX-License-Identifier: MIT OR Apache-2.0

//! The validation boundary (ADR-0028): every check runs before any
//! instance is mutated.

use core::ops::Range;
use core::sync::atomic::{AtomicBool, Ordering};

use tessera_rl_core::{Bool8, RecordLayout};

use crate::error::EngineError;
use crate::layout::ActionLayout;
use crate::options::AutoReset;

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

/// Validates actions like `check_and_copy_actions`, without copying.
pub(crate) fn check_actions_only<H: RecordLayout>(
    layout: &ActionLayout,
    auto_reset: AutoReset,
    headers: &[H],
    input: ActionInput<'_>,
) -> Result<(), EngineError> {
    let discrete = expected_len(
        "discrete",
        &[headers.len(), H::AGENTS, layout.discrete.len()],
        input.discrete.len(),
    )?;
    let continuous = expected_len(
        "continuous",
        &[headers.len(), H::AGENTS, layout.continuous.len()],
        input.continuous.len(),
    )?;
    check_inputs(layout, auto_reset, headers, input, discrete, continuous)
}

/// Every check of the validation boundary, without mutation.
fn check_inputs<H: RecordLayout>(
    layout: &ActionLayout,
    auto_reset: AutoReset,
    headers: &[H],
    input: ActionInput<'_>,
    discrete_len: usize,
    continuous_len: usize,
) -> Result<(), EngineError> {
    check_len("discrete", discrete_len, input.discrete.len())?;
    check_len("continuous", continuous_len, input.continuous.len())?;
    check_rows(
        layout,
        auto_reset,
        headers,
        input,
        (0..headers.len()).map(|i| (i, i)),
    )
}

/// Checks the instances named by `rows`: each item is `(position, id)`,
/// where `position` indexes the row block of `input` and `id` the header.
fn check_rows<H: RecordLayout>(
    layout: &ActionLayout,
    auto_reset: AutoReset,
    headers: &[H],
    input: ActionInput<'_>,
    rows: impl Iterator<Item = (usize, usize)>,
) -> Result<(), EngineError> {
    let agents = H::AGENTS;
    let (dh, ch) = (layout.discrete.len(), layout.continuous.len());
    for (position, index) in rows {
        let instance = u32::try_from(index).unwrap_or(u32::MAX);
        let Some(header) = headers.get(index) else {
            return Err(EngineError::InstanceOutOfRange {
                id: instance,
                num_envs: u32::try_from(headers.len()).unwrap_or(u32::MAX),
            });
        };
        let engine = header.engine();
        if engine.pending_reset.get() {
            // Never reset, or frozen in Disabled mode: the caller must reset.
            if engine.fresh.get() || auto_reset == AutoReset::Disabled {
                return Err(EngineError::PendingReset { id: instance });
            }
            // NextStep: this instance resets instead of stepping.
            if engine.identity.episode_index() == u32::MAX {
                return Err(EngineError::CounterExhausted {
                    id: instance,
                    counter: "episode_index",
                });
            }
            continue;
        }
        // SameStep may reset any instance in this pass.
        if auto_reset == AutoReset::SameStep && engine.identity.episode_index() == u32::MAX {
            return Err(EngineError::CounterExhausted {
                id: instance,
                counter: "episode_index",
            });
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
            let base = position * agents + slot;
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
    Ok(())
}

/// Validates a step for every instance and copies the required actions
/// into the internal buffers; values of slots that are not required are
/// stored as zero and never validated.
pub(crate) fn check_and_copy_actions<H: RecordLayout>(
    layout: &ActionLayout,
    auto_reset: AutoReset,
    headers: &[H],
    input: ActionInput<'_>,
    discrete_out: &mut [i32],
    continuous_out: &mut [f32],
) -> Result<(), EngineError> {
    let agents = H::AGENTS;
    let (dh, ch) = (layout.discrete.len(), layout.continuous.len());
    check_inputs(
        layout,
        auto_reset,
        headers,
        input,
        discrete_out.len(),
        continuous_out.len(),
    )?;
    // Validated: copy, zeroing slots whose actions are not required.
    for (index, header) in headers.iter().enumerate() {
        let pending = header.engine().pending_reset.get();
        for (slot, row) in header.agents().iter().enumerate() {
            let required = !pending && row.alive.get() && row.acting.get();
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

/// The product of `factors`, with checked arithmetic. An overflow is an
/// `ActionLength` error whose `expected` is `usize::MAX`: no real buffer
/// can hold that many values, so the call is refused before any copy.
fn expected_len(
    buffer: &'static str,
    factors: &[usize],
    actual: usize,
) -> Result<usize, EngineError> {
    let mut product: usize = 1;
    for &factor in factors {
        match product.checked_mul(factor) {
            Some(next) => product = next,
            None => {
                return Err(EngineError::ActionLength {
                    buffer,
                    expected: usize::MAX,
                    actual,
                });
            }
        }
    }
    Ok(product)
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

// Split-phase modes (ADR-0047): the checks of `send` and `recv` over
// caller-side state, and the action copy restricted to the admitted rows.
// Each check runs before anything is mutated, so a failure leaves every
// buffer and flag as it was.

/// Checks an Async `send(ids)`: the call names at least one instance, every
/// id is in range and appears once, and none is in flight or covered by a
/// live token. The flags are read with Acquire, so a token dropped on
/// another thread is seen with every read it made.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the split-phase modes (ADR-0047)")
)]
pub(crate) fn check_send_ids(
    ids: &[u32],
    seen: &mut [Bool8],
    in_flight: &[AtomicBool],
    live: &[AtomicBool],
) -> Result<(), EngineError> {
    if ids.is_empty() {
        return Err(EngineError::EmptySend);
    }
    check_ids(ids, seen)?;
    for &id in ids {
        let index = usize::try_from(id).unwrap_or(usize::MAX);
        if in_flight
            .get(index)
            .is_some_and(|f| f.load(Ordering::Acquire))
        {
            return Err(EngineError::InFlight { id });
        }
        if live.get(index).is_some_and(|f| f.load(Ordering::Acquire)) {
            return Err(EngineError::Borrowed { id });
        }
    }
    Ok(())
}

/// Checks a Pipelined `send(half)` and returns the half's index: it is 0
/// or 1 and is neither in flight nor covered by a live token.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the split-phase modes (ADR-0047)")
)]
pub(crate) fn check_send_half(
    half: u32,
    in_flight: &[AtomicBool; 2],
    live: &[AtomicBool; 2],
) -> Result<usize, EngineError> {
    let (index, short) = match half {
        0 => (0, 0_u8),
        1 => (1, 1_u8),
        _ => return Err(EngineError::InvalidHalf { half }),
    };
    if in_flight
        .get(index)
        .is_some_and(|f| f.load(Ordering::Acquire))
    {
        return Err(EngineError::HalfInFlight { half: short });
    }
    if live.get(index).is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(EngineError::HalfBorrowed { half: short });
    }
    Ok(index)
}

/// Pipelined cycle admission (ADR-0047 section 6): a half's step of cycle
/// `cycle` reads context version `cycle - 2`, so it is admitted only once
/// that cycle is committed. Cycles 0 and 1 read the initial context.
/// `committed` is the newest committed cycle, if any.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the split-phase modes (ADR-0047)")
)]
pub(crate) fn check_cycle(cycle: u64, committed: Option<u64>) -> Result<(), EngineError> {
    let Some(needed) = cycle.checked_sub(2) else {
        return Ok(());
    };
    match committed {
        Some(done) if done >= needed => Ok(()),
        _ => Err(EngineError::CycleAhead { cycle, needed }),
    }
}

/// Checks that a `recv` can return without waiting forever: something is
/// in flight or ready, and in Async enough of them exist to fill a batch.
/// `batch_size` is 1 in Pipelined, where a half is one result.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the split-phase modes (ADR-0047)")
)]
pub(crate) fn check_recv(in_flight: u32, ready: u32, batch_size: u32) -> Result<(), EngineError> {
    if in_flight == 0 && ready == 0 {
        return Err(EngineError::NothingInFlight);
    }
    // An overflowing sum exceeds any u32 batch size, so it is reachable.
    if in_flight
        .checked_add(ready)
        .is_some_and(|sum| sum < batch_size)
    {
        return Err(EngineError::BatchUnreachable {
            in_flight,
            ready,
            batch_size,
        });
    }
    Ok(())
}

/// Validates the actions of an Async `send(ids)` and copies them into the
/// rows of those ids only. `input` holds one row block per id, in `ids`
/// order. The ids must already have passed [`check_send_ids`]. Rows of
/// other instances are never written, since workers may be reading them.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the split-phase modes (ADR-0047)")
)]
pub(crate) fn check_and_copy_subset_actions<H: RecordLayout>(
    layout: &ActionLayout,
    auto_reset: AutoReset,
    headers: &[H],
    ids: &[u32],
    input: ActionInput<'_>,
    discrete_out: &mut [i32],
    continuous_out: &mut [f32],
) -> Result<(), EngineError> {
    let rows = ids
        .iter()
        .enumerate()
        .map(|(position, &id)| (position, usize::try_from(id).unwrap_or(usize::MAX)));
    check_and_copy_rows(
        layout,
        auto_reset,
        headers,
        ids.len(),
        rows,
        input,
        discrete_out,
        continuous_out,
    )
}

/// Validates the actions of a Pipelined `send(half)` and copies them into
/// the rows of `range` only. `input` holds one row block per instance of
/// the range, in order.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the split-phase modes (ADR-0047)")
)]
pub(crate) fn check_and_copy_range_actions<H: RecordLayout>(
    layout: &ActionLayout,
    auto_reset: AutoReset,
    headers: &[H],
    range: Range<usize>,
    input: ActionInput<'_>,
    discrete_out: &mut [i32],
    continuous_out: &mut [f32],
) -> Result<(), EngineError> {
    if range.start > range.end || range.end > headers.len() {
        return Err(EngineError::InstanceOutOfRange {
            id: u32::try_from(range.end).unwrap_or(u32::MAX),
            num_envs: u32::try_from(headers.len()).unwrap_or(u32::MAX),
        });
    }
    let count = range.len();
    let start = range.start;
    let rows = range.map(move |index| (index - start, index));
    check_and_copy_rows(
        layout,
        auto_reset,
        headers,
        count,
        rows,
        input,
        discrete_out,
        continuous_out,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "one private helper behind the two public shapes"
)]
fn check_and_copy_rows<H: RecordLayout>(
    layout: &ActionLayout,
    auto_reset: AutoReset,
    headers: &[H],
    count: usize,
    rows: impl Iterator<Item = (usize, usize)> + Clone,
    input: ActionInput<'_>,
    discrete_out: &mut [i32],
    continuous_out: &mut [f32],
) -> Result<(), EngineError> {
    let agents = H::AGENTS;
    let (dh, ch) = (layout.discrete.len(), layout.continuous.len());
    let n = headers.len();
    let lengths = [
        ("discrete", [count, agents, dh], input.discrete.len()),
        ("continuous", [count, agents, ch], input.continuous.len()),
        ("discrete", [n, agents, dh], discrete_out.len()),
        ("continuous", [n, agents, ch], continuous_out.len()),
    ];
    for (buffer, factors, actual) in lengths {
        let expected = expected_len(buffer, &factors, actual)?;
        check_len(buffer, expected, actual)?;
    }
    check_rows(layout, auto_reset, headers, input, rows.clone())?;
    // Validated: copy the admitted rows only.
    for (position, index) in rows {
        let Some(header) = headers.get(index) else {
            continue;
        };
        let pending = header.engine().pending_reset.get();
        for (slot, row) in header.agents().iter().enumerate() {
            let required = !pending && row.alive.get() && row.acting.get();
            let from = position * agents + slot;
            let to = index * agents + slot;
            copy_block(
                required,
                input.discrete,
                from * dh,
                discrete_out,
                to * dh,
                dh,
                0,
            );
            copy_block(
                required,
                input.continuous,
                from * ch,
                continuous_out,
                to * ch,
                ch,
                0.0,
            );
        }
    }
    Ok(())
}

fn copy_block<T: Copy>(
    required: bool,
    src: &[T],
    from: usize,
    dst: &mut [T],
    to: usize,
    len: usize,
    zero: T,
) {
    let (Some(from_end), Some(to_end)) = (from.checked_add(len), to.checked_add(len)) else {
        return;
    };
    let Some(dst) = dst.get_mut(to..to_end) else {
        return;
    };
    match src.get(from..from_end) {
        Some(src) if required => dst.copy_from_slice(src),
        _ => dst.fill(zero),
    }
}

#[cfg(test)]
mod tests;
