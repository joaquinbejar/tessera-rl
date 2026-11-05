// SPDX-License-Identifier: MIT OR Apache-2.0

//! The validation boundary (ADR-0028): every check runs before any
//! instance is mutated.

use tessera_rl_core::{Bool8, RecordLayout};

use crate::error::EngineError;
use crate::layout::ActionLayout;
use crate::options::AutoReset;
use crate::sys::sync::{AtomicBool, Ordering};

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
    for (position, index) in rows {
        let instance = u32::try_from(index).unwrap_or(u32::MAX);
        let Some(header) = headers.get(index) else {
            return Err(EngineError::InstanceOutOfRange {
                id: instance,
                num_envs: u32::try_from(headers.len()).unwrap_or(u32::MAX),
            });
        };
        check_one(layout, auto_reset, header, instance, position, input)?;
    }
    Ok(())
}

/// Every check of one instance: `header` is instance `instance`, whose
/// actions are row block `position` of `input`.
fn check_one<H: RecordLayout>(
    layout: &ActionLayout,
    auto_reset: AutoReset,
    header: &H,
    instance: u32,
    position: usize,
    input: ActionInput<'_>,
) -> Result<(), EngineError> {
    let agents = H::AGENTS;
    let (dh, ch) = (layout.discrete.len(), layout.continuous.len());
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
        return Ok(());
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
            copy_row(
                required,
                input.discrete,
                base * dh,
                discrete_out,
                base * dh,
                dh,
                0,
            );
            copy_row(
                required,
                input.continuous,
                base * ch,
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
    // A zero factor makes the product zero whatever the others are, so it
    // is checked first and never reported as an overflow.
    if factors.contains(&0) {
        return Ok(0);
    }
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

/// Copies `len` values from `src[from..]` to `dst[to..]` when the action is
/// required, and zeroes them otherwise. Out-of-bounds requests are skipped;
/// every caller checked the lengths first.
fn copy_row<T: Copy>(
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
    loom,
    allow(dead_code, reason = "the split-phase modes are not built under loom")
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
    loom,
    allow(dead_code, reason = "the split-phase modes are not built under loom")
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

/// The rows of one instance admitted to a split-phase send: its header,
/// read only, and its action rows, `[A, heads]`.
pub(crate) struct ActionRow<'a, H> {
    /// The instance header.
    pub(crate) header: &'a H,
    /// Discrete actions of the instance, `[A, discrete heads]`.
    pub(crate) discrete: &'a mut [i32],
    /// Continuous actions of the instance, `[A, continuous heads]`.
    pub(crate) continuous: &'a mut [f32],
}

/// Access to the rows of the ids an Async `send` admits, by position in the
/// call. An implementation hands out only those rows, one at a time, so no
/// reference ever spans a row another thread may use (ADR-0047 invariants
/// 3 and 4). The engine implementation is `buffer::rows::Admitted`, from
/// `RawRows::admitted`.
pub(crate) trait AdmittedRows<H> {
    /// The number of admitted ids.
    fn len(&self) -> usize;
    /// The number of instances in the pool (for out-of-range errors).
    fn instances(&self) -> usize;
    /// The id at `position`.
    fn id(&self, position: usize) -> Option<u32>;
    /// The rows of the id at `position`.
    fn row(&mut self, position: usize) -> Option<ActionRow<'_, H>>;
}

/// Validates the actions of an Async `send(ids)` and copies them into the
/// rows of those ids only. `input` holds one row block per id, in the order
/// of `rows`. All ids are validated in a first pass and copied in a second,
/// so a failure writes nothing. The ids must already have passed
/// [`check_send_ids`].
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the split-phase modes (ADR-0047)")
)]
pub(crate) fn check_and_copy_subset_actions<H: RecordLayout, R: AdmittedRows<H>>(
    layout: &ActionLayout,
    auto_reset: AutoReset,
    rows: &mut R,
    input: ActionInput<'_>,
) -> Result<(), EngineError> {
    let agents = H::AGENTS;
    let (dh, ch) = (layout.discrete.len(), layout.continuous.len());
    let count = rows.len();
    for (buffer, heads, actual) in [
        ("discrete", dh, input.discrete.len()),
        ("continuous", ch, input.continuous.len()),
    ] {
        check_len(
            buffer,
            expected_len(buffer, &[count, agents, heads], actual)?,
            actual,
        )?;
    }
    for position in 0..count {
        let (id, row) = admitted(rows, position)?;
        check_row_lengths(&row, agents, dh, ch)?;
        check_one(layout, auto_reset, row.header, id, position, input)?;
    }
    // Validated: copy, one admitted row at a time.
    for position in 0..count {
        let (_, mut row) = admitted(rows, position)?;
        copy_instance(&mut row, input, position, agents, dh, ch);
    }
    Ok(())
}

/// Validates the actions of a Pipelined `send(half)` and copies them into
/// the half's rows. `headers`, `discrete_out` and `continuous_out` are the
/// half's own views (the Pipelined mode takes them from the half's raw
/// action parts, `RawRows::range` gives the same rows), so nothing outside
/// the half is borrowed; `first_id` is the id of their first instance, for errors.
/// `input` holds one row block per instance of the half, in order.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the mode runs the checks and the copy separately; tests use both at once"
    )
)]
#[cfg_attr(
    all(test, loom),
    allow(dead_code, reason = "the split-phase modes are not built under loom")
)]
pub(crate) fn check_and_copy_range_actions<H: RecordLayout>(
    layout: &ActionLayout,
    auto_reset: AutoReset,
    first_id: usize,
    headers: &[H],
    input: ActionInput<'_>,
    discrete_out: &mut [i32],
    continuous_out: &mut [f32],
) -> Result<(), EngineError> {
    check_range_actions(
        layout,
        auto_reset,
        first_id,
        headers,
        input,
        (discrete_out.len(), continuous_out.len()),
    )?;
    copy_range_actions(layout, headers, input, discrete_out, continuous_out);
    Ok(())
}

/// The checks of [`check_and_copy_range_actions`] without the copy, so a
/// caller can run more checks (the `debug-checks` legality pass) between
/// the validation and the copy. `out_lens` are the lengths of the half's
/// discrete and continuous action rows.
#[cfg_attr(
    loom,
    allow(dead_code, reason = "the split-phase modes are not built under loom")
)]
pub(crate) fn check_range_actions<H: RecordLayout>(
    layout: &ActionLayout,
    auto_reset: AutoReset,
    first_id: usize,
    headers: &[H],
    input: ActionInput<'_>,
    out_lens: (usize, usize),
) -> Result<(), EngineError> {
    let agents = H::AGENTS;
    let (dh, ch) = (layout.discrete.len(), layout.continuous.len());
    let count = headers.len();
    for (buffer, heads, actual) in [
        ("discrete", dh, input.discrete.len()),
        ("continuous", ch, input.continuous.len()),
        ("discrete", dh, out_lens.0),
        ("continuous", ch, out_lens.1),
    ] {
        check_len(
            buffer,
            expected_len(buffer, &[count, agents, heads], actual)?,
            actual,
        )?;
    }
    for (position, header) in headers.iter().enumerate() {
        let id = first_id
            .checked_add(position)
            .and_then(|i| u32::try_from(i).ok())
            .unwrap_or(u32::MAX);
        check_one(layout, auto_reset, header, id, position, input)?;
    }
    Ok(())
}

/// The copy of [`check_and_copy_range_actions`], after its checks passed:
/// slots whose actions are not required are zeroed.
#[cfg_attr(
    loom,
    allow(dead_code, reason = "the split-phase modes are not built under loom")
)]
pub(crate) fn copy_range_actions<H: RecordLayout>(
    layout: &ActionLayout,
    headers: &[H],
    input: ActionInput<'_>,
    discrete_out: &mut [i32],
    continuous_out: &mut [f32],
) {
    let agents = H::AGENTS;
    let (dh, ch) = (layout.discrete.len(), layout.continuous.len());
    for (position, header) in headers.iter().enumerate() {
        let pending = header.engine().pending_reset.get();
        for (slot, row) in header.agents().iter().enumerate() {
            let required = !pending && row.alive.get() && row.acting.get();
            let at = position * agents + slot;
            copy_row(
                required,
                input.discrete,
                at * dh,
                discrete_out,
                at * dh,
                dh,
                0,
            );
            copy_row(
                required,
                input.continuous,
                at * ch,
                continuous_out,
                at * ch,
                ch,
                0.0,
            );
        }
    }
}

/// The id and rows at `position`; a missing row is an implementation bug,
/// reported as an out-of-range id rather than a panic.
fn admitted<H, R: AdmittedRows<H>>(
    rows: &mut R,
    position: usize,
) -> Result<(u32, ActionRow<'_, H>), EngineError> {
    let id = rows.id(position).unwrap_or(u32::MAX);
    let num_envs = u32::try_from(rows.instances()).unwrap_or(u32::MAX);
    match rows.row(position) {
        Some(row) => Ok((id, row)),
        None => Err(EngineError::InstanceOutOfRange { id, num_envs }),
    }
}

fn check_row_lengths<H>(
    row: &ActionRow<'_, H>,
    agents: usize,
    dh: usize,
    ch: usize,
) -> Result<(), EngineError> {
    for (buffer, heads, actual) in [
        ("discrete", dh, row.discrete.len()),
        ("continuous", ch, row.continuous.len()),
    ] {
        check_len(
            buffer,
            expected_len(buffer, &[agents, heads], actual)?,
            actual,
        )?;
    }
    Ok(())
}

fn copy_instance<H: RecordLayout>(
    row: &mut ActionRow<'_, H>,
    input: ActionInput<'_>,
    position: usize,
    agents: usize,
    dh: usize,
    ch: usize,
) {
    let pending = row.header.engine().pending_reset.get();
    for (slot, agent) in row.header.agents().iter().enumerate() {
        let required = !pending && agent.alive.get() && agent.acting.get();
        let from = position * agents + slot;
        copy_row(
            required,
            input.discrete,
            from * dh,
            row.discrete,
            slot * dh,
            dh,
            0,
        );
        copy_row(
            required,
            input.continuous,
            from * ch,
            row.continuous,
            slot * ch,
            ch,
            0.0,
        );
    }
}

#[cfg(test)]
mod tests;
