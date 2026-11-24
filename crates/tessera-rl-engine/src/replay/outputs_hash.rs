// SPDX-License-Identifier: MIT OR Apache-2.0

//! The outputs hash of a host call (ADR-0072 section 4) and its refusal rows
//! (ADR-0070 item 9).
//!
//! The hash is XXH3-64 over, for each touched id in the call's order, that
//! instance's rows of: every observation leaf (raw bytes, leaf order), the
//! rewards (`f32` bits), the agent terminations and truncations (one byte
//! each), `aborted`, `abort_code`, `pending_reset`, then, for a refusing
//! host, `refused` and `refuse_code`. Integers are little-endian; flags are
//! one byte, 0 or 1. Every output is laid out `[N, ...]`, so an instance's
//! row is its `1/N` share of the buffer.

use tessera_rl_core::Bool8;
use tessera_rl_core::encoding::{HashSink, Sink};

use crate::outputs::Outputs;

use super::reader::RefusalRow;

/// The row of instance `i` in a buffer of `num_envs` equal rows.
fn row<T>(buffer: &[T], num_envs: usize, i: usize) -> &[T] {
    let width = buffer.len().checked_div(num_envs).unwrap_or(0);
    let start = i.saturating_mul(width);
    buffer
        .get(start..start.saturating_add(width))
        .unwrap_or_default()
}

fn flags(sink: &mut HashSink, buffer: &[Bool8], num_envs: usize, i: usize) {
    for f in row(buffer, num_envs, i) {
        sink.write(&[u8::from(f.get())]);
    }
}

fn words(sink: &mut HashSink, buffer: &[u32], num_envs: usize, i: usize) {
    for w in row(buffer, num_envs, i) {
        sink.write(&w.to_le_bytes());
    }
}

/// The outputs hash of the instances `ids`, in that order.
pub(crate) fn outputs_hash(
    out: &Outputs,
    num_envs: usize,
    ids: impl Iterator<Item = usize>,
    refuses: bool,
) -> u64 {
    let mut sink = HashSink::new();
    for i in ids {
        let mut leaf = 0;
        while let Some(obs) = out.observation(leaf) {
            sink.write(row(obs, num_envs, i));
            leaf += 1;
        }
        for r in row(out.rewards(), num_envs, i) {
            sink.write(&r.to_bits().to_le_bytes());
        }
        flags(&mut sink, out.agent_terminated(), num_envs, i);
        flags(&mut sink, out.agent_truncated(), num_envs, i);
        flags(&mut sink, out.aborted(), num_envs, i);
        words(&mut sink, out.abort_code(), num_envs, i);
        flags(&mut sink, out.pending_reset(), num_envs, i);
        if refuses {
            flags(&mut sink, out.refused(), num_envs, i);
            words(&mut sink, out.refuse_code(), num_envs, i);
        }
    }
    sink.digest()
}

/// The refused agents of the last call, sparse, in (instance, slot) order.
pub(crate) fn refusal_rows(out: &Outputs, num_envs: usize) -> Vec<RefusalRow> {
    let codes = out.refuse_code();
    let agents = codes.len().checked_div(num_envs).unwrap_or(0);
    let mut rows = Vec::new();
    for (k, &code) in codes.iter().enumerate() {
        let refused = out.refused().get(k).is_some_and(|f| f.get());
        if !refused || code == 0 {
            continue;
        }
        let (Some(instance), Some(slot)) = (k.checked_div(agents), k.checked_rem(agents)) else {
            continue;
        };
        if let (Ok(instance), Ok(slot)) = (u32::try_from(instance), u32::try_from(slot)) {
            rows.push(RefusalRow {
                instance,
                slot,
                code,
            });
        }
    }
    rows
}
