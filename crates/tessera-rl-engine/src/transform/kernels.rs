// SPDX-License-Identifier: MIT OR Apache-2.0

//! The transform operations on one agent's leaf row (ADR-0078 item 3).
//! Infallible and allocation-free: sizes are fixed by the plan, and a
//! slice of the wrong size leaves its output untouched.

use tessera_rl_core::space::DType;

use super::plan::{LeafPlan, Op, STACK_HEADER};
use super::stats::{Applied, Moments, normalize};

/// What an observation does to a frame stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StackMode {
    /// A new observation of a continuing episode (step): pushed onto the
    /// ring, or filling it when the agent is new (a respawn).
    Push,
    /// The first observation of an episode (reset, plain restore): fills
    /// the ring (ADR-0078 item 4.2).
    Fresh,
    /// A re-observation of an exactly restored instance (fork, pool
    /// snapshots, `restore_with_transforms`): the ring is read as it is.
    Keep,
}

/// Runs one leaf's operations for one agent: `raw` is what `E::observe`
/// wrote, `out` the exported row. `state` is the agent's arena slice,
/// `applied` the pool statistics, `accum` the chunk accumulator (when
/// accumulating).
#[expect(
    clippy::too_many_arguments,
    reason = "one call site per pass; the parts are disjoint borrows"
)]
pub(crate) fn run_leaf(
    plan: &LeafPlan,
    raw: &[u8],
    out: &mut [u8],
    scratch: &mut [u8],
    state: &mut [u8],
    generation: u32,
    mode: StackMode,
    applied: &[Applied],
    mut accum: Option<&mut [Moments]>,
) {
    let half = scratch.len() / 2;
    let (a, b) = scratch.split_at_mut(half);
    // Where the current input is: 0 = raw, 1 = a, 2 = b.
    let mut src = 0u8;
    let count = plan.ops.len();
    for (j, op) in plan.ops.iter().enumerate() {
        let n = op.out_bytes();
        let last = j + 1 == count;
        let acc = accum.as_deref_mut();
        match (src, last) {
            (0, true) => apply(op, raw, out, state, generation, mode, applied, acc),
            (0, false) => {
                if let Some(dst) = a.get_mut(..n) {
                    apply(op, raw, dst, state, generation, mode, applied, acc);
                }
                src = 1;
            }
            (1, true) => apply(op, a, out, state, generation, mode, applied, acc),
            (1, false) => {
                if let Some(dst) = b.get_mut(..n) {
                    apply(op, a, dst, state, generation, mode, applied, acc);
                }
                src = 2;
            }
            (_, true) => apply(op, b, out, state, generation, mode, applied, acc),
            (_, false) => {
                if let Some(dst) = a.get_mut(..n) {
                    apply(op, b, dst, state, generation, mode, applied, acc);
                }
                src = 1;
            }
        }
    }
}

#[expect(clippy::too_many_arguments, reason = "the operation's disjoint inputs")]
fn apply(
    op: &Op,
    input: &[u8],
    output: &mut [u8],
    state: &mut [u8],
    generation: u32,
    mode: StackMode,
    applied: &[Applied],
    accum: Option<&mut [Moments]>,
) {
    match *op {
        Op::Stack { frame, k, offset } => {
            let len = STACK_HEADER + frame * k;
            if let (Some(ring), Some(frame_in)) =
                (state.get_mut(offset..offset + len), input.get(..frame))
            {
                stack(ring, frame_in, output, frame, k, generation, mode);
            }
        }
        Op::OneHot { dtype, elements, n } => one_hot(input, output, dtype, elements, n),
        Op::Normalize {
            dtype,
            elements,
            clip,
            first,
        } => {
            if let Some(stats) = applied.get(first..first + elements) {
                let acc = accum.and_then(|a| a.get_mut(first..first + elements));
                normalize_row(input, output, dtype, stats, clip, acc);
            }
        }
    }
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    bytes
        .get(at..at + 4)
        .and_then(|b| b.try_into().ok())
        .map_or(0, u32::from_le_bytes)
}

fn write_u32(bytes: &mut [u8], at: usize, value: u32) {
    if let Some(b) = bytes.get_mut(at..at + 4) {
        b.copy_from_slice(&value.to_le_bytes());
    }
}

/// A ring of `k` frames after an 8-byte header: the oldest frame's index
/// plus one (`0` for an empty ring) and the agent generation, both
/// little-endian `u32` (canonical bytes, ADR-0011).
fn stack(
    ring: &mut [u8],
    frame_in: &[u8],
    output: &mut [u8],
    frame: usize,
    k: usize,
    generation: u32,
    mode: StackMode,
) {
    let (header, frames) = ring.split_at_mut(STACK_HEADER);
    let stored = read_u32(header, 0);
    let fresh = match mode {
        StackMode::Fresh => true,
        StackMode::Push => stored == 0 || read_u32(header, 4) != generation,
        StackMode::Keep => stored == 0,
    };
    let mut oldest = usize::try_from(stored).unwrap_or(1).saturating_sub(1) % k.max(1);
    if fresh {
        for slot in frames.chunks_exact_mut(frame) {
            slot.copy_from_slice(frame_in);
        }
        oldest = 0;
        write_u32(header, 4, generation);
    } else if mode == StackMode::Push {
        if let Some(slot) = frames.get_mut(oldest * frame..(oldest + 1) * frame) {
            slot.copy_from_slice(frame_in);
        }
        oldest = (oldest + 1) % k;
    }
    write_u32(header, 0, u32::try_from(oldest + 1).unwrap_or(1));
    for (j, dst) in output.chunks_exact_mut(frame).take(k).enumerate() {
        let at = (oldest + j) % k;
        if let Some(src) = frames.get(at * frame..(at + 1) * frame) {
            dst.copy_from_slice(src);
        }
    }
}

/// The value of element `i` of an integer row, as `i64`.
fn int_at(input: &[u8], dtype: DType, i: usize) -> Option<i64> {
    let size = dtype.size();
    let b = input.get(i * size..(i + 1) * size)?;
    Some(match dtype {
        DType::U8 => i64::from(*b.first()?),
        DType::I8 => i64::from(i8::from_ne_bytes([*b.first()?])),
        DType::U16 => i64::from(u16::from_ne_bytes(b.try_into().ok()?)),
        DType::I16 => i64::from(i16::from_ne_bytes(b.try_into().ok()?)),
        DType::U32 => i64::from(u32::from_ne_bytes(b.try_into().ok()?)),
        DType::I32 => i64::from(i32::from_ne_bytes(b.try_into().ok()?)),
        DType::I64 => i64::from_ne_bytes(b.try_into().ok()?),
        DType::Bool | DType::F32 | DType::F64 => return None,
    })
}

/// The value of element `i` of a numeric row, as `f64`.
fn value_at(input: &[u8], dtype: DType, i: usize) -> Option<f64> {
    let size = dtype.size();
    let b = input.get(i * size..(i + 1) * size)?;
    match dtype {
        DType::F32 => Some(f64::from(f32::from_ne_bytes(b.try_into().ok()?))),
        DType::F64 => Some(f64::from_ne_bytes(b.try_into().ok()?)),
        #[expect(
            clippy::cast_precision_loss,
            reason = "i64 leaves beyond 2^53 round to the nearest f64, as numpy does"
        )]
        _ => int_at(input, dtype, i).map(|v| v as f64),
    }
}

fn one_hot(input: &[u8], output: &mut [u8], dtype: DType, elements: usize, n: usize) {
    output.fill(0);
    for i in 0..elements {
        let Some(v) = int_at(input, dtype, i) else {
            continue;
        };
        if let Ok(class) = usize::try_from(v)
            && class < n
            && let Some(cell) = output.get_mut(i * n + class)
        {
            *cell = 1;
        }
    }
}

fn normalize_row(
    input: &[u8],
    output: &mut [u8],
    dtype: DType,
    stats: &[Applied],
    clip: f32,
    mut accum: Option<&mut [Moments]>,
) {
    for (i, (dst, &applied)) in output
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(stats)
        .enumerate()
    {
        let Some(x) = value_at(input, dtype, i) else {
            continue;
        };
        if let Some(m) = accum.as_deref_mut().and_then(|a| a.get_mut(i)) {
            m.add(x);
        }
        *dst = normalize(x, applied, clip).to_ne_bytes();
    }
}

#[cfg(test)]
mod tests {
    use super::{StackMode, run_leaf};
    use crate::transform::plan::{LeafPlan, Op, STACK_HEADER};
    use crate::transform::stats::{Applied, Moments};
    use tessera_rl_core::space::DType;

    fn stack_plan(frame: usize, k: usize) -> LeafPlan {
        LeafPlan {
            leaf: 0,
            raw: frame,
            ops: vec![Op::Stack {
                frame,
                k,
                offset: 0,
            }],
        }
    }

    fn run(
        plan: &LeafPlan,
        raw: &[u8],
        state: &mut [u8],
        generation: u32,
        mode: StackMode,
    ) -> Vec<u8> {
        let out_len = plan.ops.last().map_or(plan.raw, Op::out_bytes);
        let mut out = vec![0xAA; out_len];
        let mut scratch = vec![0; 256];
        run_leaf(
            plan,
            raw,
            &mut out,
            &mut scratch,
            state,
            generation,
            mode,
            &[],
            None,
        );
        out
    }

    #[test]
    fn test_stack_fresh_fills_then_push_shifts_oldest_first() {
        let plan = stack_plan(1, 3);
        let mut state = vec![0; STACK_HEADER + 3];
        assert_eq!(run(&plan, &[7], &mut state, 0, StackMode::Fresh), [7, 7, 7]);
        assert_eq!(run(&plan, &[8], &mut state, 0, StackMode::Push), [7, 7, 8]);
        assert_eq!(run(&plan, &[9], &mut state, 0, StackMode::Push), [7, 8, 9]);
        // The ring wraps.
        assert_eq!(
            run(&plan, &[10], &mut state, 0, StackMode::Push),
            [8, 9, 10]
        );
        assert_eq!(
            run(&plan, &[11], &mut state, 0, StackMode::Push),
            [9, 10, 11]
        );
    }

    #[test]
    fn test_stack_keep_reads_without_pushing() {
        let plan = stack_plan(2, 2);
        let mut state = vec![0; STACK_HEADER + 4];
        run(&plan, &[1, 1], &mut state, 0, StackMode::Fresh);
        run(&plan, &[2, 2], &mut state, 0, StackMode::Push);
        let kept = state.clone();
        assert_eq!(
            run(&plan, &[2, 2], &mut state, 0, StackMode::Keep),
            [1, 1, 2, 2]
        );
        assert_eq!(state, kept);
    }

    #[test]
    fn test_stack_push_after_respawn_refills() {
        let plan = stack_plan(1, 3);
        let mut state = vec![0; STACK_HEADER + 3];
        run(&plan, &[1], &mut state, 4, StackMode::Fresh);
        run(&plan, &[2], &mut state, 4, StackMode::Push);
        // A new generation never sees the previous agent's frames.
        assert_eq!(run(&plan, &[5], &mut state, 5, StackMode::Push), [5, 5, 5]);
    }

    #[test]
    fn test_stack_push_on_an_empty_ring_fills() {
        let plan = stack_plan(1, 2);
        let mut state = vec![0; STACK_HEADER + 2];
        assert_eq!(run(&plan, &[3], &mut state, 0, StackMode::Push), [3, 3]);
    }

    #[test]
    fn test_one_hot_out_of_range_rows_are_all_false() {
        let plan = LeafPlan {
            leaf: 0,
            raw: 12,
            ops: vec![Op::OneHot {
                dtype: DType::I32,
                elements: 3,
                n: 3,
            }],
        };
        let raw: Vec<u8> = [2i32, -1, 3].iter().flat_map(|v| v.to_ne_bytes()).collect();
        let out = run(&plan, &raw, &mut [], 0, StackMode::Push);
        assert_eq!(out, [0, 0, 1, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_normalize_then_stack_chains_and_accumulates_raw_values() {
        let plan = LeafPlan {
            leaf: 0,
            raw: 2,
            ops: vec![
                Op::Normalize {
                    dtype: DType::U8,
                    elements: 2,
                    clip: 100.0,
                    first: 0,
                },
                Op::Stack {
                    frame: 8,
                    k: 2,
                    offset: 0,
                },
            ],
        };
        let mut state = vec![0; STACK_HEADER + 16];
        let mut out = vec![0; 16];
        let mut scratch = vec![0; 64];
        let applied = [Applied::default(); 2];
        let mut acc = [Moments::default(); 2];
        run_leaf(
            &plan,
            &[3, 250],
            &mut out,
            &mut scratch,
            &mut state,
            0,
            StackMode::Fresh,
            &applied,
            Some(&mut acc),
        );
        let floats: Vec<f32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect();
        assert_eq!(floats, [3.0, 100.0, 3.0, 100.0]);
        assert_eq!((acc[0].count, acc[0].mean, acc[1].mean), (1.0, 3.0, 250.0));
    }
}
