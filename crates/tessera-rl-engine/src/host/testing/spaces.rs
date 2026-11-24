// SPDX-License-Identifier: MIT OR Apache-2.0

//! `check_spaces` (#351, SPEC 10.2): outputs conform to the declared
//! spaces.

use tessera_rl_core::HostEnvironment;
use tessera_rl_core::space::{BoxBounds, DType, Space};

use super::super::HostLayout;
use super::actions::Sampler;
use super::error::CheckError;
use super::run::{CheckRun, build};
use crate::outputs::Outputs;
use crate::pool::Reseed;

/// What [`check_spaces`] saw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SpacesReport {
    /// Calls checked, the first reset included.
    pub calls: usize,
    /// Observation values checked against their space.
    pub values: u64,
}

/// Checks that `E`'s outputs conform to its declared spaces.
///
/// Builds one Inline pool, checks its layout (one mask block per discrete
/// head, of the head's size, and a mask buffer of `[N, A, mask length]`),
/// then resets it and steps it `run.calls` times with seeded legal
/// actions. After every call, for every alive agent of every instance that
/// was not aborted (an aborted instance's rows are zeros by contract):
///
/// - every observation value lies in its leaf's space: `Discrete` and
///   `MultiDiscrete` choices in range, `MultiBinary` and `Bool` values 0 or
///   1, `Box` values within their bounds (a NaN is never within);
/// - every reward, of every slot, is finite (under `debug-checks` the
///   engine poisons the pool on a NaN reward first: [`CheckError::Engine`]);
/// - an acting agent of a running episode has a legal choice in every
///   discrete head whose mask block has no parent.
///
/// # Errors
///
/// [`CheckError::Layout`], [`CheckError::OutsideSpace`],
/// [`CheckError::RewardNotFinite`], [`CheckError::Invariant`] for a head
/// with no legal choice, or [`CheckError::Engine`].
pub fn check_spaces<E: HostEnvironment, const A: usize>(
    cfg: &dyn Fn() -> E::Config,
    run: &CheckRun,
) -> Result<SpacesReport, CheckError> {
    let mut pool = build::<E, A>(cfg, run, run.seed)?;
    // Owned: the pool steps while the leaf spaces are borrowed.
    let observation_space = pool.layout().observation_space.clone();
    let spaces = leaf_spaces(&observation_space);
    check_layout(pool.layout(), spaces.len())?;
    let mut sampler = Sampler::new(run.action_seed);
    let mut values = 0u64;
    pool.reset_all(Reseed::Keep)?;
    values = values.saturating_add(check_call(0, pool.layout(), &spaces, pool.outputs()?)?);
    for call in 1..=run.calls {
        let input = sampler.fill(pool.layout(), pool.outputs()?);
        pool.step(input)?;
        values = values.saturating_add(check_call(call, pool.layout(), &spaces, pool.outputs()?)?);
    }
    Ok(SpacesReport {
        calls: run.calls.saturating_add(1),
        values,
    })
}

fn check_layout(layout: &HostLayout, leaves: usize) -> Result<(), CheckError> {
    let fail = |what| Err(CheckError::Layout { what });
    if layout.leaves.len() != leaves {
        return fail("observation leaves");
    }
    if layout.mask_len == 0 {
        return if layout.mask_heads.is_empty() {
            Ok(())
        } else {
            fail("mask heads without a mask row")
        };
    }
    if layout.mask_heads.len() != layout.discrete_heads.len() {
        return fail("one mask block per discrete head");
    }
    for (shape, &n) in layout.mask_heads.iter().zip(&layout.discrete_heads) {
        if usize::try_from(n).ok() != Some(shape.size) {
            return fail("a mask block of the head's size");
        }
        let parents = shape
            .parent_sizes
            .iter()
            .flatten()
            .try_fold(1usize, |acc, &p| acc.checked_mul(p));
        let end = parents
            .and_then(|p| p.checked_mul(shape.size))
            .and_then(|len| shape.offset.checked_add(len));
        if end.is_none_or(|end| end > layout.mask_len) {
            return fail("a mask block inside the mask row");
        }
    }
    Ok(())
}

/// Checks one call's outputs; returns the observation values checked.
fn check_call(
    call: usize,
    layout: &HostLayout,
    spaces: &[(String, &Space)],
    o: &Outputs,
) -> Result<u64, CheckError> {
    let (n, a) = (layout.num_envs, layout.agents);
    let rows = n.checked_mul(a).ok_or(CheckError::Layout {
        what: "rows overflow",
    })?;
    if o.masks().len() != rows.saturating_mul(layout.mask_len) {
        return Err(CheckError::Layout {
            what: "a mask buffer of [N, A, mask length]",
        });
    }
    let mut values = 0u64;
    for row in 0..rows {
        let (instance, agent) = (row / a.max(1), row % a.max(1));
        if let Some(&value) = o.rewards().get(row)
            && !value.is_finite()
        {
            return Err(CheckError::RewardNotFinite {
                call,
                instance,
                agent,
                value,
            });
        }
        let flag = |v: &[tessera_rl_core::Bool8], i: usize| v.get(i).is_some_and(|b| b.get());
        if !flag(o.alive(), row) || flag(o.aborted(), instance) {
            continue;
        }
        for (k, (name, space)) in spaces.iter().enumerate() {
            let Some(leaf) = o.observation(k) else {
                return Err(CheckError::Layout {
                    what: "an observation output per leaf",
                });
            };
            let per = leaf.len() / rows.max(1);
            let Some(bytes) =
                leaf.get(row.saturating_mul(per)..row.saturating_add(1).saturating_mul(per))
            else {
                continue;
            };
            values = values.saturating_add(check_leaf(bytes, space).map_err(
                |(element, value, what)| CheckError::OutsideSpace {
                    call,
                    instance,
                    agent,
                    leaf: name.clone(),
                    element,
                    value,
                    what,
                },
            )?);
        }
        let running = !flag(o.pending_reset(), instance)
            && !flag(o.episode_terminated(), instance)
            && !flag(o.episode_truncated(), instance);
        if running && flag(o.acting(), row) && !has_legal_choices(layout, o, row) {
            return Err(CheckError::Invariant {
                call,
                instance,
                what: "an acting agent has a discrete head with no legal choice",
            });
        }
    }
    Ok(values)
}

/// Whether every parentless head of `row`'s mask has a legal choice.
fn has_legal_choices(layout: &HostLayout, o: &Outputs, row: usize) -> bool {
    if layout.mask_len == 0 {
        return true;
    }
    let Some(mask) = row
        .checked_mul(layout.mask_len)
        .and_then(|s| o.masks().get(s..s.checked_add(layout.mask_len)?))
    else {
        return true;
    };
    layout
        .mask_heads
        .iter()
        .filter(|h| h.parent_sizes == [None, None])
        .all(|h| {
            h.offset
                .checked_add(h.size)
                .and_then(|end| mask.get(h.offset..end))
                .is_none_or(|block| block.iter().any(|b| b.get()))
        })
}

/// A value read from an observation, with its element and the broken rule.
type Outside = (usize, String, &'static str);

/// Checks one agent's leaf bytes; returns the values checked.
fn check_leaf(bytes: &[u8], space: &Space) -> Result<u64, Outside> {
    let dtype = leaf_dtype(space);
    let size = dtype_size(dtype);
    let mut count = 0u64;
    for (k, chunk) in bytes.chunks_exact(size).enumerate() {
        let v = decode(dtype, chunk);
        let outside = |what| Err((k, v.to_string(), what));
        match space {
            Space::Discrete { n } => {
                if !v.in_int(0, i128::from(*n) - 1) {
                    return outside("a Discrete choice in 0..n");
                }
            }
            Space::MultiDiscrete { nvec } => {
                let n = nvec.get(k).copied().unwrap_or(0);
                if !v.in_int(0, i128::from(n) - 1) {
                    return outside("a MultiDiscrete choice in 0..nvec[k]");
                }
            }
            Space::MultiBinary { .. } => {
                if !v.in_int(0, 1) {
                    return outside("a MultiBinary value 0 or 1");
                }
            }
            Space::Box { bounds, .. } => {
                if dtype == DType::Bool && !v.in_int(0, 1) {
                    return outside("a bool value 0 or 1");
                }
                if !in_bounds(&v, bounds, k) {
                    return outside("a Box value within its bounds");
                }
            }
            Space::Dict(_) | Space::Tuple(_) | Space::Set { .. } => {}
        }
        count = count.saturating_add(1);
    }
    Ok(count)
}

/// One decoded observation value.
#[derive(Debug, Clone, Copy)]
enum Value {
    Int(i128),
    Float(f64),
}

impl Value {
    fn in_int(self, low: i128, high: i128) -> bool {
        match self {
            Self::Int(v) => (low..=high).contains(&v),
            Self::Float(_) => false,
        }
    }
}

impl core::fmt::Display for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
        }
    }
}

/// `bounds` at element `k`: one value broadcasts.
fn bound<T: Copy>(v: &[T], k: usize) -> Option<T> {
    if v.len() == 1 {
        v.first().copied()
    } else {
        v.get(k).copied()
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "integer bounds against a float dtype: comparison only"
)]
#[expect(
    clippy::many_single_char_names,
    reason = "value, bounds, index, low and high read as the maths they are"
)]
fn in_bounds(v: &Value, bounds: &BoxBounds, k: usize) -> bool {
    let (lo, hi) = match bounds {
        BoxBounds::Int { low, high } => match (bound(low, k), bound(high, k)) {
            (Some(l), Some(h)) => (Value::Int(i128::from(l)), Value::Int(i128::from(h))),
            _ => return false,
        },
        BoxBounds::UInt { low, high } => match (bound(low, k), bound(high, k)) {
            (Some(l), Some(h)) => (Value::Int(i128::from(l)), Value::Int(i128::from(h))),
            _ => return false,
        },
        BoxBounds::Float { low, high } => match (bound(low, k), bound(high, k)) {
            (Some(l), Some(h)) => (Value::Float(l), Value::Float(h)),
            _ => return false,
        },
    };
    let as_f = |x: Value| match x {
        Value::Int(i) => i as f64,
        Value::Float(f) => f,
    };
    match (*v, lo, hi) {
        (Value::Int(x), Value::Int(l), Value::Int(h)) => (l..=h).contains(&x),
        (x, l, h) => {
            let x = as_f(x);
            // NaN fails both comparisons.
            as_f(l) <= x && x <= as_f(h)
        }
    }
}

fn leaf_dtype(space: &Space) -> DType {
    match space {
        Space::Box { dtype, .. } => *dtype,
        Space::MultiBinary { .. } => DType::Bool,
        _ => DType::I32,
    }
}

const fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::Bool | DType::U8 | DType::I8 => 1,
        DType::U16 | DType::I16 => 2,
        DType::U32 | DType::I32 | DType::F32 => 4,
        DType::I64 | DType::F64 => 8,
    }
}

/// Decodes one element in the platform's byte order, as the outputs hold it.
fn decode(dtype: DType, b: &[u8]) -> Value {
    fn arr<const N: usize>(b: &[u8]) -> [u8; N] {
        <[u8; N]>::try_from(b).unwrap_or([0; N])
    }
    match dtype {
        DType::Bool | DType::U8 => Value::Int(i128::from(u8::from_ne_bytes(arr(b)))),
        DType::I8 => Value::Int(i128::from(i8::from_ne_bytes(arr(b)))),
        DType::U16 => Value::Int(i128::from(u16::from_ne_bytes(arr(b)))),
        DType::I16 => Value::Int(i128::from(i16::from_ne_bytes(arr(b)))),
        DType::U32 => Value::Int(i128::from(u32::from_ne_bytes(arr(b)))),
        DType::I32 => Value::Int(i128::from(i32::from_ne_bytes(arr(b)))),
        DType::I64 => Value::Int(i128::from(i64::from_ne_bytes(arr(b)))),
        DType::F32 => Value::Float(f64::from(f32::from_ne_bytes(arr(b)))),
        DType::F64 => Value::Float(f64::from_ne_bytes(arr(b))),
    }
}

/// The leaf spaces of `space`, in the order of the observation outputs
/// (`HostLayout::leaves`), with their paths.
fn leaf_spaces(space: &Space) -> Vec<(String, &Space)> {
    fn add<'a>(space: &'a Space, name: String, out: &mut Vec<(String, &'a Space)>) {
        let join = |part: &str| {
            if name.is_empty() {
                part.to_owned()
            } else {
                format!("{name}.{part}")
            }
        };
        match space {
            Space::Dict(entries) => {
                for (key, inner) in entries {
                    add(inner, join(key), out);
                }
            }
            Space::Tuple(items) => {
                for (i, inner) in items.iter().enumerate() {
                    add(inner, join(&i.to_string()), out);
                }
            }
            Space::Set { .. } => {}
            leaf => out.push((name, leaf)),
        }
    }
    let mut out = Vec::new();
    add(space, String::new(), &mut out);
    out
}
