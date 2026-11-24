// SPDX-License-Identifier: MIT OR Apache-2.0

//! The conversions.

use burn::prelude::{Bool, Device, Int, Tensor, TensorData};
use tessera_rl_core::Bool8;
use tessera_rl_engine::Outputs;

/// Why a conversion was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AdapterError {
    /// The pool has no observation leaf with this index.
    #[error("no observation leaf {leaf}")]
    NoLeaf {
        /// The leaf index.
        leaf: usize,
    },
    /// The leaf's bytes are not whole rows of `row_len` `f32`s.
    #[error("{bytes} bytes are not whole rows of {row_len} f32")]
    Shape {
        /// The leaf's byte length.
        bytes: usize,
        /// The requested row length.
        row_len: usize,
    },
    /// An empty batch (no rows).
    #[error("empty batch")]
    Empty,
}

/// Observation leaf `leaf` as `f32`, shape `[rows, row_len]`. The leaf's
/// dtype must be `f32` (its bytes are read as little-endian `f32`).
///
/// # Errors
///
/// [`AdapterError::NoLeaf`], [`AdapterError::Shape`] or
/// [`AdapterError::Empty`].
pub fn f32_leaf(
    out: &Outputs,
    leaf: usize,
    row_len: usize,
    device: &Device,
) -> Result<Tensor<2>, AdapterError> {
    let bytes = out.observation(leaf).ok_or(AdapterError::NoLeaf { leaf })?;
    let row_bytes = row_len.checked_mul(4).filter(|&b| b > 0);
    let rows = match row_bytes {
        Some(rb) if bytes.len() % rb == 0 => bytes.len() / rb,
        _ => {
            return Err(AdapterError::Shape {
                bytes: bytes.len(),
                row_len,
            });
        }
    };
    if rows == 0 {
        return Err(AdapterError::Empty);
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| <[u8; 4]>::try_from(c).map_or(0.0, f32::from_le_bytes))
        .collect();
    Ok(Tensor::from_data(TensorData::new(values, [rows, row_len]), device))
}

/// Rewards, shape `[rows]`.
///
/// # Errors
///
/// [`AdapterError::Empty`].
pub fn rewards(out: &Outputs, device: &Device) -> Result<Tensor<1>, AdapterError> {
    let r = out.rewards();
    if r.is_empty() {
        return Err(AdapterError::Empty);
    }
    Ok(Tensor::from_data(TensorData::new(r.to_vec(), [r.len()]), device))
}

/// `Bool8` flags (for example `episode_terminated()`) as `0.0`/`1.0`,
/// shape `[len]`.
///
/// # Errors
///
/// [`AdapterError::Empty`].
pub fn flags(flags: &[Bool8], device: &Device) -> Result<Tensor<1>, AdapterError> {
    if flags.is_empty() {
        return Err(AdapterError::Empty);
    }
    let v: Vec<f32> = flags.iter().map(|f| if f.get() { 1.0 } else { 0.0 }).collect();
    Ok(Tensor::from_data(TensorData::new(v, [flags.len()]), device))
}

/// Action masks, shape `[rows, choices]`.
///
/// # Errors
///
/// [`AdapterError::Shape`] or [`AdapterError::Empty`].
pub fn masks(
    out: &Outputs,
    choices: usize,
    device: &Device,
) -> Result<Tensor<2, Bool>, AdapterError> {
    let m = out.masks();
    if choices == 0 || m.len() % choices != 0 {
        return Err(AdapterError::Shape {
            bytes: m.len(),
            row_len: choices,
        });
    }
    if m.is_empty() {
        return Err(AdapterError::Empty);
    }
    let v: Vec<bool> = m.iter().map(|f| f.get()).collect();
    Ok(Tensor::from_data(TensorData::new(v, [m.len() / choices, choices]), device))
}

/// Discrete actions from an integer tensor, for `ActionInput::discrete`.
#[must_use]
pub fn actions(actions: Tensor<1, Int>) -> Vec<i32> {
    actions
        .into_data()
        .convert::<i64>()
        .iter::<i64>()
        .map(|a| i32::try_from(a).unwrap_or(i32::MAX))
        .collect()
}
