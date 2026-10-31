// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed bounds of box spaces (ADR-0014).

use super::DType;
use crate::error::SpaceError;

/// Per-element bounds of a box space, typed to match its dtype: signed
/// integers (and `Bool` as 0 to 1) use `Int`, unsigned integers `UInt`,
/// floats `Float`. Each vector has one value for all elements or one per
/// element.
#[derive(Debug, Clone, PartialEq)]
pub enum BoxBounds {
    /// Bounds for `I8`, `I16`, `I32`, `I64` and `Bool`.
    Int {
        /// Lower bounds (inclusive).
        low: Vec<i64>,
        /// Upper bounds (inclusive).
        high: Vec<i64>,
    },
    /// Bounds for `U8`, `U16` and `U32`.
    UInt {
        /// Lower bounds (inclusive).
        low: Vec<u64>,
        /// Upper bounds (inclusive).
        high: Vec<u64>,
    },
    /// Bounds for `F32` and `F64`; infinities allowed, NaN not.
    Float {
        /// Lower bounds (inclusive).
        low: Vec<f64>,
        /// Upper bounds (inclusive).
        high: Vec<f64>,
    },
}

fn invalid(reason: &'static str) -> SpaceError {
    SpaceError::InvalidBounds { reason }
}

fn check_lengths<T>(low: &[T], high: &[T], elements: u64) -> Result<(), SpaceError> {
    let fits = |v: &[T]| v.len() == 1 || u64::try_from(v.len()) == Ok(elements);
    if !fits(low) || !fits(high) || low.len() != high.len() {
        return Err(invalid("bounds need one value or one per element"));
    }
    Ok(())
}

impl BoxBounds {
    /// Validates the bounds for `dtype` and a shape with `elements`
    /// elements.
    ///
    /// # Errors
    ///
    /// Returns [`SpaceError::InvalidBounds`] if the variant does not match
    /// the dtype, a bound is outside the dtype's range, a float bound is NaN
    /// or (for `F32`) not representable, or a range is empty.
    pub fn validate(&self, dtype: DType, elements: u64) -> Result<(), SpaceError> {
        match (self, dtype) {
            (
                Self::Int { low, high },
                DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::Bool,
            ) => {
                check_lengths(low, high, elements)?;
                let (min, max) = match dtype {
                    DType::I8 => (i64::from(i8::MIN), i64::from(i8::MAX)),
                    DType::I16 => (i64::from(i16::MIN), i64::from(i16::MAX)),
                    DType::I32 => (i64::from(i32::MIN), i64::from(i32::MAX)),
                    DType::Bool => (0, 1),
                    _ => (i64::MIN, i64::MAX),
                };
                check_ordered(low, high, |v| (min..=max).contains(v))
            }
            (Self::UInt { low, high }, DType::U8 | DType::U16 | DType::U32) => {
                check_lengths(low, high, elements)?;
                let max = match dtype {
                    DType::U8 => u64::from(u8::MAX),
                    DType::U16 => u64::from(u16::MAX),
                    _ => u64::from(u32::MAX),
                };
                check_ordered(low, high, |v| *v <= max)
            }
            (Self::Float { low, high }, DType::F32 | DType::F64) => {
                check_lengths(low, high, elements)?;
                if low.iter().chain(high).any(|v| v.is_nan()) {
                    return Err(invalid("NaN bound"));
                }
                if dtype == DType::F32 {
                    // Every F32 bound must be exactly an f32; never rounded
                    // silently.
                    if let Some(&value) = low.iter().chain(high).find(|v| !exact_f32(**v)) {
                        return Err(SpaceError::InexactF32Bound {
                            value,
                            nearest: to_f32(value),
                        });
                    }
                }
                check_ordered(low, high, |_| true)?;
                let positive_inf = |v: &&f64| v.is_infinite() && v.is_sign_positive();
                let negative_inf = |v: &&f64| v.is_infinite() && v.is_sign_negative();
                if low.iter().any(|v| positive_inf(&v)) || high.iter().any(|v| negative_inf(&v)) {
                    return Err(invalid("empty range at infinity"));
                }
                Ok(())
            }
            _ => Err(invalid("bound type does not match the dtype")),
        }
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "rounding is the point: exactness is checked"
)]
fn to_f32(v: f64) -> f32 {
    v as f32
}

/// Whether `v` is exactly representable as `f32` (infinities included,
/// signed zeros preserved).
fn exact_f32(v: f64) -> bool {
    f64::from(to_f32(v)).to_bits() == v.to_bits()
}

/// Checks every bound with `in_range`, then `low <= high` per element
/// (broadcasting a single value).
fn check_ordered<T: PartialOrd>(
    low: &[T],
    high: &[T],
    in_range: impl Fn(&T) -> bool,
) -> Result<(), SpaceError> {
    fn at<T>(v: &[T], i: usize) -> Option<&T> {
        if v.len() == 1 { v.first() } else { v.get(i) }
    }
    if !low.iter().chain(high).all(in_range) {
        return Err(invalid("bound outside the dtype's range"));
    }
    let n = low.len().max(high.len());
    for i in 0..n {
        match (at(low, i), at(high, i)) {
            (Some(l), Some(h)) if l <= h => {}
            _ => return Err(invalid("low is greater than high")),
        }
    }
    Ok(())
}
