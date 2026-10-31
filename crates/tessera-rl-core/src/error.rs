// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed errors of the core crate (ADR-0022).

use thiserror::Error;

/// A game invariant that a decoded state violates (the `Validate` trait).
///
/// Carries a static code instead of a `String` so validation never
/// allocates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("state invariant violated: {code}")]
pub struct InvariantViolation {
    /// A short, stable description of the violated invariant.
    pub code: &'static str,
}

/// Errors decoding a canonical encoding (ADR-0029).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DecodeError {
    /// The input ended early.
    #[error("unexpected end of input: needed {needed} bytes, {remaining} remaining")]
    UnexpectedEof {
        /// Bytes needed by the next field.
        needed: usize,
        /// Bytes left in the input.
        remaining: usize,
    },
    /// The input has bytes after the last field.
    #[error("{count} trailing bytes after the encoding")]
    TrailingBytes {
        /// Number of unread bytes.
        count: usize,
    },
    /// A `Bool8` byte was neither 0 nor 1.
    #[error("invalid bool8 byte {byte}")]
    InvalidBool8 {
        /// The byte read.
        byte: u8,
    },
    /// A `Reserved` byte was not zero.
    #[error("non-zero reserved byte {byte}")]
    NonZeroReserved {
        /// The byte read.
        byte: u8,
    },
    /// A `StateEnum` representation matched no variant.
    #[error("invalid discriminant {value} for enum {name}")]
    InvalidEnum {
        /// The enum's name.
        name: &'static str,
        /// The representation read.
        value: i64,
    },
    /// A float field held NaN without `#[state(allow_nan)]`.
    #[error("nan in a float field (bits {bits:#018x})")]
    NanInState {
        /// The raw bits, widened to 64 bits.
        bits: u64,
    },
    /// An inactive `Pool` entry did not hold `T::default()`.
    #[error("inactive pool entry {index} is not the default value")]
    InactivePoolEntry {
        /// The entry index.
        index: usize,
    },
    /// The decoded state failed `Validate`.
    #[error(transparent)]
    Invariant(#[from] InvariantViolation),
}

/// Errors encoding into a caller-provided buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum EncodeError {
    /// The buffer length differs from the encoded length.
    #[error("buffer holds {actual} bytes, encoding needs {expected}")]
    BufferLength {
        /// The encoded length.
        expected: usize,
        /// The buffer length.
        actual: usize,
    },
}

/// Errors in a space definition (ADR-0014).
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SpaceError {
    /// A discrete space or dimension with zero or more than 2^31 choices.
    #[error("invalid number of discrete choices {n}: must be in 1..=2^31")]
    InvalidChoices {
        /// The number of choices.
        n: u32,
    },
    /// A box or binary space with an empty shape or a zero dimension.
    #[error("invalid shape {shape:?}")]
    InvalidShape {
        /// The offending shape.
        shape: Vec<u32>,
    },
    /// Box bounds that do not fit the dtype or the shape.
    #[error("invalid box bounds: {reason}")]
    InvalidBounds {
        /// What is wrong.
        reason: &'static str,
    },
    /// An `F32` box bound that is not exactly an `f32`.
    #[error("f32 box bound {value} is not exactly representable; the nearest f32 is {nearest}")]
    InexactF32Bound {
        /// The bound given.
        value: f64,
        /// The nearest `f32`.
        nearest: f32,
    },
    /// A composite space nested deeper than the supported depth.
    #[error("space nesting depth {depth} exceeds the maximum of {max}")]
    TooDeep {
        /// The depth reached.
        depth: usize,
        /// The maximum depth.
        max: usize,
    },
    /// A dict or tuple with no entries, or a set with zero capacity.
    #[error("empty composite space")]
    Empty,
    /// A dict with a repeated key.
    #[error("duplicate dict key {key:?}")]
    DuplicateKey {
        /// The repeated key.
        key: String,
    },
}

/// Errors validating a configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("invalid config field {field}: {reason}")]
pub struct ConfigError {
    /// The offending field.
    pub field: &'static str,
    /// Why it is invalid.
    pub reason: &'static str,
}

/// Any error from the core crate.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CoreError {
    /// See [`DecodeError`].
    #[error(transparent)]
    Decode(#[from] DecodeError),
    /// See [`EncodeError`].
    #[error(transparent)]
    Encode(#[from] EncodeError),
    /// See [`SpaceError`].
    #[error(transparent)]
    Space(#[from] SpaceError),
    /// See [`ConfigError`].
    #[error(transparent)]
    Config(#[from] ConfigError),
}
