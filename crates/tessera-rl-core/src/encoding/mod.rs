// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lossless canonical encoding and `StateHash` (SPEC 8.3, ADR-0029).
//!
//! A record encodes as its fields in declaration order, little-endian, at
//! fixed widths. Float bits are preserved exactly, including signed zeros
//! and NaN payloads. Decoding validates every field (`Bool8`, `Reserved`,
//! `StateEnum`, NaN, `Pool` occupancy) and never panics.
//!
//! Encoding writes to a [`Sink`]: a fixed slice, a streaming XXH3 hasher
//! (so hashing never allocates) or, for persistence, a `Vec`.
//!
//! This module has no `unsafe` code.

mod decoder;
mod encoder;
mod eq;
mod hash;
mod impls;

pub use decoder::Decoder;
pub use encoder::{Encoder, HashSink, Sink, SliceSink};
pub use eq::canonical_eq;
pub use hash::{StateHash, StateHasher, checksum};
pub use impls::{
    DecodeAllowNan, decode_f32_allow_nan, decode_f64_allow_nan, decode_state_enum,
    encode_state_enum,
};

use crate::error::{DecodeError, EncodeError};

/// A type with a canonical encoding.
///
/// `#[derive(EnvState)]` implements it for state structs by encoding the
/// fields in declaration order.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be used as a state field",
    label = "not a plain-data state type",
    note = "use #[derive(StateEnum)] for enums and #[derive(EnvState)] for structs; other fields must be fixed-width integers, floats, Fx32, Fx64, Bool8, Reserved<N>, arrays or Pool (ADR-0029)"
)]
pub trait Canonical: Sized {
    /// The exact number of bytes [`Canonical::encode`] writes.
    const ENCODED_LEN: usize;

    /// Writes the canonical encoding.
    fn encode<S: Sink>(&self, out: &mut Encoder<S>);

    /// Reads and validates a value.
    ///
    /// # Errors
    ///
    /// Returns a [`DecodeError`] if the input is too short or a field holds
    /// an invalid value.
    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError>;
}

/// Encodes `value` into `buf`, which must be exactly
/// [`Canonical::ENCODED_LEN`] bytes long.
///
/// # Errors
///
/// Returns [`EncodeError::BufferLength`] if `buf` has another length.
pub fn encode_to_slice<T: Canonical>(value: &T, buf: &mut [u8]) -> Result<(), EncodeError> {
    if buf.len() != T::ENCODED_LEN {
        return Err(EncodeError::BufferLength {
            expected: T::ENCODED_LEN,
            actual: buf.len(),
        });
    }
    let mut out = Encoder::new(SliceSink::new(buf));
    value.encode(&mut out);
    debug_assert!(out.sink().is_full(), "ENCODED_LEN does not match encode");
    Ok(())
}

/// Encodes `value` into a new `Vec`. For persistence only; never on the
/// hot path.
#[must_use]
pub fn encode_to_vec<T: Canonical>(value: &T) -> Vec<u8> {
    let mut out = Encoder::new(Vec::with_capacity(T::ENCODED_LEN));
    value.encode(&mut out);
    out.into_sink()
}

/// Decodes a value that must use all of `bytes`.
///
/// # Errors
///
/// Returns a [`DecodeError`] for invalid fields, a short input or trailing
/// bytes.
pub fn decode_from_slice<T: Canonical>(bytes: &[u8]) -> Result<T, DecodeError> {
    let mut input = Decoder::new(bytes);
    let value = T::decode(&mut input)?;
    input.finish()?;
    Ok(value)
}
