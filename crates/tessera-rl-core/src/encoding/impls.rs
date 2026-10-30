// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Canonical` implementations for the field types allowed in state
//! (ADR-0029).

use tessera_rl_num::identity::{BranchSalt, LogicalId, RngIdentity, Seed};
use tessera_rl_num::{Fx32, Fx64};

use super::{Canonical, Decoder, Encoder, Sink};
use crate::bool8::Bool8;
use crate::error::DecodeError;
use crate::reserved::Reserved;
use crate::state_enum::StateEnum;
use crate::state_init::StateInit;

macro_rules! canonical_int {
    ($($ty:ty),*) => {$(
        impl Canonical for $ty {
            const ENCODED_LEN: usize = size_of::<$ty>();

            #[inline]
            fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
                out.write_bytes(&self.to_le_bytes());
            }

            #[inline]
            fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
                Ok(<$ty>::from_le_bytes(input.read_array()?))
            }
        }
    )*};
}

canonical_int!(u8, u16, u32, u64, i8, i16, i32, i64);

impl Canonical for f32 {
    const ENCODED_LEN: usize = 4;

    #[inline]
    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        out.write_bytes(&self.to_bits().to_le_bytes());
    }

    /// Rejects NaN; see [`decode_f32_allow_nan`].
    #[inline]
    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        let value = decode_f32_allow_nan(input)?;
        if value.is_nan() {
            return Err(DecodeError::NanInState {
                bits: u64::from(value.to_bits()),
            });
        }
        Ok(value)
    }
}

impl Canonical for f64 {
    const ENCODED_LEN: usize = 8;

    #[inline]
    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        out.write_bytes(&self.to_bits().to_le_bytes());
    }

    /// Rejects NaN; see [`decode_f64_allow_nan`].
    #[inline]
    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        let value = decode_f64_allow_nan(input)?;
        if value.is_nan() {
            return Err(DecodeError::NanInState {
                bits: value.to_bits(),
            });
        }
        Ok(value)
    }
}

/// Decodes an `f32` field annotated `#[state(allow_nan)]`, keeping any NaN
/// payload.
///
/// # Errors
///
/// Returns [`DecodeError::UnexpectedEof`] on a short input.
#[inline]
pub fn decode_f32_allow_nan(input: &mut Decoder<'_>) -> Result<f32, DecodeError> {
    Ok(f32::from_bits(u32::from_le_bytes(input.read_array()?)))
}

/// Decodes an `f64` field annotated `#[state(allow_nan)]`, keeping any NaN
/// payload.
///
/// # Errors
///
/// Returns [`DecodeError::UnexpectedEof`] on a short input.
#[inline]
pub fn decode_f64_allow_nan(input: &mut Decoder<'_>) -> Result<f64, DecodeError> {
    Ok(f64::from_bits(u64::from_le_bytes(input.read_array()?)))
}

impl Canonical for Bool8 {
    const ENCODED_LEN: usize = 1;

    #[inline]
    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        out.write_bytes(&[self.to_byte()]);
    }

    #[inline]
    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        let [byte] = input.read_array()?;
        Self::from_byte(byte).ok_or(DecodeError::InvalidBool8 { byte })
    }
}

impl<const N: usize> Canonical for Reserved<N> {
    const ENCODED_LEN: usize = N;

    #[inline]
    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        out.write_bytes(&[0; N]);
    }

    #[inline]
    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        let bytes: [u8; N] = input.read_array()?;
        match bytes.iter().find(|&&b| b != 0) {
            Some(&byte) => Err(DecodeError::NonZeroReserved { byte }),
            None => Ok(Self::ZERO),
        }
    }
}

impl<T: Canonical + StateInit, const N: usize> Canonical for [T; N] {
    const ENCODED_LEN: usize = T::ENCODED_LEN * N;

    #[inline]
    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        for item in self {
            item.encode(out);
        }
    }

    #[inline]
    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        let mut items = <[T; N]>::state_default();
        for item in &mut items {
            *item = T::decode(input)?;
        }
        Ok(items)
    }
}

impl Canonical for Fx32 {
    const ENCODED_LEN: usize = 4;

    #[inline]
    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        self.to_raw().encode(out);
    }

    #[inline]
    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        i32::decode(input).map(Self::from_raw)
    }
}

impl Canonical for Fx64 {
    const ENCODED_LEN: usize = 8;

    #[inline]
    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        self.to_raw().encode(out);
    }

    #[inline]
    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        i64::decode(input).map(Self::from_raw)
    }
}

/// The 32-byte RNG identity, in its ADR-0007 field order.
impl Canonical for RngIdentity {
    const ENCODED_LEN: usize = 32;

    #[inline]
    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        self.seed().get().encode(out);
        self.branch_salt().get().encode(out);
        self.logical_id().get().encode(out);
        self.episode_index().encode(out);
        self.step_index().encode(out);
        Reserved::<4>::ZERO.encode(out);
    }

    #[inline]
    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        let seed = Seed::new(u64::decode(input)?);
        let salt = BranchSalt::new(u64::decode(input)?);
        let lineage = LogicalId::new(u32::decode(input)?);
        let episode = u32::decode(input)?;
        let step = u32::decode(input)?;
        Reserved::<4>::decode(input)?;
        Ok(Self::from_parts(seed, salt, lineage, episode, step))
    }
}

/// Encodes a `StateEnum` through its representation. Used by the
/// generated `Canonical` impls of state structs.
#[inline]
pub fn encode_state_enum<E, S>(value: E, out: &mut Encoder<S>)
where
    E: StateEnum,
    E::Repr: Canonical,
    S: Sink,
{
    value.to_repr().encode(out);
}

/// Decodes and validates a `StateEnum`.
///
/// # Errors
///
/// Returns [`DecodeError::InvalidEnum`] if the representation matches no
/// variant, or [`DecodeError::UnexpectedEof`] on a short input.
#[inline]
pub fn decode_state_enum<E>(input: &mut Decoder<'_>) -> Result<E, DecodeError>
where
    E: StateEnum,
    E::Repr: Canonical,
{
    let repr = E::Repr::decode(input)?;
    E::from_repr(repr).ok_or(DecodeError::InvalidEnum {
        name: E::NAME,
        value: repr.into(),
    })
}
