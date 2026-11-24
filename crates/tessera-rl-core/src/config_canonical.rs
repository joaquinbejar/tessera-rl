// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decoding the canonical form of a config (ADR-0030), the inverse of
//! [`crate::config::canonical_bytes`]: a replay rebuilds its pool from the
//! canonical bytes in its header (ADR-0072, #377).
//!
//! `#[derive(EnvConfig)]` implements [`CanonicalDecode`]: it reads the
//! static entries back in their sorted order and checks every key, type tag
//! and nested entry count against the type; adaptive fields are not in the
//! canonical form, so they keep their defaults (a replay's keyframe carries
//! their effective values in the record headers). [`decode_canonical`]
//! checks the format version, that every byte was read, the config's own
//! `validate`, and that encoding the result gives back exactly the input,
//! so a decoded config is one the encoder could have produced.

use tessera_rl_num::{Fx32, Fx64};
use thiserror::Error;

use crate::config::{
    CONFIG_FORMAT_VERSION, CanonicalConfig, ConfigValue, NESTED_TAG, canonical_bytes,
};
use crate::encoding::{Canonical, Decoder};
use crate::env::EnvConfig;
use crate::error::{ConfigError, DecodeError};

/// Why canonical config bytes do not decode into a config.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum CanonicalDecodeError {
    /// The first byte is not [`CONFIG_FORMAT_VERSION`].
    #[error("canonical config format version {found}, this library reads {CONFIG_FORMAT_VERSION}")]
    Version {
        /// The byte read.
        found: u8,
    },
    /// An entry's key is not the one the type expects at that position.
    #[error("expected the key {expected:?} next")]
    Key {
        /// The key the type expects.
        expected: &'static str,
    },
    /// An entry's type tag is not the field's.
    #[error("key {key:?}: type tag {found}, expected {expected}")]
    Tag {
        /// The key.
        key: &'static str,
        /// The tag read.
        found: u8,
        /// The field's tag.
        expected: u8,
    },
    /// A nested config's entry count is not the type's.
    #[error("key {key:?}: {found} nested entries, expected {expected}")]
    NestedCount {
        /// The key.
        key: &'static str,
        /// The count read.
        found: u16,
        /// The type's count.
        expected: u16,
    },
    /// A boolean byte is neither 0 nor 1.
    #[error("key {key:?}: boolean byte {byte}")]
    Bool {
        /// The key.
        key: &'static str,
        /// The byte read.
        byte: u8,
    },
    /// A value or the framing does not decode.
    #[error(transparent)]
    Decode(#[from] DecodeError),
    /// The decoded config fails its own `validate`.
    #[error(transparent)]
    Invalid(#[from] ConfigError),
    /// Encoding the decoded config does not give back the input.
    #[error("the decoded config does not encode back to the same bytes")]
    NotCanonical,
    /// An asset field's hash is not that of the asset decoded from empty
    /// input: the canonical form holds only the hash (ADR-0077), so any
    /// other asset cannot be rebuilt from it. Build the pool from the
    /// config, with its assets, instead.
    #[error("key {key:?}: the asset with this hash is not in the canonical form")]
    Asset {
        /// The key.
        key: &'static str,
    },
}

/// A config that reads its own canonical entries back
/// (`#[derive(EnvConfig)]` implements it).
pub trait CanonicalDecode: CanonicalConfig + Sized {
    /// Reads the static entries, in canonical order, into a config whose
    /// other fields keep their defaults.
    ///
    /// # Errors
    ///
    /// The first entry that does not match the type.
    fn read_entries(input: &mut Decoder<'_>) -> Result<Self, CanonicalDecodeError>;
}

/// Decodes `bytes`, the canonical form of a `C` (the format version, then
/// the entries), into a validated config that encodes back to exactly
/// `bytes`.
///
/// # Errors
///
/// [`CanonicalDecodeError`] for another version, a mismatched entry,
/// trailing bytes, a config its `validate` refuses, or bytes that are not
/// the canonical form of what they decode to.
pub fn decode_canonical<C>(bytes: &[u8]) -> Result<C, CanonicalDecodeError>
where
    C: CanonicalDecode + EnvConfig,
{
    let mut input = Decoder::new(bytes);
    let [version] = input.read_array::<1>()?;
    if version != CONFIG_FORMAT_VERSION {
        return Err(CanonicalDecodeError::Version { found: version });
    }
    let cfg = C::read_entries(&mut input)?;
    input.finish()?;
    cfg.validate()?;
    if canonical_bytes(&cfg) != bytes {
        return Err(CanonicalDecodeError::NotCanonical);
    }
    Ok(cfg)
}

/// Reads one entry's key and checks it is `expected`. Used by generated
/// code.
///
/// # Errors
///
/// [`CanonicalDecodeError::Key`] for another key.
#[doc(hidden)]
pub fn read_key(
    input: &mut Decoder<'_>,
    expected: &'static str,
) -> Result<(), CanonicalDecodeError> {
    let len = usize::from(u16::from_le_bytes(input.read_array::<2>()?));
    if len != expected.len() {
        return Err(CanonicalDecodeError::Key { expected });
    }
    for &want in expected.as_bytes() {
        let [got] = input.read_array::<1>()?;
        if got != want {
            return Err(CanonicalDecodeError::Key { expected });
        }
    }
    Ok(())
}

/// Reads a type tag and checks it is `expected`.
fn read_tag(
    input: &mut Decoder<'_>,
    key: &'static str,
    expected: u8,
) -> Result<(), CanonicalDecodeError> {
    let [found] = input.read_array::<1>()?;
    if found == expected {
        Ok(())
    } else {
        Err(CanonicalDecodeError::Tag {
            key,
            found,
            expected,
        })
    }
}

/// A primitive config value that reads itself back. Used by generated
/// code.
#[doc(hidden)]
pub trait ConfigValueDecode: ConfigValue + Sized {
    /// Reads the value after its tag.
    ///
    /// # Errors
    ///
    /// The value's decode error.
    fn read_value(input: &mut Decoder<'_>, key: &'static str)
    -> Result<Self, CanonicalDecodeError>;
}

macro_rules! value_decode {
    ($($ty:ty),* $(,)?) => {$(
        impl ConfigValueDecode for $ty {
            #[inline]
            fn read_value(
                input: &mut Decoder<'_>,
                _key: &'static str,
            ) -> Result<Self, CanonicalDecodeError> {
                Ok(<$ty as Canonical>::decode(input)?)
            }
        }
    )*};
}

value_decode!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Fx32, Fx64);

impl ConfigValueDecode for bool {
    fn read_value(
        input: &mut Decoder<'_>,
        key: &'static str,
    ) -> Result<Self, CanonicalDecodeError> {
        match input.read_array::<1>()? {
            [0] => Ok(false),
            [1] => Ok(true),
            [byte] => Err(CanonicalDecodeError::Bool { key, byte }),
        }
    }
}

/// An asset field reads its 16 hash bytes back (ADR-0077). Only the asset
/// decoded from empty input (the type's empty or default asset, as the
/// Grid's open layout) can be rebuilt; any other hash is
/// [`CanonicalDecodeError::Asset`].
impl<T: crate::asset::AssetData> ConfigValueDecode for crate::asset::Asset<T> {
    fn read_value(
        input: &mut Decoder<'_>,
        key: &'static str,
    ) -> Result<Self, CanonicalDecodeError> {
        let hash = crate::asset::AssetHash(input.read_array::<16>()?);
        match Self::from_bytes(&[]) {
            Ok(empty) if empty.hash() == hash => Ok(empty),
            _ => Err(CanonicalDecodeError::Asset { key }),
        }
    }
}

/// Reads a primitive entry (key, tag, value). Used by generated code.
///
/// # Errors
///
/// A mismatched key or tag, or a value that does not decode.
#[doc(hidden)]
pub fn read_entry<V: ConfigValueDecode>(
    input: &mut Decoder<'_>,
    key: &'static str,
) -> Result<V, CanonicalDecodeError> {
    read_key(input, key)?;
    read_tag(input, key, V::TAG)?;
    V::read_value(input, key)
}

/// Reads a nested-config entry (key, tag, entry count, entries). Used by
/// generated code.
///
/// # Errors
///
/// A mismatched key, tag or count, or a nested entry that does not match.
#[doc(hidden)]
pub fn read_nested_entry<C: CanonicalDecode>(
    input: &mut Decoder<'_>,
    key: &'static str,
) -> Result<C, CanonicalDecodeError> {
    read_key(input, key)?;
    read_tag(input, key, NESTED_TAG)?;
    let found = u16::from_le_bytes(input.read_array::<2>()?);
    if found != C::STATIC_ENTRIES {
        return Err(CanonicalDecodeError::NestedCount {
            key,
            found,
            expected: C::STATIC_ENTRIES,
        });
    }
    C::read_entries(input)
}
