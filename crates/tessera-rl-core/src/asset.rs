// SPDX-License-Identifier: MIT OR Apache-2.0

//! Content-addressed environment assets (ADR-0030, ADR-0077).
//!
//! An asset (a map, a unit table) is an immutable value shared by every
//! instance of a pool. It lives in the config as an [`Asset<T>`] field:
//! kernels read it with [`Asset::get`], and the config's canonical form holds
//! only its 128-bit content hash.
//!
//! The hash is XXH3-128 of `canonical_bytes(decode(input))`: the input need
//! not be canonical, and two serialisations of one asset hash equal
//! (ADR-0077 point 2). [`AssetData::decode`] and
//! [`AssetData::canonical_bytes`] are pure and platform-independent.

use std::borrow::Cow;
use std::sync::Arc;

use thiserror::Error;

/// The largest raw input [`Asset::from_bytes`] accepts: 16 MiB
/// (ADR-0077 point 4).
pub const MAX_ASSET_INPUT: usize = 16 * 1024 * 1024;

/// Tag of an asset value in the canonical config form (ADR-0030 tags 1 to
/// 14 are taken).
pub const ASSET_TAG: u8 = 15;

/// Why an asset could not be built.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum AssetError {
    /// The raw input is longer than [`MAX_ASSET_INPUT`].
    #[error("asset input of {len} bytes exceeds {MAX_ASSET_INPUT}")]
    InputTooLarge {
        /// The input length.
        len: usize,
    },
    /// The decoded asset would exceed its type's `MAX_DECODED_BYTES`.
    #[error("decoded asset exceeds its limit of {limit} bytes")]
    DecodedTooLarge {
        /// The type's limit.
        limit: usize,
    },
    /// The input is not a valid asset of this type.
    #[error("invalid asset: {reason}")]
    Invalid {
        /// Why.
        reason: &'static str,
    },
    /// No bundled asset has this name.
    #[error("no bundled asset named `{name}`")]
    UnknownBundled {
        /// The name.
        name: String,
    },
}

/// An asset type: decodable from bytes, with a canonical byte form.
///
/// Both functions MUST be pure and platform-independent (no locale, no I/O,
/// no global state, fixed-width little-endian integers), and
/// `canonical_bytes(decode(canonical_bytes(x))) == canonical_bytes(x)`.
/// [`check_asset_fixpoint`] tests it.
pub trait AssetData: Send + Sync + Sized + 'static {
    /// Bound on the decoded size, checked by `decode` before it allocates.
    const MAX_DECODED_BYTES: usize;

    /// Decodes an asset; the input need not be canonical.
    ///
    /// # Errors
    ///
    /// [`AssetError::Invalid`] or [`AssetError::DecodedTooLarge`].
    fn decode(bytes: &[u8]) -> Result<Self, AssetError>;

    /// The canonical bytes, which are hashed.
    fn canonical_bytes(&self) -> Cow<'_, [u8]>;

    /// The bytes of a bundled asset named `name`, if the type ships one
    /// (typically `include_bytes!`).
    fn bundled(name: &str) -> Option<&'static [u8]>;
}

/// The 128-bit content hash of an asset, little-endian bytes of
/// XXH3-128.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AssetHash(pub [u8; 16]);

impl AssetHash {
    /// XXH3-128 of `bytes`.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        Self(xxhash_rust::xxh3::xxh3_128(bytes).to_le_bytes())
    }
}

/// An immutable, content-addressed asset shared by every instance.
#[derive(Debug)]
pub struct Asset<T: AssetData> {
    value: Arc<T>,
    hash: AssetHash,
}

impl<T: AssetData> Clone for Asset<T> {
    fn clone(&self) -> Self {
        Self {
            value: Arc::clone(&self.value),
            hash: self.hash,
        }
    }
}

/// Equal assets have equal content hashes.
impl<T: AssetData> PartialEq for Asset<T> {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
    }
}

impl<T: AssetData> Eq for Asset<T> {}

/// The default asset is `T::default()`, hashed.
impl<T: AssetData + Default> Default for Asset<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: AssetData> Asset<T> {
    /// Wraps `value`, hashing its canonical bytes.
    #[must_use]
    pub fn new(value: T) -> Self {
        let hash = AssetHash::of(&value.canonical_bytes());
        Self {
            value: Arc::new(value),
            hash,
        }
    }

    /// Decodes `bytes` (at most [`MAX_ASSET_INPUT`]) and hashes the
    /// canonical form of the result.
    ///
    /// # Errors
    ///
    /// [`AssetError::InputTooLarge`], or the type's decode error.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AssetError> {
        if bytes.len() > MAX_ASSET_INPUT {
            return Err(AssetError::InputTooLarge { len: bytes.len() });
        }
        T::decode(bytes).map(Self::new)
    }

    /// The bundled asset `name`.
    ///
    /// # Errors
    ///
    /// [`AssetError::UnknownBundled`], or the type's decode error.
    pub fn bundled(name: &str) -> Result<Self, AssetError> {
        let bytes = T::bundled(name).ok_or_else(|| AssetError::UnknownBundled {
            name: name.to_owned(),
        })?;
        Self::from_bytes(bytes)
    }

    /// The asset value.
    #[inline]
    #[must_use]
    pub fn get(&self) -> &T {
        &self.value
    }

    /// The content hash.
    #[inline]
    #[must_use]
    pub const fn hash(&self) -> AssetHash {
        self.hash
    }
}

/// A config default naming a bundled asset: the generated `Default` calls
/// it. A missing or invalid bundled asset is a bug of the environment
/// crate (its tests build the default config); it falls back to
/// `T::default()` and logs an error rather than panicking.
#[must_use]
pub fn bundled_or_default<T: AssetData + Default>(name: &str) -> Asset<T> {
    Asset::bundled(name).unwrap_or_else(|e| {
        tracing::error!(name, error = %e, "bundled asset default unavailable");
        Asset::new(T::default())
    })
}

/// One asset of a config, as persisted headers list it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetEntry {
    /// The asset name (`#[config(asset = "...")]`, or the field path).
    pub name: String,
    /// Its content hash.
    pub hash: AssetHash,
}

/// Checks the [`AssetData`] contract on `sample`: the canonical bytes are
/// a fixpoint of decode, and `sample` hashes as its canonical form. For
/// asset types' tests.
///
/// # Errors
///
/// A description of the first violation.
pub fn check_asset_fixpoint<T: AssetData>(sample: &[u8]) -> Result<(), String> {
    let first = T::decode(sample).map_err(|e| format!("sample does not decode: {e}"))?;
    let canonical = first.canonical_bytes().into_owned();
    let second =
        T::decode(&canonical).map_err(|e| format!("canonical bytes do not decode: {e}"))?;
    if second.canonical_bytes() != canonical.as_slice() {
        return Err("canonical bytes are not a fixpoint of decode".to_owned());
    }
    let a = Asset::<T>::from_bytes(sample).map_err(|e| e.to_string())?;
    let b = Asset::<T>::from_bytes(&canonical).map_err(|e| e.to_string())?;
    if a.hash() != b.hash() {
        return Err("sample and its canonical form hash differently".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests;
