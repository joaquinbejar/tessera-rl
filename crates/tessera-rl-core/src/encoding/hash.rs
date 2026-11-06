// SPDX-License-Identifier: MIT OR Apache-2.0

//! `StateHash`: XXH3-64 of the canonical encoding (SPEC 8.3, ADR-0029).

use super::{Canonical, Encoder, HashSink};

/// The XXH3-64 (seed 0) hash of a canonical encoding. Independent of
/// storage layout, chunk size and padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct StateHash(u64);

impl StateHash {
    /// Hashes one value.
    #[must_use]
    #[inline]
    pub fn of<T: Canonical>(value: &T) -> Self {
        let mut hasher = StateHasher::new();
        hasher.add(value);
        hasher.finish()
    }

    /// Wraps a raw hash.
    #[must_use]
    #[inline]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the raw hash.
    #[must_use]
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Hashes the canonical encodings of several values in order, for example
/// a record header followed by its state. Streams into XXH3; never
/// allocates.
#[derive(Debug, Default)]
pub struct StateHasher {
    out: Encoder<HashSink>,
}

impl StateHasher {
    /// Starts an empty hash.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self {
            out: Encoder::new(HashSink::new()),
        }
    }

    /// Appends the canonical encoding of `value`.
    #[inline]
    pub fn add<T: Canonical>(&mut self, value: &T) {
        value.encode(&mut self.out);
    }

    /// Returns the hash of everything added.
    #[must_use]
    #[inline]
    pub fn finish(&self) -> StateHash {
        StateHash(self.out.sink().digest())
    }
}

impl Default for Encoder<HashSink> {
    #[inline]
    fn default() -> Self {
        Self::new(HashSink::new())
    }
}

/// XXH3-64 of `bytes`: the checksum of persisted files (ADR-0011). It detects
/// accidents and is not authentication.
#[must_use]
pub fn checksum(bytes: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(bytes)
}
