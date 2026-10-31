// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded reads over an encoded byte slice.

use crate::error::DecodeError;

/// Reads canonical encodings from a byte slice. Every read is bounds
/// checked and returns an error instead of panicking.
#[derive(Debug)]
pub struct Decoder<'a> {
    bytes: &'a [u8],
}

impl<'a> Decoder<'a> {
    /// Reads from the start of `bytes`.
    #[must_use]
    #[inline]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    /// Reads the next `N` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::UnexpectedEof`] if fewer than `N` remain.
    #[inline]
    pub fn read_array<const N: usize>(&mut self) -> Result<[u8; N], DecodeError> {
        let Some((head, rest)) = self.bytes.split_first_chunk::<N>() else {
            return Err(DecodeError::UnexpectedEof {
                needed: N,
                remaining: self.bytes.len(),
            });
        };
        self.bytes = rest;
        Ok(*head)
    }

    /// Returns the number of unread bytes.
    #[must_use]
    #[inline]
    pub const fn remaining(&self) -> usize {
        self.bytes.len()
    }

    /// Checks that the whole input was read.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::TrailingBytes`] if bytes remain.
    #[inline]
    pub const fn finish(&self) -> Result<(), DecodeError> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            Err(DecodeError::TrailingBytes {
                count: self.bytes.len(),
            })
        }
    }
}
