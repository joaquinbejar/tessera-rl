// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encoding sinks.

use xxhash_rust::xxh3::Xxh3;

/// A destination for encoded bytes.
pub trait Sink {
    /// Appends `bytes`.
    fn write(&mut self, bytes: &[u8]);
}

/// Writes into a fixed slice. Writes past the end are dropped and recorded,
/// so a length mismatch is detectable without panicking.
#[derive(Debug)]
pub struct SliceSink<'a> {
    buf: &'a mut [u8],
    pos: usize,
    overflowed: bool,
}

impl<'a> SliceSink<'a> {
    /// Starts writing at the beginning of `buf`.
    #[must_use]
    #[inline]
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            overflowed: false,
        }
    }

    /// Returns whether exactly the whole slice was written.
    #[must_use]
    #[inline]
    pub fn is_full(&self) -> bool {
        !self.overflowed && self.pos == self.buf.len()
    }
}

impl Sink for SliceSink<'_> {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let end = self.pos.checked_add(bytes.len());
        match end.and_then(|end| self.buf.get_mut(self.pos..end)) {
            Some(dst) => {
                dst.copy_from_slice(bytes);
                self.pos += bytes.len();
            }
            None => self.overflowed = true,
        }
    }
}

/// Feeds a streaming XXH3-64 hasher. Never allocates.
#[derive(Clone)]
pub struct HashSink(Xxh3);

impl HashSink {
    /// A hasher with seed 0.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self(Xxh3::new())
    }

    /// Returns the XXH3-64 digest of everything written.
    #[must_use]
    #[inline]
    pub fn digest(&self) -> u64 {
        self.0.digest()
    }
}

impl Default for HashSink {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for HashSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HashSink").finish_non_exhaustive()
    }
}

impl Sink for HashSink {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
}

impl Sink for Vec<u8> {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

/// Writes canonical encodings into a sink.
#[derive(Debug)]
pub struct Encoder<S> {
    sink: S,
}

impl<S: Sink> Encoder<S> {
    /// Wraps a sink.
    #[must_use]
    #[inline]
    pub fn new(sink: S) -> Self {
        Self { sink }
    }

    /// Appends raw bytes.
    #[inline]
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.sink.write(bytes);
    }

    /// Returns the sink.
    #[must_use]
    #[inline]
    pub fn sink(&self) -> &S {
        &self.sink
    }

    /// Consumes the encoder and returns the sink.
    #[must_use]
    #[inline]
    pub fn into_sink(self) -> S {
        self.sink
    }
}
