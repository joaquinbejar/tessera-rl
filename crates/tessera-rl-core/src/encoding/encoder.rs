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
///
/// Small writes (a canonical encoding writes one scalar at a time) are
/// gathered in a fixed buffer of [`HashSink::BUFFER_LEN`] bytes and passed
/// to XXH3 in blocks: the per-call cost of a streaming update, not the hash,
/// dominated (#387). XXH3's streaming result does not depend on how its input
/// is split, so the digest is the same as for one call over all the bytes.
#[derive(Clone)]
pub struct HashSink {
    state: Xxh3,
    buf: [u8; HashSink::BUFFER_LEN],
    len: usize,
}

impl HashSink {
    /// Bytes gathered before they are passed to XXH3.
    pub const BUFFER_LEN: usize = 512;

    /// A hasher with seed 0.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self {
            state: Xxh3::new(),
            buf: [0; Self::BUFFER_LEN],
            len: 0,
        }
    }

    /// Returns the XXH3-64 digest of everything written. Leaves the sink
    /// unchanged, so it may be called again or written to further.
    #[must_use]
    #[inline]
    pub fn digest(&self) -> u64 {
        match self.buf.get(..self.len) {
            Some(tail) if !tail.is_empty() => {
                let mut state = self.state.clone();
                state.update(tail);
                state.digest()
            }
            _ => self.state.digest(),
        }
    }

    /// Passes the gathered bytes to XXH3 and empties the buffer.
    #[inline]
    fn flush(&mut self) {
        if let Some(pending) = self.buf.get(..self.len) {
            self.state.update(pending);
        }
        self.len = 0;
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
        if let Some(end) = self.len.checked_add(bytes.len())
            && let Some(dst) = self.buf.get_mut(self.len..end)
        {
            dst.copy_from_slice(bytes);
            self.len = end;
            return;
        }
        self.flush();
        match self.buf.get_mut(..bytes.len()) {
            // Fits in the emptied buffer.
            Some(dst) => {
                dst.copy_from_slice(bytes);
                self.len = bytes.len();
            }
            // Larger than the buffer: no point in copying it.
            None => self.state.update(bytes),
        }
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
