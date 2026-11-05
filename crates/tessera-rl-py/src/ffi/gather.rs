// SPDX-License-Identifier: MIT OR Apache-2.0

//! The preallocated buffer an Async batch is gathered into (ADR-0047
//! section 9): one region per output field, reused at every `recv`.
//!
//! Views read it through raw pointers, under the overwrite semantics of SPEC
//! 5.3: the generation moves before a write, so a view is stale before its
//! rows change, and reading during a `recv` is the caller's race, as with
//! every borrowed output. The one [`GatherWriter`] is held behind the
//! binding's mutex, so writes never overlap.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU64, Ordering};

use std::sync::Arc;

use tessera_rl_engine::OutputField;

struct Region {
    field: OutputField,
    /// `u64` words, so every element type is aligned.
    words: UnsafeCell<Box<[u64]>>,
    bytes: usize,
}

/// The shared side: what views keep alive and read.
pub(crate) struct GatherBuffer {
    regions: Vec<Region>,
    generation: AtomicU64,
}

// SAFETY: the words are written only through the unique `GatherWriter`, and
// `GatherBuffer` itself only hands out raw pointers to them; all other
// fields are immutable or atomic.
#[expect(unsafe_code, reason = "audited module: see the SAFETY comment")]
unsafe impl Sync for GatherBuffer {}

// SAFETY: as above; the words are plain data.
#[expect(unsafe_code, reason = "audited module: see the SAFETY comment")]
unsafe impl Send for GatherBuffer {}

impl GatherBuffer {
    /// The write generation (a view taken at `g` is stale once it differs).
    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// The start and byte length of `field`'s region.
    pub(crate) fn region(&self, field: OutputField) -> Option<(*const u8, usize)> {
        let region = self.regions.iter().find(|r| r.field == field)?;
        // Only a pointer is formed here, never a reference to the words.
        let start = region.words.get().cast_const();
        // SAFETY: `start` points to the live Box, which is never replaced;
        // reading the fat pointer does not touch the words.
        #[expect(unsafe_code, reason = "audited module: see the SAFETY comment")]
        let data = unsafe { core::ptr::addr_of!(**start) }.cast::<u8>();
        Some((data, region.bytes))
    }
}

/// The unique writer of a [`GatherBuffer`].
pub(crate) struct GatherWriter(Arc<GatherBuffer>);

impl GatherWriter {
    /// A buffer with `fields`, each of the given byte length.
    pub(crate) fn new(fields: &[(OutputField, usize)]) -> Self {
        let regions = fields
            .iter()
            .map(|&(field, bytes)| Region {
                field,
                words: UnsafeCell::new(vec![0u64; bytes.div_ceil(8)].into_boxed_slice()),
                bytes,
            })
            .collect();
        Self(Arc::new(GatherBuffer {
            regions,
            generation: AtomicU64::new(0),
        }))
    }

    pub(crate) fn buffer(&self) -> Arc<GatherBuffer> {
        Arc::clone(&self.0)
    }

    /// Ends the window of every view; the next generation's writes follow.
    pub(crate) fn begin(&mut self) -> u64 {
        self.0.generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Writes `bytes` at byte offset `at` of `field`; false if it does not
    /// fit or the field is unknown.
    pub(crate) fn write(&mut self, field: OutputField, at: usize, bytes: &[u8]) -> bool {
        let Some((start, len)) = self.0.region(field) else {
            return false;
        };
        if at.checked_add(bytes.len()).is_none_or(|end| end > len) {
            return false;
        }
        // SAFETY: `at + bytes.len() <= len`, the region's size, so the
        // offset stays inside its allocation.
        #[expect(unsafe_code, reason = "audited module: see the SAFETY comment")]
        let target = unsafe { start.cast_mut().add(at) };
        // SAFETY: the range fits (above), and this is the only writer
        // (`&mut self`, one per buffer). A concurrent view read is excluded
        // by the borrow window of SPEC 5.3.
        #[expect(unsafe_code, reason = "audited module: see the SAFETY comment")]
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), target, bytes.len());
        }
        true
    }

    /// A copy of the first `len` bytes of `field`.
    pub(crate) fn copy(&self, field: OutputField, len: usize) -> Vec<u8> {
        let Some((start, region)) = self.0.region(field) else {
            return Vec::new();
        };
        let len = len.min(region);
        // SAFETY: `len` is within the region; this is the only writer and it
        // is not writing (`&self`).
        #[expect(unsafe_code, reason = "audited module: see the SAFETY comment")]
        unsafe { core::slice::from_raw_parts(start, len) }.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_write_is_bounded_and_generation_moves_first() {
        let mut w = GatherWriter::new(&[(OutputField::Rewards, 8)]);
        let b = w.buffer();
        assert_eq!(b.generation(), 0);
        assert_eq!(w.begin(), 1);
        assert_eq!(b.generation(), 1);
        assert!(w.write(OutputField::Rewards, 4, &[1, 2, 3, 4]));
        assert!(!w.write(OutputField::Rewards, 5, &[1, 2, 3, 4]));
        assert!(!w.write(OutputField::Alive, 0, &[1]));
        assert_eq!(
            w.copy(OutputField::Rewards, 8),
            vec![0, 0, 0, 0, 1, 2, 3, 4]
        );
        assert_eq!(b.region(OutputField::Rewards).unwrap().0 as usize % 8, 0);
    }
}
