// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-instance and per-half export generations (ADR-0047 section 5).
//!
//! A split-phase mode keeps one generation per instance and one per
//! Pipelined half. A generation advances only when an admission or a
//! mutation of those rows succeeds, never on a failed call, and a view
//! records the generations of its rows when it is created: the view is
//! stale as soon as any of them has moved. The counters are caller-side
//! state, read by borrowed views on other threads, so they are atomics; no
//! worker ever touches them, and they never read a header a worker may be
//! writing.
//!
//! All storage is allocated once in [`Generations::new`]; reading and
//! advancing allocate nothing.

use thiserror::Error;

use crate::sys::sync::{AtomicU64, Ordering};

/// The number of Pipelined halves.
pub(crate) const HALVES: usize = 2;

/// Why a generation could not be read or advanced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum GenerationError {
    /// The index is past the end of the counters.
    #[error("generation index {index} is out of range for {len} counters")]
    OutOfRange {
        /// The requested index.
        index: usize,
        /// The number of counters.
        len: usize,
    },
    /// The counter is at `u64::MAX` and cannot advance.
    #[error("generation counter {index} is exhausted")]
    Exhausted {
        /// The counter's index.
        index: usize,
    },
}

/// Preallocated generations for `n` instances and [`HALVES`] halves.
#[derive(Debug)]
pub(crate) struct Generations {
    ids: Box<[AtomicU64]>,
    halves: [AtomicU64; HALVES],
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the split-phase modes (ADR-0047)")
)]
impl Generations {
    /// Counters for `n` instances, all at zero.
    #[must_use]
    pub(crate) fn new(n: usize) -> Self {
        Self {
            ids: (0..n).map(|_| AtomicU64::new(0)).collect(),
            halves: [AtomicU64::new(0), AtomicU64::new(0)],
        }
    }

    /// The number of instance counters.
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    /// The generation of instance `id`.
    pub(crate) fn id(&self, id: usize) -> Result<u64, GenerationError> {
        cell(&self.ids, id).map(|c| c.load(Ordering::Acquire))
    }

    /// The generation of half `half`.
    pub(crate) fn half(&self, half: usize) -> Result<u64, GenerationError> {
        cell(&self.halves, half).map(|c| c.load(Ordering::Acquire))
    }

    /// Advances instance `id` and returns its new generation. Called only
    /// after the admission or mutation of that instance has succeeded.
    pub(crate) fn advance_id(&self, id: usize) -> Result<u64, GenerationError> {
        advance(cell(&self.ids, id)?, id)
    }

    /// Advances half `half` and returns its new generation.
    pub(crate) fn advance_half(&self, half: usize) -> Result<u64, GenerationError> {
        advance(cell(&self.halves, half)?, half)
    }
}

fn cell(cells: &[AtomicU64], index: usize) -> Result<&AtomicU64, GenerationError> {
    cells.get(index).ok_or(GenerationError::OutOfRange {
        index,
        len: cells.len(),
    })
}

/// Checked increment: an exhausted counter is an error, never a wrap, so a
/// view can never see an old generation again (no ABA).
fn advance(cell: &AtomicU64, index: usize) -> Result<u64, GenerationError> {
    let mut current = cell.load(Ordering::Acquire);
    loop {
        let next = current
            .checked_add(1)
            .ok_or(GenerationError::Exhausted { index })?;
        match cell.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Ok(next),
            Err(seen) => current = seen,
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::{AtomicU64, GenerationError, Generations, HALVES, advance};

    #[test]
    fn test_generations_new_starts_every_counter_at_zero() {
        let g = Generations::new(3);
        assert_eq!(g.len(), 3);
        for id in 0..3 {
            assert_eq!(g.id(id), Ok(0));
        }
        for half in 0..HALVES {
            assert_eq!(g.half(half), Ok(0));
        }
    }

    #[test]
    fn test_generations_advance_moves_only_that_counter() {
        let g = Generations::new(3);
        assert_eq!(g.advance_id(1), Ok(1));
        assert_eq!(g.advance_id(1), Ok(2));
        assert_eq!(g.advance_half(0), Ok(1));
        assert_eq!(g.id(0), Ok(0));
        assert_eq!(g.id(1), Ok(2));
        assert_eq!(g.id(2), Ok(0));
        assert_eq!(g.half(0), Ok(1));
        assert_eq!(g.half(1), Ok(0));
    }

    #[test]
    fn test_generations_out_of_range_index_is_a_typed_error() {
        let g = Generations::new(2);
        let id = GenerationError::OutOfRange { index: 2, len: 2 };
        assert_eq!(g.id(2), Err(id));
        assert_eq!(g.advance_id(2), Err(id));
        let half = GenerationError::OutOfRange {
            index: HALVES,
            len: HALVES,
        };
        assert_eq!(g.half(HALVES), Err(half));
        assert_eq!(g.advance_half(HALVES), Err(half));
        assert_eq!(Generations::new(0).id(0).map_err(|_| ()), Err(()));
    }

    #[test]
    fn test_generations_exhausted_counter_is_an_error_and_unchanged() {
        let cell = AtomicU64::new(u64::MAX);
        assert_eq!(
            advance(&cell, 4),
            Err(GenerationError::Exhausted { index: 4 })
        );
        assert_eq!(cell.load(core::sync::atomic::Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn test_generations_concurrent_advances_are_all_counted() {
        let g = Generations::new(1);
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    for _ in 0..50 {
                        assert!(g.advance_id(0).is_ok());
                    }
                });
            }
        });
        assert_eq!(g.id(0), Ok(200));
    }
}
