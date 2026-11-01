// SPDX-License-Identifier: MIT OR Apache-2.0

//! Search pool options.

use core::num::NonZeroU32;

/// Default memory budget of a search pool's node storage: 1 GiB.
pub const DEFAULT_BUDGET: u64 = 1 << 30;

/// What expansion does when every node slot is in use (ADR-0043 section 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Eviction {
    /// Exceeding the capacity is an error (the conservative default).
    #[default]
    Strict,
    /// Evict the evictable node with the lowest allocation number, and its
    /// subtree.
    Oldest,
}

/// Construction options of a [`SearchPool`](crate::SearchPool).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SearchOptions {
    /// Node slots, the root included.
    pub capacity: NonZeroU32,
    /// Children stepped per expansion call: the scratch pool's size.
    pub batch: NonZeroU32,
    /// Behaviour on ring exhaustion.
    pub eviction: Eviction,
    /// Upper bound on the node storage, in bytes. The error names the bytes
    /// per node: the record, the node and one edge per action.
    pub budget_bytes: u64,
}

impl SearchOptions {
    /// `capacity` node slots and `batch` children per call, strict, with
    /// the default budget.
    #[must_use]
    pub const fn new(capacity: NonZeroU32, batch: NonZeroU32) -> Self {
        Self {
            capacity,
            batch,
            eviction: Eviction::Strict,
            budget_bytes: DEFAULT_BUDGET,
        }
    }
}
