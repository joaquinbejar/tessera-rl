// SPDX-License-Identifier: MIT OR Apache-2.0

//! Node handles and edge statistics.

/// A node of a [`SearchPool`](crate::SearchPool): its slot and the slot's
/// generation when the node was made. A handle whose generation no longer
/// matches the slot is stale and is rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeHandle {
    pub(crate) slot: u32,
    pub(crate) generation: u32,
}

impl NodeHandle {
    /// The node's slot.
    #[must_use]
    pub const fn slot(&self) -> u32 {
        self.slot
    }

    /// The slot's generation when the node was made.
    #[must_use]
    pub const fn generation(&self) -> u32 {
        self.generation
    }
}

/// The statistics of one edge (a parent's action), ADR-0043 section 2.
///
/// `visits` (`N`) is an integer count and `value` (`W`) the total backed-up
/// value from the view of the player to move at the parent; the virtual
/// loss is a separate integer count, so removing it restores `N` and `W`
/// exactly.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EdgeStats {
    /// Visit count `N(s, a)`.
    pub visits: u32,
    /// Total value `W(s, a)`.
    pub value: f64,
    /// Virtual losses currently applied (ADR-0043 section 3).
    pub virtual_loss: u32,
    /// Prior `P(s, a)`.
    pub prior: f64,
    /// Whether the edge has a child node. An evicted child reverts the
    /// edge to unexpanded and keeps its statistics.
    pub expanded: bool,
}
