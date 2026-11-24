// SPDX-License-Identifier: MIT OR Apache-2.0

//! The reset source as pool context (ADR-0075 item 7, SPEC 9.4): what a
//! pool snapshot records about it. The archive itself is named by its
//! digest, never stored; the weights are stored in full.

/// The reset source of a pool context.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ResetSourceContext {
    /// `Environment::reset` (the default).
    #[default]
    Fresh,
    /// Archive resets from the archive with this digest and these weights.
    Archive {
        /// The archive's digest ([`crate::Archive::digest`]).
        digest: u64,
        /// One weight per entry.
        weights: Vec<u32>,
    },
}
