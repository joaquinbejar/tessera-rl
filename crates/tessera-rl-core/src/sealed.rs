// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hidden supertraits that seal traits only derives implement. Public so
//! the derives' generated code in other crates can name them.
//!
//! Derive only, not a stable API: this module and its items may change or
//! disappear in any release; never implement or name them by hand.

/// Seals `ObservationLayout`: implemented by `#[derive(Observation)]` only.
/// Derive only, not a stable API.
#[doc(hidden)]
pub trait ObservationLayoutSealed {}

/// Seals `LaneTuple`: implemented for `()` and tuples of `LaneField`
/// only (ADR-0090).
#[doc(hidden)]
pub trait LaneTupleSealed {}
