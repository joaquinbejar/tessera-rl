// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hidden supertraits that seal traits only derives implement. Public so
//! the derives' generated code in other crates can name them; not part of
//! the API.

/// Seals `ObservationLayout`: implemented by `#[derive(Observation)]` only.
#[doc(hidden)]
pub trait ObservationLayoutSealed {}
