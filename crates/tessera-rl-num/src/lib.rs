// SPDX-License-Identifier: MIT OR Apache-2.0

//! Numeric foundations: Philox RNG, `Fx32`/`Fx64` fixed point, frozen math
//! and the floating-point scope guard (ADR-0007, ADR-0008).

// Randomness (SPEC 8.1, ADR-0007). The Philox block, the mixer and the
// draw core are internal; environments use `StepRng` and `ObsRng`.
pub(crate) mod draw;
pub mod identity;
pub mod key;
pub(crate) mod philox;
pub mod rng;
pub(crate) mod splitmix;

// Fixed point (SPEC 8.2, ADR-0008).
mod fixed;
pub mod fx32;
pub mod fx64;

pub use fx32::Fx32;
pub use fx64::Fx64;

// Floating point (SPEC 8.2, ADR-0008).
pub mod fp;
pub mod math;
