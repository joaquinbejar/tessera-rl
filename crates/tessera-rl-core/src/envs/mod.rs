// SPDX-License-Identifier: MIT OR Apache-2.0

//! Environment registration (SPEC 10, ADR-0032, ADR-0054).
//!
//! [`register!`](crate::register) submits an [`EnvRegistration`] for a type
//! that implements [`Persist`](crate::Persist); [`registered`] lists every
//! submission linked into the binary, sorted by id. This is not the
//! active-engine registry of SPEC 6.1.
//!
//! The registry is per binary: an environment compiled into another cdylib
//! or wheel is not listed here (ADR-0032 keeps the `tessera_rl.envs`
//! entry-point group for that). Records are collected at the first lookup,
//! never before `main`; a duplicate id is an error there.

mod lookup;
mod record;
mod register;

pub use lookup::{RegistryError, registered};
pub use record::{EnvRegistration, same_id};

/// Items the [`register!`](crate::register) expansion names. Not public API.
#[doc(hidden)]
pub mod __private {
    pub use inventory;
}
