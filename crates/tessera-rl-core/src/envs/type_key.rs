// SPDX-License-Identifier: MIT OR Apache-2.0

//! Which Rust type a `register!` id belongs to, so a pool of a deprecated
//! environment can be recognised from its type (ADR-0080 item 6).

use core::any::TypeId;

use super::deprecation::Deprecation;
use super::lookup::registered;

/// A `register!` submission tying an id to its Rust type. Not public API:
/// read through [`deprecation_of`].
#[doc(hidden)]
pub struct TypeKey {
    id: &'static str,
    type_id: fn() -> TypeId,
}

impl TypeKey {
    /// The key of `E` under `id`.
    #[must_use]
    pub const fn of<E: 'static>(id: &'static str) -> Self {
        Self {
            id,
            type_id: TypeId::of::<E>,
        }
    }
}

inventory::collect!(TypeKey);

/// The id and deprecation of the environment type `E`, if `register!`
/// registered it as deprecated in this binary (ADR-0080 item 6). `None` when `E` is not
/// registered, not deprecated, or the registry has a duplicate id. Reads
/// the registry: call it at pool construction, never in a step.
#[must_use]
pub fn deprecation_of<E: 'static>() -> Option<(&'static str, Deprecation)> {
    let want = TypeId::of::<E>();
    let id = inventory::iter::<TypeKey>
        .into_iter()
        .find(|k| (k.type_id)() == want)?
        .id;
    registered()
        .ok()?
        .iter()
        .find(|r| r.id() == id)?
        .deprecation()
        .map(|d| (id, d))
}
