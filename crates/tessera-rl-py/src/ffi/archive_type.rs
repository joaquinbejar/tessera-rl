// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native.Archive` type: an opaque, immutable archive of
//! saved states (ADR-0075), built by an environment's `_make_archive` and
//! attached with `set_reset_source`. It holds the engine's `Archive<E>`
//! without its environment type, as a safe body of the typed layer of
//! `tessera_rl_pyffi` (ADR-0062 item 8); an environment of another type
//! refuses it.

use std::any::Any;
use std::sync::Arc;

use tessera_rl_pyffi::ffi::capi::{self, CResult, Gil, Mine, Ref};
use tessera_rl_pyffi::ffi::object::{Body, MethodRow, MethodTable, TypeObject, TypeSpec};

/// An engine `Archive<E>` of some environment `E`, type-erased.
pub(crate) type AnyArchive = Arc<dyn Any + Send + Sync>;

/// An `Archive` object's body.
pub(crate) struct ArchiveBody(AnyArchive);

impl Body for ArchiveBody {}

static TYPE: TypeObject<ArchiveBody> = TypeObject::new(&SPEC);

static SPEC: TypeSpec<ArchiveBody> = TypeSpec::new(
    c"tessera_rl._native.Archive",
    c"An opaque, immutable archive of saved states (ADR-0075); use tessera_rl.Archive.",
    &METHODS,
);

static METHODS: MethodTable<ArchiveBody> = MethodTable::new(&[MethodRow::END]);

/// Wraps `archive` in a new object.
pub(crate) fn new_object(g: Gil<'_>, archive: AnyArchive) -> CResult<Mine<'_>> {
    TYPE.instantiate(g, ArchiveBody(archive))
}

/// The archive held by `obj`, which must be exactly an `Archive`.
pub(crate) fn archive_of(g: Gil<'_>, obj: Ref<'_>) -> CResult<AnyArchive> {
    TYPE.body(obj)
        .map(|b| Arc::clone(&b.0))
        .ok_or_else(|| capi::raise(g, capi::type_error(), "expected a tessera_rl.Archive"))
}

/// Creates the type and adds it to `module`.
pub(super) fn register(g: Gil<'_>, module: Ref<'_>) -> CResult<()> {
    TYPE.create(g, module, c"Archive")
}
