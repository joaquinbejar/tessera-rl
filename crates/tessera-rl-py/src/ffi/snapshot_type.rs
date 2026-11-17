// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native.PoolSnapshot` type: an opaque whole-pool
//! snapshot of a split-phase environment, as a safe body of the typed layer
//! of `tessera_rl_pyffi` (ADR-0062 item 8).

use tessera_rl_pyffi::ffi::capi::{self, CResult, Gil, Mine, Ref};
use tessera_rl_pyffi::ffi::object::{Body, MethodRow, MethodTable, TypeObject, TypeSpec};

use crate::split::Snapshot;

/// A `PoolSnapshot` object's body.
pub(crate) struct SnapshotBody(Snapshot);

impl Body for SnapshotBody {}

static TYPE: TypeObject<SnapshotBody> = TypeObject::new(&SPEC);

static SPEC: TypeSpec<SnapshotBody> = TypeSpec::new(
    c"tessera_rl._native.PoolSnapshot",
    c"An opaque whole-pool snapshot; restores only the environment that took it.",
    &METHODS,
);

static METHODS: MethodTable<SnapshotBody> = MethodTable::new(&[MethodRow::END]);

/// Wraps `snapshot` in a new object.
pub(crate) fn new_object(g: Gil<'_>, snapshot: Snapshot) -> CResult<Mine<'_>> {
    TYPE.instantiate(g, SnapshotBody(snapshot))
}

/// The snapshot held by `obj`, which must be exactly a `PoolSnapshot`.
pub(crate) fn snapshot_of(g: Gil<'_>, obj: Ref<'_>) -> CResult<Snapshot> {
    TYPE.body(obj).map(|b| b.0.clone()).ok_or_else(|| {
        capi::raise(
            g,
            capi::type_error(),
            "expected a snapshot from snapshot_all",
        )
    })
}

/// Creates the type and adds it to `module`.
pub(super) fn register(g: Gil<'_>, module: Ref<'_>) -> CResult<()> {
    TYPE.create(g, module, c"PoolSnapshot")
}
