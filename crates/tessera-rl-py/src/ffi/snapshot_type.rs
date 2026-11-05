// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native.PoolSnapshot` heap type: an opaque whole-pool
//! snapshot of a split-phase environment. The `extern "C"` function here is
//! only reachable through the thunks of `thunks`.

use core::ffi::c_int;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use pyo3::ffi;

use super::capi::{self, CResult, Gil, Mine, Ref};
use super::tables::{slot_dealloc, slot_doc, slot_end};
use super::{Table, dispose, thunks};
use crate::split::Snapshot;

/// The C layout of a `PoolSnapshot` object.
#[repr(C)]
struct SnapshotObject {
    ob_base: ffi::PyObject,
    /// Owned; null only before construction and after deallocation.
    snapshot: *mut Snapshot,
}

/// The type, set at module initialization (a borrowed reference: the module
/// keeps it alive).
static TYPE: AtomicPtr<ffi::PyObject> = AtomicPtr::new(ptr::null_mut());

/// Wraps `snapshot` in a new object.
pub(crate) fn new_object(g: Gil, snapshot: Snapshot) -> CResult<Mine> {
    let ty = TYPE.load(Ordering::Acquire);
    if ty.is_null() {
        return Err(capi::raise(
            g,
            capi::runtime_error(),
            "PoolSnapshot type not initialized",
        ));
    }
    // SAFETY: our heap type (not GC-tracked), whose basic size is
    // `SnapshotObject`.
    let obj = unsafe { capi::alloc_instance(g, ty) }?;
    // SAFETY: `obj` is a fresh, zeroed `SnapshotObject` nobody else sees yet.
    unsafe {
        (*obj.as_ref().as_ptr().cast::<SnapshotObject>()).snapshot =
            Box::into_raw(Box::new(snapshot));
    };
    Ok(obj)
}

/// The snapshot held by `obj`, which must be exactly a `PoolSnapshot`.
pub(crate) fn snapshot_of(g: Gil, obj: Ref<'_>) -> CResult<Snapshot> {
    let raw = obj.as_ptr();
    let is_ours = capi::is_instance_of(obj, TYPE.load(Ordering::Acquire));
    if !is_ours {
        return Err(capi::raise(
            g,
            capi::type_error(),
            "expected a snapshot from snapshot_all",
        ));
    }
    // SAFETY: the type check above: `raw` is a `SnapshotObject`.
    let held = unsafe { (*raw.cast::<SnapshotObject>()).snapshot };
    // SAFETY: a non-null `snapshot` is the Box made in `new_object`, freed
    // only in `dealloc`.
    unsafe { held.as_ref() }
        .cloned()
        .ok_or_else(|| capi::raise(g, capi::runtime_error(), "PoolSnapshot not initialized"))
}

/// `tp_dealloc`: a cleanup entry, which never fails (ADR-0041).
pub(super) unsafe extern "C" fn dealloc(obj: *mut ffi::PyObject) {
    // SAFETY: `obj` is a live `SnapshotObject`.
    let raw = unsafe { (*obj.cast::<SnapshotObject>()).snapshot };
    // SAFETY: as above; cleared before the box is freed.
    unsafe { (*obj.cast::<SnapshotObject>()).snapshot = ptr::null_mut() };
    if !raw.is_null() {
        // SAFETY: the Box made in `new_object`, owned by this object.
        let boxed = unsafe { Box::from_raw(raw) };
        if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(boxed)))
        {
            dispose(payload);
        }
    }
    // SAFETY: `obj` is a live instance of a heap type of this module.
    unsafe { capi::free_instance(obj) };
}

static SLOTS: Table<[ffi::PyType_Slot; 3]> = Table([
    slot_dealloc(thunks::SNAPSHOT_DEALLOC),
    slot_doc(c"An opaque whole-pool snapshot; restores only the environment that took it."),
    slot_end(),
]);

/// Creates the type and adds it to `module`.
pub(super) fn register(g: Gil, module: Ref<'_>) -> CResult<()> {
    let mut spec = ffi::PyType_Spec {
        name: c"tessera_rl._native.PoolSnapshot".as_ptr(),
        basicsize: c_int::try_from(core::mem::size_of::<SnapshotObject>()).unwrap_or(c_int::MAX),
        itemsize: 0,
        // Not GC-tracked: it holds no Python object.
        flags: core::ffi::c_uint::try_from(ffi::Py_TPFLAGS_DEFAULT).unwrap_or(0),
        slots: SLOTS.0.as_ptr().cast_mut(),
    };
    let ty = capi::type_from_spec(g, &mut spec)?;
    capi::module_add(g, module, c"PoolSnapshot", &ty)?;
    TYPE.store(ty.as_ref().as_ptr(), Ordering::Release);
    Ok(())
}
