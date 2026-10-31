// SPDX-License-Identifier: MIT OR Apache-2.0

//! The C boundary with CPython. Audited `unsafe` module (ADR-0041).
//!
//! - `thunks` defines every boundary thunk (ADR-0041) and is the only
//!   module that can make a `ThunkFn`.
//! - `tables` builds every C table row; function entries take a
//!   `ThunkFn`, so a raw function pointer cannot be registered.
//! - `inner` holds an environment object's state behind the reentrancy
//!   guard.

#![expect(unsafe_code, reason = "audited module: the CPython C boundary")]

/// A static C table holding raw pointers, shared read-only with CPython.
#[repr(transparent)]
pub(crate) struct Table<T>(pub(crate) T);

// SAFETY: the tables are written once at compile time and only read
// afterwards (by CPython, holding the interpreter lock).
unsafe impl<T> Sync for Table<T> {}

/// The body of every entry point: attached, panics contained (no unwind
/// crosses the C boundary), errors raised as Python exceptions.
pub(crate) fn entry(
    f: impl FnOnce(pyo3::Python<'_>) -> pyo3::PyResult<*mut pyo3::ffi::PyObject>,
) -> *mut pyo3::ffi::PyObject {
    // SAFETY: every entry point is called by CPython with the interpreter
    // attached to this thread.
    let py = unsafe { pyo3::Python::assume_attached() };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(py))) {
        Ok(Ok(obj)) => obj,
        Ok(Err(err)) => {
            err.restore(py);
            core::ptr::null_mut()
        }
        Err(payload) => {
            dispose(payload);
            pyo3::exceptions::PyRuntimeError::new_err("internal panic in Tessera").restore(py);
            core::ptr::null_mut()
        }
    }
}

/// Disposes of a panic payload without running its `Drop`, which could
/// itself panic across the C boundary: it is leaked.
pub(crate) fn dispose(payload: Box<dyn core::any::Any + Send>) {
    core::mem::forget(payload);
}

/// Frees a heap-type instance after its own fields were released: the
/// type's `tp_free`, then the type reference every heap instance holds.
///
/// # Safety
///
/// `obj` must be a live instance of a heap type, allocated by its
/// `tp_alloc`, in its `tp_dealloc`, attached.
pub(crate) unsafe fn free_heap_object(obj: *mut pyo3::ffi::PyObject) {
    // SAFETY: `obj` is a live object.
    let ty = unsafe { pyo3::ffi::Py_TYPE(obj) };
    // SAFETY: `ty` is a heap type, which always has a tp_free slot.
    let free = unsafe { pyo3::ffi::PyType_GetSlot(ty, pyo3::ffi::Py_tp_free) };
    if !free.is_null() {
        // SAFETY: the tp_free slot holds a `freefunc`.
        let free: unsafe extern "C" fn(*mut core::ffi::c_void) =
            unsafe { core::mem::transmute(free) };
        // SAFETY: `obj` was allocated by this type's tp_alloc.
        unsafe { free(obj.cast()) };
    }
    // SAFETY: instances of heap types hold a reference to their type,
    // released after the instance is freed.
    unsafe { pyo3::ffi::Py_DECREF(ty.cast()) };
}

mod dlpack;
mod env_type;
pub(crate) mod inner;
mod module;
mod tables;
mod thunks;
pub(crate) mod view_type;
