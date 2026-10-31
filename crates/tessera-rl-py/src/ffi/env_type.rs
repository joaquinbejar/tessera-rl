// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native.Env` heap type: object layout, allocation,
//! deallocation, the method table and the module-level constructors. The
//! `extern "C"` functions here are only reachable through the thunks of
//! `thunks`.

use core::ffi::c_int;
use core::ptr;

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple, PyType};

use super::inner::Inner;
use super::tables::{
    method, method_end, method_kw, slot_dealloc, slot_doc, slot_end, slot_methods,
};
use super::{Table, dispose, entry, free_heap_object, thunks};
use crate::api;

/// The C layout of an `Env` object.
#[repr(C)]
struct EnvObject {
    ob_base: ffi::PyObject,
    /// Owned; null only before construction and after deallocation.
    inner: *mut Inner,
}

static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// Wraps `inner` in a new `Env` object.
fn new_object(py: Python<'_>, inner: Inner) -> PyResult<*mut ffi::PyObject> {
    let ty = TYPE
        .get(py)
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Env type not initialized"))?;
    let ty_ptr = ty.as_ptr().cast::<ffi::PyTypeObject>();
    // SAFETY: attached; `ty_ptr` is our heap type, whose basic size is
    // `EnvObject`. PyType_GenericAlloc zero-fills and takes a reference to
    // the heap type, as instances of heap types require.
    let obj = unsafe { ffi::PyType_GenericAlloc(ty_ptr, 0) };
    if obj.is_null() {
        return Err(PyErr::fetch(py));
    }
    let raw = Box::into_raw(Box::new(inner));
    // SAFETY: `obj` is a fresh, zeroed `EnvObject` nobody else sees yet.
    unsafe { (*obj.cast::<EnvObject>()).inner = raw };
    Ok(obj)
}

/// The shared state of `slf`.
///
/// # Safety
///
/// `slf` must be `self` of an `Env` method, passed by CPython while
/// attached. Only `&Inner` is handed out: exclusive access to the pool goes
/// through the guard in `Inner`.
unsafe fn inner<'a>(slf: *mut ffi::PyObject) -> PyResult<&'a Inner> {
    // SAFETY: per the contract, `slf` is an `EnvObject`.
    let raw = unsafe { (*slf.cast::<EnvObject>()).inner };
    // SAFETY: a non-null `inner` is the live Box made in `new_object`,
    // freed only in `dealloc`, which runs after the last reference.
    unsafe { raw.as_ref() }
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Env object not initialized"))
}

/// A method without arguments, calling `$api(py, &Inner)`.
macro_rules! noargs {
    ($impl_fn:ident, $api:path) => {
        pub(super) unsafe extern "C" fn $impl_fn(
            slf: *mut ffi::PyObject,
            _args: *mut ffi::PyObject,
        ) -> *mut ffi::PyObject {
            entry(|py| {
                // SAFETY: CPython passes `self` of an `Env` method.
                let inner = unsafe { inner(slf) }?;
                Ok($api(py, inner)?.into_ptr())
            })
        }
    };
}

/// A method with positional arguments, calling `$api(py, &Inner, args)`.
macro_rules! varargs {
    ($impl_fn:ident, $api:path) => {
        pub(super) unsafe extern "C" fn $impl_fn(
            slf: *mut ffi::PyObject,
            args: *mut ffi::PyObject,
        ) -> *mut ffi::PyObject {
            entry(|py| {
                // SAFETY: CPython passes `self` of an `Env` method.
                let inner = unsafe { inner(slf) }?;
                // SAFETY: METH_VARARGS passes a borrowed tuple.
                let args = unsafe { Bound::from_borrowed_ptr(py, args) };
                let args = args.cast::<PyTuple>().map_err(PyErr::from)?;
                Ok($api(py, inner, args)?.into_ptr())
            })
        }
    };
}

noargs!(layout, api::layout);
noargs!(outputs, api::outputs);
noargs!(is_poisoned, api::is_poisoned);
noargs!(close, api::close);
varargs!(reset, api::reset);
varargs!(step, api::step);
varargs!(snapshot, api::snapshot);
varargs!(import_record, api::import_record);
varargs!(fork, api::fork);

static METHODS: Table<[ffi::PyMethodDef; 10]> = Table([
    method(
        c"layout",
        thunks::LAYOUT,
        ffi::METH_NOARGS,
        c"Shapes and dtypes of the outputs.",
    ),
    method(
        c"outputs",
        thunks::OUTPUTS,
        ffi::METH_NOARGS,
        c"Copies of the last call's outputs.",
    ),
    method(
        c"is_poisoned",
        thunks::IS_POISONED,
        ffi::METH_NOARGS,
        c"Whether the pool is poisoned.",
    ),
    method(
        c"close",
        thunks::CLOSE,
        ffi::METH_NOARGS,
        c"Frees the pool.",
    ),
    method(
        c"reset",
        thunks::RESET,
        ffi::METH_VARARGS,
        c"reset(seed=None)",
    ),
    method(
        c"step",
        thunks::STEP,
        ffi::METH_VARARGS,
        c"step(actions: bytes of int32)",
    ),
    method(
        c"snapshot",
        thunks::SNAPSHOT,
        ffi::METH_VARARGS,
        c"snapshot(id) -> bytes",
    ),
    method(
        c"import_record",
        thunks::IMPORT_RECORD,
        ffi::METH_VARARGS,
        c"import_record(id, record)",
    ),
    method(c"fork", thunks::FORK, ffi::METH_VARARGS, c"fork(src, dst)"),
    method_end(),
]);

/// `tp_dealloc`: a cleanup entry, which never fails or rejects (ADR-0041).
/// CPython calls it once, attached, when the reference count reaches zero.
pub(super) unsafe extern "C" fn dealloc(obj: *mut ffi::PyObject) {
    let env = obj.cast::<EnvObject>();
    // SAFETY: `obj` is a live `EnvObject` (above).
    let raw = unsafe { (*env).inner };
    // SAFETY: as above; cleared before the box is freed.
    unsafe { (*env).inner = ptr::null_mut() };
    if !raw.is_null() {
        // SAFETY: a non-null `inner` is the Box made in `new_object`, owned
        // by this object; no entry can run on an object being deallocated.
        let boxed = unsafe { Box::from_raw(raw) };
        // SAFETY: CPython calls tp_dealloc attached.
        let py = unsafe { Python::assume_attached() };
        // Dropping a pool joins its workers: detached. A panic is
        // contained and its payload disposed of without unwinding.
        if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            py.detach(move || drop(boxed));
        })) {
            dispose(payload);
        }
    }
    // SAFETY: `obj` is a live `Env` instance, in its tp_dealloc.
    unsafe { free_heap_object(obj) };
}

static SLOTS: Table<[ffi::PyType_Slot; 4]> = Table([
    slot_dealloc(thunks::DEALLOC),
    slot_methods(&METHODS.0),
    slot_doc(c"A pool of Tessera environments (use tessera_rl.make)."),
    slot_end(),
]);

/// Creates the `Env` type and adds it to `module`.
pub(super) fn register(py: Python<'_>, module: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    let mut spec = ffi::PyType_Spec {
        name: c"tessera_rl._native.Env".as_ptr(),
        basicsize: c_int::try_from(core::mem::size_of::<EnvObject>()).unwrap_or(c_int::MAX),
        itemsize: 0,
        flags: core::ffi::c_uint::try_from(ffi::Py_TPFLAGS_DEFAULT).unwrap_or(0),
        slots: SLOTS.0.as_ptr().cast_mut(),
    };
    // SAFETY: attached; the spec lives through the call, its slot table is
    // static, and CPython copies what it keeps.
    let raw = unsafe { ffi::PyType_FromSpec(&raw mut spec) };
    // SAFETY: `raw` is a new reference or null with an exception set.
    let ty = unsafe { Bound::from_owned_ptr_or_err(py, raw) }?;
    let ty = ty.cast_into::<PyType>().map_err(PyErr::from)?;
    module.add("Env", &ty)?;
    let _ = TYPE.set(py, ty.unbind());
    Ok(())
}

/// A module function that builds an `Env` from keyword arguments.
macro_rules! maker {
    ($impl_fn:ident, $api:path) => {
        pub(super) unsafe extern "C" fn $impl_fn(
            _module: *mut ffi::PyObject,
            _args: *mut ffi::PyObject,
            kwargs: *mut ffi::PyObject,
        ) -> *mut ffi::PyObject {
            entry(|py| {
                // SAFETY: METH_VARARGS | METH_KEYWORDS passes a borrowed dict
                // or null.
                let kwargs = unsafe { Bound::from_borrowed_ptr_or_opt(py, kwargs) };
                let kwargs = kwargs
                    .as_ref()
                    .map(|k| k.cast::<PyDict>())
                    .transpose()
                    .map_err(PyErr::from)?;
                let inner = $api(py, kwargs)?;
                new_object(py, inner)
            })
        }
    };
}

maker!(make_cartpole, api::make_cartpole);
maker!(make_tag, api::make_tag);

/// The module-level constructors, for the module's method table.
pub(super) const MAKERS: [ffi::PyMethodDef; 2] = [
    method_kw(
        c"make_cartpole",
        thunks::MAKE_CARTPOLE,
        c"make_cartpole(**pool_options) -> Env",
    ),
    method_kw(
        c"make_tag",
        thunks::MAKE_TAG,
        c"make_tag(**pool_options, **tag_config) -> Env",
    ),
];
