// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native.Env` heap type: object layout, allocation,
//! deallocation and the method table, every C entry reached through a
//! boundary thunk.

use core::ffi::c_int;
use core::ptr;

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple, PyType};

use super::{Table, entry};
use crate::api::{self, Inner};

/// The C layout of an `Env` object.
#[repr(C)]
struct EnvObject {
    ob_base: ffi::PyObject,
    /// Owned; null only while being constructed or after deallocation.
    inner: *mut Inner,
}

static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// Wraps `inner` in a new `Env` object.
pub(super) fn new_object(py: Python<'_>, inner: Inner) -> PyResult<*mut ffi::PyObject> {
    let ty = TYPE
        .get(py)
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Env type not initialized"))?;
    let ty_ptr = ty.as_ptr().cast::<ffi::PyTypeObject>();
    // SAFETY: attached; `ty_ptr` is our heap type, whose basic size is
    // `EnvObject`. PyType_GenericAlloc zero-fills and takes a reference to
    // the heap type, as required for instances of heap types.
    let obj = unsafe { ffi::PyType_GenericAlloc(ty_ptr, 0) };
    if obj.is_null() {
        return Err(PyErr::fetch(py));
    }
    // SAFETY: `obj` is a fresh, zeroed `EnvObject`.
    unsafe { (*obj.cast::<EnvObject>()).inner = Box::into_raw(Box::new(inner)) };
    Ok(obj)
}

/// The `Inner` of `slf`, or an error if it is not an initialized `Env`.
///
/// # Safety
///
/// `slf` must be a live object passed by CPython as `self` of an `Env`
/// method, with the interpreter attached; the reference is used only for
/// the duration of the call (the reentrancy guard in `Inner` excludes
/// concurrent use while the interpreter is detached).
unsafe fn inner<'a>(slf: *mut ffi::PyObject) -> PyResult<&'a mut Inner> {
    // SAFETY: per the contract, `slf` is an `EnvObject`.
    let raw = unsafe { (*slf.cast::<EnvObject>()).inner };
    // SAFETY: non-null `inner` is a live, owned Box until deallocation.
    unsafe { raw.as_mut() }
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Env object not initialized"))
}

/// A method with no arguments.
macro_rules! noargs {
    ($impl_fn:ident, $thunk:ident, $api:path) => {
        unsafe extern "C" fn $impl_fn(
            slf: *mut ffi::PyObject,
            _args: *mut ffi::PyObject,
        ) -> *mut ffi::PyObject {
            entry(|py| {
                // SAFETY: CPython passes `self` of an `Env` method.
                let inner = unsafe { inner(slf) }?;
                Ok($api(py, inner)?.into_ptr())
            })
        }
        thunk!($thunk => $impl_fn: fn(*mut ffi::PyObject, *mut ffi::PyObject) -> *mut ffi::PyObject);
    };
}

/// A method with positional arguments.
macro_rules! varargs {
    ($impl_fn:ident, $thunk:ident, $api:path) => {
        unsafe extern "C" fn $impl_fn(
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
        thunk!($thunk => $impl_fn: fn(*mut ffi::PyObject, *mut ffi::PyObject) -> *mut ffi::PyObject);
    };
}

noargs!(layout, thunk_layout, api::layout);
noargs!(outputs, thunk_outputs, api::outputs);
noargs!(is_poisoned, thunk_is_poisoned, api::is_poisoned);
noargs!(close, thunk_close, api::close);
varargs!(reset, thunk_reset, api::reset);
varargs!(step, thunk_step, api::step);
varargs!(snapshot, thunk_snapshot, api::snapshot);
varargs!(import_record, thunk_import_record, api::import_record);
varargs!(fork, thunk_fork, api::fork);

const fn method(
    name: &'static core::ffi::CStr,
    f: unsafe extern "C" fn(*mut ffi::PyObject, *mut ffi::PyObject) -> *mut ffi::PyObject,
    flags: c_int,
    doc: &'static core::ffi::CStr,
) -> ffi::PyMethodDef {
    ffi::PyMethodDef {
        ml_name: name.as_ptr(),
        ml_meth: ffi::PyMethodDefPointer { PyCFunction: f },
        ml_flags: flags,
        ml_doc: doc.as_ptr(),
    }
}

static METHODS: Table<[ffi::PyMethodDef; 10]> = Table([
    method(
        c"layout",
        thunk_layout,
        ffi::METH_NOARGS,
        c"Shapes and dtypes of the outputs.",
    ),
    method(
        c"outputs",
        thunk_outputs,
        ffi::METH_NOARGS,
        c"Copies of the last call's outputs.",
    ),
    method(
        c"is_poisoned",
        thunk_is_poisoned,
        ffi::METH_NOARGS,
        c"Whether the pool is poisoned.",
    ),
    method(c"close", thunk_close, ffi::METH_NOARGS, c"Frees the pool."),
    method(
        c"reset",
        thunk_reset,
        ffi::METH_VARARGS,
        c"reset(seed=None)",
    ),
    method(
        c"step",
        thunk_step,
        ffi::METH_VARARGS,
        c"step(actions: bytes of int32)",
    ),
    method(
        c"snapshot",
        thunk_snapshot,
        ffi::METH_VARARGS,
        c"snapshot(id) -> bytes",
    ),
    method(
        c"import_record",
        thunk_import_record,
        ffi::METH_VARARGS,
        c"import_record(id, record)",
    ),
    method(c"fork", thunk_fork, ffi::METH_VARARGS, c"fork(src, dst)"),
    ffi::PyMethodDef::zeroed(),
]);

unsafe extern "C" fn dealloc(obj: *mut ffi::PyObject) {
    // A cleanup entry: never fails, never rejects (ADR-0041). CPython calls
    // tp_dealloc once, attached, when the reference count reaches zero;
    // `obj` is an `EnvObject` of our heap type.
    let env = obj.cast::<EnvObject>();
    // SAFETY: `env` is a live `EnvObject` (above).
    let raw = unsafe { (*env).inner };
    // SAFETY: as above; the field is cleared before the box is freed.
    unsafe { (*env).inner = ptr::null_mut() };
    if !raw.is_null() {
        // SAFETY: a non-null `inner` is the Box created in `new_object`,
        // owned by this object and freed only here.
        let boxed = unsafe { Box::from_raw(raw) };
        // SAFETY: CPython calls tp_dealloc attached.
        let py = unsafe { Python::assume_attached() };
        // Dropping a pool joins its workers: do it detached. A panic is
        // contained; cleanup never unwinds into CPython.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            py.detach(move || drop(boxed));
        }));
    }
    // SAFETY: `obj` is a live object.
    let ty = unsafe { ffi::Py_TYPE(obj) };
    // SAFETY: `ty` is our heap type, which always has a tp_free slot.
    let free = unsafe { ffi::PyType_GetSlot(ty, ffi::Py_tp_free) };
    if !free.is_null() {
        // SAFETY: the tp_free slot has the `freefunc` signature.
        let free: unsafe extern "C" fn(*mut core::ffi::c_void) =
            unsafe { core::mem::transmute(free) };
        // SAFETY: `obj` was allocated by this type's tp_alloc.
        unsafe { free(obj.cast()) };
    }
    // SAFETY: instances of heap types hold a reference to their type,
    // released after the instance is freed.
    unsafe { ffi::Py_DECREF(ty.cast()) };
}

thunk!(thunk_dealloc => dealloc: fn(*mut ffi::PyObject) -> ());

static SLOTS: Table<[ffi::PyType_Slot; 4]> = Table([
    ffi::PyType_Slot {
        slot: ffi::Py_tp_dealloc,
        pfunc: thunk_dealloc as *mut core::ffi::c_void,
    },
    ffi::PyType_Slot {
        slot: ffi::Py_tp_methods,
        pfunc: METHODS.0.as_ptr().cast_mut().cast(),
    },
    ffi::PyType_Slot {
        slot: ffi::Py_tp_doc,
        pfunc: c"A pool of Tessera environments (use tessera_rl.make)."
            .as_ptr()
            .cast_mut()
            .cast(),
    },
    ffi::PyType_Slot {
        slot: 0,
        pfunc: ptr::null_mut(),
    },
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
    // SAFETY: attached; the spec and its slot table outlive the call (the
    // tables are static) and CPython copies what it keeps.
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
    ($impl_fn:ident, $thunk:ident, $api:path) => {
        unsafe extern "C" fn $impl_fn(
            _module: *mut ffi::PyObject,
            _args: *mut ffi::PyObject,
            kwargs: *mut ffi::PyObject,
        ) -> *mut ffi::PyObject {
            entry(|py| {
                // SAFETY: METH_VARARGS | METH_KEYWORDS passes a borrowed
                // dict or null.
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
        thunk!($thunk => $impl_fn: fn(*mut ffi::PyObject, *mut ffi::PyObject, *mut ffi::PyObject) -> *mut ffi::PyObject);
    };
}

maker!(make_cartpole, thunk_make_cartpole, api::make_cartpole);
maker!(make_tag, thunk_make_tag, api::make_tag);

/// The module-level constructors, for the module's method table.
pub(super) const MAKERS: [ffi::PyMethodDef; 2] = [
    ffi::PyMethodDef {
        ml_name: c"make_cartpole".as_ptr(),
        ml_meth: ffi::PyMethodDefPointer {
            PyCFunctionWithKeywords: thunk_make_cartpole,
        },
        ml_flags: ffi::METH_VARARGS | ffi::METH_KEYWORDS,
        ml_doc: c"make_cartpole(**pool_options) -> Env".as_ptr(),
    },
    ffi::PyMethodDef {
        ml_name: c"make_tag".as_ptr(),
        ml_meth: ffi::PyMethodDefPointer {
            PyCFunctionWithKeywords: thunk_make_tag,
        },
        ml_flags: ffi::METH_VARARGS | ffi::METH_KEYWORDS,
        ml_doc: c"make_tag(**pool_options, **tag_config) -> Env".as_ptr(),
    },
];
