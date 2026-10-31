// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native` module: its definition, method table and
//! initialization, all reached through boundary thunks.

use core::ffi::c_int;
use core::ptr;

use pyo3::ffi;

use super::{Table, entry, env_type};
use pyo3::prelude::*;

unsafe extern "C" fn version(
    _module: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    let text = env!("CARGO_PKG_VERSION");
    // SAFETY: called by CPython with the interpreter attached; the bytes
    // are valid UTF-8 of the given length.
    unsafe { ffi::PyUnicode_FromStringAndSize(text.as_ptr().cast(), text.len().cast_signed()) }
}

thunk!(thunk_version => version: fn(*mut ffi::PyObject, *mut ffi::PyObject) -> *mut ffi::PyObject);

static METHODS: Table<[ffi::PyMethodDef; 4]> = Table([
    ffi::PyMethodDef {
        ml_name: c"version".as_ptr(),
        ml_meth: ffi::PyMethodDefPointer {
            PyCFunction: thunk_version,
        },
        ml_flags: ffi::METH_NOARGS,
        ml_doc: c"The Tessera version.".as_ptr(),
    },
    env_type::MAKERS[0],
    env_type::MAKERS[1],
    ffi::PyMethodDef::zeroed(),
]);

static mut MODULE: ffi::PyModuleDef = ffi::PyModuleDef {
    m_base: ffi::PyModuleDef_HEAD_INIT,
    m_name: c"tessera_rl._native".as_ptr(),
    m_doc: c"Native Tessera bindings.".as_ptr(),
    m_size: -1,
    m_methods: ptr::null_mut(),
    m_slots: ptr::null_mut(),
    m_traverse: None,
    m_clear: None,
    m_free: None,
};

unsafe extern "C" fn init() -> *mut ffi::PyObject {
    entry(|py| {
        // CPython calls the module initializer once, attached, before any
        // other entry; MODULE is only touched here.
        let def = ptr::addr_of_mut!(MODULE);
        // SAFETY: single-threaded initialization (above); `def` is valid.
        unsafe { (*def).m_methods = METHODS.0.as_ptr().cast_mut() };
        // SAFETY: attached; `def` is a static module definition.
        let raw = unsafe { ffi::PyModule_Create2(def, ffi::PYTHON_API_VERSION as c_int) };
        // SAFETY: `raw` is a new reference or null with an exception set.
        let module = unsafe { Bound::from_owned_ptr_or_err(py, raw) }?;
        let module = module
            .cast_into::<pyo3::types::PyModule>()
            .map_err(PyErr::from)?;
        crate::errors::register(py, &module)?;
        env_type::register(py, &module)?;
        Ok(module.into_ptr())
    })
}

thunk!(
    /// The module initializer CPython looks up by name.
    #[unsafe(no_mangle)]
    PyInit__native => init: fn() -> *mut ffi::PyObject
);
