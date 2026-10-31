// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native` module: its definition, method table and
//! initialization. The `extern "C"` functions here are only reachable
//! through the thunks of `thunks`.

use core::cell::UnsafeCell;
use core::ffi::c_int;
use core::ptr;

use pyo3::ffi;
use pyo3::prelude::*;

use super::tables::{method, method_end};
use super::{Table, entry, env_type, thunks};

pub(super) unsafe extern "C" fn version(
    _module: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    let text = env!("CARGO_PKG_VERSION");
    // SAFETY: called by CPython with the interpreter attached; the bytes
    // are valid UTF-8 of the given length.
    unsafe { ffi::PyUnicode_FromStringAndSize(text.as_ptr().cast(), text.len().cast_signed()) }
}

static METHODS: Table<[ffi::PyMethodDef; 4]> = Table([
    method(
        c"version",
        thunks::VERSION,
        ffi::METH_NOARGS,
        c"The Tessera version.",
    ),
    env_type::MAKERS[0],
    env_type::MAKERS[1],
    method_end(),
]);

/// The module definition, fully initialized at compile time and never
/// written by Rust; CPython owns its `m_base` after `PyModule_Create2`.
static MODULE: Table<UnsafeCell<ffi::PyModuleDef>> = Table(UnsafeCell::new(ffi::PyModuleDef {
    m_base: ffi::PyModuleDef_HEAD_INIT,
    m_name: c"tessera_rl._native".as_ptr(),
    m_doc: c"Native Tessera bindings.".as_ptr(),
    m_size: -1,
    m_methods: METHODS.0.as_ptr().cast_mut(),
    m_slots: ptr::null_mut(),
    m_traverse: None,
    m_clear: None,
    m_free: None,
}));

pub(super) unsafe extern "C" fn init() -> *mut ffi::PyObject {
    entry(|py| {
        // SAFETY: attached; MODULE is a valid static definition that only
        // CPython writes (its `m_base`), as `PyModule_Create2` requires.
        let raw =
            unsafe { ffi::PyModule_Create2(MODULE.0.get(), ffi::PYTHON_API_VERSION as c_int) };
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
