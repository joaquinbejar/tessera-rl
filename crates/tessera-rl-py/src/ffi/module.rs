// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native` module: its definition, method table and
//! initialization. The `extern "C"` functions here are only reachable
//! through the thunks of `thunks`.

use core::cell::UnsafeCell;
use core::ffi::c_int;
use core::ptr;

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyString;

use super::boundary::{self, Policy};
use super::tables::{method, method_end};
use super::{Table, entry, env_type, thunks, view_type};
use crate::errors::{Kind, error};

pub(super) unsafe extern "C" fn version(
    _module: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|py| Ok(PyString::new(py, env!("CARGO_PKG_VERSION")).into_ptr()))
}

/// `build_profile()`: `"release"` or `"debug"`, as the native module was
/// compiled (benchmark reports record it).
pub(super) unsafe extern "C" fn build_profile(
    _module: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    let text = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    entry(|py| Ok(PyString::new(py, text).into_ptr()))
}

/// `set_fp_policy(name)`: `"canonicalize"` (the default) or `"reject"`
/// (ADR-0041).
pub(super) unsafe extern "C" fn set_fp_policy(
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|py| {
        // SAFETY: METH_VARARGS passes a borrowed tuple.
        let args = unsafe { Bound::from_borrowed_ptr(py, args) };
        let (name,): (String,) = args.extract()?;
        let policy = match name.as_str() {
            "canonicalize" => Policy::Canonicalize,
            "reject" => Policy::Reject,
            other => {
                return Err(error(
                    py,
                    Kind::Argument,
                    format!(
                        "the floating-point policy is \"canonicalize\" or \"reject\", not {other:?}"
                    ),
                ));
            }
        };
        boundary::set_policy(policy);
        Ok(py.None().into_ptr())
    })
}

/// `get_fp_policy()`: the current policy's name.
pub(super) unsafe extern "C" fn get_fp_policy(
    _module: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|py| {
        let name = match boundary::policy() {
            Policy::Canonicalize => "canonicalize",
            Policy::Reject => "reject",
        };
        Ok(PyString::new(py, name).into_ptr())
    })
}

/// `fp_non_canonical_entries()`: entries made with non-default caller
/// controls, process-wide (observability, ADR-0037).
pub(super) unsafe extern "C" fn fp_non_canonical_entries(
    _module: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|py| {
        Ok(tessera_rl_num::fp::non_canonical_entries()
            .into_pyobject(py)?
            .into_ptr())
    })
}

static METHODS: Table<[ffi::PyMethodDef; 8]> = Table([
    method(
        c"version",
        thunks::VERSION,
        ffi::METH_NOARGS,
        c"The Tessera version.",
    ),
    method(
        c"set_fp_policy",
        thunks::SET_FP_POLICY,
        ffi::METH_VARARGS,
        c"set_fp_policy(name): \"canonicalize\" (default) or \"reject\".",
    ),
    method(
        c"get_fp_policy",
        thunks::GET_FP_POLICY,
        ffi::METH_NOARGS,
        c"The floating-point policy for non-default caller controls.",
    ),
    method(
        c"fp_non_canonical_entries",
        thunks::FP_NON_CANONICAL_ENTRIES,
        ffi::METH_NOARGS,
        c"Entries made with non-default caller floating-point controls.",
    ),
    method(
        c"build_profile",
        thunks::BUILD_PROFILE,
        ffi::METH_NOARGS,
        c"The native build profile: release or debug.",
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
        view_type::register(py, &module)?;
        Ok(module.into_ptr())
    })
}
