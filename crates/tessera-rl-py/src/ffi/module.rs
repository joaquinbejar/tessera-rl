// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native` module: its definition, method table and
//! initialization. The `extern "C"` functions here are only reachable
//! through the thunks of `thunks`.

use core::cell::UnsafeCell;
use core::ptr;

use pyo3::ffi;

use super::boundary::{self, Policy};
use super::capi::{self, Ref};
use super::tables::{method, method_end};
use super::{
    Table, entry, env_type, fork, search_group_type, search_type, snapshot_type, thunks, view_type,
};
use crate::errors::{Kind, error};

pub(super) unsafe extern "C" fn version(
    _module: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|g| capi::string(g, env!("CARGO_PKG_VERSION")))
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
    entry(|g| capi::string(g, text))
}

/// `fp_boundary()`: which native floating-point boundary this build
/// carries (ADR-0041), or that there is none on this target.
pub(super) unsafe extern "C" fn fp_boundary(
    _module: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|g| capi::string(g, thunks::BOUNDARY))
}

/// `set_fp_policy(name)`: `"canonicalize"` (the default) or `"reject"`
/// (ADR-0041).
pub(super) unsafe extern "C" fn set_fp_policy(
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|g| {
        // SAFETY: METH_VARARGS passes a borrowed tuple, alive for the call.
        let args = unsafe { Ref::from_ptr(args) }
            .ok_or_else(|| capi::raise(g, capi::type_error(), "missing arguments"))?;
        let items = capi::tuple_items(g, args, "the arguments")?;
        let [name] = items.as_slice() else {
            return Err(error(
                g,
                Kind::Argument,
                "set_fp_policy takes one argument".to_owned(),
            ));
        };
        let policy = match capi::to_string(g, *name, "the policy")?.as_str() {
            "canonicalize" => Policy::Canonicalize,
            "reject" => Policy::Reject,
            other => {
                return Err(error(
                    g,
                    Kind::Argument,
                    format!(
                        "the floating-point policy is \"canonicalize\" or \"reject\", not {other:?}"
                    ),
                ));
            }
        };
        boundary::set_policy(policy);
        Ok(capi::none(g))
    })
}

/// `get_fp_policy()`: the current policy's name.
pub(super) unsafe extern "C" fn get_fp_policy(
    _module: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|g| {
        let name = match boundary::policy() {
            Policy::Canonicalize => "canonicalize",
            Policy::Reject => "reject",
        };
        capi::string(g, name)
    })
}

/// `fp_non_canonical_entries()`: entries made with non-default caller
/// controls, process-wide (observability, ADR-0037).
pub(super) unsafe extern "C" fn fp_non_canonical_entries(
    _module: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|g| capi::int_u64(g, tessera_rl_num::fp::non_canonical_entries()))
}

/// `_fail_after(n)` (test hooks only): the `n`-th next container insertion
/// fails.
#[cfg(feature = "test-hooks")]
pub(super) unsafe extern "C" fn fail_after(
    _module: *mut ffi::PyObject,
    arg: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|g| {
        // SAFETY: METH_O passes a borrowed object, alive for the call.
        let arg = unsafe { Ref::from_ptr(arg) }
            .ok_or_else(|| capi::raise(g, capi::type_error(), "missing argument"))?;
        capi::fail_after(capi::to_u32(g, arg, "n")?);
        Ok(capi::none(g))
    })
}

const METHOD_COUNT: usize = if cfg!(feature = "test-hooks") { 17 } else { 16 };

static METHODS: Table<[ffi::PyMethodDef; METHOD_COUNT]> = Table([
    method(
        c"_after_fork_child",
        thunks::AFTER_FORK,
        ffi::METH_NOARGS,
        c"The after_in_child hook of the fork guard.",
    ),
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
        c"fp_boundary",
        thunks::FP_BOUNDARY,
        ffi::METH_NOARGS,
        c"Which native floating-point boundary this build carries.",
    ),
    method(
        c"build_profile",
        thunks::BUILD_PROFILE,
        ffi::METH_NOARGS,
        c"The native build profile: release or debug.",
    ),
    env_type::MAKERS[0],
    env_type::MAKERS[1],
    env_type::MAKERS[2],
    env_type::MAKERS[3],
    env_type::MAKERS[4],
    env_type::MAKERS[5],
    env_type::MAKERS[6],
    search_type::MAKER,
    search_group_type::MAKER,
    #[cfg(feature = "test-hooks")]
    method(
        c"_fail_after",
        thunks::FAIL_AFTER,
        ffi::METH_O,
        c"Test hook: the n-th next container insertion fails.",
    ),
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
    entry(|g| {
        // SAFETY: MODULE is a valid static definition that only CPython
        // writes (its `m_base`), as `PyModule_Create2` requires.
        let module = unsafe { capi::module_create(g, MODULE.0.get()) }?;
        crate::errors::register(g, module.as_ref())?;
        env_type::register(g, module.as_ref())?;
        view_type::register(g, module.as_ref())?;
        snapshot_type::register(g, module.as_ref())?;
        search_type::register(g, module.as_ref())?;
        search_group_type::register(g, module.as_ref())?;
        fork::register(g, module.as_ref())?;
        Ok(module)
    })
}
