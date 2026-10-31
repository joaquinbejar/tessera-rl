// SPDX-License-Identifier: MIT OR Apache-2.0

//! The only constructors of C table rows. Function entries take a
//! [`ThunkFn`], so a table cannot hold a raw function pointer.

use core::ffi::{CStr, c_int, c_void};
use core::ptr;

use pyo3::ffi;

use super::thunks::{CFunction, CFunctionKw, Destructor, ThunkFn};

/// A method row with positional arguments (`METH_NOARGS` or `METH_VARARGS`).
pub(crate) const fn method(
    name: &'static CStr,
    thunk: ThunkFn<CFunction>,
    flags: c_int,
    doc: &'static CStr,
) -> ffi::PyMethodDef {
    ffi::PyMethodDef {
        ml_name: name.as_ptr(),
        ml_meth: ffi::PyMethodDefPointer {
            PyCFunction: thunk.get(),
        },
        ml_flags: flags,
        ml_doc: doc.as_ptr(),
    }
}

/// A method row with keyword arguments (`METH_VARARGS | METH_KEYWORDS`).
pub(crate) const fn method_kw(
    name: &'static CStr,
    thunk: ThunkFn<CFunctionKw>,
    doc: &'static CStr,
) -> ffi::PyMethodDef {
    ffi::PyMethodDef {
        ml_name: name.as_ptr(),
        ml_meth: ffi::PyMethodDefPointer {
            PyCFunctionWithKeywords: thunk.get(),
        },
        ml_flags: ffi::METH_VARARGS | ffi::METH_KEYWORDS,
        ml_doc: doc.as_ptr(),
    }
}

/// The terminating method row.
pub(crate) const fn method_end() -> ffi::PyMethodDef {
    ffi::PyMethodDef::zeroed()
}

/// A type slot holding a destructor thunk (`Py_tp_dealloc`).
pub(crate) const fn slot_destructor(slot: c_int, thunk: ThunkFn<Destructor>) -> ffi::PyType_Slot {
    ffi::PyType_Slot {
        slot,
        pfunc: thunk.get() as *mut c_void,
    }
}

/// A type slot holding data, not a function (`Py_tp_methods`, `Py_tp_doc`).
pub(crate) const fn slot_data(slot: c_int, data: *const c_void) -> ffi::PyType_Slot {
    ffi::PyType_Slot {
        slot,
        pfunc: data.cast_mut(),
    }
}

/// The terminating type slot.
pub(crate) const fn slot_end() -> ffi::PyType_Slot {
    ffi::PyType_Slot {
        slot: 0,
        pfunc: ptr::null_mut(),
    }
}
