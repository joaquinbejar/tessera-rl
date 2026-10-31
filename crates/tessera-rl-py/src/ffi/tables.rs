// SPDX-License-Identifier: MIT OR Apache-2.0

//! The only constructors of C table rows. Function entries take a
//! [`ThunkFn`], so a table cannot hold a raw function pointer.

use core::ffi::{CStr, c_int, c_void};
use core::ptr;

use pyo3::ffi;

use super::thunks::{CFunction, CFunctionKw, Destructor, GetBuffer, ThunkFn};

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

// Each slot constructor fixes its slot id and the type of its payload, so a
// function slot can only hold a thunk and a data slot only its data.

/// The `Py_tp_dealloc` slot.
pub(crate) const fn slot_dealloc(thunk: ThunkFn<Destructor>) -> ffi::PyType_Slot {
    ffi::PyType_Slot {
        slot: ffi::Py_tp_dealloc,
        pfunc: thunk.get() as *mut c_void,
    }
}

/// The `Py_bf_getbuffer` slot.
pub(crate) const fn slot_getbuffer(thunk: ThunkFn<GetBuffer>) -> ffi::PyType_Slot {
    ffi::PyType_Slot {
        slot: ffi::Py_bf_getbuffer,
        pfunc: thunk.get() as *mut c_void,
    }
}

/// The `Py_tp_methods` slot. `methods` must end with [`method_end`]; a
/// table that does not is rejected when the static is evaluated.
pub(crate) const fn slot_methods(methods: &'static [ffi::PyMethodDef]) -> ffi::PyType_Slot {
    let terminated = match methods.last() {
        Some(last) => last.ml_name.is_null(),
        None => false,
    };
    assert!(terminated, "a method table must end with method_end()");
    ffi::PyType_Slot {
        slot: ffi::Py_tp_methods,
        pfunc: methods.as_ptr().cast_mut().cast(),
    }
}

/// The `Py_tp_doc` slot.
pub(crate) const fn slot_doc(doc: &'static CStr) -> ffi::PyType_Slot {
    ffi::PyType_Slot {
        slot: ffi::Py_tp_doc,
        pfunc: doc.as_ptr().cast_mut().cast(),
    }
}

/// The terminating type slot.
pub(crate) const fn slot_end() -> ffi::PyType_Slot {
    ffi::PyType_Slot {
        slot: 0,
        pfunc: ptr::null_mut(),
    }
}
