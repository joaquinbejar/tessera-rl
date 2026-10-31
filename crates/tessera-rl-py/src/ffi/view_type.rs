// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native.OutputView` heap type: one borrowed output
//! region (ADR-0003), exported read-only through the buffer protocol and
//! DLPack. A view holds an `Arc` clone of the output storage, so its memory
//! outlives the environment; its contents are only guaranteed until the
//! next engine call that writes outputs, which `is_stale` reports.
//!
//! The `extern "C"` functions here are only reachable through the thunks
//! of `thunks`.

use core::ffi::c_int;
use core::ptr;
use std::sync::Arc;

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple, PyType};
use tessera_rl_engine::{OutputDtype, OutputStorage};

use super::tables::{
    method, method_end, method_kw, slot_dealloc, slot_doc, slot_end, slot_getbuffer, slot_methods,
};
use super::{Table, dispose, dlpack, entry, free_heap_object, thunks};
use crate::errors::{Kind, error};
use crate::view::{Elem, ViewSpec};

/// What a view object owns.
struct View {
    storage: Arc<OutputStorage>,
    data: *const u8,
    bytes: usize,
    generation: u64,
    elem: Elem,
    shape: Box<[isize]>,
    strides: Box<[isize]>,
}

/// The C layout of an `OutputView` object.
#[repr(C)]
struct ViewObject {
    ob_base: ffi::PyObject,
    /// Owned; null only before construction and after deallocation.
    view: *mut View,
}

static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();

const fn region_itemsize(dtype: OutputDtype) -> usize {
    match dtype {
        OutputDtype::F32 | OutputDtype::U32 => 4,
        _ => 1,
    }
}

/// A new view of `spec` in `storage`, at the storage's current generation.
pub(crate) fn new_view<'py>(
    py: Python<'py>,
    storage: &Arc<OutputStorage>,
    spec: &ViewSpec,
) -> PyResult<Bound<'py, PyAny>> {
    let region = storage
        .region(spec.field)
        .ok_or_else(|| error(py, Kind::Argument, format!("no output {:?}", spec.field)))?;
    let bytes = region.len.checked_mul(region_itemsize(region.dtype));
    let elements = spec
        .shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d));
    if bytes.is_none() || elements.and_then(|e| e.checked_mul(spec.elem.itemsize)) != bytes {
        return Err(error(
            py,
            Kind::Argument,
            format!("output {:?} does not match its layout", spec.field),
        ));
    }
    let too_large = || {
        pyo3::exceptions::PyBufferError::new_err(format!(
            "output {:?} is too large to export",
            spec.field
        ))
    };
    let shape: Box<[isize]> = spec
        .shape
        .iter()
        .map(|&d| isize::try_from(d).map_err(|_| too_large()))
        .collect::<PyResult<_>>()?;
    // Row-major strides, in bytes.
    let mut strides = vec![0isize; shape.len()];
    let mut acc = isize::try_from(spec.elem.itemsize).map_err(|_| too_large())?;
    for (stride, &dim) in strides.iter_mut().zip(shape.iter()).rev() {
        *stride = acc;
        acc = acc.checked_mul(dim).ok_or_else(too_large)?;
    }
    let bytes = bytes.ok_or_else(too_large)?;
    // The buffer protocol describes sizes and dimensions as `Py_ssize_t`
    // and `int`: refuse now what `getbuffer` could not describe.
    isize::try_from(bytes).map_err(|_| too_large())?;
    c_int::try_from(shape.len()).map_err(|_| too_large())?;
    let view = View {
        storage: Arc::clone(storage),
        data: region.ptr,
        bytes,
        generation: storage.generation(),
        elem: spec.elem,
        shape,
        strides: strides.into_boxed_slice(),
    };
    let ty = TYPE.get(py).ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("OutputView type not initialized")
    })?;
    // SAFETY: attached; our heap type, whose basic size is `ViewObject`;
    // PyType_GenericAlloc zero-fills and takes a reference to the type.
    let obj = unsafe { ffi::PyType_GenericAlloc(ty.as_ptr().cast(), 0) };
    if obj.is_null() {
        return Err(PyErr::fetch(py));
    }
    // SAFETY: `obj` is a fresh, zeroed `ViewObject` nobody else sees yet.
    unsafe { (*obj.cast::<ViewObject>()).view = Box::into_raw(Box::new(view)) };
    // SAFETY: `obj` is a new reference.
    Ok(unsafe { Bound::from_owned_ptr(py, obj) })
}

/// The view of `slf`.
///
/// # Safety
///
/// `slf` must be an `OutputView` passed by CPython while attached.
unsafe fn view<'a>(slf: *mut ffi::PyObject) -> PyResult<&'a View> {
    // SAFETY: per the contract, `slf` is a `ViewObject`.
    let raw = unsafe { (*slf.cast::<ViewObject>()).view };
    // SAFETY: a non-null `view` is the live Box made in `new_view`, freed
    // only in `dealloc`; it is never mutated after construction.
    unsafe { raw.as_ref() }
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("OutputView not initialized"))
}

/// `bf_getbuffer`: a read-only, C-contiguous export. A request for a
/// writable buffer, or for Fortran order the view does not have, raises
/// `BufferError` (with `obj` null).
pub(super) unsafe extern "C" fn getbuffer(
    slf: *mut ffi::PyObject,
    out: *mut ffi::Py_buffer,
    flags: c_int,
) -> c_int {
    let done = entry(|_py| {
        // SAFETY: CPython passes an `OutputView` as `slf`.
        let v = unsafe { view(slf) }?;
        if flags & ffi::PyBUF_WRITABLE != 0 {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "borrowed outputs are read-only",
            ));
        }
        let undescribable = || pyo3::exceptions::PyBufferError::new_err("view too large to export");
        let len = isize::try_from(v.bytes).map_err(|_| undescribable())?;
        let itemsize = isize::try_from(v.elem.itemsize).map_err(|_| undescribable())?;
        let ndim = c_int::try_from(v.shape.len()).map_err(|_| undescribable())?;
        // Views are C-contiguous, which satisfies C and ANY requests. They
        // are also F-contiguous only when at most one dimension exceeds 1
        // (or the view is empty): refuse an F request otherwise.
        let f_contiguous = v.shape.contains(&0) || v.shape.iter().filter(|&&d| d > 1).count() <= 1;
        if flags & ffi::PyBUF_F_CONTIGUOUS == ffi::PyBUF_F_CONTIGUOUS && !f_contiguous {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "borrowed outputs are C-contiguous, not Fortran-contiguous",
            ));
        }
        let nd = flags & ffi::PyBUF_ND == ffi::PyBUF_ND;
        let strided = flags & ffi::PyBUF_STRIDES == ffi::PyBUF_STRIDES;
        let buffer = ffi::Py_buffer {
            buf: v.data.cast_mut().cast(),
            obj: slf,
            len,
            itemsize,
            readonly: 1,
            ndim: if nd { ndim } else { 1 },
            format: if flags & ffi::PyBUF_FORMAT != 0 {
                v.elem.format.as_ptr().cast_mut()
            } else {
                ptr::null_mut()
            },
            shape: if nd {
                v.shape.as_ptr().cast_mut()
            } else {
                ptr::null_mut()
            },
            strides: if strided {
                v.strides.as_ptr().cast_mut()
            } else {
                ptr::null_mut()
            },
            suboffsets: ptr::null_mut(),
            internal: ptr::null_mut(),
        };
        // SAFETY: the buffer keeps a reference to its exporter, released by
        // PyBuffer_Release.
        unsafe { ffi::Py_INCREF(slf) };
        // SAFETY: `out` is the caller's Py_buffer to fill.
        unsafe { out.write(buffer) };
        Ok(slf)
    });
    if done.is_null() {
        if !out.is_null() {
            // SAFETY: on failure the protocol requires `obj` to be null.
            unsafe { (*out).obj = ptr::null_mut() };
        }
        -1
    } else {
        0
    }
}

/// `generation()`: the write generation the view was taken at.
pub(super) unsafe extern "C" fn generation(
    slf: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|py| {
        // SAFETY: CPython passes `self` of an `OutputView` method.
        let v = unsafe { view(slf) }?;
        Ok(v.generation.into_pyobject(py)?.into_ptr())
    })
}

/// `is_stale()`: whether an engine call has written outputs since the view
/// was taken (its borrow window ended).
pub(super) unsafe extern "C" fn is_stale(
    slf: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|py| {
        // SAFETY: CPython passes `self` of an `OutputView` method.
        let v = unsafe { view(slf) }?;
        let stale = v.storage.generation() != v.generation;
        Ok(pyo3::types::PyBool::new(py, stale).to_owned().into_ptr())
    })
}

/// `__dlpack_device__()`: always the CPU, `(kDLCPU, 0)`.
pub(super) unsafe extern "C" fn dlpack_device(
    _slf: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|py| Ok((1i32, 0i32).into_pyobject(py)?.into_ptr()))
}

fn kwarg<'py>(
    kwargs: Option<&Bound<'py, PyDict>>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(kwargs
        .map(|d| d.get_item(key))
        .transpose()?
        .flatten()
        .filter(|v| !v.is_none()))
}

/// `__dlpack__(*, stream=None, max_version=None, dl_device=None,
/// copy=None)`: a capsule of the view, versioned and read-only when the
/// consumer accepts DLPack 1.x, legacy otherwise. A legacy capsule has no
/// read-only flag: its consumer gets a writable alias, and writing through
/// it corrupts the outputs until the next call overwrites them.
pub(super) unsafe extern "C" fn dlpack(
    slf: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|py| {
        // SAFETY: CPython passes `self` of an `OutputView` method.
        let v = unsafe { view(slf) }?;
        // SAFETY: METH_VARARGS | METH_KEYWORDS passes a borrowed tuple.
        let args = unsafe { Bound::from_borrowed_ptr(py, args) };
        // SAFETY: and a borrowed dict or null.
        let kwargs = unsafe { Bound::from_borrowed_ptr_or_opt(py, kwargs) };
        if args.cast::<PyTuple>().map_err(PyErr::from)?.len() != 0 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "__dlpack__ takes keyword arguments only",
            ));
        }
        let kwargs = kwargs
            .as_ref()
            .map(|k| k.cast::<PyDict>())
            .transpose()
            .map_err(PyErr::from)?;
        if let Some(stream) = kwarg(kwargs, "stream")?
            && stream.extract::<i64>().ok() != Some(-1)
        {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "CPU outputs take no stream",
            ));
        }
        if let Some(device) = kwarg(kwargs, "dl_device")?
            && device.extract::<(i32, i32)>().ok() != Some((1, 0))
        {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "outputs live on the CPU (kDLCPU, 0)",
            ));
        }
        if let Some(copy) = kwarg(kwargs, "copy")?
            && copy.is_truthy()?
        {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "borrowed views do not copy; use outputs=\"owned\"",
            ));
        }
        let versioned = match kwarg(kwargs, "max_version")? {
            Some(max) => max.extract::<(u32, u32)>()?.0 >= 1,
            None => false,
        };
        let owner: dlpack::Owner = v.storage.clone();
        let src = dlpack::Source {
            owner: &owner,
            data: v.data,
            dl_code: v.elem.dl_code,
            itemsize: v.elem.itemsize,
            shape: &v.shape,
        };
        dlpack::capsule(py, &src, versioned, true)
    })
}

static METHODS: Table<[ffi::PyMethodDef; 5]> = Table([
    method_kw(
        c"__dlpack__",
        thunks::VIEW_DLPACK,
        c"__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None): a read-only DLPack 1.0 capsule when max_version >= (1, 0); otherwise a legacy capsule, which cannot be marked read-only (writing through it corrupts the outputs until the next call).",
    ),
    method(
        c"__dlpack_device__",
        thunks::VIEW_DLPACK_DEVICE,
        ffi::METH_NOARGS,
        c"(kDLCPU, 0)",
    ),
    method(
        c"generation",
        thunks::VIEW_GENERATION,
        ffi::METH_NOARGS,
        c"The write generation this view was taken at.",
    ),
    method(
        c"is_stale",
        thunks::VIEW_IS_STALE,
        ffi::METH_NOARGS,
        c"Whether outputs were written since this view was taken.",
    ),
    method_end(),
]);

/// `tp_dealloc`: a cleanup entry, which never fails (ADR-0041).
pub(super) unsafe extern "C" fn dealloc(obj: *mut ffi::PyObject) {
    // SAFETY: `obj` is a live `ViewObject`.
    let raw = unsafe { (*obj.cast::<ViewObject>()).view };
    // SAFETY: as above; cleared before the box is freed.
    unsafe { (*obj.cast::<ViewObject>()).view = ptr::null_mut() };
    if !raw.is_null() {
        // SAFETY: the Box made in `new_view`, owned by this object.
        let boxed = unsafe { Box::from_raw(raw) };
        if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(boxed)))
        {
            dispose(payload);
        }
    }
    // SAFETY: `obj` is a live instance of a heap type of this module.
    unsafe { free_heap_object(obj) };
}

static SLOTS: Table<[ffi::PyType_Slot; 5]> = Table([
    slot_dealloc(thunks::VIEW_DEALLOC),
    slot_getbuffer(thunks::VIEW_GETBUFFER),
    slot_methods(&METHODS.0),
    slot_doc(c"A borrowed, read-only view of one output buffer (valid until the next call)."),
    slot_end(),
]);

/// Creates the `OutputView` type and adds it to `module`.
pub(super) fn register(py: Python<'_>, module: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    let mut spec = ffi::PyType_Spec {
        name: c"tessera_rl._native.OutputView".as_ptr(),
        basicsize: c_int::try_from(core::mem::size_of::<ViewObject>()).unwrap_or(c_int::MAX),
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
    module.add("OutputView", &ty)?;
    let _ = TYPE.set(py, ty.unbind());
    Ok(())
}
