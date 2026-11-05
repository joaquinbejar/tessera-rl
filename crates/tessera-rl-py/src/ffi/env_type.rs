// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native.Env` heap type: object layout, allocation,
//! deallocation, the method table and the module-level constructors. The
//! `extern "C"` functions here are only reachable through the thunks of
//! `thunks`.

use core::ffi::c_int;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use pyo3::ffi;

use super::capi::{self, CResult, Gil, Mine, Ref};
use super::inner::Inner;
use super::tables::{
    method, method_end, method_kw, slot_dealloc, slot_doc, slot_end, slot_methods,
};
use super::{Table, dispose, entry, thunks};
use crate::api;

/// The C layout of an `Env` object.
#[repr(C)]
struct EnvObject {
    ob_base: ffi::PyObject,
    /// Owned; null only before construction and after deallocation.
    inner: *mut Inner,
}

/// The `Env` type, set at module initialization (a borrowed reference: the
/// module keeps it alive).
static TYPE: AtomicPtr<ffi::PyObject> = AtomicPtr::new(ptr::null_mut());

/// Wraps `inner` in a new `Env` object.
fn new_object(g: Gil, inner: Inner) -> CResult<Mine> {
    let ty = TYPE.load(Ordering::Acquire);
    if ty.is_null() {
        return Err(capi::raise(
            g,
            capi::runtime_error(),
            "Env type not initialized",
        ));
    }
    // SAFETY: our heap type (not GC-tracked), whose basic size is
    // `EnvObject`.
    let obj = unsafe { capi::alloc_instance(g, ty) }?;
    let raw = Box::into_raw(Box::new(inner));
    // SAFETY: `obj` is a fresh, zeroed `EnvObject` nobody else sees yet.
    unsafe { (*obj.as_ref().as_ptr().cast::<EnvObject>()).inner = raw };
    Ok(obj)
}

/// The shared state of `slf`.
///
/// # Safety
///
/// `slf` must be `self` of an `Env` method, passed by CPython while
/// attached. Only `&Inner` is handed out: exclusive access to the pool goes
/// through the guard in `Inner`.
unsafe fn inner<'a>(g: Gil, slf: *mut ffi::PyObject) -> CResult<&'a Inner> {
    // SAFETY: per the contract, `slf` is an `EnvObject`.
    let raw = unsafe { (*slf.cast::<EnvObject>()).inner };
    // SAFETY: a non-null `inner` is the live Box made in `new_object`,
    // freed only in `dealloc`, which runs after the last reference.
    unsafe { raw.as_ref() }
        .ok_or_else(|| capi::raise(g, capi::runtime_error(), "Env object not initialized"))
}

/// A method without arguments, calling `$api(g, &Inner)`.
macro_rules! noargs {
    ($impl_fn:ident, $api:path) => {
        pub(super) unsafe extern "C" fn $impl_fn(
            slf: *mut ffi::PyObject,
            _args: *mut ffi::PyObject,
        ) -> *mut ffi::PyObject {
            entry(|g| {
                // SAFETY: CPython passes `self` of an `Env` method.
                let inner = unsafe { inner(g, slf) }?;
                $api(g, inner)
            })
        }
    };
}

/// A method with positional arguments, calling `$api(g, &Inner, args)`.
macro_rules! varargs {
    ($impl_fn:ident, $api:path) => {
        pub(super) unsafe extern "C" fn $impl_fn(
            slf: *mut ffi::PyObject,
            args: *mut ffi::PyObject,
        ) -> *mut ffi::PyObject {
            entry(|g| {
                // SAFETY: CPython passes `self` of an `Env` method.
                let inner = unsafe { inner(g, slf) }?;
                // SAFETY: METH_VARARGS passes a borrowed tuple, alive for
                // the call.
                let args = unsafe { Ref::from_ptr(args) }
                    .ok_or_else(|| capi::raise(g, capi::type_error(), "missing arguments"))?;
                $api(g, inner, args)
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
        c"reset(seed=None, ids=None)",
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
        let g = unsafe { Gil::assume() };
        // Dropping a pool joins its workers: detached. A panic is
        // contained and its payload disposed of without unwinding.
        if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            g.detach(move || drop(boxed));
        })) {
            dispose(payload);
        }
    }
    // SAFETY: `obj` is a live `Env` instance, in its tp_dealloc.
    unsafe { capi::free_instance(obj) };
}

static SLOTS: Table<[ffi::PyType_Slot; 4]> = Table([
    slot_dealloc(thunks::DEALLOC),
    slot_methods(&METHODS.0),
    slot_doc(c"A pool of Tessera environments (use tessera_rl.make)."),
    slot_end(),
]);

/// Creates the `Env` type and adds it to `module`.
pub(super) fn register(g: Gil, module: Ref<'_>) -> CResult<()> {
    let mut spec = ffi::PyType_Spec {
        name: c"tessera_rl._native.Env".as_ptr(),
        basicsize: c_int::try_from(core::mem::size_of::<EnvObject>()).unwrap_or(c_int::MAX),
        itemsize: 0,
        // Not GC-tracked (no Py_TPFLAGS_HAVE_GC): instances hold no Python
        // objects, and allocating one never runs `gc.collect` (capi).
        flags: core::ffi::c_uint::try_from(ffi::Py_TPFLAGS_DEFAULT).unwrap_or(0),
        slots: SLOTS.0.as_ptr().cast_mut(),
    };
    let ty = capi::type_from_spec(g, &mut spec)?;
    capi::module_add(g, module, c"Env", &ty)?;
    TYPE.store(ty.as_ref().as_ptr(), Ordering::Release);
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
            entry(|g| {
                // SAFETY: METH_VARARGS | METH_KEYWORDS passes a borrowed dict
                // or null, alive for the call.
                let kwargs = unsafe { Ref::from_ptr(kwargs) };
                let inner = $api(g, kwargs)?;
                new_object(g, inner)
            })
        }
    };
}

maker!(make_cartpole, api::make_cartpole);
maker!(make_tag, api::make_tag);
maker!(make_connect4, api::make_connect4);
maker!(make_leduc, api::make_leduc);

/// The module-level constructors, for the module's method table.
pub(super) const MAKERS: [ffi::PyMethodDef; 4] = [
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
    method_kw(
        c"make_connect4",
        thunks::MAKE_CONNECT4,
        c"make_connect4(**pool_options) -> Env",
    ),
    method_kw(
        c"make_leduc",
        thunks::MAKE_LEDUC,
        c"make_leduc(**pool_options) -> Env",
    ),
];
