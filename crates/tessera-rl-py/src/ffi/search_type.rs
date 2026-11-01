// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native.Search` heap type: a PUCT search (ADR-0043),
//! its object layout, allocation, deallocation, method table and module
//! constructor. The `extern "C"` functions here are only reachable through
//! the thunks of `thunks`.

use core::ffi::c_int;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use pyo3::ffi;

use super::capi::{self, CResult, Gil, Mine, Ref};
use super::tables::{
    method, method_end, method_kw, slot_dealloc, slot_doc, slot_end, slot_methods,
};
use super::{Table, dispose, entry, thunks};
use crate::search_api::{self as api, SearchInner};

/// The C layout of a `Search` object.
#[repr(C)]
struct SearchObject {
    ob_base: ffi::PyObject,
    /// Owned; null only before construction and after deallocation.
    inner: *mut SearchInner,
}

/// The `Search` type, set at module initialization (a borrowed reference:
/// the module keeps it alive).
static TYPE: AtomicPtr<ffi::PyObject> = AtomicPtr::new(ptr::null_mut());

/// Wraps `inner` in a new `Search` object.
fn new_object(g: Gil, inner: SearchInner) -> CResult<Mine> {
    let ty = TYPE.load(Ordering::Acquire);
    if ty.is_null() {
        return Err(capi::raise(
            g,
            capi::runtime_error(),
            "Search type not initialized",
        ));
    }
    // SAFETY: our heap type (not GC-tracked), whose basic size is
    // `SearchObject`.
    let obj = unsafe { capi::alloc_instance(g, ty) }?;
    let raw = Box::into_raw(Box::new(inner));
    // SAFETY: `obj` is a fresh, zeroed `SearchObject` nobody else sees yet.
    unsafe { (*obj.as_ref().as_ptr().cast::<SearchObject>()).inner = raw };
    Ok(obj)
}

/// The shared state of `slf`.
///
/// # Safety
///
/// `slf` must be `self` of a `Search` method, passed by CPython while
/// attached. Only `&SearchInner` is handed out: exclusive access goes
/// through the guard in `Inner`.
unsafe fn inner<'a>(g: Gil, slf: *mut ffi::PyObject) -> CResult<&'a SearchInner> {
    // SAFETY: per the contract, `slf` is a `SearchObject`.
    let raw = unsafe { (*slf.cast::<SearchObject>()).inner };
    // SAFETY: a non-null `inner` is the live Box made in `new_object`,
    // freed only in `dealloc`, which runs after the last reference.
    unsafe { raw.as_ref() }
        .ok_or_else(|| capi::raise(g, capi::runtime_error(), "Search object not initialized"))
}

/// A method without arguments, calling `$api(g, &SearchInner)`.
macro_rules! noargs {
    ($impl_fn:ident, $api:path) => {
        pub(super) unsafe extern "C" fn $impl_fn(
            slf: *mut ffi::PyObject,
            _args: *mut ffi::PyObject,
        ) -> *mut ffi::PyObject {
            entry(|g| {
                // SAFETY: CPython passes `self` of an `Search` method.
                let inner = unsafe { inner(g, slf) }?;
                $api(g, inner)
            })
        }
    };
}

/// A method with positional arguments, calling `$api(g, &SearchInner, args)`.
macro_rules! varargs {
    ($impl_fn:ident, $api:path) => {
        pub(super) unsafe extern "C" fn $impl_fn(
            slf: *mut ffi::PyObject,
            args: *mut ffi::PyObject,
        ) -> *mut ffi::PyObject {
            entry(|g| {
                // SAFETY: CPython passes `self` of an `Search` method.
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
noargs!(root_visits, api::root_visits);
noargs!(calls, api::calls);
noargs!(tree_hash, api::tree_hash);
varargs!(set_root, api::set_root);
varargs!(run, api::run);
varargs!(select_action, api::select_action);

static METHODS: Table<[ffi::PyMethodDef; 8]> = Table([
    method(
        c"layout",
        thunks::SEARCH_LAYOUT,
        ffi::METH_NOARGS,
        c"Actions and observation bytes per leaf.",
    ),
    method(
        c"set_root",
        thunks::SEARCH_SET_ROOT,
        ffi::METH_VARARGS,
        c"set_root(record: bytes): replaces the tree by this root.",
    ),
    method(
        c"run",
        thunks::SEARCH_RUN,
        ffi::METH_VARARGS,
        c"run(evaluator): one search call (all or nothing).",
    ),
    method(
        c"root_visits",
        thunks::SEARCH_ROOT_VISITS,
        ffi::METH_NOARGS,
        c"The root's visit counts.",
    ),
    method(
        c"select_action",
        thunks::SEARCH_SELECT_ACTION,
        ffi::METH_VARARGS,
        c"select_action(temperature: float) -> int",
    ),
    method(
        c"calls",
        thunks::SEARCH_CALLS,
        ffi::METH_NOARGS,
        c"Completed search calls.",
    ),
    method(
        c"tree_hash",
        thunks::SEARCH_TREE_HASH,
        ffi::METH_NOARGS,
        c"A digest of the whole tree.",
    ),
    method_end(),
]);

/// `tp_dealloc`: a cleanup entry, which never fails or rejects (ADR-0041).
/// CPython calls it once, attached, when the reference count reaches zero.
pub(super) unsafe extern "C" fn dealloc(obj: *mut ffi::PyObject) {
    let env = obj.cast::<SearchObject>();
    // SAFETY: `obj` is a live `SearchObject` (above).
    let raw = unsafe { (*env).inner };
    // SAFETY: as above; cleared before the box is freed.
    unsafe { (*env).inner = ptr::null_mut() };
    if !raw.is_null() {
        // SAFETY: a non-null `inner` is the Box made in `new_object`, owned
        // by this object; no entry can run on an object being deallocated.
        let boxed = unsafe { Box::from_raw(raw) };
        // SAFETY: CPython calls tp_dealloc attached.
        let g = unsafe { Gil::assume() };
        // Dropping the search frees its pools: detached. A panic is
        // contained and its payload disposed of without unwinding.
        if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            g.detach(move || drop(boxed));
        })) {
            dispose(payload);
        }
    }
    // SAFETY: `obj` is a live `Search` instance, in its tp_dealloc.
    unsafe { capi::free_instance(obj) };
}

static SLOTS: Table<[ffi::PyType_Slot; 4]> = Table([
    slot_dealloc(thunks::SEARCH_DEALLOC),
    slot_methods(&METHODS.0),
    slot_doc(c"A PUCT search over an environment (use tessera_rl.search.Search)."),
    slot_end(),
]);

/// Creates the `Search` type and adds it to `module`.
pub(super) fn register(g: Gil, module: Ref<'_>) -> CResult<()> {
    let mut spec = ffi::PyType_Spec {
        name: c"tessera_rl._native.Search".as_ptr(),
        basicsize: c_int::try_from(core::mem::size_of::<SearchObject>()).unwrap_or(c_int::MAX),
        itemsize: 0,
        // Not GC-tracked: instances hold no Python objects (the evaluator
        // is borrowed for one `run` call only).
        flags: core::ffi::c_uint::try_from(ffi::Py_TPFLAGS_DEFAULT).unwrap_or(0),
        slots: SLOTS.0.as_ptr().cast_mut(),
    };
    let ty = capi::type_from_spec(g, &mut spec)?;
    capi::module_add(g, module, c"Search", &ty)?;
    TYPE.store(ty.as_ref().as_ptr(), Ordering::Release);
    Ok(())
}

/// `make_search(**kwargs) -> Search`.
pub(super) unsafe extern "C" fn make_search(
    _module: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|g| {
        // SAFETY: METH_VARARGS | METH_KEYWORDS passes a borrowed dict or
        // null, alive for the call.
        let kwargs = unsafe { Ref::from_ptr(kwargs) };
        let inner = api::make_search(g, kwargs)?;
        new_object(g, inner)
    })
}

/// The module-level constructor, for the module's method table.
pub(super) const MAKER: ffi::PyMethodDef = method_kw(
    c"make_search",
    thunks::MAKE_SEARCH,
    c"make_search(**search_options) -> Search",
);
