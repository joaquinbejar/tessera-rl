// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native.SearchGroup` heap type: searches sharing
//! evaluator batches (ADR-0045),
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
use crate::search_group_api::{self as api, GroupInner};

/// The C layout of a `SearchGroup` object.
#[repr(C)]
struct GroupObject {
    ob_base: ffi::PyObject,
    /// Owned; null only before construction and after deallocation.
    inner: *mut GroupInner,
}

/// The `SearchGroup` type, set at module initialization (a borrowed reference:
/// the module keeps it alive).
static TYPE: AtomicPtr<ffi::PyObject> = AtomicPtr::new(ptr::null_mut());

/// Wraps `inner` in a new `SearchGroup` object.
fn new_object(g: Gil, inner: GroupInner) -> CResult<Mine> {
    let ty = TYPE.load(Ordering::Acquire);
    if ty.is_null() {
        return Err(capi::raise(
            g,
            capi::runtime_error(),
            "SearchGroup type not initialized",
        ));
    }
    // SAFETY: our heap type (not GC-tracked), whose basic size is
    // `GroupObject`.
    let obj = unsafe { capi::alloc_instance(g, ty) }?;
    let raw = Box::into_raw(Box::new(inner));
    // SAFETY: `obj` is a fresh, zeroed `GroupObject` nobody else sees yet.
    unsafe { (*obj.as_ref().as_ptr().cast::<GroupObject>()).inner = raw };
    Ok(obj)
}

/// The shared state of `slf`.
///
/// # Safety
///
/// `slf` must be `self` of a `SearchGroup` method, passed by CPython while
/// attached. Only `&GroupInner` is handed out: exclusive access goes
/// through the guard in `Inner`.
unsafe fn inner<'a>(g: Gil, slf: *mut ffi::PyObject) -> CResult<&'a GroupInner> {
    // SAFETY: per the contract, `slf` is a `GroupObject`.
    let raw = unsafe { (*slf.cast::<GroupObject>()).inner };
    // SAFETY: a non-null `inner` is the live Box made in `new_object`,
    // freed only in `dealloc`, which runs after the last reference.
    unsafe { raw.as_ref() }.ok_or_else(|| {
        capi::raise(
            g,
            capi::runtime_error(),
            "SearchGroup object not initialized",
        )
    })
}

/// A method without arguments, calling `$api(g, &GroupInner)`.
macro_rules! noargs {
    ($impl_fn:ident, $api:path) => {
        pub(super) unsafe extern "C" fn $impl_fn(
            slf: *mut ffi::PyObject,
            _args: *mut ffi::PyObject,
        ) -> *mut ffi::PyObject {
            entry(|g| {
                // SAFETY: CPython passes `self` of a `SearchGroup` method.
                let inner = unsafe { inner(g, slf) }?;
                $api(g, inner)
            })
        }
    };
}

/// A method with positional arguments, calling `$api(g, &GroupInner, args)`.
macro_rules! varargs {
    ($impl_fn:ident, $api:path) => {
        pub(super) unsafe extern "C" fn $impl_fn(
            slf: *mut ffi::PyObject,
            args: *mut ffi::PyObject,
        ) -> *mut ffi::PyObject {
            entry(|g| {
                // SAFETY: CPython passes `self` of a `SearchGroup` method.
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
noargs!(last_failure, api::last_failure);
varargs!(set_root, api::set_root);
varargs!(set_active, api::set_active);
varargs!(active, api::active);
varargs!(run, api::run);
varargs!(root_visits, api::root_visits);
varargs!(select_action, api::select_action);
varargs!(improved_policy, api::improved_policy);
varargs!(considered, api::considered);
varargs!(calls, api::calls);
varargs!(tree_hash, api::tree_hash);

static METHODS: Table<[ffi::PyMethodDef; 13]> = Table([
    method(
        c"layout",
        thunks::GROUP_LAYOUT,
        ffi::METH_NOARGS,
        c"Trees, actions, observation bytes per leaf and the memory bound.",
    ),
    method(
        c"last_failure",
        thunks::GROUP_LAST_FAILURE,
        ffi::METH_NOARGS,
        c"The last failed call: None or (slot, message, user_raised).",
    ),
    method(
        c"set_root",
        thunks::GROUP_SET_ROOT,
        ffi::METH_VARARGS,
        c"set_root(i, record: bytes): replaces slot i's tree by this root.",
    ),
    method(
        c"set_active",
        thunks::GROUP_SET_ACTIVE,
        ffi::METH_VARARGS,
        c"set_active(i, flag: bool): whether slot i takes part in the next call.",
    ),
    method(
        c"active",
        thunks::GROUP_ACTIVE,
        ffi::METH_VARARGS,
        c"active(i) -> bool",
    ),
    method(
        c"run",
        thunks::GROUP_RUN,
        ffi::METH_VARARGS,
        c"run(evaluator): one group call (all or nothing).",
    ),
    method(
        c"root_visits",
        thunks::GROUP_ROOT_VISITS,
        ffi::METH_VARARGS,
        c"root_visits(i): slot i's root visit counts.",
    ),
    method(
        c"select_action",
        thunks::GROUP_SELECT_ACTION,
        ffi::METH_VARARGS,
        c"select_action(i, temperature: float) -> int",
    ),
    method(
        c"improved_policy",
        thunks::GROUP_IMPROVED_POLICY,
        ffi::METH_VARARGS,
        c"improved_policy(i): slot i's last Gumbel improved policy, or None.",
    ),
    method(
        c"considered",
        thunks::GROUP_CONSIDERED,
        ffi::METH_VARARGS,
        c"considered(i): the actions slot i's last Gumbel call considered, or None.",
    ),
    method(
        c"calls",
        thunks::GROUP_CALLS,
        ffi::METH_VARARGS,
        c"calls(i): slot i's completed calls.",
    ),
    method(
        c"tree_hash",
        thunks::GROUP_TREE_HASH,
        ffi::METH_VARARGS,
        c"tree_hash(i): a digest of slot i's tree.",
    ),
    method_end(),
]);

/// `tp_dealloc`: a cleanup entry, which never fails or rejects (ADR-0041).
/// CPython calls it once, attached, when the reference count reaches zero.
pub(super) unsafe extern "C" fn dealloc(obj: *mut ffi::PyObject) {
    let env = obj.cast::<GroupObject>();
    // SAFETY: `obj` is a live `GroupObject` (above).
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
    // SAFETY: `obj` is a live `SearchGroup` instance, in its tp_dealloc.
    unsafe { capi::free_instance(obj) };
}

static SLOTS: Table<[ffi::PyType_Slot; 4]> = Table([
    slot_dealloc(thunks::GROUP_DEALLOC),
    slot_methods(&METHODS.0),
    slot_doc(c"PUCT searches sharing evaluator batches (use tessera_rl.search.SearchGroup)."),
    slot_end(),
]);

/// Creates the `SearchGroup` type and adds it to `module`.
pub(super) fn register(g: Gil, module: Ref<'_>) -> CResult<()> {
    let mut spec = ffi::PyType_Spec {
        name: c"tessera_rl._native.SearchGroup".as_ptr(),
        basicsize: c_int::try_from(core::mem::size_of::<GroupObject>()).unwrap_or(c_int::MAX),
        itemsize: 0,
        // Not GC-tracked: instances hold no Python objects (the evaluator
        // is borrowed for one `run` call only).
        flags: core::ffi::c_uint::try_from(ffi::Py_TPFLAGS_DEFAULT).unwrap_or(0),
        slots: SLOTS.0.as_ptr().cast_mut(),
    };
    let ty = capi::type_from_spec(g, &mut spec)?;
    capi::module_add(g, module, c"SearchGroup", &ty)?;
    TYPE.store(ty.as_ref().as_ptr(), Ordering::Release);
    Ok(())
}

/// `make_search_group(**kwargs) -> SearchGroup`.
pub(super) unsafe extern "C" fn make_search_group(
    _module: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|g| {
        // SAFETY: METH_VARARGS | METH_KEYWORDS passes a borrowed dict or
        // null, alive for the call.
        let kwargs = unsafe { Ref::from_ptr(kwargs) };
        let inner = api::make_search_group(g, kwargs)?;
        new_object(g, inner)
    })
}

/// The module-level constructor, for the module's method table.
pub(super) const MAKER: ffi::PyMethodDef = method_kw(
    c"make_search_group",
    thunks::MAKE_SEARCH_GROUP,
    c"make_search_group(trees, slot_options=None, **search_options) -> SearchGroup",
);
