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

/// The state of `slf`, refusing an object inherited through a fork before any
/// lock or flag of it is touched.
///
/// # Safety
///
/// As [`inner`].
unsafe fn guarded<'a>(g: Gil, slf: *mut ffi::PyObject) -> CResult<&'a Inner> {
    // SAFETY: per the contract.
    let inner = unsafe { inner(g, slf) }?;
    inner.check_fork(g)?;
    Ok(inner)
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
                let inner = unsafe { guarded(g, slf) }?;
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
                let inner = unsafe { guarded(g, slf) }?;
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

/// `close()`: on an inherited object a no-op, as `dealloc`: it neither joins
/// nor refuses, so `with` blocks and cleanup paths in a child still finish.
pub(super) unsafe extern "C" fn close(
    slf: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    entry(|g| {
        // SAFETY: CPython passes `self` of an `Env` method.
        let inner = unsafe { inner(g, slf) }?;
        if inner.inherited() {
            return Ok(capi::none(g));
        }
        api::close(g, inner)
    })
}

varargs!(recv, api::recv);
varargs!(restore, crate::async_api::restore);
noargs!(snapshot_all, api::snapshot_all);
varargs!(restore_all, api::restore_all);
varargs!(send, api::send);
varargs!(reset, api::reset);
varargs!(step, api::step);
varargs!(snapshot, api::snapshot);
varargs!(illegal_actions, api::illegal_actions);
varargs!(state_hash, api::state_hash);
varargs!(import_record, api::import_record);
varargs!(fork, api::fork);
varargs!(encode, api::encode);
varargs!(load, api::load);

static METHODS: Table<[ffi::PyMethodDef; 19]> = Table([
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
        c"illegal_actions",
        thunks::ILLEGAL_ACTIONS,
        ffi::METH_VARARGS,
        c"illegal_actions(id) -> per-agent illegal actions this episode",
    ),
    method(
        c"state_hash",
        thunks::STATE_HASH,
        ffi::METH_VARARGS,
        c"state_hash(id) -> the instance's state hash",
    ),
    method(
        c"import_record",
        thunks::IMPORT_RECORD,
        ffi::METH_VARARGS,
        c"import_record(id, record)",
    ),
    method(
        c"encode",
        thunks::ENCODE,
        ffi::METH_VARARGS,
        c"encode(ids) -> bytes",
    ),
    method(
        c"load",
        thunks::LOAD,
        ffi::METH_VARARGS,
        c"load(data, ids, max_instances, max_decoded_bytes)",
    ),
    method(
        c"send",
        thunks::SEND,
        ffi::METH_VARARGS,
        c"send(half, actions: bytes of int32, delivered)",
    ),
    method(
        c"recv",
        thunks::RECV,
        ffi::METH_VARARGS,
        c"recv(delivered) -> the oldest sent half's outputs",
    ),
    method(
        c"restore",
        thunks::RESTORE,
        ffi::METH_VARARGS,
        c"restore(ids, snapshots, delivered)",
    ),
    method(
        c"snapshot_all",
        thunks::SNAPSHOT_ALL,
        ffi::METH_NOARGS,
        c"snapshot_all() -> an opaque whole-pool snapshot",
    ),
    method(
        c"restore_all",
        thunks::RESTORE_ALL,
        ffi::METH_VARARGS,
        c"restore_all(snapshot) -> both halves' outputs",
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
        if boxed.inherited() {
            // SAFETY-relevant, no `unsafe` here: this runs only in a forked
            // child, whose copy of the workers does not exist and whose locks
            // may be held forever, so `Drop` could hang or abort. The memory
            // is leaked by design, for the rest of the child's life.
            core::mem::forget(boxed);
            // SAFETY: `obj` is a live `Env` instance, in its tp_dealloc.
            unsafe { capi::free_instance(obj) };
            return;
        }
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

#[cfg(feature = "test-hooks")]
maker!(make_panic_counter, api::make_panic_counter);
#[cfg(feature = "test-hooks")]
maker!(make_continuous_probe, api::make_continuous_probe);
#[cfg(feature = "test-hooks")]
maker!(make_multi_probe, api::make_multi_probe);
maker!(make_cartpole, api::make_cartpole);
maker!(make_pendulum, api::make_pendulum);
maker!(make_acrobot, api::make_acrobot);
maker!(make_mountaincar, api::make_mountaincar);
maker!(make_tag, api::make_tag);
maker!(make_rts, api::make_rts);
maker!(make_grid, api::make_grid);
maker!(make_keydoor, api::make_keydoor);
maker!(make_snake, api::make_snake);
maker!(make_brickwall, api::make_brickwall);
maker!(make_paddleduel, api::make_paddleduel);
maker!(make_connect4, api::make_connect4);
maker!(make_leduc, api::make_leduc);
maker!(make_othello, api::make_othello);
maker!(make_go9, api::make_go9);

/// The module-level constructors, for the module's method table.
pub(super) const MAKERS: [ffi::PyMethodDef; 15] = [
    method_kw(
        c"make_cartpole",
        thunks::MAKE_CARTPOLE,
        c"make_cartpole(**pool_options) -> Env",
    ),
    method_kw(
        c"make_acrobot",
        thunks::MAKE_ACROBOT,
        c"make_acrobot(**pool_options) -> Env",
    ),
    method_kw(
        c"make_mountaincar",
        thunks::MAKE_MOUNTAINCAR,
        c"make_mountaincar(**pool_options) -> Env",
    ),
    method_kw(
        c"make_pendulum",
        thunks::MAKE_PENDULUM,
        c"make_pendulum(**pool_options) -> Env",
    ),
    method_kw(
        c"make_tag",
        thunks::MAKE_TAG,
        c"make_tag(**pool_options, **tag_config) -> Env",
    ),
    method_kw(
        c"make_rts",
        thunks::MAKE_RTS,
        c"make_rts(**pool_options, **rts_config) -> Env",
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
    method_kw(
        c"make_othello",
        thunks::MAKE_OTHELLO,
        c"make_othello(**pool_options) -> Env",
    ),
    method_kw(
        c"make_go9",
        thunks::MAKE_GO9,
        c"make_go9(**pool_options, komi_half_points=15) -> Env",
    ),
    method_kw(
        c"make_grid",
        thunks::MAKE_GRID,
        c"make_grid(**pool_options, size=16, max_steps=200) -> Env",
    ),
    method_kw(
        c"make_keydoor",
        thunks::MAKE_KEYDOOR,
        c"make_keydoor(**pool_options, size=8, agents=1) -> Env",
    ),
    method_kw(
        c"make_snake",
        thunks::MAKE_SNAKE,
        c"make_snake(**pool_options, size=16, agents=1, obs=\"pixels\") -> Env",
    ),
    method_kw(
        c"make_brickwall",
        thunks::MAKE_BRICKWALL,
        c"make_brickwall(**pool_options, frameskip=4, lives=5, obs=\"pixels\") -> Env",
    ),
    method_kw(
        c"make_paddleduel",
        thunks::MAKE_PADDLEDUEL,
        c"make_paddleduel(**pool_options, agents=1, frameskip=4, max_score=11, bounce=1, obs=\"pixels\") -> Env",
    ),
];
