// SPDX-License-Identifier: MIT OR Apache-2.0

//! Python output allocators (ADR-0084 items 1, 4 and 4b). Part of the
//! audited `ffi` module.
//!
//! [`output_allocator`] turns a Python object with
//! `allocate(len, align) -> (address, owner)` into an engine
//! [`OutputAllocator`]. The engine calls it once per output storage, at
//! construction, usually with the interpreter released (the bindings build
//! pools detached), so every call takes the interpreter with the
//! re-entrant `PyGILState_Ensure`.
//!
//! The returned `owner` keeps the memory alive. It is released exactly
//! once, by the thread that drops the last reference to the storage: that
//! thread takes the interpreter the same way, calls `owner.wait()` if the
//! owner has one (the last device copy that reads the memory, bounded by
//! one copy of the outputs), and only then drops its reference. A release
//! after the interpreter has finalized leaks the owner instead of calling
//! into a dead interpreter.
//!
//! Trust: the memory is used without any further check than the engine's
//! own (length and alignment, ADR-0084 item 2). The Python allocator must
//! keep `address..address + len` valid and writable until its owner is
//! released, never write it, and never hand its own handle to the memory to
//! user code (the guarantees of `ExternalAlloc::new`). Only Tessera's own
//! allocators (`tessera_rl._cuda`) and its tests pass one.

use core::ptr::NonNull;
use std::sync::Arc;

use pyo3::ffi;
use tessera_rl_engine::{ExternalAlloc, OutputAllocator};

use super::capi::{self, Gil, Ref};
use super::thunks;

/// A strong reference to a Python object, usable from any thread: every
/// use, the final decref included, happens with the interpreter taken
/// ([`with_interpreter`]).
struct Held {
    obj: NonNull<ffi::PyObject>,
    /// Call `obj.wait()` before the final decref (an allocation's owner).
    wait_on_drop: bool,
}

// SAFETY: the object is only touched with the interpreter taken on the
// current thread (`with_interpreter`), which serializes every access; the
// pointer itself is plain data.
unsafe impl Send for Held {}
// SAFETY: as above: shared access only reads the pointer, and every use of
// the object goes through `with_interpreter`.
unsafe impl Sync for Held {}

impl Held {
    /// A new strong reference to `obj`.
    fn new(_g: Gil<'_>, obj: Ref<'_>, wait_on_drop: bool) -> Self {
        let ptr = obj.as_ptr();
        // SAFETY: attached (the token) and a live object; an incref runs
        // no Python code.
        unsafe { ffi::Py_IncRef(ptr) };
        // A live object is never null; `Ref` holds a non-null pointer.
        let obj = NonNull::new(ptr).unwrap_or(NonNull::dangling());
        Self { obj, wait_on_drop }
    }

    /// A borrowed reference, for as long as `self`.
    fn get(&self) -> Option<Ref<'_>> {
        // SAFETY: our strong reference keeps the object alive for `self`.
        unsafe { Ref::from_ptr(self.obj.as_ptr()) }
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        let wait = self.wait_on_drop;
        let released = with_interpreter(|g| {
            if wait {
                wait_for(g, self);
            }
            // SAFETY: our strong reference, released once; a decref of a
            // user object can run finalizers, so it is a callout.
            unsafe { thunks::c_decref(self.obj.as_ptr()) };
        });
        // `None`: the interpreter is gone, and the reference leaks
        // (documented in the module notes).
        let _ = released;
    }
}

/// `obj.wait()`, if `obj` has a `wait`; any exception it raises is
/// discarded (the caller restores the indicator it found).
fn wait_for(g: Gil<'_>, held: &Held) {
    let Some(obj) = held.get() else {
        return;
    };
    let Ok(wait) = capi::get_attr(g, obj, c"wait") else {
        capi::clear_pending(g);
        return;
    };
    let waited = capi::tuple(g, Vec::new()).and_then(|args| capi::call(g, wait.as_ref(), &args));
    if waited.is_err() {
        capi::clear_pending(g);
    }
}

/// Runs `f` with the interpreter taken on this thread: re-entrant (a
/// thread that holds it already, such as a deleter's, keeps it), with the
/// pending exception saved and restored around `f`, and panics contained.
/// `None` when the interpreter is not initialized (before start-up or after
/// finalization): `f` does not run.
fn with_interpreter<R>(f: impl FnOnce(Gil<'_>) -> R) -> Option<R> {
    // SAFETY: reads a process flag; callable without the interpreter.
    if unsafe { ffi::Py_IsInitialized() } == 0 {
        return None;
    }
    // SAFETY: the interpreter is initialized; the re-entrant ensure takes
    // it on this thread, or keeps it if this thread holds it already.
    let state = unsafe { ffi::PyGILState_Ensure() };
    // SAFETY: attached until the release below; the token stays inside.
    let g = unsafe { Gil::assume() };
    let pending = capi::fetch_pending(g);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(g)));
    capi::put_back_pending(g, pending);
    // SAFETY: the matching release of the ensure above, on the same thread.
    unsafe { ffi::PyGILState_Release(state) };
    match result {
        Ok(r) => Some(r),
        Err(payload) => {
            super::dispose(payload);
            None
        }
    }
}

/// A Python allocator as an engine [`OutputAllocator`].
struct PyAllocator(Held);

impl OutputAllocator for PyAllocator {
    fn allocate(&self, len: usize, align: usize) -> Option<ExternalAlloc> {
        with_interpreter(|g| allocate_in(g, &self.0, len, align)).flatten()
    }
}

/// `allocator.allocate(len, align)`: an `(address, owner)` tuple with an
/// exact `int` address. Anything else, an exception included, is `None`
/// (the engine then refuses the pool with `OutputAllocation`).
fn allocate_in(g: Gil<'_>, allocator: &Held, len: usize, align: usize) -> Option<ExternalAlloc> {
    let obj = allocator.get()?;
    let method = capi::get_attr(g, obj, c"allocate").ok()?;
    let len_obj = capi::int_u64(g, u64::try_from(len).ok()?).ok()?;
    let align_obj = capi::int_u64(g, u64::try_from(align).ok()?).ok()?;
    let args = capi::tuple(g, vec![len_obj, align_obj]).ok()?;
    let out = capi::call(g, method.as_ref(), &args).ok()?;
    let items = capi::tuple_items(g, out.as_ref(), "allocate's result").ok()?;
    let [address, owner] = items.as_slice() else {
        return None;
    };
    let address = usize::try_from(capi::to_u64(g, *address, "the address").ok()?).ok()?;
    let ptr = NonNull::new(core::ptr::with_exposed_provenance_mut::<u8>(address))?;
    let owner = Held::new(g, *owner, true);
    // SAFETY: the Python allocator's contract (module notes): the range is
    // valid and writable until `owner` is released, only the engine writes
    // it, others read it only inside a borrow window (the bindings' device
    // copy holds the window until its event, and `owner.wait()` waits for
    // the last one), and its own handle never reaches user code. `owner`'s
    // drop is safe on any thread (`Held`).
    Some(unsafe { ExternalAlloc::new(ptr, len, Box::new(owner)) })
}

/// The engine allocator of the Python object `allocator`, which must have
/// `allocate(len, align) -> (address, owner)` (module notes). The object
/// is held for the allocator's life.
#[must_use]
pub fn output_allocator(g: Gil<'_>, allocator: Ref<'_>) -> Arc<dyn OutputAllocator> {
    Arc::new(PyAllocator(Held::new(g, allocator, false)))
}

// No Rust unit tests here: every path calls into libpython, which a
// workspace build with `pyo3/extension-module` (the wheel's feature) does
// not link into test binaries. tests/python/test_device.py covers the
// adoption, the refusals, the owner's wait and its release.
