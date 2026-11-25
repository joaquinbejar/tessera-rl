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
//! The finalization window: the leak covers releases only after
//! `Py_IsInitialized` reads 0, which happens late in `Py_FinalizeEx`. A
//! release on a thread CPython does not know (a non-Python thread dropping
//! the last export) earlier during finalization still enters
//! `PyGILState_Ensure`, which can hang on 3.12 and later. Release every
//! device pool and its exports before the interpreter exits.
//!
//! Exceptions from `allocate`: an `Exception` refuses the pool with
//! `ConfigError`, its class name and message appended; any other
//! `BaseException` (`KeyboardInterrupt`, `SystemExit`) is raised again
//! unchanged by the bindings.
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
    fn new(g: Gil<'_>, obj: Ref<'_>, wait_on_drop: bool) -> Self {
        Self {
            obj: capi::new_strong(g, obj),
            wait_on_drop,
        }
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
    if !may_enter(capi::interpreter_initialized()) {
        return None;
    }
    // SAFETY: the interpreter is initialized (a release during
    // finalization is the documented exception); released below.
    let state = unsafe { capi::ensure() };
    // SAFETY: attached until the release below; the token stays inside.
    let g = unsafe { Gil::assume() };
    let pending = capi::fetch_pending(g);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(g)));
    capi::put_back_pending(g, pending);
    // SAFETY: the matching release of the ensure above, on the same thread.
    unsafe { capi::release(state) };
    match result {
        Ok(r) => Some(r),
        Err(payload) => {
            super::dispose(payload);
            None
        }
    }
}

/// Why the last `allocate` on this thread was refused, beyond the
/// engine's own reason: the engine's `OutputAllocation` cannot carry it, so
/// the bindings add it to the error they raise (construction runs on the
/// caller's thread).
pub enum AllocationFailure {
    /// A message for the `ConfigError`.
    Message(String),
    /// A `BaseException` that is not an `Exception`, to raise again
    /// unchanged.
    Interrupt(capi::Stashed),
}

std::thread_local! {
    static LAST_ERROR: core::cell::RefCell<Option<AllocationFailure>> = const { core::cell::RefCell::new(None) };
}

impl core::fmt::Debug for AllocationFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Message(m) => f.debug_tuple("Message").field(m).finish(),
            Self::Interrupt(_) => f.write_str("Interrupt(..)"),
        }
    }
}

/// Records `failure`. An older one is replaced: `allocate` runs once per
/// construction, and `output_allocator` clears any left over first.
fn remember(failure: AllocationFailure) {
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(failure));
}

/// Records the exception `raised` proves is set; it is taken out (the
/// caller restores the indicator it found).
fn remember_exception(g: Gil<'_>, raised: capi::Raised) {
    if !capi::pending_is_exception(g) {
        remember(AllocationFailure::Interrupt(capi::Stashed::take(g, raised)));
        return;
    }
    #[cfg(any(feature = "native-module", feature = "host"))]
    let message = {
        let (name, message, _) = capi::pending_name_and_message(g, raised);
        format!("allocate raised {name}: {message}")
    };
    #[cfg(not(any(feature = "native-module", feature = "host")))]
    let message = {
        let _ = raised;
        String::from("allocate raised an exception")
    };
    remember(AllocationFailure::Message(message));
}

/// Takes why the last `allocate` on this thread was refused, if it said.
#[must_use]
pub fn take_allocation_error() -> Option<AllocationFailure> {
    LAST_ERROR.with(|e| e.borrow_mut().take())
}

/// Forgets any refusal left by an earlier construction on this thread (a
/// stashed interrupt is raised and discarded, under the interpreter).
pub fn clear_allocation_error(g: Gil<'_>) {
    if let Some(AllocationFailure::Interrupt(stashed)) = take_allocation_error() {
        let _ = stashed.raise_again(g);
        capi::clear_pending(g);
    }
}

/// Why `address..address + len` cannot be adopted before the engine's own
/// checks: a range that wraps the address space, or a start not aligned to
/// `align`. `None` when both hold.
fn range_refusal(address: usize, len: usize, align: usize) -> Option<String> {
    if address.checked_add(len).is_none() {
        return Some(format!("allocate returned address {address:#x}, whose {len} bytes overflow the address space"));
    }
    match address.checked_rem(align) {
        Some(0) => None,
        _ => Some(format!("allocate returned address {address:#x}, not aligned to {align}")),
    }
}

/// Whether a release may enter the interpreter: only while it is
/// initialized (`Py_IsInitialized`); otherwise the owner leaks.
const fn may_enter(initialized: core::ffi::c_int) -> bool {
    initialized != 0
}

/// A Python allocator as an engine [`OutputAllocator`].
struct PyAllocator(Held);

impl OutputAllocator for PyAllocator {
    fn allocate(&self, len: usize, align: usize) -> Option<ExternalAlloc> {
        let _ = take_allocation_error();
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
    let out = match capi::call(g, method.as_ref(), &args) {
        Ok(out) => out,
        Err(raised) => {
            remember_exception(g, raised);
            return None;
        }
    };
    let items = capi::tuple_items(g, out.as_ref(), "allocate's result").ok()?;
    let [address, owner] = items.as_slice() else {
        return None;
    };
    let address = usize::try_from(capi::to_u64(g, *address, "the address").ok()?).ok()?;
    if let Some(why) = range_refusal(address, len, align) {
        remember(AllocationFailure::Message(why));
        return None;
    }
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
    // A construction starts here: no refusal of an earlier one survives.
    clear_allocation_error(g);
    Arc::new(PyAllocator(Held::new(g, allocator, false)))
}

// Only the libpython-free decision is unit tested here: a workspace build
// with `pyo3/extension-module` links no libpython into test binaries.
// tests/python/test_device.py covers adoption, refusals, the owner's wait
// and its release; the leak after finalization is not exercised.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_may_enter_uninitialized_interpreter_false() {
        assert!(!may_enter(0));
        assert!(may_enter(1));
    }

    #[test]
    fn test_take_allocation_error_takes_once() {
        remember(AllocationFailure::Message("boom".to_owned()));
        assert!(matches!(take_allocation_error(), Some(AllocationFailure::Message(m)) if m == "boom"));
        assert!(take_allocation_error().is_none());
    }

    #[test]
    fn test_range_refusal_wrapping_range_refused() {
        let why = range_refusal(usize::MAX - 8, 16, 16384);
        assert!(why.is_some_and(|w| w.contains("overflow")));
    }

    #[test]
    fn test_range_refusal_misaligned_refused() {
        let why = range_refusal(16384 + 64, 16, 16384);
        assert!(why.is_some_and(|w| w.contains("not aligned to 16384")));
    }

    #[test]
    fn test_range_refusal_aligned_in_range_accepted() {
        assert_eq!(range_refusal(16384 * 3, 16384, 16384), None);
        assert!(range_refusal(16384, 16, 0).is_some());
    }
}
