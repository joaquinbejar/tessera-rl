// SPDX-License-Identifier: MIT OR Apache-2.0

//! The only module of the bindings that calls the CPython C API (ADR-0041
//! callouts). Audited `unsafe` module.
//!
//! Rust runs under canonical floating-point controls from the boundary
//! thunk to its return. Any C-API call that can run Python code, and so
//! change the controls under Rust's feet, goes through a **callout**
//! (`thunks::callout!`): the caller's controls around the C call, the
//! canonical ones restored (and a change counted) before Rust resumes.
//!
//! **The rule** for choosing, by function:
//!
//! - **Callout**: every GC-tracked allocation (`PyList_New`, `PyTuple_New`,
//!   `PyDict_New`, `PyModule_Create2`, `PyType_FromSpec`,
//!   `PyErr_NewExceptionWithDoc`), because it can run `gc.collect` and so
//!   any `__del__`; every call that mutates a GC-tracked container
//!   (`PyList_Append`, `PyTuple_SetItem`, `PyDict_SetItemString`) or
//!   decrefs one (`Py_DecRef` of a container [`Mine`]), because Python code
//!   (a `gc` callback, `gc.get_objects`) can find our containers and store
//!   user objects in them, so a replaced or dropped value may run a
//!   finalizer; setting an exception (`PyErr_SetString`, which normalizes:
//!   an allocation); a conversion that may raise (`PyLong_As*`,
//!   `PyUnicode_AsUTF8AndSize`); a buffer request, its copy and its
//!   release on an exact `memoryview` (`PyObject_GetBuffer`,
//!   `PyBuffer_ToContiguous`, `PyBuffer_Release`); and a warning.
//! - **Direct** (cannot run Python code): non-GC allocations (`str`,
//!   `bytes`, `bytearray`, `int`, `float`, capsules, and instances of our
//!   own types, which are not GC-tracked); reads of exact built-in types
//!   (`PyTuple_Size`/`GetItem`, `PyList_Size`/`GetItem`, `PyDict_Next`,
//!   `PyBytes_AsString`/`Size`); `PyErr_Fetch` and `PyErr_Occurred`;
//!   reference increments; releasing and reacquiring the interpreter; and a
//!   decref of a leaf [`Mine`]. `PyErr_Clear` and `PyErr_Restore` are
//!   callouts: dropping or replacing a pending exception can run finalizers
//!   through its traceback.
//!
//! **Decref invariant.** Owned references are [`Mine`]. A leaf (not
//! GC-tracked: an object we created that holds no other object) cannot
//! reach a user object, so its decref is direct. Every other owned
//! reference is dropped through a callout (`Mine::drop`): a GC-tracked
//! container we created (Python code running in any callout may find it
//! and store user objects in it), a call result, and a user object read
//! from a list or dict ([`list_items`], [`dict_items`]), on success and on
//! every failure or unwind path.
//!
//! **Borrow invariant.** A [`Ref`] is valid for its lifetime, enforced by
//! the types: the arguments CPython passes live for the call; an item of a
//! tuple lives as long as the tuple (tuples are immutable); a [`Mine`]'s
//! borrow lives as long as the borrow. Items of lists and dicts, which
//! Python code can change during any callout, are returned owned, never
//! borrowed. No safe function relies on its caller for this.
//!
//! **Exact types.** Arguments are read only from exact built-in types
//! (`int`, `bool`, `str`, `bytes`, `memoryview`, `tuple`, `list`, `dict`,
//! `None`); any
//! other object raises `TypeError` without calling its `__index__`,
//! `__bool__`, `__iter__` or anything else. The Python layer converts user
//! objects (with `int()`, NumPy) before calling the native module, under
//! the caller's own controls, which is the one place non-exact objects
//! are accepted.

#![expect(unsafe_code, reason = "audited module: the CPython C API")]
#![expect(
    clippy::missing_errors_doc,
    reason = "every `Err(Raised)` here means a Python exception is set (the C-API failure convention)"
)]

use core::ffi::{CStr, c_int};
use core::marker::PhantomData;
use core::ptr::{self, NonNull};

use pyo3::ffi;

use super::thunks;

/// Proof that the interpreter is attached to this thread. Only entries
/// (`ffi::entry` and the typed shims) make one, scoped to the call: `'py`
/// is the entry call's lifetime, so the token, and every [`Ref`] and
/// [`Mine`] made under it, cannot outlive the call.
///
/// The token is `!Send` and `!Sync`: it cannot reach another thread, and
/// [`Gil::detach`] takes only `Send` closures, so it never enters the code
/// that runs with the interpreter released.
///
/// ```compile_fail
/// use tessera_rl_pyffi::ffi::capi::Gil;
/// fn escape(g: Gil<'_>) -> Gil<'static> {
///     g // the call's token cannot become 'static
/// }
/// ```
///
/// ```compile_fail
/// use tessera_rl_pyffi::ffi::capi::Gil;
/// fn inside(g: Gil<'_>) {
///     g.detach(|| {
///         drop(g); // not Send: cannot enter the detached closure
///     });
/// }
/// ```
///
/// ```compile_fail
/// use tessera_rl_pyffi::ffi::capi::{self, Gil};
/// fn out(g: Gil<'_>) {
///     // An object reference is not Send: it cannot leave it either.
///     let _ = g.detach(|| capi::none(g));
/// }
/// ```
#[derive(Clone, Copy)]
pub struct Gil<'py>(PhantomData<(&'py (), *mut ())>);

impl Gil<'_> {
    /// # Safety
    ///
    /// The interpreter must be attached to the calling thread for as long
    /// as the token is used.
    pub(crate) const unsafe fn assume() -> Self {
        Self(PhantomData)
    }

    /// Runs `f` with the interpreter released (the engine's work). `F` and
    /// `R` are `Send`, so no interpreter token or object reference (all
    /// `!Send`) enters or leaves the detached closure.
    pub fn detach<F, R>(self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        struct Restore(*mut ffi::PyThreadState);
        impl Drop for Restore {
            fn drop(&mut self) {
                // SAFETY: the state saved by `PyEval_SaveThread` on this
                // thread; reacquiring runs no Python code.
                unsafe { ffi::PyEval_RestoreThread(self.0) };
            }
        }
        // SAFETY: attached (the token); releasing runs no Python code.
        let saved = Restore(unsafe { ffi::PyEval_SaveThread() });
        let result = f();
        drop(saved);
        result
    }
}

/// `Py_IsInitialized()` (direct: reads a process flag, callable without
/// the interpreter; runs no Python code).
#[must_use]
pub(crate) fn interpreter_initialized() -> core::ffi::c_int {
    // SAFETY: callable at any time, attached or not.
    unsafe { ffi::Py_IsInitialized() }
}

/// The state of a re-entrant [`ensure`], for its [`release`].
pub(crate) struct Ensured(ffi::PyGILState_STATE);

/// `PyGILState_Ensure()` (direct: takes the interpreter on this thread, or
/// keeps it if this thread holds it already; runs no Python code).
///
/// # Safety
///
/// The interpreter is initialized and not finalizing; the result goes to
/// [`release`] on the same thread.
pub(crate) unsafe fn ensure() -> Ensured {
    // SAFETY: the caller's contract.
    Ensured(unsafe { ffi::PyGILState_Ensure() })
}

/// `PyGILState_Release()` (direct: restores the thread state [`ensure`]
/// found; runs no Python code).
///
/// # Safety
///
/// `state` is the result of an [`ensure`] on this thread, released once,
/// innermost first.
#[expect(clippy::needless_pass_by_value, reason = "the state is consumed")]
pub(crate) unsafe fn release(state: Ensured) {
    // SAFETY: the caller's contract.
    unsafe { ffi::PyGILState_Release(state.0) };
}

/// A new strong reference to `r` (direct: an incref runs no Python code),
/// released by the caller exactly once.
#[must_use]
pub(crate) fn new_strong(_g: Gil<'_>, r: Ref<'_>) -> NonNull<ffi::PyObject> {
    // SAFETY: attached (the token) and a live object.
    unsafe { ffi::Py_IncRef(r.as_ptr()) };
    r.0
}

/// An exception is set (the C-API failure convention).
///
/// Only this crate makes one, where an exception was just set:
///
/// ```compile_fail
/// let _ = tessera_rl_pyffi::ffi::capi::Raised(());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Raised(());

/// Made only where an exception was just set (here, by a failed C-API
/// call, or by [`raise`]), so `Err(RAISED)` always means one is.
pub(crate) const RAISED: Raised = Raised(());

/// The result of a C-API operation: `Err` means an exception is set.
pub type CResult<T> = Result<T, Raised>;

/// A borrowed reference, valid for `'a`.
#[derive(Clone, Copy)]
pub struct Ref<'a>(NonNull<ffi::PyObject>, PhantomData<&'a ()>);

impl Ref<'_> {
    /// # Safety
    ///
    /// `ptr` must be a live object for the reference's lifetime.
    pub(crate) unsafe fn from_ptr(ptr: *mut ffi::PyObject) -> Option<Self> {
        NonNull::new(ptr).map(|p| Self(p, PhantomData))
    }

    pub(crate) const fn as_ptr(self) -> *mut ffi::PyObject {
        self.0.as_ptr()
    }

    fn is_exact(self, ty: *mut ffi::PyTypeObject) -> bool {
        // SAFETY: a live object (the type's invariant).
        unsafe { ffi::Py_TYPE(self.as_ptr()) == ty }
    }

    /// Whether this is `None`.
    #[must_use]
    pub fn is_none(self) -> bool {
        // SAFETY: reading the address of the `None` singleton.
        self.as_ptr() == unsafe { ffi::Py_None() }
    }
}

/// An owned reference to an object we created. A leaf (not GC-tracked:
/// `int`, `float`, `str`, `bytes`, capsules, our own instances) is dropped
/// with a direct decref; a GC-tracked container is dropped through a
/// callout, because Python code may have found it and stored user objects
/// in it (decref invariant).
pub struct Mine<'py>(NonNull<ffi::PyObject>, Tracked, PhantomData<Gil<'py>>);

/// Whether the garbage collector tracks the object, so Python code can
/// reach it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tracked {
    Leaf,
    Container,
}

impl Mine<'_> {
    /// Takes ownership of a new reference; `None` (an exception is set)
    /// for null.
    fn new(ptr: *mut ffi::PyObject) -> CResult<Self> {
        NonNull::new(ptr)
            .map(|p| Self(p, Tracked::Leaf, PhantomData))
            .ok_or(RAISED)
    }

    /// A new reference to `ptr`, a live user object read from a container
    /// (an item, a key or a value): owned so it stays alive whatever Python
    /// code runs next, and dropped through a callout, as a container is
    /// (the user object may hold anything).
    fn owned_user(g: Gil<'_>, ptr: *mut ffi::PyObject) -> CResult<Self> {
        let p = NonNull::new(ptr)
            .ok_or_else(|| raise(g, runtime_error(), "a container item is missing"))?;
        // SAFETY: a live object (the caller read it from a live container
        // without running Python code); incrementing runs no Python code.
        unsafe { ffi::Py_IncRef(p.as_ptr()) };
        Ok(Self(p, Tracked::Container, PhantomData))
    }

    /// As [`Mine::new`], for a GC-tracked container.
    fn container(ptr: *mut ffi::PyObject) -> CResult<Self> {
        NonNull::new(ptr)
            .map(|p| Self(p, Tracked::Container, PhantomData))
            .ok_or(RAISED)
    }

    /// A borrowed reference to the object, for as long as `self`.
    #[must_use]
    pub fn as_ref(&self) -> Ref<'_> {
        Ref(self.0, PhantomData)
    }

    /// Releases ownership, for a return value.
    pub(crate) fn into_ptr(self) -> *mut ffi::PyObject {
        let ptr = self.0.as_ptr();
        core::mem::forget(self);
        ptr
    }
}

impl Drop for Mine<'_> {
    fn drop(&mut self) {
        if self.1 == Tracked::Container {
            // SAFETY: an owned reference; a container may hold user objects
            // whose finalizers run, so the decref is a callout.
            unsafe { thunks::c_decref(self.0.as_ptr()) };
        } else {
            // SAFETY: an owned reference to a leaf: its deallocation runs
            // no Python code, so a direct decref is safe for the boundary.
            unsafe { ffi::Py_DecRef(self.0.as_ptr()) };
        }
    }
}

// -- exceptions -----------------------------------------------------------------

/// An exception class: a built-in one, or one of [`crate::errors`]'
/// (created by `register`, or bound by `bind` after checking that it is a
/// `BaseException` subclass). Made only here, so it is always a live
/// exception type object.
#[derive(Clone, Copy)]
pub struct Exc(*mut ffi::PyObject);

impl Exc {
    /// A class [`crate::errors`] created, or bound after checking it with
    /// [`is_exception_class`], and keeps alive for the process.
    pub(crate) const fn from_type(ty: *mut ffi::PyObject) -> Self {
        Self(ty)
    }
}

/// Raises `ty(message)` (callout: setting an exception allocates).
#[must_use]
pub fn raise(_g: Gil<'_>, ty: Exc, message: &str) -> Raised {
    let ty = ty.0;
    let text = std::ffi::CString::new(message.replace('\0', " ")).unwrap_or_default();
    // SAFETY: attached; `ty` is a live exception type and `text` a C
    // string alive for the call.
    unsafe { thunks::c_err_set_string(ty, text.as_ptr()) };
    RAISED
}

/// `TypeError`.
#[must_use]
pub fn type_error() -> Exc {
    // SAFETY: reading an interpreter global.
    Exc(unsafe { ffi::PyExc_TypeError })
}
/// `RuntimeError`.
#[must_use]
pub fn runtime_error() -> Exc {
    // SAFETY: reading an interpreter global.
    Exc(unsafe { ffi::PyExc_RuntimeError })
}
/// `ImportError`.
#[must_use]
pub fn import_error() -> Exc {
    // SAFETY: reading an interpreter global.
    Exc(unsafe { ffi::PyExc_ImportError })
}
/// `BufferError`.
#[must_use]
pub fn buffer_error() -> Exc {
    // SAFETY: reading an interpreter global.
    Exc(unsafe { ffi::PyExc_BufferError })
}
pub(crate) fn exception_base() -> *mut ffi::PyObject {
    // SAFETY: reading an interpreter global.
    unsafe { ffi::PyExc_Exception }
}
pub(crate) fn runtime_warning() -> *mut ffi::PyObject {
    // SAFETY: reading an interpreter global.
    unsafe { ffi::PyExc_RuntimeWarning }
}

/// A new exception class `name` (`module.Name`) deriving from `base`
/// (callout: a GC allocation).
pub(crate) fn new_exception<'py>(
    _g: Gil<'py>,
    name: &CStr,
    doc: &CStr,
    base: *mut ffi::PyObject,
) -> CResult<Mine<'py>> {
    // SAFETY: attached; static C strings and a live base type.
    Mine::container(unsafe {
        thunks::c_err_new_exception(name.as_ptr(), doc.as_ptr(), base, ptr::null_mut())
    })
}

/// Whether `r` is a class deriving from `BaseException` (direct: a flag
/// read and an MRO walk, no Python code).
#[cfg(feature = "host")]
#[must_use]
pub(crate) fn is_exception_class(r: Ref<'_>) -> bool {
    // SAFETY: a live object; `PyType_Check` reads its type's flags.
    if unsafe { ffi::PyType_Check(r.as_ptr()) } == 0 {
        return false;
    }
    // SAFETY: reading an interpreter global.
    let base = unsafe { ffi::PyExc_BaseException };
    // SAFETY: two live type objects (`r` checked above); the subtype check
    // walks the MRO without running Python code.
    unsafe {
        ffi::PyType_IsSubtype(
            r.as_ptr().cast::<ffi::PyTypeObject>(),
            base.cast::<ffi::PyTypeObject>(),
        ) != 0
    }
}

/// Whether the pending exception is an `AttributeError` (direct: a class
/// comparison that runs no Python code). False with none pending.
#[must_use]
pub(crate) fn pending_is_attribute_error(_g: Gil<'_>) -> bool {
    // SAFETY: reading an interpreter global.
    let attribute_error = unsafe { ffi::PyExc_AttributeError };
    // SAFETY: attached; a class comparison that runs no Python code.
    unsafe { ffi::PyErr_ExceptionMatches(attribute_error) != 0 }
}

/// Whether the pending exception is a `TypeError`, an `OverflowError` or
/// a `UnicodeEncodeError`: what a conversion of an exact built-in value
/// raises for a wrong type, a range, or a `str` with a lone surrogate
/// (direct: class comparisons that run no Python code). False with none
/// pending.
#[cfg(feature = "host")]
#[must_use]
pub(crate) fn pending_is_conversion_error(_g: Gil<'_>) -> bool {
    // SAFETY: reading an interpreter global.
    let type_error = unsafe { ffi::PyExc_TypeError };
    // SAFETY: reading an interpreter global.
    let overflow = unsafe { ffi::PyExc_OverflowError };
    // SAFETY: attached; a class comparison that runs no Python code.
    let is_type = unsafe { ffi::PyErr_ExceptionMatches(type_error) } != 0;
    // SAFETY: reading an interpreter global.
    let encode = unsafe { ffi::PyExc_UnicodeEncodeError };
    // SAFETY: as above.
    let is_overflow = unsafe { ffi::PyErr_ExceptionMatches(overflow) } != 0;
    // SAFETY: as above.
    is_type || is_overflow || unsafe { ffi::PyErr_ExceptionMatches(encode) } != 0
}

/// Imports the module `name` (callout: runs the import system).
#[cfg(feature = "host")]
pub(crate) fn import_module<'py>(_g: Gil<'py>, name: &CStr) -> CResult<Mine<'py>> {
    // SAFETY: attached; a C string.
    Mine::container(unsafe { thunks::c_import_module(name.as_ptr()) })
}

/// `sys.modules[name]`, created as an empty module when absent (callout:
/// reads and may write `sys.modules`). Owned as a user object: the entry
/// may be anything a user stored there.
pub(crate) fn add_module<'py>(g: Gil<'py>, name: &CStr) -> CResult<Mine<'py>> {
    // `PyImport_AddModule` returns a borrowed reference; it is in the
    // stable ABI of the abi3-py311 floor. When the floor reaches 3.13, use
    // `PyImport_AddModuleRef` (a new reference) and drop the incref below.
    // SAFETY: attached; a C string.
    let borrowed = unsafe { thunks::c_import_add_module(name.as_ptr()) };
    if borrowed.is_null() {
        return Err(RAISED);
    }
    // A borrowed reference from `sys.modules`; nothing runs between the
    // call and the incref inside `owned_user`.
    Mine::owned_user(g, borrowed)
}

/// `getattr(obj, name)` (callout: can run `__getattr__`). The result is
/// owned as a container: its decref can run finalizers.
pub(crate) fn get_attr<'py>(_g: Gil<'py>, obj: Ref<'_>, name: &CStr) -> CResult<Mine<'py>> {
    // SAFETY: attached; a live object and a C string.
    Mine::container(unsafe { thunks::c_get_attr_string(obj.as_ptr(), name.as_ptr()) })
}

/// Warns with `category` (callout: runs filters and hooks).
pub(crate) fn warn(_g: Gil<'_>, category: *mut ffi::PyObject, message: &CStr) -> CResult<()> {
    // SAFETY: attached; a live category and a static message.
    let rc = unsafe { thunks::c_err_warn(category, message.as_ptr(), 1) };
    if rc < 0 { Err(RAISED) } else { Ok(()) }
}

/// The pending exception, taken (direct: no normalization), and put back.
pub(crate) struct Pending(*mut ffi::PyObject, *mut ffi::PyObject, *mut ffi::PyObject);

/// Takes the pending exception, if any.
#[must_use]
pub(crate) fn fetch_pending(_g: Gil<'_>) -> Pending {
    let mut p = Pending(ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
    // SAFETY: attached (the token); fetch only moves the references.
    unsafe {
        ffi::PyErr_Fetch(&raw mut p.0, &raw mut p.1, &raw mut p.2);
    }
    p
}

/// Puts back exactly what [`fetch_pending`] took, even nothing (a cleanup
/// entry that must leave the error indicator as it found it).
#[expect(
    clippy::needless_pass_by_value,
    reason = "the pending exception is consumed"
)]
pub(crate) fn put_back_pending(_g: Gil<'_>, p: Pending) {
    // SAFETY: attached (the token); restores exactly what was fetched
    // (references transferred). A callout: restoring replaces any
    // exception set meanwhile, and dropping it can run finalizers.
    unsafe { thunks::c_err_restore(p.0, p.1, p.2) };
}

/// Stashes the exception `raised` proves is set into `slot`, unless
/// `slot` already holds one: then the new exception is discarded (a
/// callout: dropping it can run finalizers) and the first is kept, so no
/// stash is ever overwritten and leaked.
///
/// The callers are the Python evaluator's batch callback and its signal
/// poll (`tessera-rl-py`, `search_api.rs`), and the ONNX evaluator's
/// signal poll (`onnx_api.rs`). A second stash is unreachable today: the
/// search drivers end the call
/// on the first evaluator error (`?` after every `check` and evaluation in
/// `Mcts::run_inner` and in `SearchGroup`'s `prologue` and `round`), so
/// the bindings raise the stashed exception before the evaluator is used
/// again. The guard keeps a later driver that continues after an error
/// from leaking the first exception.
pub fn stash_first(g: Gil<'_>, slot: &mut Option<Stashed>, raised: Raised) {
    if slot.is_some() {
        let _ = raised;
        clear_pending(g);
    } else {
        *slot = Some(Stashed::take(g, raised));
    }
}

/// The message of the exception `raised` proves is set, which stays set:
/// `str()` of its value (a callout: it can run Python code), or a fixed
/// text when there is no value or `str()` fails.
#[cfg(any(feature = "native-module", feature = "host"))]
#[must_use]
pub(crate) fn pending_message(g: Gil<'_>, raised: Raised) -> (String, Raised) {
    let _ = raised;
    let p = fetch_pending(g);
    let mut message = None;
    if !p.1.is_null() {
        // SAFETY: attached, no exception pending; a live value (the
        // fetched reference, kept until it is put back). `str()` may return
        // a `str` subclass that is GC-tracked and has a `__del__`, so the
        // result is owned as a container (its decref is a callout).
        let s = unsafe { thunks::c_object_str(p.1) };
        if let Ok(s) = Mine::container(s) {
            message = to_string(g, s.as_ref(), "message").ok();
        }
        clear_pending(g);
    }
    let message = message.unwrap_or_else(|| String::from("an exception without a message"));
    put_back_pending(g, p);
    (message, RAISED)
}

/// An exception taken out of the interpreter while other work runs (a
/// rollback), to be raised again afterwards. Made only from a [`Raised`],
/// so it always holds an exception. Dropping it unraised leaks the
/// exception's references.
pub struct Stashed(Pending);

impl Stashed {
    /// Takes the exception `raised` proves is set.
    #[must_use]
    pub fn take(g: Gil<'_>, raised: Raised) -> Self {
        let _ = raised;
        Self(fetch_pending(g))
    }

    /// Raises it again.
    #[must_use]
    pub fn raise_again(self, g: Gil<'_>) -> Raised {
        if self.0.0.is_null() {
            return raise(g, runtime_error(), "no exception to raise again");
        }
        put_back_pending(g, self.0);
        RAISED
    }
}

// -- building objects -----------------------------------------------------------------

/// A new reference to a singleton.
fn singleton<'py>(p: *mut ffi::PyObject) -> Mine<'py> {
    // SAFETY: a live singleton; incrementing runs no Python code.
    unsafe { ffi::Py_IncRef(p) };
    // SAFETY: singletons are never null.
    Mine(
        unsafe { NonNull::new_unchecked(p) },
        Tracked::Leaf,
        PhantomData,
    )
}

/// `None`, a new reference.
#[must_use]
pub fn none(_g: Gil<'_>) -> Mine<'_> {
    // SAFETY: the address of the `None` singleton.
    singleton(unsafe { ffi::Py_None() })
}

/// `True` or `False`, a new reference.
#[must_use]
pub fn boolean(_g: Gil<'_>, value: bool) -> Mine<'_> {
    if value {
        // SAFETY: the address of the `True` singleton.
        singleton(unsafe { ffi::Py_True() })
    } else {
        // SAFETY: the address of the `False` singleton.
        singleton(unsafe { ffi::Py_False() })
    }
}

/// An `int` (direct: not GC-tracked).
pub fn int_u64(_g: Gil<'_>, v: u64) -> CResult<Mine<'_>> {
    // SAFETY: attached.
    Mine::new(unsafe { ffi::PyLong_FromUnsignedLongLong(v) })
}

/// An `int` from a signed value (direct: not GC-tracked).
pub fn int_i64(_g: Gil<'_>, v: i64) -> CResult<Mine<'_>> {
    // SAFETY: attached (the `Gil` token); `PyLong_FromLongLong` takes any
    // `long long` and returns a new reference or null with an exception
    // set, which `Mine::new` turns into an error.
    Mine::new(unsafe { ffi::PyLong_FromLongLong(v) })
}

/// A `float` (direct: not GC-tracked).
pub fn float(_g: Gil<'_>, v: f64) -> CResult<Mine<'_>> {
    // SAFETY: attached.
    Mine::new(unsafe { ffi::PyFloat_FromDouble(v) })
}

fn ssize(g: Gil<'_>, len: usize) -> CResult<ffi::Py_ssize_t> {
    ffi::Py_ssize_t::try_from(len).map_err(|_| raise(g, runtime_error(), "size out of range"))
}

/// A `str` from UTF-8 (direct: not GC-tracked).
pub fn string<'py>(g: Gil<'py>, s: &str) -> CResult<Mine<'py>> {
    let Ok(len) = ssize(g, s.len()) else {
        return Err(raise(g, runtime_error(), "string too long"));
    };
    // SAFETY: attached; `len` bytes of valid UTF-8.
    Mine::new(unsafe { ffi::PyUnicode_FromStringAndSize(s.as_ptr().cast(), len) })
}

/// A `bytes` copy (direct: not GC-tracked).
pub fn bytes<'py>(g: Gil<'py>, data: &[u8]) -> CResult<Mine<'py>> {
    let Ok(len) = ssize(g, data.len()) else {
        return Err(raise(g, runtime_error(), "buffer too long"));
    };
    // SAFETY: attached; `len` readable bytes.
    Mine::new(unsafe { ffi::PyBytes_FromStringAndSize(data.as_ptr().cast(), len) })
}

/// A `bytearray` copy (direct: not GC-tracked).
pub fn bytearray<'py>(g: Gil<'py>, data: &[u8]) -> CResult<Mine<'py>> {
    let Ok(len) = ssize(g, data.len()) else {
        return Err(raise(g, runtime_error(), "buffer too long"));
    };
    // SAFETY: attached; `len` readable bytes.
    Mine::new(unsafe { ffi::PyByteArray_FromStringAndSize(data.as_ptr().cast(), len) })
}

/// A new empty `list` (callout: GC-tracked).
pub fn list(_g: Gil<'_>) -> CResult<Mine<'_>> {
    // SAFETY: attached.
    Mine::container(unsafe { thunks::c_list_new(0) })
}

/// Test hook: the `n`-th next container insertion or module attribute
/// fails with `RuntimeError` (0 disables), to exercise the unwind drops
/// and a module initialization that fails part way.
#[cfg(feature = "test-hooks")]
pub fn fail_after(n: u32) {
    FAIL_AFTER.with(|c| c.set(n));
}

#[cfg(feature = "test-hooks")]
std::thread_local! {
    static FAIL_AFTER: core::cell::Cell<u32> = const { core::cell::Cell::new(0) };
}

#[cfg(feature = "test-hooks")]
fn injected(g: Gil<'_>) -> CResult<()> {
    let hit = FAIL_AFTER.with(|c| match c.get() {
        0 => false,
        n => {
            c.set(n - 1);
            n == 1
        }
    });
    if hit {
        Err(raise(g, runtime_error(), "test hook: injected failure"))
    } else {
        Ok(())
    }
}

#[cfg(not(feature = "test-hooks"))]
#[inline]
#[expect(
    clippy::unnecessary_wraps,
    reason = "the same signature as the test-hooks build"
)]
fn injected(_g: Gil<'_>) -> CResult<()> {
    Ok(())
}

/// Appends `item` to `list` (callout: it mutates a GC-tracked container;
/// the item's reference is not stolen).
pub fn list_append(g: Gil<'_>, list: &Mine<'_>, item: &Mine<'_>) -> CResult<()> {
    injected(g)?;
    // SAFETY: attached; our list and a live item.
    let rc = unsafe { thunks::c_list_append(list.as_ref().as_ptr(), item.as_ref().as_ptr()) };
    if rc < 0 { Err(RAISED) } else { Ok(()) }
}

/// A `tuple` of `items` (callout: GC-tracked; the items are ours).
pub fn tuple<'py>(g: Gil<'py>, items: Vec<Mine<'py>>) -> CResult<Mine<'py>> {
    let n = ssize(g, items.len())?;
    // SAFETY: attached.
    let t = Mine::container(unsafe { thunks::c_tuple_new(n) })?;
    for (i, item) in items.into_iter().enumerate() {
        let i = ssize(g, i)?;
        // SAFETY: our new tuple, an index below its size; the reference is
        // stolen (ownership passes to the tuple). A callout: it mutates a
        // GC-tracked container and drops any previous item.
        let rc = unsafe { thunks::c_tuple_set_item(t.as_ref().as_ptr(), i, item.into_ptr()) };
        if rc < 0 {
            return Err(RAISED);
        }
    }
    Ok(t)
}

/// `callable(*args)` (callout: it runs arbitrary Python code, under the
/// caller's floating-point controls, ADR-0041). The result may be any
/// object, so it is owned as a container (its decref is a callout too).
pub fn call<'py>(_g: Gil<'py>, callable: Ref<'_>, args: &Mine<'_>) -> CResult<Mine<'py>> {
    // SAFETY: attached; a live callable and our argument tuple, both
    // borrowed for the call.
    Mine::container(unsafe { thunks::c_object_call(callable.as_ptr(), args.as_ref().as_ptr()) })
}

/// Discards the pending exception (callout: dropping it may run
/// finalizers).
pub fn clear_pending(_g: Gil<'_>) {
    // SAFETY: attached.
    unsafe { thunks::c_err_clear() };
}

/// Runs pending signal handlers (callout: they are Python code); an
/// exception they raise (`KeyboardInterrupt`) is returned.
pub fn check_signals(_g: Gil<'_>) -> CResult<()> {
    // SAFETY: attached.
    let rc = unsafe { thunks::c_check_signals() };
    if rc < 0 { Err(RAISED) } else { Ok(()) }
}

/// A new empty `dict` (callout: GC-tracked).
pub fn dict(_g: Gil<'_>) -> CResult<Mine<'_>> {
    // SAFETY: attached.
    Mine::container(unsafe { thunks::c_dict_new() })
}

/// `dict[key] = value` with an exact `str` key (callout: it mutates a
/// GC-tracked container, and replacing a value Python code stored there
/// runs its finalizer).
pub fn dict_set(g: Gil<'_>, dict: &Mine<'_>, key: &CStr, value: &Mine<'_>) -> CResult<()> {
    injected(g)?;
    // SAFETY: attached; our dict, a static key and a live value.
    let rc = unsafe {
        thunks::c_dict_set_item_string(
            dict.as_ref().as_ptr(),
            key.as_ptr(),
            value.as_ref().as_ptr(),
        )
    };
    if rc < 0 { Err(RAISED) } else { Ok(()) }
}

/// `module.name = value` (callout: sets a module attribute).
pub fn module_add(g: Gil<'_>, module: Ref<'_>, name: &CStr, value: &Mine<'_>) -> CResult<()> {
    injected(g)?;
    // SAFETY: attached; a live module, a C string, a live value (not
    // stolen).
    let rc =
        unsafe { thunks::c_module_add(module.as_ptr(), name.as_ptr(), value.as_ref().as_ptr()) };
    if rc < 0 { Err(RAISED) } else { Ok(()) }
}

/// `module.name = ty`, for a class this crate holds for the process
/// (`errors::register`; callout: sets a module attribute).
pub(crate) fn module_add_class(
    g: Gil<'_>,
    module: Ref<'_>,
    name: &CStr,
    ty: *mut ffi::PyObject,
) -> CResult<()> {
    injected(g)?;
    // SAFETY: attached; a live module, a C string, and a class held by a
    // never-released strong reference (not stolen here).
    let rc = unsafe { thunks::c_module_add(module.as_ptr(), name.as_ptr(), ty) };
    if rc < 0 { Err(RAISED) } else { Ok(()) }
}

// -- types, modules, instances, capsules ----------------------------------------------

/// A new heap type from `spec` (callout: a GC allocation).
pub(crate) fn type_from_spec<'py>(_g: Gil<'py>, spec: &mut ffi::PyType_Spec) -> CResult<Mine<'py>> {
    // SAFETY: attached; the spec lives through the call, its slot table is
    // static, and CPython copies what it keeps.
    Mine::container(unsafe { thunks::c_type_from_spec(&raw mut *spec) })
}

/// A new module from its static definition (callout: a GC allocation).
///
/// # Safety
///
/// `def` must be a valid module definition that lives for the process.
pub(crate) unsafe fn module_create(_g: Gil<'_>, def: *mut ffi::PyModuleDef) -> CResult<Mine<'_>> {
    // SAFETY: attached; per the contract.
    Mine::container(unsafe { thunks::c_module_create(def, ffi::PYTHON_API_VERSION as c_int) })
}

/// A new, zero-filled instance of one of our heap types (direct: our types
/// are not GC-tracked, so allocating one never runs `gc.collect`).
///
/// # Safety
///
/// `ty` must be one of our heap types (no `Py_TPFLAGS_HAVE_GC`).
pub(crate) unsafe fn alloc_instance(_g: Gil<'_>, ty: *mut ffi::PyObject) -> CResult<Mine<'_>> {
    // SAFETY: attached; per the contract. It takes a reference to the type,
    // as heap-type instances require.
    Mine::new(unsafe { ffi::PyType_GenericAlloc(ty.cast(), 0) })
}

/// Frees a heap-type instance in its `tp_dealloc`, after its own fields:
/// the type's `tp_free`, then the type reference the instance held
/// (direct: the module keeps the type alive, so this never deallocates it).
///
/// # Safety
///
/// `obj` must be a live instance of one of our heap types, allocated by
/// its `tp_alloc`, in its `tp_dealloc`, attached.
pub(crate) unsafe fn free_instance(obj: *mut ffi::PyObject) {
    // SAFETY: `obj` is a live object.
    let ty = unsafe { ffi::Py_TYPE(obj) };
    // SAFETY: `ty` is a heap type, which always has a tp_free slot.
    let free = unsafe { ffi::PyType_GetSlot(ty, ffi::Py_tp_free) };
    if !free.is_null() {
        // SAFETY: the tp_free slot holds a `freefunc`.
        let free: unsafe extern "C" fn(*mut core::ffi::c_void) =
            unsafe { core::mem::transmute(free) };
        // SAFETY: `obj` was allocated by this type's tp_alloc.
        unsafe { free(obj.cast()) };
    }
    // SAFETY: instances of heap types hold a reference to their type,
    // released after the instance is freed. A type is GC-tracked, so the
    // decref is a callout.
    unsafe { thunks::c_decref(ty.cast()) };
}

/// A new strong reference to `r`, returned as a raw pointer (for a
/// `Py_buffer.obj`).
pub(crate) fn incref(r: Ref<'_>) -> *mut ffi::PyObject {
    // SAFETY: a live object; incrementing runs no Python code.
    unsafe { ffi::Py_IncRef(r.as_ptr()) };
    r.as_ptr()
}

/// A capsule holding `ptr` under `name`, with a destructor thunk (direct:
/// capsules are not GC-tracked).
///
/// # Safety
///
/// `ptr` must be valid for the destructor, and `name` must outlive the
/// capsule.
pub(crate) unsafe fn capsule<'py>(
    _g: Gil<'py>,
    ptr: *mut core::ffi::c_void,
    name: &'static CStr,
    destructor: unsafe extern "C" fn(*mut ffi::PyObject),
) -> CResult<Mine<'py>> {
    // SAFETY: attached; per the contract.
    Mine::new(unsafe { ffi::PyCapsule_New(ptr, name.as_ptr(), Some(destructor)) })
}

/// A capsule over the function pointer `f`, with no name and no
/// destructor: the form `jax.ffi.register_ffi_target` takes (an XLA FFI
/// handler, ADR-0095). Safe: the pointer is never dereferenced by Python,
/// and a function pointer lives for the process.
pub(crate) fn fn_capsule(_g: Gil<'_>, f: *const core::ffi::c_void) -> CResult<Mine<'_>> {
    // SAFETY: attached (`_g`); a null name and no destructor are allowed.
    Mine::new(unsafe { ffi::PyCapsule_New(f.cast_mut(), core::ptr::null(), None) })
}

/// The pointer of a capsule still named `name`, if it is (direct: no
/// exception is set when it is not).
///
/// # Safety
///
/// `cap` must be a live capsule.
pub(crate) unsafe fn capsule_pointer(
    cap: *mut ffi::PyObject,
    name: &CStr,
) -> Option<*mut core::ffi::c_void> {
    // SAFETY: per the contract; `PyCapsule_IsValid` sets no exception.
    if unsafe { ffi::PyCapsule_IsValid(cap, name.as_ptr()) } != 1 {
        return None;
    }
    // SAFETY: a valid capsule of this name.
    Some(unsafe { ffi::PyCapsule_GetPointer(cap, name.as_ptr()) })
}

// -- reading arguments (exact types only) -----------------------------------------------

fn wrong(g: Gil<'_>, what: &str, expected: &str) -> Raised {
    raise(
        g,
        type_error(),
        &format!("{what} must be {expected} (convert it in Python first)"),
    )
}

/// The items of an exact `tuple` (borrowed).
pub fn tuple_items<'a>(g: Gil<'_>, t: Ref<'a>, what: &str) -> CResult<Vec<Ref<'a>>> {
    if !t.is_exact(&raw mut ffi::PyTuple_Type) {
        return Err(wrong(g, what, "a tuple"));
    }
    // SAFETY: an exact tuple; its size and borrowed items run no Python code.
    let n = unsafe { ffi::PyTuple_Size(t.as_ptr()) };
    (0..n)
        .map(|i| {
            // SAFETY: as above, an index below the size.
            let item = unsafe { ffi::PyTuple_GetItem(t.as_ptr(), i) };
            // SAFETY: a borrowed item, alive while the tuple is.
            unsafe { Ref::from_ptr(item) }
                .ok_or_else(|| raise(g, runtime_error(), "a tuple item is missing"))
        })
        .collect()
}

/// Whether `r` is exactly of `ty`, one of our heap types.
pub(crate) fn is_instance_of(r: Ref<'_>, ty: *mut ffi::PyObject) -> bool {
    r.is_exact(ty.cast())
}

/// The exact built-in type of a value, for decoding (no exception and no
/// Python code either way): subclasses are [`Kind::Other`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `True` or `False`.
    Bool,
    /// An exact `int` (not a `bool`).
    Int,
    /// An exact `float`.
    Float,
    /// An exact `str`.
    Str,
    /// An exact `dict`.
    Dict,
    /// `None`.
    None,
    /// Anything else.
    Other,
}

/// The exact kind of `r`.
#[must_use]
pub fn kind(r: Ref<'_>) -> Kind {
    let ptr = r.as_ptr();
    // SAFETY: the address of the `True` singleton; nothing is read.
    let yes = unsafe { ffi::Py_True() };
    // SAFETY: the address of the `False` singleton.
    let no = unsafe { ffi::Py_False() };
    // SAFETY: the address of the `None` singleton.
    let none = unsafe { ffi::Py_None() };
    if ptr == yes || ptr == no {
        Kind::Bool
    } else if ptr == none {
        Kind::None
    } else if r.is_exact(&raw mut ffi::PyLong_Type) {
        Kind::Int
    } else if r.is_exact(&raw mut ffi::PyFloat_Type) {
        Kind::Float
    } else if r.is_exact(&raw mut ffi::PyUnicode_Type) {
        Kind::Str
    } else if r.is_exact(&raw mut ffi::PyDict_Type) {
        Kind::Dict
    } else {
        Kind::Other
    }
}

/// Whether `r` is an exact `list` (no exception either way).
#[must_use]
pub fn is_list(r: Ref<'_>) -> bool {
    r.is_exact(&raw mut ffi::PyList_Type)
}

/// The items of an exact `list`, owned: a list can change while Python
/// code runs (any callout), so a borrowed item could be freed under us. An
/// item is a user object, so its decref is a callout ([`Mine`]).
pub fn list_items<'py>(g: Gil<'py>, l: Ref<'_>, what: &str) -> CResult<Vec<Mine<'py>>> {
    if !l.is_exact(&raw mut ffi::PyList_Type) {
        return Err(wrong(g, what, "a list"));
    }
    // SAFETY: an exact list.
    let n = unsafe { ffi::PyList_Size(l.as_ptr()) };
    (0..n)
        .map(|i| {
            // SAFETY: as above, an index below the size; nothing runs
            // Python code between the size and this read.
            let item = unsafe { ffi::PyList_GetItem(l.as_ptr(), i) };
            Mine::owned_user(g, item)
        })
        .collect()
}

/// The `(key, value)` items of an exact `dict` with exact `str` keys, the
/// values owned (as [`list_items`]: the dict can change while Python code
/// runs).
pub fn dict_items<'py>(g: Gil<'py>, d: Ref<'_>, what: &str) -> CResult<Vec<(String, Mine<'py>)>> {
    if !d.is_exact(&raw mut ffi::PyDict_Type) {
        return Err(wrong(g, what, "a dict"));
    }
    // Owned first, without running Python code; the keys are decoded after.
    let mut owned = Vec::new();
    let mut pos: ffi::Py_ssize_t = 0;
    let (mut key, mut value) = (ptr::null_mut(), ptr::null_mut());
    // SAFETY: an exact dict; `PyDict_Next` reads entries without running
    // Python code.
    while unsafe { ffi::PyDict_Next(d.as_ptr(), &raw mut pos, &raw mut key, &raw mut value) } != 0 {
        if key.is_null() || value.is_null() {
            return Err(raise(
                g,
                runtime_error(),
                "a dict entry without a key or value",
            ));
        }
        owned.push((Mine::owned_user(g, key)?, Mine::owned_user(g, value)?));
    }
    owned
        .into_iter()
        .map(|(k, v)| Ok((to_string(g, k.as_ref(), "a keyword name")?, v)))
        .collect()
}

/// An exact `str`, copied (callout: decoding may raise).
pub fn to_string(g: Gil<'_>, s: Ref<'_>, what: &str) -> CResult<String> {
    if !s.is_exact(&raw mut ffi::PyUnicode_Type) {
        return Err(wrong(g, what, "a str"));
    }
    let mut len: ffi::Py_ssize_t = 0;
    // SAFETY: an exact str; the buffer stays valid while the object lives.
    let data = unsafe { thunks::c_unicode_as_utf8(s.as_ptr(), &raw mut len) };
    if data.is_null() {
        return Err(RAISED);
    }
    let len = usize::try_from(len).map_err(|_| raise(g, runtime_error(), "a negative length"))?;
    // SAFETY: `len` bytes of UTF-8 owned by the live str.
    let bytes = unsafe { core::slice::from_raw_parts(data.cast::<u8>(), len) };
    Ok(String::from_utf8_lossy(bytes).into_owned())
}

/// An exact `int` as `u64` (callout: overflow raises).
pub fn to_u64(g: Gil<'_>, v: Ref<'_>, what: &str) -> CResult<u64> {
    if !v.is_exact(&raw mut ffi::PyLong_Type) {
        return Err(wrong(g, what, "an int"));
    }
    // SAFETY: an exact int; no `__index__` is called.
    let x = unsafe { thunks::c_long_as_u64(v.as_ptr()) };
    // SAFETY: attached.
    if x == u64::MAX && !unsafe { ffi::PyErr_Occurred() }.is_null() {
        return Err(RAISED);
    }
    Ok(x)
}

/// An exact `int` as `i64` (callout: overflow raises).
pub fn to_i64(g: Gil<'_>, v: Ref<'_>, what: &str) -> CResult<i64> {
    if !v.is_exact(&raw mut ffi::PyLong_Type) {
        return Err(wrong(g, what, "an int"));
    }
    // SAFETY: an exact int.
    let x = unsafe { thunks::c_long_as_i64(v.as_ptr()) };
    // SAFETY: attached.
    if x == -1 && !unsafe { ffi::PyErr_Occurred() }.is_null() {
        return Err(RAISED);
    }
    Ok(x)
}

/// An exact `float` as `f64`. Not a callout: for an exact float,
/// `PyFloat_AsDouble` reads the value and never calls `__float__`.
pub fn to_f64(g: Gil<'_>, v: Ref<'_>, what: &str) -> CResult<f64> {
    if !v.is_exact(&raw mut ffi::PyFloat_Type) {
        return Err(wrong(g, what, "a float"));
    }
    // SAFETY: attached; an exact float.
    Ok(unsafe { ffi::PyFloat_AsDouble(v.as_ptr()) })
}

/// An exact `int` in `u32` range.
pub fn to_u32(g: Gil<'_>, v: Ref<'_>, what: &str) -> CResult<u32> {
    let x = to_u64(g, v, what)?;
    u32::try_from(x).map_err(|_| out_of_range(g, what))
}

/// The error of an integer outside its type.
#[cold]
#[must_use]
pub fn out_of_range(g: Gil<'_>, what: &str) -> Raised {
    raise(g, type_error(), &format!("{what} is out of range"))
}

/// An exact `bool`.
pub fn to_bool(g: Gil<'_>, v: Ref<'_>, what: &str) -> CResult<bool> {
    // SAFETY: the bool singletons' addresses.
    let t = unsafe { ffi::Py_True() };
    // SAFETY: the address of the `False` singleton.
    let f = unsafe { ffi::Py_False() };
    if v.as_ptr() == t {
        Ok(true)
    } else if v.as_ptr() == f {
        Ok(false)
    } else {
        Err(wrong(g, what, "a bool"))
    }
}

/// The bytes of an exact `bytes` (borrowed for the object's lifetime).
pub fn bytes_of<'a>(g: Gil<'_>, b: Ref<'a>, what: &str) -> CResult<&'a [u8]> {
    if !b.is_exact(&raw mut ffi::PyBytes_Type) {
        return Err(wrong(g, what, "bytes"));
    }
    // SAFETY: an exact bytes object: its buffer and size never fail and
    // are immutable while it lives.
    let data = unsafe { ffi::PyBytes_AsString(b.as_ptr()) };
    // SAFETY: as above.
    let len = unsafe { ffi::PyBytes_Size(b.as_ptr()) };
    let len = usize::try_from(len).map_err(|_| raise(g, runtime_error(), "a negative length"))?;
    // SAFETY: `len` immutable bytes owned by the live object.
    Ok(unsafe { core::slice::from_raw_parts(data.cast::<u8>(), len) })
}

/// Whether `b` is an exact `memoryview`.
#[must_use]
pub fn is_memoryview(b: Ref<'_>) -> bool {
    b.is_exact(&raw mut ffi::PyMemoryView_Type)
}

/// An owned copy of an exact `memoryview`'s buffer, made by
/// [`memoryview_copy`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferCopy {
    /// A copy of the buffer's bytes in C order; empty unless the buffer is
    /// C-contiguous (a buffer that is not is refused, so it is not copied).
    pub bytes: Vec<u8>,
    /// The size of one item, in bytes.
    pub itemsize: usize,
    /// The item format (`struct` syntax, `"B"` when the exporter gives none).
    pub format: Vec<u8>,
    /// Whether the buffer is C-contiguous.
    pub c_contiguous: bool,
}

/// A copy of the buffer of the exact `memoryview` `b`, made by CPython
/// (`PyBuffer_ToContiguous`) into memory Rust owns; the export is released
/// before this returns.
///
/// `TypeError` when `b` is not an exact `memoryview`; the `memoryview`'s
/// own exception (a released view: `ValueError`) when the request fails.
/// The Python layer makes the view from a user object (exact types rule:
/// the user object's exporter runs there, under the caller's controls);
/// here only `memoryview`'s own C getbuffer runs, through a callout.
///
/// The exported memory is shared: a thread that has released the
/// interpreter (NumPy inside a large copy, another C extension) can write
/// it at any time. So Rust never forms a reference over it; CPython copies
/// it, and the copy may hold a mix of old and new bytes. The caller treats
/// the copy as untrusted input: every `i32` bit pattern is a valid value,
/// and each one is validated on the copy.
pub fn memoryview_copy(g: Gil<'_>, b: Ref<'_>, what: &str) -> CResult<BufferCopy> {
    /// Releases the buffer on every exit.
    struct Release(*mut ffi::Py_buffer);
    impl Drop for Release {
        fn drop(&mut self) {
            // SAFETY: a buffer filled by a successful PyObject_GetBuffer,
            // released once.
            unsafe { thunks::c_buffer_release(self.0) };
        }
    }
    if !b.is_exact(&raw mut ffi::PyMemoryView_Type) {
        return Err(wrong(g, what, "bytes or a memoryview"));
    }
    let mut view = core::mem::MaybeUninit::<ffi::Py_buffer>::uninit();
    // SAFETY: a live exact memoryview and a writable Py_buffer; a read-only
    // request with strides and format, which a live memoryview always
    // satisfies (a released one raises).
    let rc = unsafe { thunks::c_get_buffer(b.as_ptr(), view.as_mut_ptr(), ffi::PyBUF_RECORDS_RO) };
    if rc != 0 {
        return Err(RAISED);
    }
    let release = Release(view.as_mut_ptr());
    let v = view.as_ptr();
    // SAFETY: `c"C"` is a live NUL-terminated string; its first byte is
    // read as the platform's `c_char`.
    let order = unsafe { *c"C".as_ptr() };
    // SAFETY: a filled buffer; IsContiguous reads only the struct's own
    // fields (shape, strides, itemsize), not the exported memory.
    let c_contiguous = unsafe { ffi::PyBuffer_IsContiguous(v, order) } == 1;
    // SAFETY: a filled buffer; reading its own fields (plain integers and
    // the format pointer), not the exported memory.
    let fields = unsafe { &*v };
    let (len, itemsize, format_ptr) = (fields.len, fields.itemsize, fields.format);
    let itemsize = usize::try_from(itemsize).unwrap_or(0);
    let format = if format_ptr.is_null() {
        b"B".to_vec()
    } else {
        // SAFETY: the export's NUL-terminated format string, fixed for the
        // export's lifetime (no exporter rewrites it), copied at once.
        unsafe { CStr::from_ptr(format_ptr) }.to_bytes().to_vec()
    };
    let mut bytes: Vec<u8> = Vec::new();
    let n = if c_contiguous {
        usize::try_from(len).unwrap_or(0)
    } else {
        0
    };
    if n > 0 {
        bytes.reserve_exact(n);
        // SAFETY: `n` writable bytes of owned spare capacity, which no one
        // else can reach; CPython copies the export into them (a callout: on
        // failure it sets an exception). No Rust reference to the exported
        // memory is formed.
        let rc =
            unsafe { thunks::c_buffer_to_contiguous(bytes.as_mut_ptr().cast(), v, len, order) };
        if rc != 0 {
            return Err(RAISED);
        }
        // SAFETY: the successful copy initialized all `n` bytes.
        unsafe { bytes.set_len(n) };
    }
    drop(release);
    Ok(BufferCopy {
        bytes,
        itemsize,
        format,
        c_contiguous,
    })
}

#[cfg(test)]
mod tests {
    //! The tokens are not `Send` or `Sync` (no escape to another thread or
    //! into a detached closure), checked at compile time.

    use super::{Gil, Mine, Ref};

    /// Fails to compile when `T` is `Send`: the impls for `()` and for
    /// `u8` both apply, so the call is ambiguous (static_assertions'
    /// `assert_not_impl_any`).
    trait AmbiguousIfSend<A> {
        fn some_item() {}
    }
    impl<T: ?Sized> AmbiguousIfSend<()> for T {}
    impl<T: ?Sized + Send> AmbiguousIfSend<u8> for T {}

    trait AmbiguousIfSync<A> {
        fn some_item() {}
    }
    impl<T: ?Sized> AmbiguousIfSync<()> for T {}
    impl<T: ?Sized + Sync> AmbiguousIfSync<u8> for T {}

    #[test]
    fn test_tokens_and_references_are_neither_send_nor_sync() {
        <Gil<'static> as AmbiguousIfSend<_>>::some_item();
        <Gil<'static> as AmbiguousIfSync<_>>::some_item();
        <Ref<'static> as AmbiguousIfSend<_>>::some_item();
        <Ref<'static> as AmbiguousIfSync<_>>::some_item();
        <Mine<'static> as AmbiguousIfSend<_>>::some_item();
        <Mine<'static> as AmbiguousIfSync<_>>::some_item();
    }
}
