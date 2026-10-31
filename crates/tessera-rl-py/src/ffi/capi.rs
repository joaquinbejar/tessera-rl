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
//!   `PyUnicode_AsUTF8AndSize`); and a warning.
//! - **Direct** (cannot run Python code): non-GC allocations (`str`,
//!   `bytes`, `bytearray`, `int`, `float`, capsules, and instances of our
//!   own types, which are not GC-tracked); reads of exact built-in types
//!   (`PyTuple_Size`/`GetItem`, `PyList_Size`/`GetItem`, `PyDict_Next`,
//!   `PyBytes_AsString`/`Size`); `PyErr_Fetch`,
//!   `PyErr_Occurred`, and `PyErr_Restore` with no exception set (it then
//!   decrefs nothing); reference increments; releasing and reacquiring the
//!   interpreter; and a decref of a leaf [`Mine`]. `PyErr_Clear` is never
//!   called: dropping a pending exception can run finalizers through its
//!   traceback, so it would be a callout.
//!
//! **Decref invariant.** The bindings never own a reference to a user
//! object: everything they read from Python (arguments, keyword values,
//! items) is borrowed ([`Ref`]) and never decref'd. The only owned
//! references are [`Mine`]: objects we created. A leaf (not GC-tracked)
//! cannot hold or reach a user object, so its decref is direct. A
//! GC-tracked container can: Python code running in any callout may find
//! it and store user objects in it. So its decref, on success and on every
//! failure or unwind path that drops a partly built container, is a
//! callout (`Mine::drop`). Owning any other reference that may reach a
//! user object would also need a callout decref; add one to the table
//! first.
//!
//! **Borrow invariant.** A [`Ref`] is valid only while its owner keeps the
//! object alive. A `Ref` borrowed from a container a user can reach (a
//! keyword dict, a tuple or list the caller passed) must not be used after
//! a callout that can run Python code: that code can mutate the container
//! and drop the object. Read every argument first, then allocate.
//!
//! **Exact types.** Arguments are read only from exact built-in types
//! (`int`, `bool`, `str`, `bytes`, `tuple`, `list`, `dict`, `None`); any
//! other object raises `TypeError` without calling its `__index__`,
//! `__bool__`, `__iter__` or anything else. The Python layer converts user
//! objects (with `int()`, NumPy) before calling the native module, under
//! the caller's own controls, which is the one place non-exact objects
//! are accepted.

#![expect(unsafe_code, reason = "audited module: the CPython C API")]

use core::ffi::{CStr, c_int};
use core::marker::PhantomData;
use core::ptr::{self, NonNull};

use pyo3::ffi;

use super::thunks;

/// Proof that the interpreter is attached to this thread. Only entries
/// (`ffi::entry`) make one.
#[derive(Clone, Copy)]
pub(crate) struct Gil(PhantomData<*mut ()>);

impl Gil {
    /// # Safety
    ///
    /// The interpreter must be attached to the calling thread for as long
    /// as the token is used.
    pub(crate) const unsafe fn assume() -> Self {
        Self(PhantomData)
    }

    /// Runs `f` with the interpreter released (the engine's work).
    #[expect(
        clippy::unused_self,
        reason = "the token proves the interpreter is attached"
    )]
    pub(crate) fn detach<R>(self, f: impl FnOnce() -> R) -> R {
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

/// An exception is set (the C-API failure convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Raised;

/// The result of a C-API operation: `Err` means an exception is set.
pub(crate) type CResult<T> = Result<T, Raised>;

/// A borrowed reference, valid for `'a`.
#[derive(Clone, Copy)]
pub(crate) struct Ref<'a>(NonNull<ffi::PyObject>, PhantomData<&'a ()>);

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
    pub(crate) fn is_none(self) -> bool {
        // SAFETY: reading the address of the `None` singleton.
        self.as_ptr() == unsafe { ffi::Py_None() }
    }
}

/// An owned reference to an object we created. A leaf (not GC-tracked:
/// `int`, `float`, `str`, `bytes`, capsules, our own instances) is dropped
/// with a direct decref; a GC-tracked container is dropped through a
/// callout, because Python code may have found it and stored user objects
/// in it (decref invariant).
pub(crate) struct Mine(NonNull<ffi::PyObject>, Tracked);

/// Whether the garbage collector tracks the object, so Python code can
/// reach it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tracked {
    Leaf,
    Container,
}

impl Mine {
    /// Takes ownership of a new reference; `None` (an exception is set)
    /// for null.
    fn new(ptr: *mut ffi::PyObject) -> CResult<Self> {
        NonNull::new(ptr)
            .map(|p| Self(p, Tracked::Leaf))
            .ok_or(Raised)
    }

    /// As [`Mine::new`], for a GC-tracked container.
    fn container(ptr: *mut ffi::PyObject) -> CResult<Self> {
        NonNull::new(ptr)
            .map(|p| Self(p, Tracked::Container))
            .ok_or(Raised)
    }

    pub(crate) fn as_ref(&self) -> Ref<'_> {
        Ref(self.0, PhantomData)
    }

    /// Releases ownership, for a return value.
    pub(crate) fn into_ptr(self) -> *mut ffi::PyObject {
        let ptr = self.0.as_ptr();
        core::mem::forget(self);
        ptr
    }
}

impl Drop for Mine {
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

/// Raises `ty(message)` (callout: setting an exception allocates).
pub(crate) fn raise(_g: Gil, ty: *mut ffi::PyObject, message: &str) -> Raised {
    let text = std::ffi::CString::new(message.replace('\0', " ")).unwrap_or_default();
    // SAFETY: attached; `ty` is a live exception type and `text` a C
    // string alive for the call.
    unsafe { thunks::c_err_set_string(ty, text.as_ptr()) };
    Raised
}

/// The built-in exception types the bindings raise.
pub(crate) fn type_error() -> *mut ffi::PyObject {
    // SAFETY: reading an interpreter global.
    unsafe { ffi::PyExc_TypeError }
}
pub(crate) fn runtime_error() -> *mut ffi::PyObject {
    // SAFETY: reading an interpreter global.
    unsafe { ffi::PyExc_RuntimeError }
}
pub(crate) fn buffer_error() -> *mut ffi::PyObject {
    // SAFETY: reading an interpreter global.
    unsafe { ffi::PyExc_BufferError }
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
pub(crate) fn new_exception(
    _g: Gil,
    name: &CStr,
    doc: &CStr,
    base: *mut ffi::PyObject,
) -> CResult<Mine> {
    // SAFETY: attached; static C strings and a live base type.
    Mine::container(unsafe {
        thunks::c_err_new_exception(name.as_ptr(), doc.as_ptr(), base, ptr::null_mut())
    })
}

/// Warns with `category` (callout: runs filters and hooks).
pub(crate) fn warn(_g: Gil, category: *mut ffi::PyObject, message: &CStr) -> CResult<()> {
    // SAFETY: attached; a live category and a static message.
    let rc = unsafe { thunks::c_err_warn(category, message.as_ptr(), 1) };
    if rc < 0 { Err(Raised) } else { Ok(()) }
}

/// The pending exception, taken (direct: no normalization), and put back.
pub(crate) struct Pending(*mut ffi::PyObject, *mut ffi::PyObject, *mut ffi::PyObject);

pub(crate) fn fetch_pending() -> Pending {
    let mut p = Pending(ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
    // SAFETY: attached by contract of the callers (cleanup entries); fetch
    // only moves the references.
    unsafe {
        ffi::PyErr_Fetch(&raw mut p.0, &raw mut p.1, &raw mut p.2);
    }
    p
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "the pending exception is consumed"
)]
pub(crate) fn restore_pending(p: Pending) {
    // SAFETY: restores exactly what was fetched (references transferred).
    unsafe {
        ffi::PyErr_Restore(p.0, p.1, p.2);
    }
}

// -- building objects -----------------------------------------------------------------

/// A new reference to a singleton.
fn singleton(p: *mut ffi::PyObject) -> Mine {
    // SAFETY: a live singleton; incrementing runs no Python code.
    unsafe { ffi::Py_IncRef(p) };
    // SAFETY: singletons are never null.
    Mine(unsafe { NonNull::new_unchecked(p) }, Tracked::Leaf)
}

/// `None`, a new reference.
pub(crate) fn none(_g: Gil) -> Mine {
    // SAFETY: the address of the `None` singleton.
    singleton(unsafe { ffi::Py_None() })
}

/// `True` or `False`, a new reference.
pub(crate) fn boolean(_g: Gil, value: bool) -> Mine {
    if value {
        // SAFETY: the address of the `True` singleton.
        singleton(unsafe { ffi::Py_True() })
    } else {
        // SAFETY: the address of the `False` singleton.
        singleton(unsafe { ffi::Py_False() })
    }
}

/// An `int` (direct: not GC-tracked).
pub(crate) fn int_u64(_g: Gil, v: u64) -> CResult<Mine> {
    // SAFETY: attached.
    Mine::new(unsafe { ffi::PyLong_FromUnsignedLongLong(v) })
}

/// A `float` (direct: not GC-tracked).
pub(crate) fn float(_g: Gil, v: f64) -> CResult<Mine> {
    // SAFETY: attached.
    Mine::new(unsafe { ffi::PyFloat_FromDouble(v) })
}

fn ssize(len: usize) -> CResult<ffi::Py_ssize_t> {
    ffi::Py_ssize_t::try_from(len).map_err(|_| Raised)
}

/// A `str` from UTF-8 (direct: not GC-tracked).
pub(crate) fn string(g: Gil, s: &str) -> CResult<Mine> {
    let Ok(len) = ssize(s.len()) else {
        return Err(raise(g, runtime_error(), "string too long"));
    };
    // SAFETY: attached; `len` bytes of valid UTF-8.
    Mine::new(unsafe { ffi::PyUnicode_FromStringAndSize(s.as_ptr().cast(), len) })
}

/// A `bytes` copy (direct: not GC-tracked).
pub(crate) fn bytes(g: Gil, data: &[u8]) -> CResult<Mine> {
    let Ok(len) = ssize(data.len()) else {
        return Err(raise(g, runtime_error(), "buffer too long"));
    };
    // SAFETY: attached; `len` readable bytes.
    Mine::new(unsafe { ffi::PyBytes_FromStringAndSize(data.as_ptr().cast(), len) })
}

/// A `bytearray` copy (direct: not GC-tracked).
pub(crate) fn bytearray(g: Gil, data: &[u8]) -> CResult<Mine> {
    let Ok(len) = ssize(data.len()) else {
        return Err(raise(g, runtime_error(), "buffer too long"));
    };
    // SAFETY: attached; `len` readable bytes.
    Mine::new(unsafe { ffi::PyByteArray_FromStringAndSize(data.as_ptr().cast(), len) })
}

/// A new empty `list` (callout: GC-tracked).
pub(crate) fn list(_g: Gil) -> CResult<Mine> {
    // SAFETY: attached.
    Mine::container(unsafe { thunks::c_list_new(0) })
}

/// Test hook: the `n`-th next container insertion fails with
/// `RuntimeError` (0 disables), to exercise the unwind drops.
#[cfg(feature = "test-hooks")]
pub(crate) fn fail_after(n: u32) {
    FAIL_AFTER.with(|c| c.set(n));
}

#[cfg(feature = "test-hooks")]
std::thread_local! {
    static FAIL_AFTER: core::cell::Cell<u32> = const { core::cell::Cell::new(0) };
}

#[cfg(feature = "test-hooks")]
fn injected(g: Gil) -> CResult<()> {
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
fn injected(_g: Gil) -> CResult<()> {
    Ok(())
}

/// Appends `item` to `list` (callout: it mutates a GC-tracked container;
/// the item's reference is not stolen).
pub(crate) fn list_append(g: Gil, list: &Mine, item: &Mine) -> CResult<()> {
    injected(g)?;
    // SAFETY: attached; our list and a live item.
    let rc = unsafe { thunks::c_list_append(list.as_ref().as_ptr(), item.as_ref().as_ptr()) };
    if rc < 0 { Err(Raised) } else { Ok(()) }
}

/// A `tuple` of `items` (callout: GC-tracked; the items are ours).
pub(crate) fn tuple(g: Gil, items: Vec<Mine>) -> CResult<Mine> {
    let n = ssize(items.len()).map_err(|_| raise(g, runtime_error(), "tuple too long"))?;
    // SAFETY: attached.
    let t = Mine::container(unsafe { thunks::c_tuple_new(n) })?;
    for (i, item) in items.into_iter().enumerate() {
        let i = ssize(i)?;
        // SAFETY: our new tuple, an index below its size; the reference is
        // stolen (ownership passes to the tuple). A callout: it mutates a
        // GC-tracked container and drops any previous item.
        let rc = unsafe { thunks::c_tuple_set_item(t.as_ref().as_ptr(), i, item.into_ptr()) };
        if rc < 0 {
            return Err(Raised);
        }
    }
    Ok(t)
}

/// A new empty `dict` (callout: GC-tracked).
pub(crate) fn dict(_g: Gil) -> CResult<Mine> {
    // SAFETY: attached.
    Mine::container(unsafe { thunks::c_dict_new() })
}

/// `dict[key] = value` with an exact `str` key (callout: it mutates a
/// GC-tracked container, and replacing a value Python code stored there
/// runs its finalizer).
pub(crate) fn dict_set(g: Gil, dict: &Mine, key: &CStr, value: &Mine) -> CResult<()> {
    injected(g)?;
    // SAFETY: attached; our dict, a static key and a live value.
    let rc = unsafe {
        thunks::c_dict_set_item_string(
            dict.as_ref().as_ptr(),
            key.as_ptr(),
            value.as_ref().as_ptr(),
        )
    };
    if rc < 0 { Err(Raised) } else { Ok(()) }
}

/// `module.name = value` (callout: sets a module attribute).
pub(crate) fn module_add(_g: Gil, module: Ref<'_>, name: &CStr, value: &Mine) -> CResult<()> {
    // SAFETY: attached; a live module, a C string, a live value (not
    // stolen).
    let rc =
        unsafe { thunks::c_module_add(module.as_ptr(), name.as_ptr(), value.as_ref().as_ptr()) };
    if rc < 0 { Err(Raised) } else { Ok(()) }
}

// -- types, modules, instances, capsules ----------------------------------------------

/// A new heap type from `spec` (callout: a GC allocation).
pub(crate) fn type_from_spec(_g: Gil, spec: &mut ffi::PyType_Spec) -> CResult<Mine> {
    // SAFETY: attached; the spec lives through the call, its slot table is
    // static, and CPython copies what it keeps.
    Mine::container(unsafe { thunks::c_type_from_spec(&raw mut *spec) })
}

/// A new module from its static definition (callout: a GC allocation).
///
/// # Safety
///
/// `def` must be a valid module definition that lives for the process.
pub(crate) unsafe fn module_create(_g: Gil, def: *mut ffi::PyModuleDef) -> CResult<Mine> {
    // SAFETY: attached; per the contract.
    Mine::container(unsafe { thunks::c_module_create(def, ffi::PYTHON_API_VERSION as c_int) })
}

/// A new, zero-filled instance of one of our heap types (direct: our types
/// are not GC-tracked, so allocating one never runs `gc.collect`).
///
/// # Safety
///
/// `ty` must be one of our heap types (no `Py_TPFLAGS_HAVE_GC`).
pub(crate) unsafe fn alloc_instance(_g: Gil, ty: *mut ffi::PyObject) -> CResult<Mine> {
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
pub(crate) unsafe fn capsule(
    _g: Gil,
    ptr: *mut core::ffi::c_void,
    name: &'static CStr,
    destructor: unsafe extern "C" fn(*mut ffi::PyObject),
) -> CResult<Mine> {
    // SAFETY: attached; per the contract.
    Mine::new(unsafe { ffi::PyCapsule_New(ptr, name.as_ptr(), Some(destructor)) })
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

fn wrong(g: Gil, what: &str, expected: &str) -> Raised {
    raise(
        g,
        type_error(),
        &format!("{what} must be {expected} (convert it in Python first)"),
    )
}

/// The items of an exact `tuple` (borrowed).
pub(crate) fn tuple_items<'a>(g: Gil, t: Ref<'a>, what: &str) -> CResult<Vec<Ref<'a>>> {
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
            unsafe { Ref::from_ptr(item) }.ok_or(Raised)
        })
        .collect()
}

/// Whether `r` is an exact `list` (no exception either way).
pub(crate) fn is_list(r: Ref<'_>) -> bool {
    r.is_exact(&raw mut ffi::PyList_Type)
}

/// The items of an exact `list` (borrowed).
pub(crate) fn list_items<'a>(g: Gil, l: Ref<'a>, what: &str) -> CResult<Vec<Ref<'a>>> {
    if !l.is_exact(&raw mut ffi::PyList_Type) {
        return Err(wrong(g, what, "a list"));
    }
    // SAFETY: an exact list.
    let n = unsafe { ffi::PyList_Size(l.as_ptr()) };
    (0..n)
        .map(|i| {
            // SAFETY: as above, an index below the size.
            let item = unsafe { ffi::PyList_GetItem(l.as_ptr(), i) };
            // SAFETY: a borrowed item, alive while the list is unchanged
            // (no Python code runs while we read it).
            unsafe { Ref::from_ptr(item) }.ok_or(Raised)
        })
        .collect()
}

/// The `(key, value)` items of an exact `dict` with exact `str` keys
/// (borrowed).
pub(crate) fn dict_items<'a>(g: Gil, d: Ref<'a>, what: &str) -> CResult<Vec<(String, Ref<'a>)>> {
    if !d.is_exact(&raw mut ffi::PyDict_Type) {
        return Err(wrong(g, what, "a dict"));
    }
    let mut out = Vec::new();
    let mut pos: ffi::Py_ssize_t = 0;
    let (mut key, mut value) = (ptr::null_mut(), ptr::null_mut());
    // SAFETY: an exact dict; `PyDict_Next` reads entries without running
    // Python code.
    while unsafe { ffi::PyDict_Next(d.as_ptr(), &raw mut pos, &raw mut key, &raw mut value) } != 0 {
        // SAFETY: borrowed entries of a live dict.
        let (Some(k), Some(v)) = (unsafe { Ref::from_ptr(key) }, unsafe {
            Ref::from_ptr(value)
        }) else {
            return Err(raise(
                g,
                runtime_error(),
                "a dict entry without a key or value",
            ));
        };
        out.push((to_string(g, k, "a keyword name")?, v));
    }
    Ok(out)
}

/// An exact `str`, copied (callout: decoding may raise).
pub(crate) fn to_string(g: Gil, s: Ref<'_>, what: &str) -> CResult<String> {
    if !s.is_exact(&raw mut ffi::PyUnicode_Type) {
        return Err(wrong(g, what, "a str"));
    }
    let mut len: ffi::Py_ssize_t = 0;
    // SAFETY: an exact str; the buffer stays valid while the object lives.
    let data = unsafe { thunks::c_unicode_as_utf8(s.as_ptr(), &raw mut len) };
    if data.is_null() {
        return Err(Raised);
    }
    let len = usize::try_from(len).map_err(|_| Raised)?;
    // SAFETY: `len` bytes of UTF-8 owned by the live str.
    let bytes = unsafe { core::slice::from_raw_parts(data.cast::<u8>(), len) };
    Ok(String::from_utf8_lossy(bytes).into_owned())
}

/// An exact `int` as `u64` (callout: overflow raises).
pub(crate) fn to_u64(g: Gil, v: Ref<'_>, what: &str) -> CResult<u64> {
    if !v.is_exact(&raw mut ffi::PyLong_Type) {
        return Err(wrong(g, what, "an int"));
    }
    // SAFETY: an exact int; no `__index__` is called.
    let x = unsafe { thunks::c_long_as_u64(v.as_ptr()) };
    // SAFETY: attached.
    if x == u64::MAX && !unsafe { ffi::PyErr_Occurred() }.is_null() {
        return Err(Raised);
    }
    Ok(x)
}

/// An exact `int` as `i64` (callout: overflow raises).
pub(crate) fn to_i64(g: Gil, v: Ref<'_>, what: &str) -> CResult<i64> {
    if !v.is_exact(&raw mut ffi::PyLong_Type) {
        return Err(wrong(g, what, "an int"));
    }
    // SAFETY: an exact int.
    let x = unsafe { thunks::c_long_as_i64(v.as_ptr()) };
    // SAFETY: attached.
    if x == -1 && !unsafe { ffi::PyErr_Occurred() }.is_null() {
        return Err(Raised);
    }
    Ok(x)
}

/// An exact `int` in `u32` range.
pub(crate) fn to_u32(g: Gil, v: Ref<'_>, what: &str) -> CResult<u32> {
    let x = to_u64(g, v, what)?;
    u32::try_from(x).map_err(|_| raise(g, type_error(), &format!("{what} is out of range")))
}

/// An exact `bool`.
pub(crate) fn to_bool(g: Gil, v: Ref<'_>, what: &str) -> CResult<bool> {
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
pub(crate) fn bytes_of<'a>(g: Gil, b: Ref<'a>, what: &str) -> CResult<&'a [u8]> {
    if !b.is_exact(&raw mut ffi::PyBytes_Type) {
        return Err(wrong(g, what, "bytes"));
    }
    // SAFETY: an exact bytes object: its buffer and size never fail and
    // are immutable while it lives.
    let data = unsafe { ffi::PyBytes_AsString(b.as_ptr()) };
    // SAFETY: as above.
    let len = unsafe { ffi::PyBytes_Size(b.as_ptr()) };
    let len = usize::try_from(len).map_err(|_| Raised)?;
    // SAFETY: `len` immutable bytes owned by the live object.
    Ok(unsafe { core::slice::from_raw_parts(data.cast::<u8>(), len) })
}
