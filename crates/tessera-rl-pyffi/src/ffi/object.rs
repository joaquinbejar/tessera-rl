// SPDX-License-Identifier: MIT OR Apache-2.0

//! The typed object layer (ADR-0062 item 8): Python types whose instances
//! hold a Rust [`Body`], with methods written as safe trait impls.
//!
//! - A type is described by a [`TypeSpec<T>`], built at compile time from a
//!   [`MethodTable<T>`] of [`MethodRow<T>`]s. Each row names a method trait
//!   impl whose `Body` is `T`, so a row can only sit in `T`'s table.
//! - [`TypeObject<T>`] creates the type in a module and makes instances.
//!   Its type object never leaves this crate.
//! - The generic shims here are the only code that turns CPython's raw
//!   pointers into typed values. Each runs behind a boundary thunk
//!   (`ThunkFn::of`) inside [`entry`], with a [`Gil`] scoped to the call.
//!
//! The instance layout is `Instance<T>`: the object header, then
//! `Option<T>`. The body is written exactly twice, `Some` in
//! [`TypeObject::instantiate`] before the object is handed out, and taken
//! in the type's `tp_dealloc`, when no other reference exists. A method
//! call therefore always sees `Some`; "closed" is a state inside `T`.
//! The types are not subclassable (no `Py_TPFLAGS_BASETYPE`), so an object
//! whose type is `T`'s type has exactly this layout; and they have no
//! `tp_new`, so Python code cannot create one (calling the type raises
//! `TypeError`), as before.

use core::ffi::{CStr, c_int};
use core::marker::PhantomData;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use pyo3::ffi;

use super::buffer::{self, Buffer};
use super::capi::{self, CResult, Gil, Mine, Ref};
use super::tables::{method, method_end, method_kw, slot_dealloc, slot_doc, slot_end};
use super::thunks::{
    CFunction, CFunctionEntry, CFunctionKw, CFunctionKwEntry, Destructor, DestructorEntry,
    GetBuffer, ReleaseBuffer, ThunkFn,
};
use super::{dispose, entry};

type Obj = *mut ffi::PyObject;

/// The Rust value inside a Python object of a type built here.
///
/// `Sync` because two Python threads can call methods of one object at
/// once: while one runs with the interpreter released ([`Gil::detach`]) it
/// still holds `&Self`, and a second thread can enter and form another.
pub trait Body: Send + Sync + Sized + 'static {
    /// Releases the body when its object is deallocated, attached, with
    /// panics contained. The default drops it.
    fn release(self, g: Gil<'_>) {
        let _ = g;
        drop(self);
    }
}

/// A method without arguments (`METH_NOARGS`).
pub trait MethodNoArgs {
    /// The body of the type the method belongs to.
    type Body: Body;
    /// The method.
    ///
    /// # Errors
    ///
    /// `Err` with a Python exception set.
    fn call<'py>(g: Gil<'py>, body: &Self::Body) -> CResult<Mine<'py>>;
}

/// A method with positional arguments (`METH_VARARGS`); `args` is the
/// argument tuple.
pub trait Method {
    /// The body of the type the method belongs to.
    type Body: Body;
    /// The method.
    ///
    /// # Errors
    ///
    /// `Err` with a Python exception set.
    fn call<'py>(g: Gil<'py>, body: &Self::Body, args: Ref<'py>) -> CResult<Mine<'py>>;
}

/// A method with keyword arguments (`METH_VARARGS | METH_KEYWORDS`).
pub trait MethodKw {
    /// The body of the type the method belongs to.
    type Body: Body;
    /// The method; `kwargs` is `None` when the call passed none.
    ///
    /// # Errors
    ///
    /// `Err` with a Python exception set.
    fn call<'py>(
        g: Gil<'py>,
        body: &Self::Body,
        args: Ref<'py>,
        kwargs: Option<Ref<'py>>,
    ) -> CResult<Mine<'py>>;
}

/// The C layout of an instance.
#[repr(C)]
struct Instance<T> {
    ob_base: ffi::PyObject,
    body: Option<T>,
}

/// `&T` of `slf`, an instance of `T`'s type.
///
/// # Safety
///
/// `slf` is `self` of a method of `T`'s type, passed by CPython for the
/// call: rows are built only by [`MethodRow<T>`], which sits only in a
/// [`MethodTable<T>`] of a [`TypeSpec<T>`], whose type is not subclassable,
/// so `slf` is an `Instance<T>`, alive for `'a`.
unsafe fn body_of<'a, T: Body>(g: Gil<'_>, slf: Obj) -> CResult<&'a T> {
    // SAFETY: per the contract; the body is not written while a method
    // runs (module docs), so a shared borrow is sound.
    unsafe { (*slf.cast::<Instance<T>>()).body.as_ref() }
        .ok_or_else(|| capi::raise(g, capi::runtime_error(), "object not initialized"))
}

/// A row of `T`'s method table. Rows hold static strings and thunks only.
#[repr(transparent)]
pub struct MethodRow<T: Body>(ffi::PyMethodDef, PhantomData<fn() -> T>);

// SAFETY: a row is immutable plain data (static C strings and thunk
// addresses), read by CPython only.
unsafe impl<T: Body> Sync for MethodRow<T> {}

impl<T: Body> MethodRow<T> {
    /// The row that ends a table.
    pub const END: Self = Self(method_end(), PhantomData);

    /// A `METH_NOARGS` method.
    #[must_use]
    pub const fn noargs<M: MethodNoArgs<Body = T>>(
        name: &'static CStr,
        doc: &'static CStr,
    ) -> Self {
        Self(
            method(
                name,
                ThunkFn::<CFunction>::of::<NoArgs<M>>(),
                ffi::METH_NOARGS,
                doc,
            ),
            PhantomData,
        )
    }

    /// A `METH_VARARGS` method.
    #[must_use]
    pub const fn varargs<M: Method<Body = T>>(name: &'static CStr, doc: &'static CStr) -> Self {
        Self(
            method(
                name,
                ThunkFn::<CFunction>::of::<VarArgs<M>>(),
                ffi::METH_VARARGS,
                doc,
            ),
            PhantomData,
        )
    }

    /// A `METH_VARARGS | METH_KEYWORDS` method.
    #[must_use]
    pub const fn kw<M: MethodKw<Body = T>>(name: &'static CStr, doc: &'static CStr) -> Self {
        Self(
            method_kw(name, ThunkFn::<CFunctionKw>::of::<Keywords<M>>(), doc),
            PhantomData,
        )
    }
}

/// A method table: rows ending with [`MethodRow::END`], checked when the
/// table is built (at compile time for a static).
pub struct MethodTable<T: Body> {
    rows: &'static [MethodRow<T>],
}

impl<T: Body> MethodTable<T> {
    /// A table of `rows`.
    ///
    /// # Panics
    ///
    /// If `rows` does not end with [`MethodRow::END`] or holds it earlier:
    /// for a `static`, a compile error.
    #[must_use]
    pub const fn new(rows: &'static [MethodRow<T>]) -> Self {
        let mut ends = 0;
        let mut rest = rows;
        while let Some((row, tail)) = rest.split_first() {
            if row.0.ml_name.is_null() {
                ends += 1;
            }
            rest = tail;
        }
        let last_ends = match rows.last() {
            Some(last) => last.0.ml_name.is_null(),
            None => false,
        };
        assert!(
            last_ends && ends == 1,
            "a method table ends with exactly one MethodRow::END"
        );
        Self { rows }
    }
}

/// How a type exports its buffer, if it does.
#[derive(Clone, Copy)]
struct BufferSlots {
    get: ThunkFn<GetBuffer>,
    release: ThunkFn<ReleaseBuffer>,
}

/// A type's static description.
pub struct TypeSpec<T: Body> {
    name: &'static CStr,
    doc: &'static CStr,
    methods: &'static MethodTable<T>,
    buffer: Option<BufferSlots>,
}

impl<T: Body> TypeSpec<T> {
    /// A type named `name` (qualified, `package.module.Name`).
    #[must_use]
    pub const fn new(
        name: &'static CStr,
        doc: &'static CStr,
        methods: &'static MethodTable<T>,
    ) -> Self {
        Self {
            name,
            doc,
            methods,
            buffer: None,
        }
    }

    /// The type also exports its body through the buffer protocol, as `B`
    /// describes: read-only, C-contiguous.
    #[must_use]
    pub const fn with_buffer<B: Buffer<Body = T>>(self) -> Self {
        Self {
            buffer: Some(BufferSlots {
                get: buffer::getbuffer_thunk::<B>(),
                release: buffer::releasebuffer_thunk::<T>(),
            }),
            ..self
        }
    }
}

/// A type built from a [`TypeSpec<T>`], kept in a `static`: created at
/// module initialization, then used to make and recognize instances. The
/// type object is a process-lifetime reference held here (the module keeps
/// the type alive) and never leaves this crate.
pub struct TypeObject<T: Body> {
    spec: &'static TypeSpec<T>,
    ty: AtomicPtr<ffi::PyObject>,
}

impl<T: Body> TypeObject<T> {
    /// Not yet created.
    #[must_use]
    pub const fn new(spec: &'static TypeSpec<T>) -> Self {
        Self {
            spec,
            ty: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Creates the type and adds it to `module` as `attr` (callouts: a GC
    /// allocation and a module attribute).
    ///
    /// # Errors
    ///
    /// `Err` with a Python exception set.
    pub fn create(&self, g: Gil<'_>, module: Ref<'_>, attr: &CStr) -> CResult<()> {
        const {
            assert!(
                core::mem::align_of::<Instance<T>>() <= 8,
                "a body must not need more alignment than CPython's allocator gives"
            );
        };
        let dealloc = slot_dealloc(ThunkFn::<Destructor>::of::<Dealloc<T>>());
        let methods = super::tables::slot_method_rows(self.spec.methods.rows.as_ptr().cast());
        let doc = slot_doc(self.spec.doc);
        let mut slots = [dealloc, methods, doc, slot_end(), slot_end(), slot_end()];
        if let Some(b) = self.spec.buffer
            && let [.., get, release, _] = &mut slots
        {
            *get = super::tables::slot_getbuffer(b.get);
            *release = super::tables::slot_releasebuffer(b.release);
        }
        let mut spec = ffi::PyType_Spec {
            name: self.spec.name.as_ptr(),
            basicsize: c_int::try_from(core::mem::size_of::<Instance<T>>())
                .map_err(|_| capi::raise(g, capi::runtime_error(), "object too large"))?,
            itemsize: 0,
            // Not GC-tracked (no Py_TPFLAGS_HAVE_GC): a body holds no
            // Python object, and allocating an instance never runs
            // `gc.collect` (capi). Not subclassable (no BASETYPE).
            flags: core::ffi::c_uint::try_from(ffi::Py_TPFLAGS_DEFAULT).unwrap_or(0),
            slots: slots.as_mut_ptr(),
        };
        let ty = capi::type_from_spec(g, &mut spec)?;
        capi::module_add(g, module, attr, &ty)?;
        self.ty.store(ty.as_ref().as_ptr(), Ordering::Release);
        Ok(())
    }

    /// A new object holding `value` (direct: our types are not GC-tracked).
    ///
    /// # Errors
    ///
    /// `RuntimeError` before [`TypeObject::create`]; an allocation failure.
    pub fn instantiate<'py>(&self, g: Gil<'py>, value: T) -> CResult<Mine<'py>> {
        let ty = self.ty.load(Ordering::Acquire);
        if ty.is_null() {
            return Err(capi::raise(
                g,
                capi::runtime_error(),
                "type not initialized",
            ));
        }
        // SAFETY: `ty` is this type, created from `spec` with the basic
        // size of `Instance<T>` and no GC flag.
        let obj = unsafe { capi::alloc_instance(g, ty) }?;
        let inst = obj.as_ref().as_ptr().cast::<Instance<T>>();
        // SAFETY: a fresh, zeroed instance nobody else sees yet: the body is
        // written once, without dropping the zeroed bytes.
        let slot = unsafe { &raw mut (*inst).body };
        // SAFETY: as above; `slot` is the instance's body field.
        unsafe { slot.write(Some(value)) };
        Ok(obj)
    }

    /// `&T` of `obj`, if it is an instance of this type (exactly: the type
    /// is not subclassable).
    #[must_use]
    pub fn body<'a>(&self, obj: Ref<'a>) -> Option<&'a T> {
        let ty = self.ty.load(Ordering::Acquire);
        if ty.is_null() || !capi::is_instance_of(obj, ty) {
            return None;
        }
        // SAFETY: an instance of this type is an `Instance<T>`, alive for
        // `'a`; its body is written only before it is handed out and in its
        // deallocation.
        unsafe { (*obj.as_ptr().cast::<Instance<T>>()).body.as_ref() }
    }
}

/// The shim of a [`MethodNoArgs`].
struct NoArgs<M>(PhantomData<M>);

impl<M: MethodNoArgs> CFunctionEntry for NoArgs<M> {
    unsafe extern "C" fn call(slf: Obj, _args: Obj) -> Obj {
        entry(|g| {
            // SAFETY: `self` of a method row of `M::Body`'s type (body_of).
            let body = unsafe { body_of::<M::Body>(g, slf) }?;
            M::call(g, body)
        })
    }
}

/// The shim of a [`Method`].
struct VarArgs<M>(PhantomData<M>);

impl<M: Method> CFunctionEntry for VarArgs<M> {
    unsafe extern "C" fn call(slf: Obj, args: Obj) -> Obj {
        entry(|g| {
            // SAFETY: as in `NoArgs`.
            let body = unsafe { body_of::<M::Body>(g, slf) }?;
            // SAFETY: METH_VARARGS passes a borrowed tuple, alive for the
            // call.
            let args = unsafe { Ref::from_ptr(args) }
                .ok_or_else(|| capi::raise(g, capi::type_error(), "missing arguments"))?;
            M::call(g, body, args)
        })
    }
}

/// The shim of a [`MethodKw`].
struct Keywords<M>(PhantomData<M>);

impl<M: MethodKw> CFunctionKwEntry for Keywords<M> {
    unsafe extern "C" fn call(slf: Obj, args: Obj, kwargs: Obj) -> Obj {
        entry(|g| {
            // SAFETY: as in `NoArgs`.
            let body = unsafe { body_of::<M::Body>(g, slf) }?;
            // SAFETY: METH_VARARGS | METH_KEYWORDS passes a borrowed tuple
            // and a borrowed dict or null, alive for the call.
            let args = unsafe { Ref::from_ptr(args) }
                .ok_or_else(|| capi::raise(g, capi::type_error(), "missing arguments"))?;
            // SAFETY: as above.
            let kwargs = unsafe { Ref::from_ptr(kwargs) };
            M::call(g, body, args, kwargs)
        })
    }
}

/// The `tp_dealloc` shim: a cleanup entry, which never fails or rejects
/// (ADR-0041).
struct Dealloc<T>(PhantomData<T>);

impl<T: Body> DestructorEntry for Dealloc<T> {
    unsafe extern "C" fn call(obj: Obj) {
        let inst = obj.cast::<Instance<T>>();
        // SAFETY: CPython deallocates an instance of `T`'s type (the slot is
        // set only from `T`), once, with no other reference left.
        let body = unsafe { (*inst).body.take() };
        if let Some(body) = body {
            // SAFETY: CPython calls tp_dealloc attached; the token does not
            // outlive this call.
            let g = unsafe { Gil::assume() };
            if let Err(payload) =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body.release(g)))
            {
                dispose(payload);
            }
        }
        // SAFETY: a live instance of a heap type of ours, in its tp_dealloc.
        unsafe { capi::free_instance(obj) };
    }
}

/// `&T` of an instance in a buffer slot (`buffer`).
///
/// # Safety
///
/// As [`body_of`]: `obj` is an instance of `T`'s type, alive for `'a`.
pub(super) unsafe fn instance_body<'a, T: Body>(g: Gil<'_>, obj: Obj) -> CResult<&'a T> {
    // SAFETY: per the contract.
    unsafe { body_of::<T>(g, obj) }
}

#[cfg(test)]
mod tests {
    //! What runs without an interpreter: the static checks of the typed
    //! layer. The shims and types run in the Python suite (they reference
    //! the C API, which a test binary cannot link without libpython).

    use super::*;
    use crate::ffi::buffer::Export;
    use crate::ffi::inner::Inner;

    struct Probe;
    impl Body for Probe {}

    static ROWS: [MethodRow<Probe>; 1] = [MethodRow::END];
    static TABLE: MethodTable<Probe> = MethodTable::new(&ROWS);
    static SPEC: TypeSpec<Probe> = TypeSpec::new(c"tests.Probe", c"A probe.", &TABLE);
    static TYPE: TypeObject<Probe> = TypeObject::new(&SPEC);

    const fn shared<T: Send + Sync>() {}

    #[test]
    fn test_typed_statics_and_bodies_are_shareable() {
        shared::<TypeObject<Probe>>();
        shared::<TypeSpec<Probe>>();
        shared::<MethodTable<Probe>>();
        shared::<Export>();
        // The reentrancy guard is a lock: `Sync` for a `Send` body.
        shared::<Inner<dyn Send>>();
        assert!(
            TYPE.ty.load(Ordering::Relaxed).is_null(),
            "not created before init"
        );
        assert_eq!(SPEC.methods.rows.len(), 1);
    }

    static EMPTY: [MethodRow<Probe>; 0] = [];
    static EARLY_END: [MethodRow<Probe>; 2] = [MethodRow::END, MethodRow::END];

    #[test]
    #[should_panic(expected = "exactly one MethodRow::END")]
    fn test_method_table_without_its_end_row_panics() {
        let _ = MethodTable::new(&EMPTY);
    }

    #[test]
    #[should_panic(expected = "exactly one MethodRow::END")]
    fn test_method_table_with_an_early_end_row_panics() {
        let _ = MethodTable::new(&EARLY_END);
    }
}
