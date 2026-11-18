// SPDX-License-Identifier: MIT OR Apache-2.0

//! The typed module layer (ADR-0062 item 8): a module's definition and
//! function table, its initializer, and the fixed `PyInit_*` symbols.
//!
//! An extension describes its module with a [`ModuleDef`] and a
//! [`ModuleInit`] impl, and registers that impl with [`export_module!`]
//! (an `inventory` record in the extension's own crate, ADR-0054). This
//! crate defines the initializer CPython looks up by name, as a
//! non-generic naked boundary thunk (P1 of the #272 step 3 design):
//!
//! - `PyInit__native` behind the `native-module` feature (Tessera's own
//!   `_native`);
//! - `PyInit__tessera_host` behind the `host` feature (a host-driven
//!   environment wheel, ADR-0062 D3).
//!
//! The thunk is the first code to run: it saves and canonicalizes the
//! caller's floating-point controls before any Rust frame; the record
//! lookup and the extension's initialization run inside it.
//!
//! [`export_module!`]: crate::export_module

use core::cell::UnsafeCell;
use core::ffi::CStr;
use core::marker::PhantomData;
use core::ptr;

use pyo3::ffi;

use super::capi::{self, CResult, Gil, Mine, Ref};
use super::entry;
use super::tables::{method, method_end, method_kw};
use super::thunks::{CFunction, CFunctionEntry, CFunctionKw, CFunctionKwEntry, ThunkFn};

type Obj = *mut ffi::PyObject;

/// A module function without arguments (`METH_NOARGS`).
pub trait FunctionNoArgs {
    /// The function.
    ///
    /// # Errors
    ///
    /// `Err` with a Python exception set.
    fn call(g: Gil<'_>) -> CResult<Mine<'_>>;
}

/// A module function of one argument (`METH_O`).
pub trait FunctionArg {
    /// The function.
    ///
    /// # Errors
    ///
    /// `Err` with a Python exception set.
    fn call<'py>(g: Gil<'py>, arg: Ref<'py>) -> CResult<Mine<'py>>;
}

/// A module function with positional arguments (`METH_VARARGS`).
pub trait Function {
    /// The function; `args` is the argument tuple.
    ///
    /// # Errors
    ///
    /// `Err` with a Python exception set.
    fn call<'py>(g: Gil<'py>, args: Ref<'py>) -> CResult<Mine<'py>>;
}

/// A module function with keyword arguments (`METH_VARARGS |
/// METH_KEYWORDS`).
pub trait FunctionKw {
    /// The function; `kwargs` is `None` when the call passed none.
    ///
    /// # Errors
    ///
    /// `Err` with a Python exception set.
    fn call<'py>(g: Gil<'py>, args: Ref<'py>, kwargs: Option<Ref<'py>>) -> CResult<Mine<'py>>;
}

/// A row of a module's function table. Rows hold static strings and
/// thunks only.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct FunctionRow(ffi::PyMethodDef);

// SAFETY: a row is immutable plain data (static C strings and thunk
// addresses), read by CPython only.
unsafe impl Sync for FunctionRow {}

impl FunctionRow {
    /// The row that ends a table.
    pub const END: Self = Self(method_end());

    /// A `METH_NOARGS` function.
    #[must_use]
    pub const fn noargs<F: FunctionNoArgs>(name: &'static CStr, doc: &'static CStr) -> Self {
        Self(method(
            name,
            ThunkFn::<CFunction>::of::<NoArgs<F>>(),
            ffi::METH_NOARGS,
            doc,
        ))
    }

    /// A `METH_O` function.
    #[must_use]
    pub const fn arg<F: FunctionArg>(name: &'static CStr, doc: &'static CStr) -> Self {
        Self(method(
            name,
            ThunkFn::<CFunction>::of::<Arg<F>>(),
            ffi::METH_O,
            doc,
        ))
    }

    /// A `METH_VARARGS` function.
    #[must_use]
    pub const fn varargs<F: Function>(name: &'static CStr, doc: &'static CStr) -> Self {
        Self(method(
            name,
            ThunkFn::<CFunction>::of::<VarArgs<F>>(),
            ffi::METH_VARARGS,
            doc,
        ))
    }

    /// A `METH_VARARGS | METH_KEYWORDS` function.
    #[must_use]
    pub const fn kw<F: FunctionKw>(name: &'static CStr, doc: &'static CStr) -> Self {
        Self(method_kw(
            name,
            ThunkFn::<CFunctionKw>::of::<Keywords<F>>(),
            doc,
        ))
    }
}

/// A module's function table: rows ending with [`FunctionRow::END`],
/// checked when the table is built (at compile time for a static).
pub struct FunctionTable {
    rows: &'static [FunctionRow],
}

impl FunctionTable {
    /// A table of `rows`.
    ///
    /// # Panics
    ///
    /// If `rows` does not end with [`FunctionRow::END`] or holds it
    /// earlier: for a `static`, a compile error.
    #[must_use]
    pub const fn new(rows: &'static [FunctionRow]) -> Self {
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
            "a function table ends with exactly one FunctionRow::END"
        );
        Self { rows }
    }
}

/// A module's definition, built at compile time. Rust never writes it;
/// CPython owns its `m_base` after creating the module.
pub struct ModuleDef(UnsafeCell<ffi::PyModuleDef>);

// SAFETY: written only by CPython (its `m_base`, in `PyModule_Create2`,
// attached); Rust only takes its address.
unsafe impl Sync for ModuleDef {}

impl ModuleDef {
    /// The definition of module `name` (qualified, `package.module`).
    #[must_use]
    pub const fn new(
        name: &'static CStr,
        doc: &'static CStr,
        functions: &'static FunctionTable,
    ) -> Self {
        Self(UnsafeCell::new(ffi::PyModuleDef {
            m_base: ffi::PyModuleDef_HEAD_INIT,
            m_name: name.as_ptr(),
            m_doc: doc.as_ptr(),
            m_size: -1,
            m_methods: functions
                .rows
                .as_ptr()
                .cast::<ffi::PyMethodDef>()
                .cast_mut(),
            m_slots: ptr::null_mut(),
            m_traverse: None,
            m_clear: None,
            m_free: None,
        }))
    }
}

/// An extension's module: its definition and what initialization adds to
/// it (types, exception classes, attributes).
pub trait ModuleInit: 'static {
    /// The module's definition.
    const DEF: &'static ModuleDef;

    /// Fills the new module.
    ///
    /// # Errors
    ///
    /// `Err` with a Python exception set: the import fails.
    fn init<'py>(g: Gil<'py>, module: Ref<'py>) -> CResult<()>;
}

/// Creates `I`'s module and initializes it.
fn create<I: ModuleInit>(g: Gil<'_>) -> CResult<Mine<'_>> {
    // SAFETY: `I::DEF` is a static definition built by `ModuleDef::new`,
    // which only CPython writes (its `m_base`), as `PyModule_Create2`
    // requires.
    let module = unsafe { capi::module_create(g, I::DEF.0.get()) }?;
    I::init(g, module.as_ref())?;
    Ok(module)
}

/// Test hooks only: runs a module initialization that registers the
/// exception classes and then fails, as a `ModuleInit::init` whose later
/// step fails would. The new module is freed with the error; the classes
/// must outlive it (they are held for the process).
///
/// # Errors
///
/// Always: the injected `RuntimeError`.
#[cfg(feature = "test-hooks")]
pub fn init_fails_after_register(g: Gil<'_>) -> CResult<()> {
    struct FailsAfterRegister;

    static ROWS: [FunctionRow; 1] = [FunctionRow::END];
    static TABLE: FunctionTable = FunctionTable::new(&ROWS);
    static DEF: ModuleDef = ModuleDef::new(
        c"tessera_rl._init_fails_after_register",
        c"Test hook: an initialization that fails after registering the exception classes.",
        &TABLE,
    );

    impl ModuleInit for FailsAfterRegister {
        const DEF: &'static ModuleDef = &DEF;

        fn init<'py>(g: Gil<'py>, module: Ref<'py>) -> CResult<()> {
            crate::errors::register(g, module)?;
            Err(capi::raise(
                g,
                capi::runtime_error(),
                "injected: initialization fails after register",
            ))
        }
    }

    create::<FailsAfterRegister>(g).map(drop)
}

/// Which fixed initializer a record serves.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Leaf {
    Native,
    Host,
}

/// An extension's module, registered for the fixed initializer of its
/// leaf name. Made only by [`export_module!`].
///
/// [`export_module!`]: crate::export_module
#[cfg_attr(
    not(any(feature = "native-module", feature = "host")),
    expect(
        dead_code,
        reason = "read by the fixed initializers, behind their features"
    )
)]
pub struct ModuleRecord {
    leaf: Leaf,
    create: for<'py> fn(Gil<'py>) -> CResult<Mine<'py>>,
}

impl ModuleRecord {
    /// `I` as the module of `PyInit__native`. Not public API: use
    /// [`export_module!`](crate::export_module).
    #[doc(hidden)]
    #[must_use]
    pub const fn native<I: ModuleInit>() -> Self {
        Self {
            leaf: Leaf::Native,
            create: create::<I>,
        }
    }

    /// `I` as the module of `PyInit__tessera_host`. Not public API.
    #[doc(hidden)]
    #[must_use]
    pub const fn host<I: ModuleInit>() -> Self {
        Self {
            leaf: Leaf::Host,
            create: create::<I>,
        }
    }
}

inventory::collect!(ModuleRecord);

/// Registers a [`ModuleInit`] impl for a fixed module initializer:
/// `export_module!(native: MyInit)` for `PyInit__native` (feature
/// `native-module`), `export_module!(host: MyInit)` for
/// `PyInit__tessera_host` (feature `host`). Invoke it in the final cdylib's
/// own crate, where the linker keeps the record (ADR-0054).
///
/// The record is left out of the crate's unit-test build (`cfg(test)`): it
/// keeps the whole initialization alive, and a test binary cannot link the
/// C API without libpython. That build only names the module in a constant.
#[macro_export]
macro_rules! export_module {
    (native: $init:ty) => {
        #[cfg(not(test))]
        $crate::__private::inventory::submit! {
            $crate::ffi::module_def::ModuleRecord::native::<$init>()
        }
        // The unit-test build still names the module, so its code counts
        // as used, without a record that would link it.
        #[cfg(test)]
        const _: $crate::ffi::module_def::ModuleRecord =
            $crate::ffi::module_def::ModuleRecord::native::<$init>();
    };
    (host: $init:ty) => {
        #[cfg(not(test))]
        $crate::__private::inventory::submit! {
            $crate::ffi::module_def::ModuleRecord::host::<$init>()
        }
        // The unit-test build still names the module, so its code counts
        // as used, without a record that would link it.
        #[cfg(test)]
        const _: $crate::ffi::module_def::ModuleRecord =
            $crate::ffi::module_def::ModuleRecord::host::<$init>();
    };
}

/// The module registered for `leaf`: exactly one record, or `ImportError`.
#[cfg(any(feature = "native-module", feature = "host"))]
fn init_leaf(leaf: Leaf, name: &str) -> Obj {
    entry(|g| {
        let mut found = inventory::iter::<ModuleRecord>
            .into_iter()
            .filter(|r| r.leaf == leaf);
        match (found.next(), found.next()) {
            (Some(record), None) => (record.create)(g),
            (None, _) => Err(capi::raise(
                g,
                capi::import_error(),
                &format!("{name}: no module registered (tessera_rl_pyffi::export_module!)"),
            )),
            (Some(_), Some(_)) => Err(capi::raise(
                g,
                capi::import_error(),
                &format!("{name}: more than one module registered"),
            )),
        }
    })
}

/// The target of the `PyInit__native` thunk.
#[cfg(feature = "native-module")]
pub(super) extern "C" fn init_native() -> Obj {
    init_leaf(Leaf::Native, "_native")
}

/// The target of the `PyInit__tessera_host` thunk.
#[cfg(feature = "host")]
pub(super) extern "C" fn init_host() -> Obj {
    init_leaf(Leaf::Host, "_tessera_host")
}

/// The shim of a [`FunctionNoArgs`].
struct NoArgs<F>(PhantomData<F>);

impl<F: FunctionNoArgs> CFunctionEntry for NoArgs<F> {
    unsafe extern "C" fn call(_module: Obj, _args: Obj) -> Obj {
        entry(|g| F::call(g))
    }
}

/// The shim of a [`FunctionArg`].
struct Arg<F>(PhantomData<F>);

impl<F: FunctionArg> CFunctionEntry for Arg<F> {
    unsafe extern "C" fn call(_module: Obj, arg: Obj) -> Obj {
        entry(|g| {
            // SAFETY: METH_O passes the borrowed argument, alive for the
            // call.
            let arg = unsafe { Ref::from_ptr(arg) }
                .ok_or_else(|| capi::raise(g, capi::type_error(), "missing argument"))?;
            F::call(g, arg)
        })
    }
}

/// The shim of a [`Function`].
struct VarArgs<F>(PhantomData<F>);

impl<F: Function> CFunctionEntry for VarArgs<F> {
    unsafe extern "C" fn call(_module: Obj, args: Obj) -> Obj {
        entry(|g| {
            // SAFETY: METH_VARARGS passes a borrowed tuple, alive for the
            // call.
            let args = unsafe { Ref::from_ptr(args) }
                .ok_or_else(|| capi::raise(g, capi::type_error(), "missing arguments"))?;
            F::call(g, args)
        })
    }
}

/// The shim of a [`FunctionKw`].
struct Keywords<F>(PhantomData<F>);

impl<F: FunctionKw> CFunctionKwEntry for Keywords<F> {
    unsafe extern "C" fn call(_module: Obj, args: Obj, kwargs: Obj) -> Obj {
        entry(|g| {
            // SAFETY: METH_VARARGS | METH_KEYWORDS passes a borrowed tuple
            // and a borrowed dict or null, alive for the call.
            let args = unsafe { Ref::from_ptr(args) }
                .ok_or_else(|| capi::raise(g, capi::type_error(), "missing arguments"))?;
            // SAFETY: as above.
            let kwargs = unsafe { Ref::from_ptr(kwargs) };
            F::call(g, args, kwargs)
        })
    }
}

#[cfg(test)]
mod tests {
    //! The static checks of the module layer (as `object`'s tests: no code
    //! that references the C API).

    use super::*;

    static ROWS: [FunctionRow; 1] = [FunctionRow::END];
    static TABLE: FunctionTable = FunctionTable::new(&ROWS);
    static DEF: ModuleDef = ModuleDef::new(c"tests.module", c"A module.", &TABLE);

    #[test]
    fn test_module_def_points_at_its_terminated_function_table() {
        // SAFETY: reading the static definition; nothing writes it here.
        let def = unsafe { &*DEF.0.get() };
        assert_eq!(
            def.m_methods.cast_const(),
            ROWS.as_ptr().cast::<ffi::PyMethodDef>()
        );
        assert_eq!(def.m_size, -1);
    }

    static EMPTY: [FunctionRow; 0] = [];
    static EARLY_END: [FunctionRow; 2] = [FunctionRow::END, FunctionRow::END];

    #[test]
    #[should_panic(expected = "exactly one FunctionRow::END")]
    fn test_function_table_without_its_end_row_panics() {
        let _ = FunctionTable::new(&EMPTY);
    }

    #[test]
    #[should_panic(expected = "exactly one FunctionRow::END")]
    fn test_function_table_with_an_early_end_row_panics() {
        let _ = FunctionTable::new(&EARLY_END);
    }
}
