// SPDX-License-Identifier: MIT OR Apache-2.0

//! The engine registry shared by every wheel of a process (ADR-0005,
//! SPEC 6.1, #349).
//!
//! Each wheel (Tessera's `_native`, every host wheel) links its own copy
//! of the engine, with its own statics. To count the active engines of
//! all of them, the count lives at a well-known Python location:
//!
//! - the module `sys.modules["tessera_rl._registry"]`, created by the first
//!   wheel to initialize;
//! - its attribute `counter_v1`, a capsule named
//!   `"tessera_rl._registry.counter_v1"` that points to one
//!   `AtomicUsize`, allocated by that wheel and never freed.
//!
//! Every wheel's module initialization calls [`attach`], which finds or
//! creates the capsule and hands the counter to its engine copy
//! (`tessera_rl_engine::attach_shared_registry`). Registrations are then
//! atomic operations on the shared counter and need no Python call and no
//! GIL, so a pool dropped with the GIL released, or during interpreter
//! shutdown, unregisters safely.
//!
//! **Race freedom.** `attach` runs inside module initialization, with the
//! GIL held. On a plain `sys.modules` dict and a plain module (what this
//! code creates), the lookup, the attribute read and the attribute store
//! run no Python code, so the GIL is not released between them and no
//! other thread can interleave: exactly one capsule is ever created.
//!
//! **Versions.** The name carries the protocol version. A later layout
//! adds a new attribute (`counter_v2`) beside this one. A wheel that finds
//! the module without a valid `counter_v1` capsule, or a `sys.modules`
//! entry it cannot use, warns (`RuntimeWarning`) and keeps its engine's
//! own count. Wheels built before #349 never attach: they count only their
//! own engines, as before.
//!
//! **Fork.** A forked child inherits the count, including engines whose
//! workers do not exist in the child; those are refused there (#93), and
//! the stale count only makes a new pool in the child run unpinned.

use core::ffi::CStr;
use core::sync::atomic::AtomicUsize;

use super::capi::{self, CResult, Gil};

/// The registry module's name in `sys.modules`.
pub const MODULE: &CStr = c"tessera_rl._registry";

/// The counter's attribute on the module.
pub const ATTRIBUTE: &CStr = c"counter_v1";

/// The counter capsule's name: its type and protocol version.
pub const CAPSULE: &CStr = c"tessera_rl._registry.counter_v1";

/// The counter capsule's destructor: the counter is never freed, as an
/// engine of any wheel may hold it for the process.
unsafe extern "C" fn keep(_capsule: *mut pyo3::ffi::PyObject) {}

/// Finds or creates the shared counter and attaches it to this wheel's
/// engine. Called once per module initialization.
///
/// # Errors
///
/// A failure to create the module or the capsule (memory), or a warning
/// turned into an error by a filter, fails the import. An unusable
/// existing entry only warns.
pub fn attach(g: Gil<'_>) -> CResult<()> {
    let module = capi::add_module(g, MODULE)?;
    let counter: &'static AtomicUsize = match capi::get_attr(g, module.as_ref(), ATTRIBUTE) {
        Ok(found) => {
            // SAFETY: a live object (owned above); the name check makes a
            // non-capsule or another capsule return `None`.
            #[expect(unsafe_code, reason = "audited module: capsule read")]
            let pointer = unsafe { capi::capsule_pointer(found.as_ref().as_ptr(), CAPSULE) };
            let Some(pointer) = pointer.filter(|p| !p.is_null()) else {
                return fallback(g, c"tessera_rl._registry.counter_v1 is not a counter capsule of this version: this wheel counts only its own engines (ADR-0005)");
            };
            // SAFETY: a capsule of exactly this name is only ever made by
            // `attach` below, around a leaked `AtomicUsize` that is never
            // freed (`keep`); every wheel uses it only through atomic
            // operations, and `AtomicUsize` has the same layout in all.
            #[expect(unsafe_code, reason = "audited module: shared counter")]
            let counter = unsafe { &*pointer.cast::<AtomicUsize>() };
            counter
        }
        Err(raised) => {
            if !capi::pending_is_attribute_error(g) {
                return Err(raised);
            }
            capi::clear_pending(g);
            let leaked: &'static AtomicUsize = Box::leak(Box::new(AtomicUsize::new(0)));
            let pointer = core::ptr::from_ref(leaked)
                .cast_mut()
                .cast::<core::ffi::c_void>();
            // SAFETY: `pointer` is valid for the process (leaked); `CAPSULE`
            // is `'static`; `keep` frees nothing.
            #[expect(unsafe_code, reason = "audited module: capsule creation")]
            let capsule = unsafe { capi::capsule(g, pointer, CAPSULE, keep) }?;
            if capi::module_add(g, module.as_ref(), ATTRIBUTE, &capsule).is_err() {
                capi::clear_pending(g);
                return fallback(g, c"sys.modules['tessera_rl._registry'] is not a module: this wheel counts only its own engines (ADR-0005)");
            }
            leaked
        }
    };
    if tessera_rl_engine::attach_shared_registry(counter) {
        Ok(())
    } else {
        fallback(
            g,
            c"this wheel's engine already uses another registry counter (ADR-0005)",
        )
    }
}

/// Warns that this wheel keeps its own count.
fn fallback(g: Gil<'_>, message: &CStr) -> CResult<()> {
    capi::warn(g, capi::runtime_warning(), message)
}
