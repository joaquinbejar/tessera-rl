// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tessera_rl._native.WatchHandle` type: the single consumer of a
//! pool's live watch (ADR-0083 item 7), as a safe body of the typed layer of
//! `tessera_rl_pyffi` (ADR-0062 item 8).
//!
//! The engine's handle is `Send` but not `Sync` and takes `&mut self`; a
//! Python object can be shared between threads, so the body holds it behind
//! a mutex of its own. That mutex is never on the training path and the
//! handle never takes the environment's guard: `request` and `take` only
//! touch the watch slot, so a viewer thread can call them while another
//! thread steps.

use std::sync::{Mutex, MutexGuard};

use tessera_rl_engine::{WatchError, WatchHandle};
use tessera_rl_pyffi::ffi::capi::{self, CResult, Gil, Mine, Raised, Ref};
use tessera_rl_pyffi::ffi::object::{
    Body, Method, MethodNoArgs, MethodRow, MethodTable, TypeObject, TypeSpec,
};

use crate::errors::{Kind, error};

/// A `WatchHandle` object's body: the handle, `None` once closed.
pub(crate) struct WatchBody(Mutex<Option<WatchHandle>>);

impl Body for WatchBody {}

static TYPE: TypeObject<WatchBody> = TypeObject::new(&SPEC);

static SPEC: TypeSpec<WatchBody> = TypeSpec::new(
    c"tessera_rl._native.WatchHandle",
    c"The single consumer of a pool's live watch (use Env.watch).",
    &METHODS,
);

/// Wraps `handle` in a new object.
pub(crate) fn new_object(g: Gil<'_>, handle: WatchHandle) -> CResult<Mine<'_>> {
    TYPE.instantiate(g, WatchBody(Mutex::new(Some(handle))))
}

/// Creates the type and adds it to `module`.
pub(super) fn register(g: Gil<'_>, module: Ref<'_>) -> CResult<()> {
    TYPE.create(g, module, c"WatchHandle")
}

/// The Python exception for a refused watch call.
fn watch_error(g: Gil<'_>, e: WatchError) -> Raised {
    let kind = match e {
        WatchError::Busy => Kind::Concurrency,
        WatchError::Poisoned => Kind::Poisoned,
        _ => Kind::Argument,
    };
    error(g, kind, e.to_string())
}

impl WatchBody {
    /// The handle under the body's mutex, refusing a closed one.
    fn lock(&self, g: Gil<'_>) -> CResult<MutexGuard<'_, Option<WatchHandle>>> {
        // Poisoned only by a panic in a section of this module.
        self.0
            .lock()
            .map_err(|_| error(g, Kind::Poisoned, "the watch handle is poisoned".to_owned()))
    }
}

fn closed(g: Gil<'_>) -> Raised {
    error(g, Kind::Argument, "the watch is closed".to_owned())
}

/// `request(instance)`: asks for one instance's record at the next batch
/// boundary where it is idle.
struct RequestM;

impl Method for RequestM {
    type Body = WatchBody;
    fn call<'py>(g: Gil<'py>, body: &WatchBody, args: Ref<'py>) -> CResult<Mine<'py>> {
        let a: [Ref<'_>; 1] = crate::api::exactly(g, args, "request")?;
        let [instance] = a;
        let instance = capi::to_u32(g, instance, "instance")?;
        let mut guard = body.lock(g)?;
        let handle = guard.as_mut().ok_or_else(|| closed(g))?;
        handle.request(instance).map_err(|e| watch_error(g, e))?;
        Ok(capi::none(g))
    }
}

/// `take() -> (generation, record) | None`: the served record, if one is
/// ready; never blocks.
struct TakeM;

impl MethodNoArgs for TakeM {
    type Body = WatchBody;
    fn call<'py>(g: Gil<'py>, body: &WatchBody) -> CResult<Mine<'py>> {
        let mut guard = body.lock(g)?;
        let handle = guard.as_mut().ok_or_else(|| closed(g))?;
        let mut out = vec![0; handle.record_len()];
        match handle.take(&mut out).map_err(|e| watch_error(g, e))? {
            None => Ok(capi::none(g)),
            Some(generation) => capi::tuple(
                g,
                vec![capi::int_u64(g, generation)?, capi::bytes(g, &out)?],
            ),
        }
    }
}

/// `close()`: drops the handle; the pool drops its side at its next pending
/// request, and a later `Env.watch()` replaces it. Idempotent.
struct CloseM;

impl MethodNoArgs for CloseM {
    type Body = WatchBody;
    fn call<'py>(g: Gil<'py>, body: &WatchBody) -> CResult<Mine<'py>> {
        let taken = body.lock(g)?.take();
        drop(taken);
        Ok(capi::none(g))
    }
}

/// `record_len() -> int`: the length of a record in bytes.
struct RecordLenM;

impl MethodNoArgs for RecordLenM {
    type Body = WatchBody;
    fn call<'py>(g: Gil<'py>, body: &WatchBody) -> CResult<Mine<'py>> {
        let guard = body.lock(g)?;
        let handle = guard.as_ref().ok_or_else(|| closed(g))?;
        let len = u64::try_from(handle.record_len()).unwrap_or(u64::MAX);
        capi::int_u64(g, len)
    }
}

static METHODS: MethodTable<WatchBody> = MethodTable::new(&[
    MethodRow::varargs::<RequestM>(
        c"request",
        c"request($self, instance, /)\n--\n\nrequest(instance): asks for that instance's record at the pool's next batch boundary where it is idle.",
    ),
    MethodRow::noargs::<TakeM>(
        c"take",
        c"take($self, /)\n--\n\ntake() -> (generation, record bytes) | None; never blocks.",
    ),
    MethodRow::noargs::<CloseM>(
        c"close",
        c"close($self, /)\n--\n\nDrops the handle; idempotent.",
    ),
    MethodRow::noargs::<RecordLenM>(
        c"record_len",
        c"record_len($self, /)\n--\n\nThe length of a record in bytes.",
    ),
    MethodRow::END,
]);
