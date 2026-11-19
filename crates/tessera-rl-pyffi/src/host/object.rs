// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host object (SPEC 14.4.2 draft): a host environment's pool behind
//! the reentrancy guard, with its protocol attributes and methods.

use tessera_rl_engine::DynHostPool;

use crate::ffi::capi::{self, CResult, Gil, Mine, Ref};
use crate::ffi::inner::Inner;
use crate::ffi::object::{
    Attribute, AttributeRow, AttributeTable, Body, MethodNoArgs, MethodRow, MethodTable, Release,
    TypeObject, TypeSpec,
};

use super::describe;

/// The host protocol version this module implements (SPEC 14.4.7).
pub const HOST_PROTOCOL: u64 = 1;

/// The body of a host object: its pool, until `close`.
pub struct HostBody {
    pub(super) inner: Inner<dyn DynHostPool>,
}

impl HostBody {
    /// A host object's body around `pool`.
    pub(super) fn new(pool: Box<dyn DynHostPool>) -> Self {
        Self {
            inner: Inner::new(pool, true),
        }
    }

    /// The state, refusing an object inherited through a fork before any
    /// lock or flag of it is touched.
    pub(super) fn guarded(&self, g: Gil<'_>) -> CResult<&Inner<dyn DynHostPool>> {
        self.inner.check_fork(g)?;
        Ok(&self.inner)
    }
}

impl Body for HostBody {
    /// Dropping a pool joins its workers, so it is dropped detached. In a
    /// forked child the copy of the workers does not exist and its locks may
    /// be held forever: the memory is leaked by design, as for the
    /// first-party `Env`.
    fn release(self, r: Release<'_>) {
        if self.inner.inherited() {
            core::mem::forget(self);
        } else {
            r.detach(move || drop(self));
        }
    }
}

/// `host_protocol`: 1.
struct HostProtocol;

impl Attribute for HostProtocol {
    type Body = HostBody;
    fn get<'py>(g: Gil<'py>, _: &HostBody) -> CResult<Mine<'py>> {
        capi::int_u64(g, HOST_PROTOCOL)
    }
}

/// `tessera_rl_min`: the version this module was built with.
struct TesseraMin;

impl Attribute for TesseraMin {
    type Body = HostBody;
    fn get<'py>(g: Gil<'py>, _: &HostBody) -> CResult<Mine<'py>> {
        capi::string(g, &describe::version_min())
    }
}

/// `tessera_rl_below`: the next minor version, excluded.
struct TesseraBelow;

impl Attribute for TesseraBelow {
    type Body = HostBody;
    fn get<'py>(g: Gil<'py>, _: &HostBody) -> CResult<Mine<'py>> {
        capi::string(g, &describe::version_below())
    }
}

/// `layout()`: the dict of SPEC 14.4.3 (draft, with ADR-0065's keys).
struct LayoutM;

impl MethodNoArgs for LayoutM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody) -> CResult<Mine<'py>> {
        body.guarded(g)?.with(g, |pool| describe::layout(g, &*pool))
    }
}

/// `is_poisoned()`: whether a panic poisoned the pool (ADR-0028).
struct IsPoisonedM;

impl MethodNoArgs for IsPoisonedM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody) -> CResult<Mine<'py>> {
        body.guarded(g)?
            .with(g, |pool| Ok(capi::boolean(g, pool.is_poisoned())))
    }
}

/// `close()`: releases the pool; a second `close` does nothing. Views
/// already exported keep their memory. On an inherited object a no-op.
struct CloseM;

impl MethodNoArgs for CloseM {
    type Body = HostBody;
    fn call<'py>(g: Gil<'py>, body: &HostBody) -> CResult<Mine<'py>> {
        if body.inner.inherited() {
            return Ok(capi::none(g));
        }
        let pool = body.inner.take(g)?;
        g.detach(move || drop(pool));
        Ok(capi::none(g))
    }
}

static METHODS: MethodTable<HostBody> = MethodTable::new(&[
    MethodRow::noargs::<LayoutM>(c"layout", c"The pool's layout (SPEC 14.4.3)."),
    MethodRow::noargs::<IsPoisonedM>(c"is_poisoned", c"Whether a panic poisoned the pool."),
    MethodRow::noargs::<CloseM>(c"close", c"Releases the pool."),
    MethodRow::END,
]);

static ATTRIBUTES: AttributeTable<HostBody> = AttributeTable::new(&[
    AttributeRow::get::<HostProtocol>(c"host_protocol", c"The host protocol version: 1."),
    AttributeRow::get::<TesseraMin>(
        c"tessera_rl_min",
        c"The lowest tessera_rl version this object works with (MAJOR.MINOR.PATCH).",
    ),
    AttributeRow::get::<TesseraBelow>(
        c"tessera_rl_below",
        c"The first tessera_rl version this object does not work with (MAJOR.MINOR.PATCH).",
    ),
    AttributeRow::END,
]);

static SPEC: TypeSpec<HostBody> = TypeSpec::new(
    c"tessera_host.HostEnv",
    c"A host environment's pool (host protocol 1, SPEC 14.4).",
    &METHODS,
)
.with_attributes(&ATTRIBUTES);

pub(super) static HOST_TYPE: TypeObject<HostBody> = TypeObject::new(&SPEC);

/// Adds the host object type to `module`.
pub(super) fn register(g: Gil<'_>, module: Ref<'_>) -> CResult<()> {
    HOST_TYPE.create(g, module, c"HostEnv")
}
