// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Env.watch()` (ADR-0083 item 7): attaches a live watch to the pool and
//! returns its handle object. Sync and split-phase modes; host pools raise
//! `CapabilityError`, and a second watch while a handle is alive raises
//! `ConcurrencyError`.

use crate::errors::to_py;
use crate::ffi::capi::{CResult, Gil, Mine};
use crate::ffi::inner::Inner;
use crate::ffi::watch_type;

/// `watch() -> WatchHandle`.
pub(crate) fn watch<'py>(g: Gil<'py>, inner: &Inner) -> CResult<Mine<'py>> {
    let handle = inner.with(g, |ops| ops.watch().map_err(|e| to_py(g, e)))?;
    watch_type::new_object(g, handle)
}
