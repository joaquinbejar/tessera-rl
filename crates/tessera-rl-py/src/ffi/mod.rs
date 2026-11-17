// SPDX-License-Identifier: MIT OR Apache-2.0

//! `_native`'s side of the C boundary with CPython. Audited `unsafe` module
//! (ADR-0041): `thunks` instantiates every boundary thunk of the module from
//! the shared templates of `tessera_rl_pyffi`, and the type modules build their C tables.
//! The shared boundary (templates, C-API wrappers and callouts, tables,
//! DLPack, fork generation, reentrancy guard) lives in
//! `tessera_rl_pyffi::ffi` (ADR-0062 item 8) and is re-exported here under
//! its former paths.

#![expect(unsafe_code, reason = "audited module: the CPython C boundary")]

pub(crate) use tessera_rl_pyffi::ffi::{
    Table, boundary, capi, dispose, dlpack, entry, entry_status, fork, tables,
};

/// The reentrancy guard holder, an environment pool by default.
pub(crate) mod inner {
    /// `tessera_rl_pyffi::ffi::inner::Inner` over an environment pool by
    /// default (`Env`), or a search.
    pub(crate) type Inner<T = dyn crate::env::PoolOps> = tessera_rl_pyffi::ffi::inner::Inner<T>;
}

mod env_type;
pub(crate) mod gather;
mod module;
mod search_group_type;
mod search_type;
pub(crate) mod snapshot_type;
mod thunks;
pub(crate) mod view_type;
