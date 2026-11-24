// SPDX-License-Identifier: MIT OR Apache-2.0

//! `_native`'s types and module, written against the safe typed layer of
//! `tessera_rl_pyffi` (ADR-0041, ADR-0062 item 8). No `unsafe` here: the
//! boundary thunks, the raw-to-typed step, the type and module assembly and
//! the exports all live in `tessera_rl_pyffi::ffi`, the audited module.

pub(crate) use tessera_rl_pyffi::ffi::{capi, fork};

/// The reentrancy guard holder, an environment pool by default.
pub(crate) mod inner {
    /// `tessera_rl_pyffi::ffi::inner::Inner` over an environment pool by
    /// default (`Env`), or a search.
    pub(crate) type Inner<T = dyn crate::env::PoolOps> = tessera_rl_pyffi::ffi::inner::Inner<T>;
}

mod env_type;
mod module;
mod search_group_type;
mod search_type;
pub(crate) mod snapshot_type;
pub(crate) mod view_type;
pub(crate) mod watch_type;
