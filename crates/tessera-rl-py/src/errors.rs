// SPDX-License-Identifier: MIT OR Apache-2.0

//! `_native`'s errors: the exception hierarchy and the engine-error mapping
//! live in `tessera_rl_pyffi::errors` (ADR-0062 item 8); this module maps the
//! bindings' own errors.

use crate::env::BindError;
use crate::ffi::capi::{Gil, Raised};

use tessera_rl_pyffi::errors::engine_kind;
pub(crate) use tessera_rl_pyffi::errors::{Kind, error, register};

/// Raises the Python exception for `e`.
pub(crate) fn to_py(g: Gil, e: BindError) -> Raised {
    match e {
        BindError::Engine(e @ tessera_rl_engine::EngineError::OutputAllocation { .. }) => {
            // The Python allocator's own exception, if it raised one.
            let message = match tessera_rl_pyffi::ffi::alloc::take_allocation_error() {
                Some(cause) => format!("{e}: allocate raised {cause}"),
                None => e.to_string(),
            };
            error(g, engine_kind(&e), message)
        }
        BindError::Engine(e) => error(g, engine_kind(&e), e.to_string()),
        BindError::Decode(e) => error(g, Kind::Record, e.to_string()),
        BindError::Argument(message) => error(g, Kind::Argument, message),
        BindError::Forked(message) => error(g, Kind::Forked, message),
        BindError::Persist(tessera_rl_engine::PersistError::Engine(e)) => {
            error(g, engine_kind(&e), e.to_string())
        }
        BindError::Persist(e) => error(g, Kind::Persist, e.to_string()),
        BindError::Search(tessera_rl_search::SearchError::Evaluator(e)) => {
            error(g, Kind::Evaluator, e.to_string())
        }
        BindError::Search(e) => error(g, Kind::Search, e.to_string()),
        BindError::Onnx(message) => error(g, Kind::Onnx, message),
        BindError::Capability(message) => error(g, Kind::Capability, message),
        // Divergences are returned to Python as data by `replay`; any other
        // replay refusal is a refused file, as for `load`.
        BindError::Replay(e) => error(g, Kind::Persist, e.to_string()),
    }
}
