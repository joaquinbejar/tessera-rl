// SPDX-License-Identifier: MIT OR Apache-2.0

//! Instance records, output ownership, scheduler, execution modes and
//! snapshot, restore and fork (ADR-0002, ADR-0005, ADR-0036).
//!
//! Milestones 5a and 5b: record storage, construction, the validation
//! boundary, the Inline mode, auto-reset and poisoning.

mod error;
mod layout;
mod options;
mod outputs;
mod pool;
mod validate;

pub use error::EngineError;
pub use layout::MAX_LEAVES;
pub use options::{AutoReset, CHUNK_RANGE, PoolOptions};
pub use outputs::Outputs;
pub use pool::{EnvPool, Reseed};
pub use validate::ActionInput;
