// SPDX-License-Identifier: MIT OR Apache-2.0

//! Instance records, output ownership, scheduler, execution modes and
//! snapshot, restore and fork (ADR-0002, ADR-0005, ADR-0036).
//!
//! Milestones 5a to 5d: record storage, construction, the validation
//! boundary, Inline and Sync modes, auto-reset, poisoning, and snapshot,
//! restore and fork.
//!
//! `unsafe` code lives only in the audited modules `buffer` (disjoint chunk
//! views for workers) and `sys` (the worker gang).

mod buffer;
mod error;
#[cfg(test)]
mod fault_tests;
mod layout;
mod masks;
mod options;
mod outputs;
mod pool;
mod record;
mod registry;
mod snapshot;
mod sys;
mod validate;

pub use error::EngineError;
pub use layout::MAX_LEAVES;
pub use masks::{IllegalActionPolicy, MaskProvider, Masked, NoMasks};
pub use options::{AutoReset, CHUNK_RANGE, PoolOptions, SyncOptions, WaitBackend};
pub use outputs::Outputs;
pub use pool::{EnvPool, Reseed};
pub use record::{DecodedRecord, PoolContext, Record};
pub use validate::ActionInput;
