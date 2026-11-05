// SPDX-License-Identifier: MIT OR Apache-2.0

//! Search over real environment instances (SPEC 9, ADR-0043): the bounded
//! [`SearchPool`] of node records, expanded in batch through a scratch
//! [`EnvPool`](tessera_rl_engine::EnvPool).
//!
//! The pool allocates everything at construction; expansion, eviction and
//! re-rooting allocate nothing. Every operation that can fail checks
//! everything before it changes the tree, so an error leaves the tree and
//! its counters exactly as they were.

mod determinize;
mod error;
mod evaluator;
mod evict;
mod group;
mod handle;
mod journal;
mod mcts;
mod noise;
mod options;
mod pool;

pub use determinize::{
    Determinization, ROLLOUT, SAMPLER, rollout_identity, rollout_salt, sampler_identity,
    sampler_salt,
};
pub use error::SearchError;
pub use evaluator::{Evaluator, EvaluatorError, LeafBatch, PRIOR_SUM_TOLERANCE};
pub use group::{
    CAPACITY_EXTRA, DEFAULT_GROUP_BUDGET, FIXED_PER_TREE, GroupError, GroupFailure, MAX_TREES,
    SearchGroup,
};
pub use handle::{EdgeStats, NodeHandle};
pub use mcts::{Mcts, PUCT_FORMULA, PuctConfig, RootNoise};
pub use options::{Eviction, SearchOptions};
pub use pool::SearchPool;

#[cfg(test)]
mod tests;
