// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed errors of the search pool (ADR-0022, ADR-0043). After any error,
//! the tree and its counters are unchanged.

use tessera_rl_engine::EngineError;
use thiserror::Error;

/// Errors of [`SearchPool`](crate::SearchPool).
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum SearchError {
    /// The scratch pool failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// Invalid options.
    #[error("invalid search option {option}: {reason}")]
    Options {
        /// The option.
        option: &'static str,
        /// Why it is invalid.
        reason: &'static str,
    },
    /// The node storage would exceed the budget.
    #[error("node storage needs {bytes} bytes ({per_node} per node), over the budget of {budget}")]
    Budget {
        /// Bytes needed.
        bytes: u64,
        /// Bytes per node: record, node and edges.
        per_node: u64,
        /// The budget.
        budget: u64,
    },
    /// Player-perspective search on an imperfect-information environment
    /// needs `Determinize` (ADR-0035).
    #[error("the environment has imperfect information: search needs Determinize (ADR-0035)")]
    ImperfectInformation,
    /// The action space is not a single discrete head.
    #[error("search needs a Discrete action space")]
    UnsupportedActions,
    /// A handle whose slot was freed or reused.
    #[error("stale handle: slot {slot}, generation {generation}")]
    StaleHandle {
        /// The slot.
        slot: u32,
        /// The handle's generation.
        generation: u32,
    },
    /// An action outside the action space.
    #[error("action {action} out of range for {actions} actions")]
    ActionOutOfRange {
        /// The action.
        action: u32,
        /// Actions of the space.
        actions: u32,
    },
    /// An action the parent's mask forbids.
    #[error("action {action} is illegal at its parent")]
    IllegalAction {
        /// The action.
        action: u32,
    },
    /// The edge already has a child.
    #[error("action {action} is already expanded")]
    AlreadyExpanded {
        /// The action.
        action: u32,
    },
    /// The same edge appears twice in one call.
    #[error("request {index} repeats an earlier edge")]
    DuplicateRequest {
        /// Position in the call.
        index: usize,
    },
    /// A terminal node has no children.
    #[error("the node is terminal")]
    TerminalNode,
    /// Expansion needs exactly one acting agent at the parent.
    #[error("{acting} acting agents at the parent; search needs exactly one")]
    NotSingleActor {
        /// Acting agents.
        acting: usize,
    },
    /// More requests than the batch.
    #[error("{requests} requests for a batch of {batch}")]
    BatchTooLarge {
        /// Requests.
        requests: usize,
        /// Batch size.
        batch: u32,
    },
    /// The output slice does not have one handle per request.
    #[error("{got} output handles for {expected} requests")]
    OutputLength {
        /// Requests.
        expected: usize,
        /// Handles given.
        got: usize,
    },
    /// Strict mode: no free slot (retired slots do not count).
    #[error("all {capacity} node slots are in use")]
    CapacityExceeded {
        /// Capacity.
        capacity: u32,
    },
    /// Eviction mode: no free slot and nothing evictable.
    #[error("no node is evictable")]
    Exhausted,
    /// The allocation counter would overflow.
    #[error("the allocation counter is exhausted")]
    CounterExhausted,
    /// An edge's visit count or virtual loss would overflow.
    #[error("an edge counter is exhausted")]
    VisitsExhausted,
    /// A node's hold count would overflow.
    #[error("a hold count is exhausted")]
    HoldsExhausted,
    /// `release` of a node that is not held.
    #[error("the node is not held")]
    NotHeld,
    /// Re-rooting or clearing would free a held node.
    #[error("slot {slot} is held")]
    Held {
        /// The held slot.
        slot: u32,
    },
    /// The root cannot be released by eviction or re-rooting away.
    #[error("the operation needs a non-root node")]
    RootNode,
}
