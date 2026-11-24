// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed errors of the search pool (ADR-0022, ADR-0043). After any error,
//! the tree and its counters are unchanged.

use tessera_rl_core::PolicyGroup;
use tessera_rl_engine::EngineError;
use thiserror::Error;

/// Errors of [`SearchPool`](crate::SearchPool).
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum SearchError {
    /// The scratch pool failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// The evaluator failed or returned an invalid response.
    #[error(transparent)]
    Evaluator(#[from] crate::evaluator::EvaluatorError),
    /// A terminal reward outside `[-1, 1]` or not finite (ADR-0043: the
    /// scalar contract covers two-player zero-sum games).
    #[error("a terminal reward is outside [-1, 1]")]
    TerminalValue,
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
    /// The action space is neither `Discrete` nor `MultiDiscrete`, or
    /// declares entity groups (ADR-0086).
    #[error("search needs a Discrete or MultiDiscrete action space without entity groups")]
    UnsupportedActions,
    /// A `MultiDiscrete` action space with more joint actions than
    /// [`crate::MAX_JOINT_ACTIONS`] (ADR-0086).
    #[error("{joint} joint actions, above the search maximum of {max}")]
    TooManyJointActions {
        /// The product of the head sizes (saturating).
        joint: u64,
        /// The cap.
        max: u32,
    },
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
    /// A scratch pool with heuristic slots and no searcher: neither
    /// `SearchOptions::searcher` nor a determinization agent (ADR-0093
    /// decision 2).
    #[error("a pool with heuristic slots needs a searcher: set SearchOptions::searcher (ADR-0093)")]
    SearcherRequired,
    /// `SearchOptions::searcher` and the determinization agent differ.
    #[error("searcher slot {searcher} differs from the determinization agent {determinized}")]
    SearcherMismatch {
        /// `SearchOptions::searcher`.
        searcher: usize,
        /// The determinization agent.
        determinized: usize,
    },
    /// The searcher slot is not a `Learner` slot (ADR-0093 decision 3).
    #[error("searcher slot {slot} is {group:?}, not a Learner slot")]
    SearcherNotLearner {
        /// The searcher slot.
        slot: usize,
        /// Its group.
        group: PolicyGroup,
    },
    /// A slot other than the searcher that search cannot play as an
    /// opponent: `Learner`, `External`, or `PrivilegedHeuristic` outside
    /// a privileged search (ADR-0093 decision 3).
    #[error(
        "slot {slot} is {group:?}: every slot but the searcher must be a heuristic slot (ADR-0093)"
    )]
    NotSearchable {
        /// The slot.
        slot: usize,
        /// Its group.
        group: PolicyGroup,
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
    /// A backed-up value that is NaN or infinite.
    #[error("a backed-up value is not finite")]
    NonFiniteValue,
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
    /// A determinized root waits for its reset: there is nothing to
    /// search after the episode's end (ADR-0046).
    #[error("the root waits for its reset; a determinized root needs a running episode")]
    PendingResetRoot,
    /// The root fails the readiness checks of an initialized record.
    #[error("the root is not an initialized record: {source}")]
    RootNotReady {
        /// The engine's import check that failed.
        source: EngineError,
    },
    /// The sampled state is not consistent with the information state.
    #[error("inconsistent determinization: {reason}")]
    InconsistentSample {
        /// What did not hold.
        reason: &'static str,
    },
    /// The sampled state violates the environment's invariants
    /// (`Validate`).
    #[error("the sampled state is invalid: {violation}")]
    SampleInvariant {
        /// The violated invariant.
        violation: tessera_rl_core::InvariantViolation,
    },
    /// The sampled record has a field with an invalid bit pattern.
    #[error("the sampled record has an invalid field: {source}")]
    SampleField {
        /// The decoding error of the field.
        source: tessera_rl_core::DecodeError,
    },
    /// Author code (`Determinize::sample`, `Validate`, `info_state` or
    /// `acting`) panicked while a root was determinized. The panic was
    /// contained; the tree and the scratch pool are unchanged.
    #[error("environment code panicked while a root was determinized")]
    AuthorPanicked,
    /// The environment has adaptive fields: ADR-0046 allows no
    /// determinization until adaptive values have a route into the
    /// information state.
    #[error("an environment with adaptive fields cannot be determinized yet (ADR-0046)")]
    AdaptiveDeterminize,
    /// A `Sequential` sample without exactly one acting agent.
    #[error("{acting} acting agents in a Sequential sample; exactly one is required")]
    SampleNotSingleActor {
        /// Acting agents.
        acting: usize,
    },
    /// `reroot` on a determinized tree: the retained subtree belongs to
    /// the old sample, so the tree is resampled instead (ADR-0046).
    #[error("a determinized tree is never re-rooted; set a new determinized root")]
    DeterminizedReroot,
    /// `set_root` with a full record on a determinized pool.
    #[error("a determinized pool takes its roots from set_root_determinized")]
    DeterminizedPool,
    /// A Gumbel root decision found a state its invariants exclude (a
    /// halving with pending virtual loss, or no root visit); the call is
    /// rolled back.
    #[error("Gumbel root invariant: {reason}")]
    GumbelInvariant {
        /// What did not hold.
        reason: &'static str,
    },
    /// `set_root_determinized` on a pool not made by `new_determinized`.
    #[error("the pool was not made by new_determinized")]
    NotDeterminized,
    /// A root on a scratch pool with frame stacks without its frame-stack
    /// bytes, or with the wrong number of them (ADR-0078 item 7.4): its
    /// observations would differ from the live pool's. Determinized pools
    /// do not support frame stacks.
    #[error("root frame-stack bytes: {found}, the scratch pool needs {expected}")]
    StackedRoot {
        /// `transform_state_bytes` of the scratch pool.
        expected: usize,
        /// Bytes given.
        found: usize,
    },
    /// A transform context of another transform list (ADR-0078 item 7.1).
    #[error("transform identity {found:#018x} differs from the scratch pool's {expected:#018x}")]
    TransformIdentity {
        /// The scratch pool's.
        expected: u64,
        /// The context's.
        found: u64,
    },
}
