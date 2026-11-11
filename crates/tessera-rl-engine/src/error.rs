// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed errors of the engine (ADR-0022, ADR-0028).

use tessera_rl_core::{ConfigError, SpaceError};
use thiserror::Error;

/// Errors returned by pool construction and the validation boundary. On
/// any error from `step` or `reset`, no instance was mutated.
///
/// Non-exhaustive (ADR-0055): match it with a wildcard arm.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum EngineError {
    /// The environment config is invalid.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// A space is invalid or not supported.
    #[error(transparent)]
    Space(#[from] SpaceError),
    /// A space kind the engine does not support yet.
    #[error("unsupported space: {what}")]
    UnsupportedSpace {
        /// What is not supported.
        what: &'static str,
    },
    /// Invalid pool options.
    #[error("invalid pool option {option}: {reason}")]
    Options {
        /// The option.
        option: &'static str,
        /// Why it is invalid.
        reason: &'static str,
    },
    /// An instance id is out of range.
    #[error("instance {id} out of range for a pool of {num_envs}")]
    InstanceOutOfRange {
        /// The id.
        id: u32,
        /// Pool size.
        num_envs: u32,
    },
    /// An instance id appears twice in one call.
    #[error("instance {id} appears more than once")]
    DuplicateInstance {
        /// The id.
        id: u32,
    },
    /// A seed list does not have one seed per instance.
    #[error("{seeds} seeds for {instances} instances")]
    SeedCount {
        /// Seeds given.
        seeds: usize,
        /// Instances to reseed.
        instances: usize,
    },
    /// An imported record has an inconsistent engine header.
    #[error("record {index}: {reason}")]
    InvalidRecord {
        /// Position of the record in the call.
        index: usize,
        /// What is inconsistent.
        reason: &'static str,
    },
    /// An imported record violates a game invariant (`Validate`).
    #[error("record {index}: {violation}")]
    Invariant {
        /// Position of the record in the call.
        index: usize,
        /// The violated invariant.
        violation: tessera_rl_core::InvariantViolation,
    },
    /// A `Sequential` environment named no acting agent, or more than one,
    /// in a step that does not end the episode (ADR-0013, ADR-0028). It
    /// poisons the pool.
    #[error("instance {instance}: {violation}")]
    TurnOrder {
        /// The first instance that violated the turn order.
        instance: u32,
        /// The violated invariant.
        violation: tessera_rl_core::InvariantViolation,
    },
    /// A record slice does not have one record per instance.
    #[error("{records} records for {instances} instances")]
    RecordCount {
        /// Records given.
        records: usize,
        /// Instances in the pool.
        instances: usize,
    },
    /// A label list does not have one label per destination.
    #[error("{labels} labels for {destinations} destinations")]
    LabelCount {
        /// Labels given.
        labels: usize,
        /// Destinations.
        destinations: usize,
    },
    /// A `fork_branched` label appears twice in one call.
    #[error("branch label {label} appears more than once")]
    DuplicateLabel {
        /// The label.
        label: u64,
    },
    /// The process ran out of output storage ids (2^64 pools built): ids
    /// are never reused, so no further pool can be constructed.
    #[error("the output storage id counter is exhausted")]
    StorageIdsExhausted,
    /// The pool's branch counter cannot be incremented.
    #[error("the branch counter is exhausted")]
    BranchCounterExhausted,
    /// An action buffer has the wrong length.
    #[error("{buffer} actions hold {actual} values, expected {expected}")]
    ActionLength {
        /// `discrete` or `continuous`.
        buffer: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// A discrete action is outside `0..n`.
    #[error("instance {instance}, agent {agent}, head {head}: action {value} outside 0..{n}")]
    DiscreteOutOfRange {
        /// Instance.
        instance: u32,
        /// Agent slot.
        agent: u16,
        /// Head.
        head: u32,
        /// The value.
        value: i32,
        /// Number of choices.
        n: u32,
    },
    /// A continuous action is NaN, infinite or outside its bounds.
    #[error(
        "instance {instance}, agent {agent}, head {head}: action {value} is not finite or outside its bounds"
    )]
    ContinuousOutOfRange {
        /// Instance.
        instance: u32,
        /// Agent slot.
        agent: u16,
        /// Head.
        head: u32,
        /// The value.
        value: f32,
    },
    /// A worker thread could not be spawned or failed to start; the workers
    /// already started were shut down.
    #[error("engine worker {worker} failed to start")]
    WorkerStartup {
        /// The worker that failed.
        worker: usize,
    },
    /// A worker thread died outside a job; the pool cannot run in Sync mode
    /// any more.
    #[error("an engine worker died; the pool cannot step in Sync mode")]
    WorkerDied,
    /// The worker gang's job sequence cannot be advanced (ADR-0047).
    #[error("the worker job sequence is exhausted")]
    SequenceExhausted,
    /// A job was submitted while two were already in flight, or a wait was
    /// made with nothing in flight (ADR-0047).
    #[error("the worker gang has {in_flight} jobs in flight")]
    GangQueue {
        /// Jobs in flight when the call was made.
        in_flight: u64,
    },
    /// An instance was sent and not yet returned by `recv` (ADR-0047).
    #[error("instance {id} is in flight")]
    InFlight {
        /// The id.
        id: u32,
    },
    /// A half was sent and not yet returned by `recv` (ADR-0047).
    #[error("half {half} is in flight")]
    HalfInFlight {
        /// The half, 0 or 1.
        half: u8,
    },
    /// A live token still covers the instance, so its rows cannot be
    /// written (ADR-0047).
    #[error("instance {id} is borrowed by a live token")]
    Borrowed {
        /// The id.
        id: u32,
    },
    /// A live token still covers the half (ADR-0047).
    #[error("half {half} is borrowed by a live token")]
    HalfBorrowed {
        /// The half, 0 or 1.
        half: u8,
    },
    /// A token from another pool was passed (ADR-0047).
    #[error("the token belongs to another pool")]
    WrongPool,
    /// A half index other than 0 or 1 (ADR-0047).
    #[error("half {half} does not exist; a pool has halves 0 and 1")]
    InvalidHalf {
        /// The half given.
        half: u32,
    },
    /// The call does not exist in this execution mode (ADR-0047).
    #[error("{call} is not available in {mode} mode")]
    WrongMode {
        /// The call.
        call: &'static str,
        /// The pool's mode.
        mode: &'static str,
    },
    /// `recv` was called with nothing in flight and nothing ready
    /// (ADR-0047). It never waits.
    #[error("nothing is in flight")]
    NothingInFlight,
    /// An Async `recv` can never fill a batch: fewer ids are in flight or
    /// ready than the batch size (ADR-0047). It never waits.
    #[error("{in_flight} in flight and {ready} ready cannot fill a batch of {batch_size}")]
    BatchUnreachable {
        /// Ids in flight.
        in_flight: u32,
        /// Ids completed and not yet returned, including ids pushed by
        /// `reset(ids)` and `restore(ids)`.
        ready: u32,
        /// The batch size.
        batch_size: u32,
    },
    /// Work is admitted to workers and not yet completed (ADR-0047).
    #[error("the pool is not quiescent: work is in flight")]
    NotQuiescent,
    /// A Pipelined half was sent for cycle `cycle` before the context
    /// version of cycle `cycle - 2` was committed (ADR-0047).
    #[error("cycle {cycle} cannot start before cycle {needed} is committed")]
    CycleAhead {
        /// The cycle the send would start.
        cycle: u64,
        /// The cycle whose commit it needs.
        needed: u64,
    },
    /// An Async `send` named no instance (ADR-0047).
    #[error("a send must name at least one instance")]
    EmptySend,
    /// The pool cannot be split into two halves of whole chunks
    /// (ADR-0047).
    #[error("the pool cannot be split into two halves of chunks of {chunk_size}")]
    HalfTooSmall {
        /// The chunk size.
        chunk_size: u32,
    },
    /// Async does not support `SoA` lane environments yet (ADR-0047,
    /// ADR-0002).
    #[error("Async mode does not support SoA lane environments")]
    LaneEnvUnsupported,
    /// An engine was created inside an engine worker (ADR-0005).
    #[error("engines cannot be created inside an engine worker")]
    NestedEngine,
    /// The pool is poisoned after a panic in environment code; only a full
    /// reset (or, later, a full restore) or dropping the pool is accepted
    /// (ADR-0028).
    #[error("the pool is poisoned by an earlier panic; reset all instances to recover")]
    Poisoned,
    /// Environment code panicked during this call. The pool is now
    /// poisoned; there is no rollback.
    #[error("environment code panicked; the pool is poisoned")]
    Panicked,
    /// A required action is valid but illegal under the current mask
    /// (`check_actions`, and every step with `debug-checks`).
    #[error(
        "instance {instance}, agent {agent}, head {head}: action {value} is illegal under the mask"
    )]
    IllegalAction {
        /// Instance.
        instance: u32,
        /// Agent slot.
        agent: u16,
        /// Head.
        head: u32,
        /// Entity group of the head (ADR-0051); 0 when none are declared.
        group: u32,
        /// The value.
        value: i32,
    },
    /// The action heads' mask declaration is invalid (ADR-0051, SPEC 11.3).
    #[error("invalid mask layout: {what}")]
    InvalidMaskLayout {
        /// The rule broken.
        what: &'static str,
    },
    /// The mask buffer would exceed the pool's mask budget (SPEC 11.3).
    #[error("mask buffer needs {bytes} bytes, over the budget of {budget}")]
    MaskBudgetExceeded {
        /// Bytes needed (`u64::MAX` on overflow).
        bytes: u64,
        /// The budget.
        budget: u64,
    },
    /// With `debug-checks`, an invalid agent event: the environment spawned
    /// an agent on a slot that was alive or died in the same step (SPEC
    /// 9.1), or reported the death of an agent that was already dead. This
    /// is an environment bug (or an inconsistent imported record): the pool
    /// is poisoned.
    #[error("instance {instance}: invalid agent event on agent slot {agent}")]
    InvalidAgentEvent {
        /// The instance.
        instance: u32,
        /// The agent slot.
        agent: u32,
    },
    /// With `debug-checks`, a NaN in a reward or in a state field not
    /// annotated `#[state(allow_nan)]` (ADR-0055). This is an environment
    /// bug: the pool is poisoned. Never constructed without the feature.
    #[error("instance {instance}: NaN in {site}")]
    NanDetected {
        /// The instance.
        instance: u32,
        /// Where the NaN is.
        site: NanSite,
    },
    /// With `debug-checks` and the engine's `TrackingAllocator` installed,
    /// environment code allocated inside a chunk step (ADR-0004, ADR-0055):
    /// the pool is poisoned. Never constructed without the feature.
    #[error("{count} allocations inside a chunk step")]
    AllocationInChunkStep {
        /// Allocations counted in the chunk steps of the call.
        count: u64,
    },
    /// The instance has not been reset yet, or its episode ended in
    /// `Disabled` auto-reset mode.
    #[error("instance {id} is pending reset")]
    PendingReset {
        /// The id.
        id: u32,
    },
    /// The instance's step or episode counter would overflow.
    #[error("instance {id}: {counter} exhausted")]
    CounterExhausted {
        /// The id.
        id: u32,
        /// `step_index` or `episode_index`.
        counter: &'static str,
    },
}

/// Where `debug-checks` found a NaN (ADR-0055).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanSite {
    /// The reward of an agent slot.
    Reward {
        /// The agent slot.
        agent: u32,
    },
    /// A state field: the top-level field that holds the NaN.
    State {
        /// The field name.
        field: &'static str,
    },
}

impl core::fmt::Display for NanSite {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Reward { agent } => write!(f, "the reward of agent slot {agent}"),
            Self::State { field } => write!(f, "state field `{field}`"),
        }
    }
}
