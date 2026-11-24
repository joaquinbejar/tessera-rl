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
    /// A host environment refused to start (`HostError::Construction`,
    /// ADR-0062).
    #[error("host environment construction failed: {reason}")]
    HostConstruction {
        /// What failed, as the environment reported it.
        reason: String,
    },
    /// A space kind the engine does not support yet.
    #[error("unsupported space: {what}")]
    UnsupportedSpace {
        /// What is not supported.
        what: &'static str,
    },
    /// Two exported observation leaves share a name, for example a leaf
    /// `units_mask` beside a set `units` (ADR-0082 point 2).
    #[error("duplicate observation leaf name {name:?}")]
    DuplicateLeafName {
        /// The name exported twice.
        name: String,
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
    /// A host pool without snapshots (ADR-0073): its environment does not
    /// implement `HostSnapshot`, or the type-erased pool was not built with
    /// `HostPool::with_dyn_snapshots`.
    #[error("this host environment has no snapshots")]
    NoSnapshots,
    /// A host environment's `max_snapshot_len` is over
    /// `MAX_HOST_SNAPSHOT_BYTES` (ADR-0073); nothing was reserved.
    #[error("max_snapshot_len is {max} bytes, over the limit of {cap}")]
    SnapshotBound {
        /// The environment's bound.
        max: usize,
        /// The engine's limit.
        cap: usize,
    },
    /// The allocator refused the bytes reserved for a snapshot.
    #[error("could not reserve {bytes} bytes for a host snapshot")]
    SnapshotAllocation {
        /// Bytes asked for.
        bytes: usize,
    },
    /// A host environment refused a record's bytes
    /// (`HostSnapshot::check_snapshot`, ADR-0073); nothing was restored.
    #[error("record {index}: snapshot rejected ({code}): {reason}")]
    SnapshotRejected {
        /// Position of the record in the call.
        index: usize,
        /// The environment's code.
        code: u32,
        /// The environment's reason.
        reason: &'static str,
    },
    /// A host environment's `save` wrote more than its
    /// `max_snapshot_len` (ADR-0073 item 1): an author bug that poisons the
    /// pool.
    #[error("instance {instance}: save wrote {len} bytes, over max_snapshot_len {max}")]
    SnapshotTooLong {
        /// The instance.
        instance: u32,
        /// Bytes written.
        len: usize,
        /// The environment's bound.
        max: usize,
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
    /// An archive or its reset source is invalid (ADR-0075).
    #[error("archive: {reason}")]
    Archive {
        /// What is wrong.
        reason: &'static str,
    },
    /// A pool context names an archive other than the one given, or the
    /// pool's archive is of another environment or config (ADR-0075).
    #[error("archive mismatch: expected digest {expected:016x}, found {found:016x}")]
    ArchiveMismatch {
        /// The digest the pool or context requires.
        expected: u64,
        /// The digest given (0 when no archive was given).
        found: u64,
    },
    /// A transform list the pool cannot run, or a transform operation it
    /// refuses (ADR-0078).
    #[error(transparent)]
    Transform(#[from] crate::transform::TransformError),
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
    /// An epoch barrier on an Async pool whose adaptive subsystems are not
    /// all frozen (ADR-0081 item 2.1, ADR-0026).
    #[error("the epoch barrier needs every adaptive subsystem frozen in Async mode")]
    AdaptiveInAsync,
    /// A reduction counter of the epoch barrier would overflow `u64`.
    #[error("epoch barrier: {counter} overflows")]
    StatisticsOverflow {
        /// The counter.
        counter: &'static str,
    },
    /// A live watch is already attached to this pool, with its handle
    /// alive (ADR-0083 item 7).
    #[error("a watch is already attached to this pool")]
    WatchAttached,
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
    /// Async does not support lane-wise (`SoA`) states (ADR-0047, ADR-0090
    /// decision 4.5).
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
    /// A host environment's `validate` refused the instance's joint action
    /// (ADR-0065 item 2); nothing was stepped.
    #[error("instance {instance}: {reason} (code {code})")]
    ActionRejected {
        /// The first instance, in instance order, whose action was refused.
        instance: u32,
        /// The environment's code.
        code: u32,
        /// The environment's reason.
        reason: &'static str,
    },
    /// A host environment's `refuse` refused an agent slot that was not
    /// alive and acting (ADR-0070 item 2). The instance was not stepped. It
    /// poisons the pool.
    #[error("instance {instance}: refused agent slot {agent}, which is not acting")]
    RefusedNotActing {
        /// The first instance, in instance order, with such a refusal.
        instance: u32,
        /// The first such slot of that instance.
        agent: u32,
    },
    /// An in-engine heuristic wrote an action out of range or illegal under
    /// the mask (ADR-0079 item 3.2): an environment bug. It poisons the
    /// pool, in every build.
    #[error("instance {instance}: the heuristic of agent slot {slot} wrote an invalid action")]
    HeuristicAction {
        /// The first instance, in instance order, with an invalid action.
        instance: u32,
        /// The first such slot of that instance.
        slot: u16,
    },
    /// An invalid policy map, or one set at the wrong time (ADR-0079 item
    /// 2): a slot out of range, a heuristic group the environment cannot
    /// drive, or a map set after the first reset.
    #[error("invalid policy map: {reason}")]
    PolicyMap {
        /// The rule broken.
        reason: &'static str,
    },
    /// A pool context whose policy map digest is not this pool's (ADR-0079
    /// item 2.3); nothing was written.
    #[error("policy map digest {found:#018x} does not match the pool's {expected:#018x}")]
    PolicyMapMismatch {
        /// This pool's digest.
        expected: u64,
        /// The context's digest.
        found: u64,
    },
    /// A mode that does not run heuristic slots yet (ADR-0079 stage A:
    /// Pipelined and Async refuse a map with a heuristic slot); the pool
    /// is handed back.
    #[error("{mode} mode does not support heuristic slots yet")]
    HeuristicSlotsUnsupported {
        /// The mode.
        mode: &'static str,
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
