// SPDX-License-Identifier: MIT OR Apache-2.0

//! The typed errors of recording and replaying (ADR-0072).

use crate::error::EngineError;
use crate::persist::PersistError;

use super::format::FORMAT_VERSION;

/// Which check of a replay failed (ADR-0072 section 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceKind {
    /// A call that was accepted when recording failed on replay without
    /// poisoning the pool.
    CallFailed,
    /// A call poisoned the pool on replay, but the file records no poison.
    UnexpectedPoison,
    /// The file records a poisoning call, and the call did not poison.
    NotPoisoned,
    /// A checkpoint's `StateHash` differs from the replayed instance's.
    Checkpoint,
    /// A periodic keyframe's record differs from the replayed instance's.
    Keyframe,
    /// The pool's branch counter differs from the keyframe's.
    BranchCounter,
    /// The validated import path refused a state-write keyframe.
    KeyframeRejected,
    /// A recorded rejection replayed as an accepted call, or as a rejection
    /// with another instance or code (ADR-0069 item 8); or a recorded
    /// accepted call was rejected.
    RejectionMismatch,
    /// A stepped call's refusal rows differ (ADR-0070 item 9).
    RefusalMismatch,
    /// A call's outputs hash differs (ADR-0072 section 4).
    OutputsMismatch,
    /// A keyframe's reset source differs from the replayed pool's
    /// (ADR-0075 item 7).
    ResetSource,
    /// An Async `recv` could not deliver the recorded ids: an id the
    /// recording received was never sent on replay, or nothing was left in
    /// flight.
    RecvMismatch,
}

/// Why a replay file could not be written, read or replayed.
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    /// The header's canonical config bytes do not decode into the
    /// environment's config (#377).
    #[error("replay config: {0}")]
    Config(#[from] tessera_rl_core::CanonicalDecodeError),
    /// The pool refused a call.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// A shared ADR-0011 header field is invalid, or the file does not match
    /// the pool.
    #[error(transparent)]
    Persist(#[from] PersistError),
    /// The writer failed.
    #[error("replay writer: {0}")]
    Io(#[from] std::io::Error),
    /// The first bytes are not the replay magic.
    #[error("not a Tessera replay file")]
    BadMagic,
    /// A format version this library does not read.
    #[error("replay format version {found}, this library reads {FORMAT_VERSION}")]
    UnsupportedVersion {
        /// The file's version.
        found: u32,
    },
    /// The header checksum does not match.
    #[error("replay header checksum mismatch: the file is damaged")]
    HeaderChecksum,
    /// The checksum over the events does not match.
    #[error("replay event checksum mismatch: the file is damaged")]
    EventChecksum,
    /// The end event's count differs from the events in the file.
    #[error("the end event counts {recorded} events, the file holds {found}")]
    EventCount {
        /// The count in the end event.
        recorded: u64,
        /// The events found.
        found: u64,
    },
    /// The file has no end event.
    #[error("the replay file has no end event")]
    MissingEnd,
    /// Bytes follow the end event.
    #[error("bytes follow the end event")]
    TrailingBytes,
    /// An event tag this format version does not define.
    #[error("unknown replay event tag {tag}")]
    UnknownTag {
        /// The tag.
        tag: u8,
    },
    /// A tag reserved for a later implementation (Rejected, Adaptive, Recv).
    #[error("replay event tag {tag} is reserved and not supported by this library")]
    ReservedTag {
        /// The tag.
        tag: u8,
    },
    /// A header feature this library does not implement yet (host pools,
    /// `from_reset_only` files, outputs hashes).
    #[error("unsupported replay feature: {feature}")]
    Unsupported {
        /// The feature.
        feature: &'static str,
    },
    /// An event body is malformed.
    #[error("event {event_index}: malformed {field}")]
    Malformed {
        /// The event, counted from 0 after the header.
        event_index: u64,
        /// What is wrong.
        field: &'static str,
    },
    /// A record in a keyframe is invalid.
    #[error("event {event_index}: record {index}: {source}")]
    Record {
        /// The event.
        event_index: u64,
        /// The record within the keyframe.
        index: usize,
        /// The field error.
        #[source]
        source: tessera_rl_core::DecodeError,
    },
    /// Decoding an event would hold more than `max_decoded_bytes`.
    #[error("decoding needs {bytes} bytes, over the limit of {limit}")]
    TooLarge {
        /// Bytes needed (`u64::MAX` on overflow).
        bytes: u64,
        /// The limit.
        limit: u64,
    },
    /// The allocator refused.
    #[error("could not allocate a decoded event")]
    Allocation,
    /// The pool has adaptive config values, whose events are reserved
    /// (ADR-0072 section 3, tag 7).
    #[error("recording a pool with adaptive config values is not supported")]
    AdaptiveUnsupported,
    /// A call poisoned the recorded pool; the recorder accepts only
    /// `finish`.
    #[error("the recorded pool was poisoned; the recording is closed")]
    RecorderClosed,
    /// The replay differs from the recording (ADR-0072 section 5).
    #[error(
        "replay diverged at event {event_index} ({kind:?}, instance {instance:?}); last matching checkpoint {last_matching_checkpoint:?}"
    )]
    Diverged {
        /// The event whose check failed.
        event_index: u64,
        /// The check.
        kind: DivergenceKind,
        /// The instance, when the check is per instance.
        instance: Option<u32>,
        /// The event index of the last checkpoint that matched, if any.
        last_matching_checkpoint: Option<u64>,
    },
}
