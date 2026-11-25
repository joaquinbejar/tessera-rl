// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reading replay files: the header, the framing and the trailer are checked
//! at open; event bodies are parsed one at a time (ADR-0072 sections 2, 6
//! and 8).
//!
//! The bytes are untrusted. Nothing is cast, every length is checked against
//! the bytes left before anything is allocated, and the input plus a decoded
//! event never passes `max_decoded_bytes`.

use tessera_rl_core::StateHash;
use tessera_rl_core::encoding::checksum;
use tessera_rl_num::identity::Seed;

use crate::persist::{DecodeLimits, FileHeader, PersistError, Reader, u64_of};

use super::ReplayError;
use super::cursor::ReplayCursor;
use super::format::{
    Body, END_BODY_LEN, EventTag, FLAG_ASYNC, FLAG_CONFIG_CHECKED, FLAG_FROM_RESET_ONLY,
    FLAG_HOST_SNAPSHOTS, FLAG_OUTPUTS_HASH, FORMAT_VERSION, FRAME_LEN, IDS_ALL, IDS_LIST,
    MAX_CONFIG_BYTES, REPLAY_MAGIC, StateWriteOp,
};
use super::v1_fields::{
    ArchiveHeader, ResetSourceRecord, read_adaptive, read_extension, read_reset_from,
    read_reset_source,
};

/// The instances an event touches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ids {
    /// Every instance, in id order.
    All,
    /// These ids, in this order.
    List(Vec<u32>),
}

/// The `Reseed` of a recorded reset, owned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordedReseed {
    /// `Reseed::Keep`.
    Keep,
    /// `Reseed::Base`.
    Base(Seed),
    /// `Reseed::List`.
    List(Vec<Seed>),
    /// `Reseed::Each`.
    Each(Vec<Option<Seed>>),
}

impl RecordedReseed {
    /// The pool argument.
    #[must_use]
    pub fn as_reseed(&self) -> crate::pool::Reseed<'_> {
        use crate::pool::Reseed;
        match self {
            Self::Keep => Reseed::Keep,
            Self::Base(seed) => Reseed::Base(*seed),
            Self::List(seeds) => Reseed::List(seeds),
            Self::Each(seeds) => Reseed::Each(seeds),
        }
    }
}

/// One event of a replay file, decoded.
#[derive(Debug, Clone, PartialEq)]
pub enum Event<'a> {
    /// A `reset` (ids) or `reset_all` (`Ids::All`) call.
    Reset {
        /// The reset instances.
        ids: Ids,
        /// The reseed argument.
        reseed: RecordedReseed,
        /// The outputs hash of the call, in files that carry one.
        outputs_hash: Option<u64>,
    },
    /// A `step` call.
    Step {
        /// The stepped instances.
        ids: Ids,
        /// The discrete actions as submitted.
        discrete: Vec<i32>,
        /// The continuous actions as submitted, as bits.
        continuous_bits: Vec<u32>,
        /// The outputs hash of the call, in files that carry one.
        outputs_hash: Option<u64>,
        /// The refused agents of the call, for a refusing host (ADR-0070
        /// item 9); empty otherwise.
        refusals: Vec<RefusalRow>,
    },
    /// A whole-call rejection of a notifying host (ADR-0069 item 8).
    Rejected {
        /// The full joint action of the call.
        discrete: Vec<i32>,
        /// The continuous actions, as bits.
        continuous_bits: Vec<u32>,
        /// The reported instance.
        instance: u32,
        /// The environment's code.
        code: u32,
    },
    /// A state write; a keyframe of `ids` follows.
    StateWrite {
        /// The pool call.
        op: StateWriteOp,
        /// The written instances.
        ids: Ids,
        /// The pool's branch counter after the call.
        branch_counter: u64,
    },
    /// Canonical records of `ids`.
    Keyframe {
        /// The pool's branch counter.
        branch_counter: u64,
        /// The pool's reset source (ADR-0075 item 7).
        reset_source: ResetSourceRecord,
        /// The instances.
        ids: Ids,
        /// One canonical record (ADR-0029) per id, `record_len` bytes each;
        /// untrusted.
        records: &'a [u8],
        /// In a `has_league` file (ADR-0088 erratum 3): one `u32` policy id
        /// per id, little-endian, after the records; empty otherwise.
        assignments: &'a [u8],
    },
    /// `StateHash`es of `ids`.
    Checkpoint {
        /// The instances.
        ids: Ids,
        /// One hash per id.
        hashes: Vec<StateHash>,
    },
    /// An Async `recv`: the ids it returned, in order, and the outputs
    /// hash of those ids when the file carries one.
    Recv {
        /// The received ids, in the order the token lists them.
        ids: Vec<u32>,
        /// The outputs hash of the ids, in that order.
        outputs_hash: Option<u64>,
    },
    /// The preceding call poisoned the pool.
    Poisoned,
    /// New adaptive targets (ADR-0081 item 7): per changed instance, its id
    /// and its canonical adaptive values (`value_len` bytes each).
    Adaptive {
        /// Bytes of one instance's adaptive values.
        value_len: usize,
        /// The changed instances, ids strictly increasing.
        entries: Vec<(u32, &'a [u8])>,
    },
    /// The league's view (ADR-0088 section 8, erratum 3): first in a
    /// `has_league` file, then after each barrier that changed it. The
    /// body is decoded by the sync replay, which reports its errors with
    /// this event's index.
    League {
        /// The event body.
        body: &'a [u8],
    },
    /// A `set_reset_source` call (ADR-0075 item 7).
    ResetSource(ResetSourceRecord),
    /// A `reset_from` call (ADR-0075 item 5): instance ids and the archive
    /// entry for each.
    ResetFrom {
        /// The reset instances, distinct.
        ids: Vec<u32>,
        /// One archive entry index per id.
        entries: Vec<u32>,
    },
}

/// One refused agent of a stepped call (ADR-0070 item 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefusalRow {
    /// The instance.
    pub instance: u32,
    /// The agent slot.
    pub slot: u32,
    /// The refuse code.
    pub code: u32,
}

/// The host fields of a host pool's replay header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostInfo {
    /// The host protocol: 1, or 2 for a refusing host.
    pub host_protocol: u8,
    /// `HostEnvironment::NOTIFIES_REJECTION`.
    pub notifies: bool,
    /// `HostEnvironment::REFUSES_AGENTS`.
    pub refuses: bool,
    /// `Reset`, `Step` and `Recv` events carry an outputs hash.
    pub outputs_hash: bool,
    /// Which kind of host file.
    pub kind: HostFileKind,
}

/// The two kinds of host file (ADR-0072 section 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostFileKind {
    /// `from_reset_only`: the first event is a reset of every instance; no
    /// keyframes, checkpoints or state writes; no config in the header.
    FromReset,
    /// A host with snapshots (ADR-0073): keyframes of host records and
    /// checkpoints, a checked canonical config.
    Snapshots {
        /// The ADR-0073 identity digest.
        identity: u64,
    },
}

impl HostInfo {
    /// Whether the config bytes and `config_hash` are filled and checked:
    /// exactly for a file with snapshots.
    #[must_use]
    pub const fn config_checked(&self) -> bool {
        matches!(self.kind, HostFileKind::Snapshots { .. })
    }

    /// The identity digest of a file with snapshots.
    #[must_use]
    pub const fn identity(&self) -> Option<u64> {
        match self.kind {
            HostFileKind::Snapshots { identity } => Some(identity),
            HostFileKind::FromReset => None,
        }
    }
}

/// The replay extension of the header, after the ADR-0011 fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each bool is an independent header flag of the file format"
)]
pub struct ReplayHeader<'a> {
    /// Instances in the recorded pool.
    pub num_envs: u32,
    /// Bytes of one canonical record.
    pub record_len: u32,
    /// The canonical config bytes.
    pub config: &'a [u8],
    /// Agents per instance.
    pub agents: u32,
    /// Discrete action heads per agent.
    pub discrete_heads: u32,
    /// Continuous action heads per agent.
    pub continuous_heads: u32,
    /// `Some` for a host pool's file.
    pub host: Option<HostInfo>,
    /// Recorded on an `AsyncPool` (flag bit4): the file may hold `Recv`
    /// events and replays only with `Replay::run_async`.
    pub async_pool: bool,
    /// `Reset`, `Step` and `Recv` events carry an outputs hash (flag bit1).
    pub outputs_hash: bool,
    /// The policy map (ADR-0079 item 6): one group code per agent slot (0
    /// learner, 1 external, 2 heuristic, 3 privileged heuristic, 4 opponent
    /// (ADR-0088)); empty for
    /// the default map, every slot a learner.
    pub policy_map: &'a [u8],
    /// Whether the pool has a curriculum (ADR-0081 item 7): only then may
    /// the file hold `Adaptive` events.
    pub has_curriculum: bool,
    /// Whether the pool has a league (ADR-0088 item 8): only then may the
    /// file hold `League` events.
    pub has_league: bool,
    /// The archive reset source at recording start (ADR-0075), if any.
    pub archive: Option<ArchiveHeader<'a>>,
    /// The pool's chunk size; 0 when not recorded (no check).
    pub chunk_size: u32,
}

/// What `scan` needs to know which tags a file may hold.
#[derive(Debug, Clone, Copy)]
struct TagRules {
    host: Option<HostInfo>,
    async_pool: bool,
    has_curriculum: bool,
    has_league: bool,
}

/// A parsed and checked header, before the events are scanned.
struct Opened<'a> {
    file: FileHeader<'a>,
    header: ReplayHeader<'a>,
    rules: TagRules,
    /// The bytes after the header: the events and the end event.
    rest: &'a [u8],
    input_len: u64,
}

/// An opened replay file: header, framing and trailer checked.
pub struct Replay<'a> {
    pub(crate) file: FileHeader<'a>,
    pub(crate) header: ReplayHeader<'a>,
    events: &'a [u8],
    event_count: u64,
    input_len: u64,
    pub(crate) limits: DecodeLimits,
}

impl core::fmt::Debug for Replay<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Replay")
            .field("header", &self.header)
            .field("event_count", &self.event_count)
            .finish_non_exhaustive()
    }
}

impl<'a> Replay<'a> {
    /// Opens a replay file: checks the magic, the version, the header
    /// checksum and fields, the limits, every event frame and tag, and the
    /// trailer's count and checksum. Event bodies are parsed later, by
    /// [`Replay::events`].
    ///
    /// Known gap: the whole file is held in memory as `bytes`; it is not
    /// streamed from a reader as ADR-0072 section 8 says.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayError::BadMagic`] for another file kind (a snapshot
    /// file included), [`ReplayError::UnsupportedVersion`] for another
    /// version, [`ReplayError::UnknownTag`] or [`ReplayError::ReservedTag`]
    /// for a tag this library does not replay, and another typed error for
    /// anything but a sound file within `limits`.
    pub fn open(bytes: &'a [u8], limits: DecodeLimits) -> Result<Self, ReplayError> {
        let Opened {
            file,
            header,
            rules,
            rest,
            input_len,
        } = Self::open_header(bytes, limits)?;
        let (events, event_count) = scan(rest, rules)?;
        Ok(Self {
            file,
            header,
            events,
            event_count,
            input_len,
            limits,
        })
    }

    /// Opens `bytes` again for a [`ReplayCursor`] built from the same bytes:
    /// the header is parsed and checked as [`Replay::open`] does, but the
    /// event frames are not scanned again; the cursor's event length and
    /// count stand in for the scan. This keeps each step of a cursor held
    /// across calls (the Python bindings own the bytes between calls) at the
    /// cost of the header plus the events it advances over, never a rescan
    /// of the whole file.
    ///
    /// The bytes must be the ones the cursor was built from. Other bytes of
    /// the same length are not unsafe (every event is still parsed with
    /// checked lengths, and a header that does not parse is refused), but
    /// they replay as whatever they hold.
    ///
    /// # Errors
    ///
    /// The header errors of [`Replay::open`], and
    /// [`ReplayError::Malformed`] if `bytes` is not as long as the file the
    /// cursor was built from.
    pub fn reopen(
        bytes: &'a [u8],
        limits: DecodeLimits,
        cursor: &ReplayCursor,
    ) -> Result<Self, ReplayError> {
        let Opened {
            file,
            header,
            rest,
            input_len,
            ..
        } = Self::open_header(bytes, limits)?;
        let (events_len, event_count, built_from) = cursor.file_shape();
        let mismatch = ReplayError::Malformed {
            event_index: 0,
            field: "a cursor reopened with other bytes",
        };
        if input_len != built_from {
            return Err(mismatch);
        }
        let events = rest.get(..events_len).ok_or(mismatch)?;
        Ok(Self {
            file,
            header,
            events,
            event_count,
            input_len,
            limits,
        })
    }

    /// The header part of [`Replay::open`]: the parsed headers, the tag
    /// rules and the bytes after the header.
    fn open_header(bytes: &'a [u8], limits: DecodeLimits) -> Result<Opened<'a>, ReplayError> {
        let input_len = u64_of(bytes.len());
        if input_len > limits.max_decoded_bytes {
            return Err(ReplayError::TooLarge {
                bytes: input_len,
                limit: limits.max_decoded_bytes,
            });
        }
        let mut r = Reader(bytes);
        if r.array::<8>()? != REPLAY_MAGIC {
            return Err(ReplayError::BadMagic);
        }
        let found = r.u32()?;
        if found != FORMAT_VERSION {
            return Err(ReplayError::UnsupportedVersion { found });
        }
        let mut r = Reader(bytes);
        let file = FileHeader::read(&mut r)?;
        let config_len = usize::try_from(r.u32()?).map_err(|_| PersistError::Truncated)?;
        if config_len > MAX_CONFIG_BYTES {
            return Err(PersistError::FieldTooLong {
                field: "config",
                len: config_len,
                cap: MAX_CONFIG_BYTES,
            }
            .into());
        }
        let config = r.take(config_len)?;
        let agents = r.u32()?;
        let discrete_heads = r.u32()?;
        let continuous_heads = r.u32()?;
        let [flags, host_protocol, notifies, refuses] = r.array::<4>()?;
        let identity = if host_protocol != 0 && flags & FLAG_HOST_SNAPSHOTS != 0 {
            Some(r.u64()?)
        } else {
            None
        };
        let (ext, used) = read_extension(r.0, agents)?;
        r.take(used)?;
        let header_len = bytes
            .len()
            .checked_sub(r.0.len())
            .ok_or(PersistError::Truncated)?;
        let stored = r.u64()?;
        let header_bytes = bytes.get(..header_len).ok_or(PersistError::Truncated)?;
        if checksum(header_bytes) != stored {
            return Err(ReplayError::HeaderChecksum);
        }
        let host = check_replay_fields(flags, host_protocol, notifies, refuses, identity)?;
        check_file(&file, host, config, limits)?;
        let header = ReplayHeader {
            num_envs: file.count,
            record_len: file.record_len,
            config,
            agents,
            discrete_heads,
            continuous_heads,
            host,
            async_pool: flags & FLAG_ASYNC != 0,
            outputs_hash: flags & FLAG_OUTPUTS_HASH != 0,
            policy_map: ext.policy_map,
            has_curriculum: ext.has_curriculum,
            has_league: ext.has_league,
            archive: ext.archive,
            chunk_size: ext.chunk_size,
        };
        let rules = TagRules {
            host,
            async_pool: header.async_pool,
            has_curriculum: header.has_curriculum,
            has_league: header.has_league,
        };
        Ok(Opened {
            file,
            header,
            rules,
            rest: r.0,
            input_len,
        })
    }

    /// The replay extension of the header.
    #[must_use]
    pub const fn header(&self) -> &ReplayHeader<'a> {
        &self.header
    }

    /// Events before the trailer.
    #[must_use]
    pub const fn event_count(&self) -> u64 {
        self.event_count
    }

    /// The events in file order, each parsed when reached.
    #[must_use]
    pub const fn events(&self) -> Events<'a> {
        self.events_from(self.events, 0)
    }

    /// The events from byte `offset` of the event bytes, the first of them
    /// numbered `index`: `offset` must be a frame start that
    /// [`Replay::event_offsets`] returned, with its position as `index`.
    /// An offset past the end gives no events.
    #[must_use]
    pub fn events_at(&self, offset: usize, index: u64) -> Events<'a> {
        self.events_from(self.events.get(offset..).unwrap_or(&[]), index)
    }

    /// The byte offset of every event's frame in the event bytes, in file
    /// order (event `i` at position `i`): one walk over the frames, which
    /// [`Replay::open`] already checked.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Allocation`] if the list cannot be allocated, and
    /// [`ReplayError::Malformed`] if a frame does not fit (not possible after
    /// a successful open).
    pub fn event_offsets(&self) -> Result<Vec<usize>, ReplayError> {
        let n = usize::try_from(self.event_count).map_err(|_| ReplayError::Allocation)?;
        let mut out = alloc(n)?;
        let mut at: usize = 0;
        while at < self.events.len() {
            let len = self
                .events
                .get(at..)
                .and_then(|rest| rest.get(1..FRAME_LEN))
                .and_then(|l| <[u8; 4]>::try_from(l).ok())
                .and_then(|l| usize::try_from(u32::from_le_bytes(l)).ok())
                .ok_or(ReplayError::Malformed {
                    event_index: u64_of(out.len()),
                    field: "event frame",
                })?;
            out.push(at);
            at = at
                .checked_add(FRAME_LEN)
                .and_then(|a| a.checked_add(len))
                .ok_or(PersistError::Truncated)?;
        }
        Ok(out)
    }

    /// The tag byte of the frame at `offset`, if any.
    pub(super) fn tag_at(&self, offset: usize) -> Option<u8> {
        self.events.get(offset).copied()
    }

    /// The length of the event bytes and the length of the whole input.
    pub(super) const fn shape(&self) -> (usize, u64) {
        (self.events.len(), self.input_len)
    }

    const fn events_from(&self, rest: &'a [u8], index: u64) -> Events<'a> {
        Events {
            rest,
            index,
            num_envs: self.header.num_envs,
            record_len: self.header.record_len,
            host: self.header.host,
            hashes: self.header.outputs_hash,
            has_league: self.header.has_league,
            input_len: self.input_len,
            limits: self.limits,
        }
    }
}

/// The checks of the shared header that depend on the replay fields: an
/// unchecked host config is empty, the tier fits the file kind, a host file
/// holds no assets, and the instance count is within `limits`.
fn check_file(
    file: &FileHeader<'_>,
    host: Option<HostInfo>,
    config: &[u8],
    limits: DecodeLimits,
) -> Result<(), ReplayError> {
    if let Some(h) = host
        && !h.config_checked()
        && (!config.is_empty() || file.hash != 0)
    {
        return Err(PersistError::InvalidField {
            field: "config of an unchecked host file",
        }
        .into());
    }
    file.check_fields_for(host.is_some())?;
    // ADR-0077: host replays hold no assets (host snapshots neither).
    if host.is_some() && file.asset_count != 0 {
        return Err(PersistError::Incompatible { field: "assets" }.into());
    }
    if file.count == 0 {
        return Err(PersistError::InvalidField {
            field: "instance_count",
        }
        .into());
    }
    if file.count > limits.max_instances {
        return Err(PersistError::TooManyInstances {
            count: file.count,
            limit: limits.max_instances,
        }
        .into());
    }
    Ok(())
}

fn check_replay_fields(
    flags: u8,
    host_protocol: u8,
    notifies: u8,
    refuses: u8,
    identity: Option<u64>,
) -> Result<Option<HostInfo>, ReplayError> {
    let invalid = |field: &'static str| -> Result<Option<HostInfo>, ReplayError> {
        Err(PersistError::InvalidField { field }.into())
    };
    let known = FLAG_FROM_RESET_ONLY
        | FLAG_OUTPUTS_HASH
        | FLAG_CONFIG_CHECKED
        | FLAG_HOST_SNAPSHOTS
        | FLAG_ASYNC;
    if flags & !known != 0 {
        return invalid("flags");
    }
    if notifies > 1 {
        return invalid("notifies_rejection");
    }
    if refuses > 1 {
        return invalid("refuses_agents");
    }
    if host_protocol == 0 {
        if notifies != 0 || refuses != 0 || flags & (FLAG_CONFIG_CHECKED | FLAG_HOST_SNAPSHOTS) != 0
        {
            return invalid("host fields of a first-party file");
        }
        if flags & FLAG_FROM_RESET_ONLY != 0 {
            return Err(ReplayError::Unsupported {
                feature: "from_reset_only first-party replay files",
            });
        }
        // The outputs hash is optional in an async file (its `Recv` events),
        // and not supported in a synchronous first-party file.
        if flags & FLAG_OUTPUTS_HASH != 0 && flags & FLAG_ASYNC == 0 {
            return Err(ReplayError::Unsupported {
                feature: "outputs hashes in first-party replay files",
            });
        }
        return Ok(None);
    }
    if flags & FLAG_ASYNC != 0 {
        return invalid("flags: async host files");
    }
    if host_protocol > 2 {
        return invalid("host_protocol");
    }
    // ADR-0070 item 10: protocol 2 exactly for a refusing host.
    if (refuses == 1) != (host_protocol == 2) {
        return invalid("refuses_agents");
    }
    // ADR-0072 section 4: a host with snapshots (ADR-0073) has keyframes,
    // checkpoints and a checked config, and the outputs hash is optional; a
    // host without them starts from a reset and needs the outputs hash.
    let snapshots = flags & FLAG_HOST_SNAPSHOTS != 0;
    let from_reset = flags & FLAG_FROM_RESET_ONLY != 0;
    let outputs_hash = flags & FLAG_OUTPUTS_HASH != 0;
    let config_checked = flags & FLAG_CONFIG_CHECKED != 0;
    if snapshots == from_reset {
        return invalid("flags: exactly one of host_snapshots and from_reset_only");
    }
    if from_reset && !outputs_hash {
        return invalid("flags: a from_reset_only file needs outputs hashes");
    }
    if snapshots != config_checked {
        return invalid("flags: config_checked exactly for a snapshot host");
    }
    let kind = match identity {
        Some(identity) if snapshots => HostFileKind::Snapshots { identity },
        None if !snapshots => HostFileKind::FromReset,
        _ => return invalid("identity"),
    };
    Ok(Some(HostInfo {
        host_protocol,
        notifies: notifies == 1,
        refuses: refuses == 1,
        outputs_hash,
        kind,
    }))
}

/// Checks every frame, tag and the trailer; returns the event bytes before
/// the end event and their count. Allocates nothing.
fn scan(bytes: &[u8], rules: TagRules) -> Result<(&[u8], u64), ReplayError> {
    let mut rest = bytes;
    let mut count: u64 = 0;
    loop {
        let (frame, tail) = rest
            .split_at_checked(FRAME_LEN)
            .ok_or(ReplayError::MissingEnd)?;
        let [tag, l0, l1, l2, l3] =
            <[u8; FRAME_LEN]>::try_from(frame).map_err(|_| ReplayError::MissingEnd)?;
        let len = u32::from_le_bytes([l0, l1, l2, l3]);
        let tag = EventTag::from_u8(tag).ok_or(ReplayError::UnknownTag { tag })?;
        check_tag(tag, rules, count)?;
        let body_len = usize::try_from(len).map_err(|_| PersistError::Truncated)?;
        let (body, after) = tail
            .split_at_checked(body_len)
            .ok_or(PersistError::Truncated)?;
        if tag == EventTag::End {
            if len != END_BODY_LEN {
                return Err(ReplayError::Malformed {
                    event_index: count,
                    field: "end event length",
                });
            }
            if !after.is_empty() {
                return Err(ReplayError::TrailingBytes);
            }
            let mut b = Body::new(body, count);
            let recorded = b.u64("event count")?;
            let sum = b.u64("events checksum")?;
            let events_len = bytes
                .len()
                .checked_sub(rest.len())
                .ok_or(PersistError::Truncated)?;
            let events = bytes.get(..events_len).ok_or(PersistError::Truncated)?;
            if checksum(events) != sum {
                return Err(ReplayError::EventChecksum);
            }
            if recorded != count {
                return Err(ReplayError::EventCount {
                    recorded,
                    found: count,
                });
            }
            return Ok((events, count));
        }
        count = count.checked_add(1).ok_or(PersistError::Truncated)?;
        rest = after;
    }
}

/// Which tags a file may hold: `Adaptive` only with a curriculum; `League`
/// only with a league; `Recv` only in an async file; `Rejected` only in a
/// notifying host's file; keyframes, checkpoints and state writes never in a
/// `from_reset_only` host file. The first event is a keyframe, a reset for
/// a `from_reset_only` file, or the `League` event of a `has_league` file,
/// whose second event is then the keyframe (ADR-0088 erratum 3).
fn check_tag(tag: EventTag, rules: TagRules, count: u64) -> Result<(), ReplayError> {
    let TagRules {
        host,
        async_pool,
        has_curriculum,
        has_league,
    } = rules;
    let notifies = host.is_some_and(|h| h.notifies);
    let allowed = (tag == EventTag::Rejected && notifies)
        || (tag == EventTag::Recv && async_pool)
        || (tag == EventTag::Adaptive && has_curriculum)
        || (tag == EventTag::League && has_league);
    if tag.is_reserved() && !allowed {
        return Err(ReplayError::ReservedTag { tag: tag as u8 });
    }
    // ADR-0075 item 8: archives are for first-party pools.
    if host.is_some() && matches!(tag, EventTag::ResetSource | EventTag::ResetFrom) {
        return Err(ReplayError::Unsupported {
            feature: "archive resets in host replay files",
        });
    }
    let from_reset = host.is_some_and(|h| h.identity().is_none());
    if from_reset
        && matches!(
            tag,
            EventTag::Keyframe | EventTag::Checkpoint | EventTag::StateWrite
        )
    {
        return Err(ReplayError::Unsupported {
            feature: "keyframes, checkpoints and state writes in from_reset_only files",
        });
    }
    let first = if from_reset {
        EventTag::Reset
    } else if has_league {
        EventTag::League
    } else {
        EventTag::Keyframe
    };
    if has_league && count == 1 && tag != EventTag::Keyframe {
        return Err(ReplayError::Malformed {
            event_index: 1,
            field: "second event (the keyframe after a has_league file's League event)",
        });
    }
    if count == 0 && tag != first {
        return Err(ReplayError::Malformed {
            event_index: 0,
            field: "first event (a keyframe of every instance, a from_reset_only file's reset, or a has_league file's League event)",
        });
    }
    Ok(())
}

/// The events of a [`Replay`], parsed one at a time.
#[derive(Debug, Clone)]
pub struct Events<'a> {
    rest: &'a [u8],
    index: u64,
    num_envs: u32,
    record_len: u32,
    host: Option<HostInfo>,
    hashes: bool,
    has_league: bool,
    input_len: u64,
    limits: DecodeLimits,
}

impl<'a> Iterator for Events<'a> {
    type Item = Result<(u64, Event<'a>), ReplayError>;

    fn next(&mut self) -> Option<Self::Item> {
        let (frame, tail) = self.rest.split_at_checked(FRAME_LEN)?;
        let [tag, l0, l1, l2, l3] = <[u8; FRAME_LEN]>::try_from(frame).ok()?;
        let len = usize::try_from(u32::from_le_bytes([l0, l1, l2, l3])).ok()?;
        let (body, after) = tail.split_at_checked(len)?;
        self.rest = after;
        let index = self.index;
        self.index = self.index.saturating_add(1);
        let event = self.parse(tag, body, index);
        if event.is_err() {
            // One error ends the iteration.
            self.rest = &[];
        }
        Some(event.map(|e| (index, e)))
    }
}

impl<'a> Events<'a> {
    fn parse(&self, tag: u8, body: &'a [u8], index: u64) -> Result<Event<'a>, ReplayError> {
        // The bytes held: the input plus, at most, a decoded copy of the body
        // widened twice (seeds with flags, records as values).
        let held = u64_of(body.len())
            .checked_mul(2)
            .and_then(|b| b.checked_add(self.input_len))
            .unwrap_or(u64::MAX);
        if held > self.limits.max_decoded_bytes {
            return Err(ReplayError::TooLarge {
                bytes: held,
                limit: self.limits.max_decoded_bytes,
            });
        }
        let mut b = Body::new(body, index);
        let tag = EventTag::from_u8(tag).ok_or(ReplayError::UnknownTag { tag })?;
        let event = match tag {
            EventTag::Reset => {
                let ids = self.ids(&mut b)?;
                let reseed = reseed(&mut b)?;
                let outputs_hash = self.call_hash(&mut b)?;
                Event::Reset {
                    ids,
                    reseed,
                    outputs_hash,
                }
            }
            EventTag::Step => self.step(&mut b)?,
            EventTag::Rejected => self.rejected(&mut b)?,
            EventTag::StateWrite => {
                let op = StateWriteOp::from_u8(b.u8("state write op")?)
                    .ok_or_else(|| b.malformed("state write op"))?;
                let ids = self.ids(&mut b)?;
                let branch_counter = b.u64("branch counter")?;
                Event::StateWrite {
                    op,
                    ids,
                    branch_counter,
                }
            }
            EventTag::Keyframe => {
                let branch_counter = b.u64("branch counter")?;
                let reset_source = read_reset_source(&mut b)?;
                let ids = self.ids(&mut b)?;
                let n = self.len_of(&ids)?;
                let records = self.keyframe_records(&mut b, body, n)?;
                let assignments = if self.has_league {
                    let bytes = n.checked_mul(4).ok_or_else(|| b.malformed("assignments"))?;
                    b.take(bytes, "assignments")?
                } else {
                    &[]
                };
                Event::Keyframe {
                    branch_counter,
                    reset_source,
                    ids,
                    records,
                    assignments,
                }
            }
            EventTag::Adaptive => {
                let (value_len, entries) = read_adaptive(&mut b, self.num_envs)?;
                Event::Adaptive { value_len, entries }
            }
            EventTag::League => Event::League {
                body: b.take(b.remaining(), "league body")?,
            },
            EventTag::ResetSource => Event::ResetSource(read_reset_source(&mut b)?),
            EventTag::ResetFrom => {
                let (ids, entries) = read_reset_from(&mut b, self.num_envs)?;
                Event::ResetFrom { ids, entries }
            }
            EventTag::Checkpoint => {
                let ids = self.ids(&mut b)?;
                let hashes = self.checkpoint_hashes(&mut b, &ids)?;
                Event::Checkpoint { ids, hashes }
            }
            EventTag::Poisoned => Event::Poisoned,
            EventTag::Recv => {
                let ids = self.ids(&mut b)?;
                let Ids::List(ids) = ids else {
                    return Err(b.malformed("recv ids"));
                };
                let outputs_hash = self.outputs_hash(&mut b)?;
                Event::Recv { ids, outputs_hash }
            }
            EventTag::End => {
                return Err(ReplayError::ReservedTag { tag: tag as u8 });
            }
        };
        b.finish()?;
        Ok(event)
    }

    fn ids(&self, b: &mut Body<'_>) -> Result<Ids, ReplayError> {
        match b.u8("ids")? {
            IDS_ALL => Ok(Ids::All),
            IDS_LIST => {
                let n = b.count(4, "ids")?;
                if u64_of(n) > u64::from(self.num_envs) {
                    return Err(b.malformed("ids"));
                }
                let mut ids = alloc(n)?;
                for _ in 0..n {
                    let id = b.u32("ids")?;
                    if id >= self.num_envs {
                        return Err(b.malformed("ids"));
                    }
                    ids.push(id);
                }
                Ok(Ids::List(ids))
            }
            _ => Err(b.malformed("ids")),
        }
    }

    /// The records of a keyframe of `n` ids: fixed-length canonical records,
    /// or for a host file `(len:u32 HostRecord)*`. `body` is the whole event
    /// body that `b` reads.
    fn keyframe_records(
        &self,
        b: &mut Body<'a>,
        body: &'a [u8],
        n: usize,
    ) -> Result<&'a [u8], ReplayError> {
        if self.host.is_none() {
            let len = usize::try_from(self.record_len).map_err(|_| b.malformed("records"))?;
            let bytes = n.checked_mul(len).ok_or_else(|| b.malformed("records"))?;
            return b.take(bytes, "records");
        }
        // Host records have variable length: `len:u32` before each.
        let start = b.remaining();
        for _ in 0..n {
            let len = usize::try_from(b.u32("host record length")?)
                .map_err(|_| b.malformed("host record length"))?;
            b.take(len, "host record")?;
        }
        let used = start.saturating_sub(b.remaining());
        body.get(body.len().saturating_sub(start)..)
            .and_then(|rest| rest.get(..used))
            .ok_or_else(|| b.malformed("host records"))
    }

    /// One `StateHash` per id, filling the rest of the body.
    fn checkpoint_hashes(
        &self,
        b: &mut Body<'_>,
        ids: &Ids,
    ) -> Result<Vec<StateHash>, ReplayError> {
        let n = self.len_of(ids)?;
        if n.checked_mul(8) != Some(b.remaining()) {
            return Err(b.malformed("checkpoint hashes"));
        }
        let mut hashes = alloc(n)?;
        for _ in 0..n {
            hashes.push(StateHash::from_raw(b.u64("checkpoint hashes")?));
        }
        Ok(hashes)
    }

    /// A `Step` body.
    fn step(&self, b: &mut Body<'_>) -> Result<Event<'a>, ReplayError> {
        let ids = self.ids(b)?;
        let (discrete, continuous_bits) = joint_action(b)?;
        let outputs_hash = self.call_hash(b)?;
        let refusals = if self.host.is_some_and(|h| h.refuses) {
            self.refusals(b)?
        } else {
            Vec::new()
        };
        Ok(Event::Step {
            ids,
            discrete,
            continuous_bits,
            outputs_hash,
            refusals,
        })
    }

    /// A `Rejected` body.
    fn rejected(&self, b: &mut Body<'_>) -> Result<Event<'a>, ReplayError> {
        let (discrete, continuous_bits) = joint_action(b)?;
        let instance = b.u32("rejected instance")?;
        if instance >= self.num_envs {
            return Err(b.malformed("rejected instance"));
        }
        let code = b.u32("rejected code")?;
        Ok(Event::Rejected {
            discrete,
            continuous_bits,
            instance,
            code,
        })
    }

    /// The outputs hash of a `Reset` or `Step`: present in a host file with
    /// the flag. An async file hashes only its `Recv` events, which receive
    /// the outputs; a send or a reset returns none.
    fn call_hash(&self, b: &mut Body<'_>) -> Result<Option<u64>, ReplayError> {
        if self.host.is_some() {
            self.outputs_hash(b)
        } else {
            Ok(None)
        }
    }

    fn outputs_hash(&self, b: &mut Body<'_>) -> Result<Option<u64>, ReplayError> {
        if self.hashes {
            Ok(Some(b.u64("outputs hash")?))
        } else {
            Ok(None)
        }
    }

    /// Sparse rows, in (instance, slot) order, each code nonzero.
    fn refusals(&self, b: &mut Body<'_>) -> Result<Vec<RefusalRow>, ReplayError> {
        let n = b.count(12, "refusal rows")?;
        let mut rows: Vec<RefusalRow> = alloc(n)?;
        for _ in 0..n {
            let row = RefusalRow {
                instance: b.u32("refusal rows")?,
                slot: b.u32("refusal rows")?,
                code: b.u32("refusal rows")?,
            };
            let ordered = rows
                .last()
                .is_none_or(|p| (p.instance, p.slot) < (row.instance, row.slot));
            if row.instance >= self.num_envs || row.code == 0 || !ordered {
                return Err(b.malformed("refusal rows"));
            }
            rows.push(row);
        }
        Ok(rows)
    }

    fn len_of(&self, ids: &Ids) -> Result<usize, ReplayError> {
        match ids {
            Ids::All => usize::try_from(self.num_envs).map_err(|_| ReplayError::Allocation),
            Ids::List(ids) => Ok(ids.len()),
        }
    }
}

fn reseed(b: &mut Body<'_>) -> Result<RecordedReseed, ReplayError> {
    Ok(match b.u8("reseed")? {
        0 => RecordedReseed::Keep,
        1 => RecordedReseed::Base(Seed::new(b.u64("reseed")?)),
        2 => {
            let n = b.count(8, "seeds")?;
            let mut seeds = alloc(n)?;
            for _ in 0..n {
                seeds.push(Seed::new(b.u64("seeds")?));
            }
            RecordedReseed::List(seeds)
        }
        3 => {
            let n = b.count(9, "seeds")?;
            let mut seeds = alloc(n)?;
            for _ in 0..n {
                let some = b.u8("seeds")?;
                let seed = b.u64("seeds")?;
                seeds.push(match (some, seed) {
                    (0, 0) => None,
                    (1, seed) => Some(Seed::new(seed)),
                    _ => return Err(b.malformed("seeds")),
                });
            }
            RecordedReseed::Each(seeds)
        }
        _ => return Err(b.malformed("reseed")),
    })
}

/// A joint action: discrete `i32`s, then continuous bits.
fn joint_action(b: &mut Body<'_>) -> Result<(Vec<i32>, Vec<u32>), ReplayError> {
    let n = b.count(4, "discrete actions")?;
    let mut discrete = alloc(n)?;
    for _ in 0..n {
        discrete.push(b.u32("discrete actions")?.cast_signed());
    }
    let n = b.count(4, "continuous actions")?;
    let mut continuous_bits = alloc(n)?;
    for _ in 0..n {
        continuous_bits.push(b.u32("continuous actions")?);
    }
    Ok((discrete, continuous_bits))
}

fn alloc<T>(n: usize) -> Result<Vec<T>, ReplayError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| ReplayError::Allocation)?;
    Ok(v)
}
