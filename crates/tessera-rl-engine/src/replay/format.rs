// SPDX-License-Identifier: MIT OR Apache-2.0

//! The byte format of replay files, version 1 (ADR-0072 sections 2 and 3,
//! with the fields ADR-0075, ADR-0077, ADR-0079 and ADR-0081 folded into
//! version 1 before any release shipped replay files, #411).
//!
//! ```text
//! file      := header event* end
//! header    := "TESRPLY\x1a" version:u32 shared extension header_checksum:u64
//! shared    := <the ADR-0011 header fields, in ADR-0011's order>, except
//!              that asset_count:u16 is followed by asset_count entries
//!              (ADR-0077 item 6): name_len:u8 name:[u8] hash:[u8; 16],
//!              names 1 to 64 bytes, at most 64 entries, compared with the
//!              pool's assets by name (`FileHeader::check_against`); a host
//!              file holds none
//! extension := config_len:u32 config:[u8]
//!              agents:u32 discrete_heads:u32 continuous_heads:u32
//!              flags:u8 host_protocol:u8 notifies:u8 refuses:u8
//!              [identity:u64]                ; flags bit3 (host snapshots)
//!              policy_map                    ; ADR-0079 item 6
//!              has_curriculum:u8             ; ADR-0081 item 7, 0 or 1
//!              has_league:u8                 ; ADR-0088 item 8, 0 or 1
//!              archive:reset_source          ; ADR-0075, at recording start
//!              chunk_size:u32                ; 0: not recorded, no check
//! policy_map:= slot_count:u16 group:u8*slot_count
//!              ; slot_count 0 (default map) or the agent count; groups 0
//!              ; learner, 1 external, 2 heuristic, 3 privileged heuristic,
//!              ; 4 opponent (ADR-0088)
//! reset_source := 0:u8                         ; Fresh
//!              | 1:u8 digest:u64 count:u32 weight:u32*count
//!              ; digest: XXH3-64 archive digest (ADR-0075 item 1);
//!              ; count 1 to 2^24; the weights sum to more than 0
//! header_checksum := XXH3-64 of every header byte before it
//! event     := tag:u8 len:u32 body:[u8; len]
//! end       := 0:u8 16:u32 event_count:u64 events_checksum:u64
//! ids       := 0:u8                           ; every instance
//!            | 1:u8 count:u32 id:u32*count
//! ```
//!
//! All integers are little-endian; floats are stored as their bits.
//!
//! Any addition after the first release bumps the replay version; the fields
//! above were folded into version 1 before any release shipped replay files.
//!
//! Flags: bit0 `from_reset_only`, bit1 `outputs_hash`, bit2
//! `config_checked`, bit3 `host_snapshots`, bit4 `async`. A first-party
//! file has none, or bit4 with bit1 optional; a host file has exactly one
//! of bit0 and bit3, bit1 with bit0, bit2 exactly with bit3, never bit4.
//!
//! Event bodies:
//!
//! - `Reset` (1): ids, then `0` Keep, `1` Base `seed:u64`, `2` List
//!   `count:u32 seed:u64*count`, or `3` Each `count:u32 (some:u8
//!   seed:u64)*count`, with a `None` written as `0` and seed `0`; then
//!   `outputs_hash:u64` in a host file with bit1.
//! - `Step` (2): ids, `count:u32 discrete:i32*count`, `count:u32
//!   continuous_bits:u32*count`; then `outputs_hash:u64` in a host file with
//!   bit1; then, for a refusing host, `count:u32 (instance:u32 slot:u32
//!   code:u32)*count`, sparse, in (instance, slot) order, codes nonzero.
//! - `Rejected` (3, notifying hosts only): `count:u32 discrete:i32*count
//!   count:u32 continuous_bits:u32*count instance:u32 code:u32`.
//! - `StateWrite` (4): `op:u8` (0 `restore`, 1 `restore_all`, 2 `import`,
//!   3 `fork`, 4 `fork_branched`, 5 `import_all`, 6 `set_lineage`; 1 and 5
//!   write every instance), the written ids, `branch_counter:u64` after
//!   the call. A `Keyframe` of those ids follows.
//! - `Keyframe` (5): `branch_counter:u64`, `reset_source` (the pool
//!   context's, ADR-0075 item 7), ids, then one record per id: a canonical
//!   record of `record_len` bytes (ADR-0029), or in a host file
//!   `len:u32 HostRecord` (ADR-0073, identity digest first); then, in a
//!   `has_league` file only, `assignment:u32` per id (ADR-0088 erratum 3).
//! - `Checkpoint` (6): ids, one `StateHash:u64` per id.
//! - `Adaptive` (7, ADR-0081 item 7, only with `has_curriculum` 1):
//!   `count:u32` (at least 1), then per changed instance `id:u32` and the
//!   canonical adaptive values (`ENCODED_LEN` of the header's `Adaptive`),
//!   ids strictly increasing and in range.
//! - `Recv` (8, async files only): a list of ids in token order, then
//!   `outputs_hash:u64` with bit1. A `recv` that reported a panicked step is
//!   a `Recv` with no ids, then `Poisoned`.
//! - `Poisoned` (9): empty.
//! - `ResetSource` (10, ADR-0075 item 7, first-party files only):
//!   `reset_source`.
//! - `ResetFrom` (11, ADR-0075 items 5 and 7, first-party files only):
//!   `count:u32` (1 to the instance count), `count` distinct ids in range,
//!   then `count` entry indices below 2^24.
//! - `League` (12, ADR-0088 section 8 with erratum 3, only with
//!   `has_league` 1): the league's whole view, `digest:u64 last:u32
//!   count:u32 kind*count threshold:u64*count`, a kind being `0:u8 tag:u64`
//!   (External), `1:u8` (Heuristic) or `2:u8 content_hash:[u8; 16]` (Onnx),
//!   count 1 to 2^16 (`replay::league`). A `has_league` file's first event
//!   is a `League` event with the view at recording start and its second
//!   the first keyframe; another `League` event follows each barrier that
//!   changed the view.
//! - `End` (0): the trailer of ADR-0072 section 2.
//!
//! Until their features land, this library writes the defaults (no
//! curriculum, a `Fresh` source) and never writes tags 7, 10 or 11; replay
//! refuses a non-default value or one of those tags as
//! `ReplayError::Unsupported`. A league pool's sync recording writes
//! `has_league` 1, code 4 and tag 12 (#384 head D); a file without a league
//! is byte-identical to one written before. Asset entries are written and checked
//! as ADR-0077 says. The policy map is written (count 0 for the default map,
//! else one code per agent slot) and checked against the pool before any
//! event (ADR-0079 item 6, #347 stage B2).
//!
//! # Async files
//!
//! Keyframes and checkpoints sit at quiescent points (no id outstanding),
//! except the keyframe after a state write, which holds the records written.
//! The final checkpoint is "last when quiescent": none is written if the
//! recording ends with ids outstanding (an erratum to ADR-0072 section 4).
//! The `Recv` outputs hash covers, for each id in order, its observation
//! leaves, rewards as `f32` bits, and agent termination and truncation flags
//! as one byte each. The branch counter is always 0.
//!
//! # Host pool files
//!
//! Host protocol 1, or 2 exactly for a refusing host (ADR-0070 item 10).
//! Shared header fields: tier 2 (`HOST_TIER`, ADR-0073) with an empty
//! descriptor, `record_len` 0, the mask provider from
//! `HostEnvironment::MASKS`, and the pool options as for any pool. With
//! snapshots (bit3): the canonical config and its hash are filled and the
//! identity digest follows the host fields; the branch counter is always 0.
//! From reset only (bit0): the config is empty and `config_hash` 0; the first
//! event is a `Reset` of every instance; there are no `Keyframe`,
//! `Checkpoint` or `StateWrite` events. A poisoning call is logged with
//! outputs hash 0 and no refusal rows. The outputs hash is defined in
//! `outputs_hash.rs`.

/// The first bytes of every replay file.
pub const REPLAY_MAGIC: [u8; 8] = *b"TESRPLY\x1a";
/// The replay format version: the ADR-0011 header's version, shared by every
/// file kind (ADR-0072 section 6).
pub const FORMAT_VERSION: u32 = crate::persist::FORMAT_VERSION;

/// Largest canonical config accepted in a header.
pub(crate) const MAX_CONFIG_BYTES: usize = 1 << 20;
/// Bytes of an event's tag and length.
pub(crate) const FRAME_LEN: usize = 5;
/// Bytes of the end event's body.
pub(crate) const END_BODY_LEN: u32 = 16;

/// Header flag: the file starts from a `Reset`, not a `Keyframe`.
pub(crate) const FLAG_FROM_RESET_ONLY: u8 = 1;
/// Header flag: `Reset`, `Step` and `Recv` events carry an outputs hash.
pub(crate) const FLAG_OUTPUTS_HASH: u8 = 2;
/// Header flag (host files): the config bytes and `config_hash` are filled
/// and checked; without it both are empty and zero.
pub(crate) const FLAG_CONFIG_CHECKED: u8 = 4;
/// Header flag (host files): the host has snapshots (ADR-0073); the host
/// extension ends with the identity digest, and the file has keyframes and
/// checkpoints of host records.
pub(crate) const FLAG_HOST_SNAPSHOTS: u8 = 8;
/// Header flag (first-party files): recorded on an `AsyncPool`; the file
/// may hold `Recv` events (tag 8), and the outputs hash (bit1) is allowed.
pub(crate) const FLAG_ASYNC: u8 = 16;

/// Ids marker: every instance.
pub(crate) const IDS_ALL: u8 = 0;
/// Ids marker: an explicit list.
pub(crate) const IDS_LIST: u8 = 1;

/// The tag of an event (ADR-0072 section 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EventTag {
    /// The trailer: event count and checksum.
    End = 0,
    /// A `reset` or `reset_all` call.
    Reset = 1,
    /// A `step` call.
    Step = 2,
    /// A rejected call of a notifying host (ADR-0069 item 8); host files
    /// only.
    Rejected = 3,
    /// A `restore`, `restore_all`, `import`, `fork` or `fork_branched` call.
    StateWrite = 4,
    /// Records of some or all instances.
    Keyframe = 5,
    /// `StateHash`es of some or all instances.
    Checkpoint = 6,
    /// New adaptive targets at an epoch barrier (ADR-0081 item 7); only in
    /// a file whose header has a curriculum.
    Adaptive = 7,
    /// An Async `recv`; only in an async file.
    Recv = 8,
    /// The preceding call poisoned the pool.
    Poisoned = 9,
    /// A `set_reset_source` call (ADR-0075 item 7).
    ResetSource = 10,
    /// A `reset_from` call (ADR-0075 items 5 and 7).
    ResetFrom = 11,
    /// League changes at an epoch barrier (ADR-0088 item 8); only in a file
    /// whose header has a league.
    League = 12,
}

impl EventTag {
    /// The tag of `byte`, or `None` for an unknown tag.
    #[must_use]
    pub const fn from_u8(byte: u8) -> Option<Self> {
        Some(match byte {
            0 => Self::End,
            1 => Self::Reset,
            2 => Self::Step,
            3 => Self::Rejected,
            4 => Self::StateWrite,
            5 => Self::Keyframe,
            6 => Self::Checkpoint,
            7 => Self::Adaptive,
            8 => Self::Recv,
            9 => Self::Poisoned,
            10 => Self::ResetSource,
            11 => Self::ResetFrom,
            12 => Self::League,
            _ => return None,
        })
    }

    /// Whether the tag is allowed only in some files: `Adaptive` with a
    /// curriculum, `League` with a league, `Recv` in an async file,
    /// `Rejected` in a notifying host's file. Elsewhere each is refused as
    /// reserved.
    #[must_use]
    pub const fn is_reserved(self) -> bool {
        matches!(
            self,
            Self::Rejected | Self::Adaptive | Self::League | Self::Recv
        )
    }
}

/// The pool call behind a `StateWrite` event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StateWriteOp {
    /// `restore`.
    Restore = 0,
    /// `restore_all`.
    RestoreAll = 1,
    /// `import`.
    Import = 2,
    /// `fork`.
    Fork = 3,
    /// `fork_branched`.
    ForkBranched = 4,
    /// `import_all` (#389): every instance, like `restore_all`.
    ImportAll = 5,
    /// `set_lineage` (#389): one instance's RNG identity; the keyframe holds
    /// its record with the new lineage.
    SetLineage = 6,
}

impl StateWriteOp {
    /// Whether the operation writes every instance (its ids must be all).
    #[must_use]
    pub const fn writes_all(self) -> bool {
        matches!(self, Self::RestoreAll | Self::ImportAll)
    }
}

impl StateWriteOp {
    /// The operation of `byte`, or `None`.
    #[must_use]
    pub const fn from_u8(byte: u8) -> Option<Self> {
        Some(match byte {
            0 => Self::Restore,
            1 => Self::RestoreAll,
            2 => Self::Import,
            3 => Self::Fork,
            4 => Self::ForkBranched,
            5 => Self::ImportAll,
            6 => Self::SetLineage,
            _ => return None,
        })
    }
}

/// Appends the ids marker and, for a list, the ids.
pub(crate) fn push_ids(out: &mut Vec<u8>, ids: Option<&[u32]>) -> Result<(), super::ReplayError> {
    match ids {
        None => out.push(IDS_ALL),
        Some(ids) => {
            out.push(IDS_LIST);
            push_len(out, ids.len())?;
            for id in ids {
                out.extend_from_slice(&id.to_le_bytes());
            }
        }
    }
    Ok(())
}

/// Appends a `u32` length.
pub(crate) fn push_len(out: &mut Vec<u8>, len: usize) -> Result<(), super::ReplayError> {
    let len = u32::try_from(len).map_err(|_| super::ReplayError::Unsupported {
        feature: "more than u32::MAX items in one event",
    })?;
    out.extend_from_slice(&len.to_le_bytes());
    Ok(())
}

/// A cursor over an event body; every read is bounds-checked.
pub(crate) struct Body<'a> {
    bytes: &'a [u8],
    event_index: u64,
}

impl<'a> Body<'a> {
    pub(crate) const fn new(bytes: &'a [u8], event_index: u64) -> Self {
        Self { bytes, event_index }
    }

    pub(crate) const fn malformed(&self, field: &'static str) -> super::ReplayError {
        super::ReplayError::Malformed {
            event_index: self.event_index,
            field,
        }
    }

    pub(crate) fn take(
        &mut self,
        n: usize,
        field: &'static str,
    ) -> Result<&'a [u8], super::ReplayError> {
        let (head, tail) = self
            .bytes
            .split_at_checked(n)
            .ok_or_else(|| self.malformed(field))?;
        self.bytes = tail;
        Ok(head)
    }

    fn array<const N: usize>(
        &mut self,
        field: &'static str,
    ) -> Result<[u8; N], super::ReplayError> {
        let bytes = self.take(N, field)?;
        bytes.try_into().map_err(|_| self.malformed(field))
    }

    pub(crate) fn u8(&mut self, field: &'static str) -> Result<u8, super::ReplayError> {
        Ok(u8::from_le_bytes(self.array(field)?))
    }

    pub(crate) fn u16(&mut self, field: &'static str) -> Result<u16, super::ReplayError> {
        Ok(u16::from_le_bytes(self.array(field)?))
    }

    pub(crate) fn u32(&mut self, field: &'static str) -> Result<u32, super::ReplayError> {
        Ok(u32::from_le_bytes(self.array(field)?))
    }

    pub(crate) fn u64(&mut self, field: &'static str) -> Result<u64, super::ReplayError> {
        Ok(u64::from_le_bytes(self.array(field)?))
    }

    /// A `u32` count of items of `width` bytes, checked against the bytes
    /// left before anything is allocated.
    pub(crate) fn count(
        &mut self,
        width: usize,
        field: &'static str,
    ) -> Result<usize, super::ReplayError> {
        let n = usize::try_from(self.u32(field)?).map_err(|_| self.malformed(field))?;
        match n.checked_mul(width) {
            Some(bytes) if bytes <= self.bytes.len() => Ok(n),
            _ => Err(self.malformed(field)),
        }
    }

    pub(crate) const fn remaining(&self) -> usize {
        self.bytes.len()
    }

    pub(crate) fn finish(self) -> Result<(), super::ReplayError> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            Err(self.malformed("trailing bytes in the event body"))
        }
    }
}
