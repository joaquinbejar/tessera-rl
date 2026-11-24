// SPDX-License-Identifier: MIT OR Apache-2.0

//! The byte format of replay files, version 1 (ADR-0072 sections 2 and 3).
//!
//! ```text
//! file    := header event* end
//! header  := "TESRPLY\x1a" version:u32 <ADR-0011 header fields>
//!            config_len:u32 config:[u8]
//!            agents:u32 discrete_heads:u32 continuous_heads:u32
//!            flags:u8 host_protocol:u8 notifies:u8 refuses:u8
//!            header_checksum:u64           ; XXH3-64 of every byte before it
//! event   := tag:u8 len:u32 body:[u8; len]
//! end     := 0:u8 16:u32 event_count:u64 events_checksum:u64
//! ids     := 0:u8                           ; every instance
//!          | 1:u8 count:u32 id:u32*count
//! ```
//!
//! All integers are little-endian; floats are stored as their bits. Event
//! bodies:
//!
//! - `Reset`: ids, then `0` Keep, `1` Base `seed:u64`, `2` List `count:u32
//!   seed:u64*count`, or `3` Each `count:u32 (some:u8 seed:u64)*count`, with
//!   a `None` written as `0` and seed `0`.
//! - `Step`: ids, `count:u32 discrete:i32*count`, `count:u32
//!   continuous_bits:u32*count`.
//! - `StateWrite`: `op:u8`, the written ids, `branch_counter:u64` after the
//!   call. A `Keyframe` of those ids follows.
//! - `Keyframe`: `branch_counter:u64`, ids, one canonical record (ADR-0029,
//!   `record_len` bytes) per id.
//! - `Checkpoint`: ids, one `StateHash:u64` per id.
//! - `Poisoned`: empty.
//! - `End` (tag 0): the trailer of ADR-0072 section 2.
//!
//! # Async files
//!
//! A first-party file recorded on an `AsyncPool` has flag bit4 (`async`),
//! and bit1 (`outputs_hash`) optionally. `Step` events hold the sent ids
//! (a list); `Recv` (tag 8) holds the ids one `recv` returned, as a list in
//! token order, then `outputs_hash:u64` when bit1 is set (XXH3-64, for each
//! id in order, of its observation leaves, rewards as `f32` bits, and agent
//! termination and truncation flags as one byte each). A `recv` that
//! reported a panicked step is a `Recv` with no ids, then `Poisoned`.
//! Keyframes and checkpoints sit at quiescent points (no id outstanding),
//! except the keyframe after a state write, which holds the records written.
//! The branch counter is always 0. `Recv` is refused in any other file, and
//! an async file is refused by every replay but `run_async`.
//!
//!
//! A host pool's file has `host_protocol` 1, or 2 exactly for a refusing
//! host (ADR-0070 item 10). Its shared header fields: tier 2
//! (`HOST_TIER`, ADR-0073) with an empty descriptor, `record_len` 0 (host
//! records have variable length), the mask provider from
//! `HostEnvironment::MASKS`, and the pool options as for any pool. Flags:
//! bit0 `from_reset_only`, bit1 `outputs_hash`, bit2 `config_checked`,
//! bit3 `host_snapshots`; exactly one of bit0 and bit3 is set.
//!
//! - **With snapshots** (bit3, ADR-0073): bit2 is set, the config bytes and
//!   `config_hash` are those of the canonical config, and the host
//!   extension ends with `identity:u64` (the ADR-0073 identity digest),
//!   before the header checksum. Events as for a first-party file, except
//!   that a `Keyframe`'s records are `(len:u32 HostRecord)*`, each the
//!   ADR-0073 encoding (identity digest first), and the branch counter is
//!   always 0 (host pools have no `fork_branched`). The outputs hash is
//!   optional (bit1).
//! - **From reset only** (bit0): bit1 is set, bit2 is clear, the config is
//!   empty and `config_hash` 0 (a host config need not be canonical). The
//!   first event is a `Reset` of every instance, made by the recorder; there
//!   are no `Keyframe`, `Checkpoint` or `StateWrite` events.
//!
//! Bodies, when bit1 is set:
//!
//! - `Reset`: as above, then `outputs_hash:u64` over the reset ids.
//! - `Step`: as above, then `outputs_hash:u64`.
//!
//! For a refusing host, `Step` then ends with `count:u32 (instance:u32
//! slot:u32 code:u32)*count`, sparse, in (instance, slot) order, every code
//! nonzero. `Rejected` (notifying hosts only): `count:u32
//! discrete:i32*count count:u32 continuous_bits:u32*count instance:u32
//! code:u32`.
//!
//! A poisoning call is logged with outputs hash 0 and no refusal rows, as
//! a poisoned pool has no outputs. The outputs hash is defined in
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
    /// Reserved: an adaptive-config event (SPEC 8.4).
    Adaptive = 7,
    /// Reserved: an Async `recv`.
    Recv = 8,
    /// The preceding call poisoned the pool.
    Poisoned = 9,
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
            _ => return None,
        })
    }

    /// Whether the tag is reserved: `Adaptive` always, `Recv` outside an
    /// async file, and
    /// `Rejected` outside a notifying host's file.
    #[must_use]
    pub const fn is_reserved(self) -> bool {
        matches!(self, Self::Rejected | Self::Adaptive | Self::Recv)
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
