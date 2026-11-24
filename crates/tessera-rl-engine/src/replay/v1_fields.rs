// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fields that accepted ADRs folded into replay format version 1 before any
//! release shipped replay files (#411): the policy map (ADR-0079 item 6), the
//! curriculum flag and `Adaptive` body (ADR-0081 item 7), the reset source
//! and `ResetSource` / `ResetFrom` bodies (ADR-0075 item 7), and the chunk
//! size. They are parsed and validated now; until their features land, every
//! file this library writes holds their defaults, and replay refuses a
//! non-default value as `ReplayError::Unsupported`.

use super::ReplayError;
use super::format::Body;
use crate::persist::PersistError;

/// Most archive entries (ADR-0075 item 1): the engine's
/// [`crate::MAX_ARCHIVE_ENTRIES`] as the `u32` the format stores.
pub const MAX_ARCHIVE_ENTRIES: u32 = {
    assert!(crate::MAX_ARCHIVE_ENTRIES <= u32::MAX as usize);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "checked to fit by the assertion above"
    )]
    let max = crate::MAX_ARCHIVE_ENTRIES as u32;
    max
};
/// Policy group codes (ADR-0079 item 6): learner, external, heuristic,
/// privileged heuristic.
pub const MAX_POLICY_GROUP: u8 = 3;

/// A pool's reset source (ADR-0075 item 7), as a `Keyframe`'s pool context
/// and a `ResetSource` event hold it: the engine's
/// [`crate::ResetSourceContext`], one type for the pool context and the
/// file (#363, agreed with #411).
pub use crate::archive::ResetSourceContext as ResetSourceRecord;

/// The archive block of the header: the source at recording start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchiveHeader<'a> {
    /// The archive digest.
    pub digest: u64,
    /// The weights, `u32` little-endian, one per entry.
    weights: &'a [u8],
}

impl ArchiveHeader<'_> {
    /// The weights.
    pub fn weights(&self) -> impl Iterator<Item = u32> + '_ {
        self.weights.as_chunks::<4>().0.iter().copied().map(word)
    }
}

/// The version 1 header fields after the host fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Extension<'a> {
    pub(crate) policy_map: &'a [u8],
    pub(crate) has_curriculum: bool,
    pub(crate) archive: Option<ArchiveHeader<'a>>,
    pub(crate) chunk_size: u32,
}

/// A little-endian `u32` from a 4-byte chunk.
/// The `(instance id, adaptive values)` entries of a tag 7 body.
type Entries<'a> = Vec<(u32, &'a [u8])>;

fn word(w: [u8; 4]) -> u32 {
    u32::from_le_bytes(w)
}

/// Reads the archive weights after their `u32` count: 1 to 2^24, summing
/// to a positive total. Returns the raw bytes.
fn weights<'a>(b: &mut Body<'a>) -> Result<&'a [u8], ReplayError> {
    let n = b.u32("archive weights")?;
    if n == 0 || n > MAX_ARCHIVE_ENTRIES {
        return Err(b.malformed("archive weights"));
    }
    let len = usize::try_from(n)
        .ok()
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| b.malformed("archive weights"))?;
    let bytes = b.take(len, "archive weights")?;
    let total: u64 = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u64::from(word(*w)))
        .sum();
    if total == 0 {
        return Err(b.malformed("archive weights"));
    }
    Ok(bytes)
}

/// Reads a reset source: `0` Fresh, or `1` then `digest:u64 count:u32
/// weight:u32*count`.
pub(crate) fn read_reset_source(b: &mut Body<'_>) -> Result<ResetSourceRecord, ReplayError> {
    match b.u8("reset source")? {
        0 => Ok(ResetSourceRecord::Fresh),
        1 => {
            let digest = b.u64("archive digest")?;
            let bytes = weights(b)?;
            let mut out = Vec::new();
            out.try_reserve_exact(bytes.len() / 4)
                .map_err(|_| ReplayError::Allocation)?;
            out.extend(bytes.as_chunks::<4>().0.iter().copied().map(word));
            Ok(ResetSourceRecord::Archive {
                digest,
                weights: out,
            })
        }
        _ => Err(b.malformed("reset source")),
    }
}

/// Reads the header extension from `rest` (the bytes after the host fields
/// and identity); returns it and the bytes it used. A field error is
/// `PersistError::InvalidField` naming the field.
pub(crate) fn read_extension(
    rest: &[u8],
    agents: u32,
) -> Result<(Extension<'_>, usize), ReplayError> {
    let invalid = |e: ReplayError| -> ReplayError {
        match e {
            ReplayError::Malformed { field, .. } => PersistError::InvalidField { field }.into(),
            e => e,
        }
    };
    let mut b = Body::new(rest, 0);
    let slots = b.u16("policy map").map_err(invalid)?;
    if slots != 0 && u32::from(slots) != agents {
        return Err(PersistError::InvalidField {
            field: "policy map",
        }
        .into());
    }
    let policy_map = b.take(usize::from(slots), "policy map").map_err(invalid)?;
    if policy_map.iter().any(|&g| g > MAX_POLICY_GROUP) {
        return Err(PersistError::InvalidField {
            field: "policy map group",
        }
        .into());
    }
    let has_curriculum = match b.u8("has_curriculum").map_err(invalid)? {
        0 => false,
        1 => true,
        _ => {
            return Err(PersistError::InvalidField {
                field: "has_curriculum",
            }
            .into());
        }
    };
    let archive = match b.u8("archive digest_present").map_err(invalid)? {
        0 => None,
        1 => {
            let digest = b.u64("archive digest").map_err(invalid)?;
            let weights = weights(&mut b).map_err(invalid)?;
            Some(ArchiveHeader { digest, weights })
        }
        _ => {
            return Err(PersistError::InvalidField {
                field: "archive digest_present",
            }
            .into());
        }
    };
    let chunk_size = b.u32("chunk_size").map_err(invalid)?;
    let used = rest.len().saturating_sub(b.remaining());
    Ok((
        Extension {
            policy_map,
            has_curriculum,
            archive,
            chunk_size,
        },
        used,
    ))
}

/// Appends the default extension: default policy map, no curriculum, no
/// archive, then `chunk_size`.
pub(crate) fn push_default_extension(out: &mut Vec<u8>, chunk_size: u32) {
    push_extension(out, chunk_size, &ResetSourceRecord::Fresh);
}

/// Appends the extension: default policy map, no curriculum, the archive
/// block of `source` (the source at recording start, ADR-0075 item 7), then
/// `chunk_size`.
pub(crate) fn push_extension(out: &mut Vec<u8>, chunk_size: u32, source: &ResetSourceRecord) {
    out.extend_from_slice(&0u16.to_le_bytes());
    out.push(0);
    match source {
        ResetSourceRecord::Fresh => out.push(0),
        ResetSourceRecord::Archive { digest, weights } => {
            out.push(1);
            out.extend_from_slice(&digest.to_le_bytes());
            push_weights(out, weights);
        }
    }
    out.extend_from_slice(&chunk_size.to_le_bytes());
}

/// Appends a reset source as `read_reset_source` reads it.
pub(crate) fn push_reset_source(out: &mut Vec<u8>, source: &ResetSourceRecord) {
    match source {
        ResetSourceRecord::Fresh => out.push(0),
        ResetSourceRecord::Archive { digest, weights } => {
            out.push(1);
            out.extend_from_slice(&digest.to_le_bytes());
            push_weights(out, weights);
        }
    }
}

/// Appends `count:u32` then the weights; the engine bounds the count to
/// 2^24 (`Archive::new`), so it fits.
fn push_weights(out: &mut Vec<u8>, weights: &[u32]) {
    let n = u32::try_from(weights.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&n.to_le_bytes());
    for w in weights {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

/// Appends a `ResetFrom` body as `read_reset_from` reads it.
pub(crate) fn push_reset_from(out: &mut Vec<u8>, ids: &[u32], entries: &[u32]) {
    let n = u32::try_from(ids.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&n.to_le_bytes());
    for id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    for e in entries {
        out.extend_from_slice(&e.to_le_bytes());
    }
}

impl ArchiveHeader<'_> {
    /// The owned reset source this header block names.
    #[must_use]
    pub fn to_record(&self) -> ResetSourceRecord {
        ResetSourceRecord::Archive {
            digest: self.digest,
            weights: self.weights().collect(),
        }
    }
}

/// The `Adaptive` body (ADR-0081 item 7): `count:u32` (at least 1), then per
/// changed instance `id:u32` and `value_len` bytes of canonical adaptive
/// values, ids strictly increasing and below `num_envs`. The value length is
/// the environment's, which the reader does not know: it is derived from the
/// body length and checked against the pool at replay.
pub(crate) fn read_adaptive<'a>(
    b: &mut Body<'a>,
    num_envs: u32,
) -> Result<(usize, Entries<'a>), ReplayError> {
    let count =
        usize::try_from(b.u32("adaptive count")?).map_err(|_| b.malformed("adaptive count"))?;
    let rest = b.remaining();
    if count == 0 || !rest.is_multiple_of(count) || rest / count <= 4 {
        return Err(b.malformed("adaptive body length"));
    }
    let value_len = rest / count - 4;
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| ReplayError::Allocation)?;
    for _ in 0..count {
        let id = b.u32("adaptive ids")?;
        if id >= num_envs || out.last().is_some_and(|&(p, _)| p >= id) {
            return Err(b.malformed("adaptive ids"));
        }
        out.push((id, b.take(value_len, "adaptive values")?));
    }
    Ok((value_len, out))
}

/// The `ResetFrom` body (ADR-0075 item 5): `count:u32` (at least 1), `count`
/// distinct ids below `num_envs`, then `count` entry indices below 2^24.
pub(crate) fn read_reset_from(
    b: &mut Body<'_>,
    num_envs: u32,
) -> Result<(Vec<u32>, Vec<u32>), ReplayError> {
    let count = b.count(8, "reset_from count")?;
    if count == 0 || u64::try_from(count).unwrap_or(u64::MAX) > u64::from(num_envs) {
        return Err(b.malformed("reset_from count"));
    }
    let mut seen = Vec::new();
    seen.try_reserve_exact(usize::try_from(num_envs).unwrap_or(0))
        .map_err(|_| ReplayError::Allocation)?;
    seen.resize(usize::try_from(num_envs).unwrap_or(0), false);
    let mut ids = Vec::new();
    ids.try_reserve_exact(count)
        .map_err(|_| ReplayError::Allocation)?;
    for _ in 0..count {
        let id = b.u32("reset_from ids")?;
        let slot = usize::try_from(id).ok().and_then(|i| seen.get_mut(i));
        match slot {
            Some(s) if !*s => *s = true,
            _ => return Err(b.malformed("reset_from ids")),
        }
        ids.push(id);
    }
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(count)
        .map_err(|_| ReplayError::Allocation)?;
    for _ in 0..count {
        let entry = b.u32("reset_from entries")?;
        if entry >= MAX_ARCHIVE_ENTRIES {
            return Err(b.malformed("reset_from entries"));
        }
        entries.push(entry);
    }
    Ok((ids, entries))
}

impl super::reader::Replay<'_> {
    /// Checks the version 1 fields whose features are not implemented yet
    /// against a pool (asset entries are compared by `check_against`,
    /// ADR-0077): a non-default policy map, a curriculum or an archive source (`Unsupported` until
    /// ADR-0079, ADR-0081 and ADR-0075 land), and the chunk size, which must
    /// equal the pool's when the header records one (non-zero).
    pub(super) fn check_v1(&self, chunk_size: u32) -> Result<(), ReplayError> {
        if self.header.policy_map.iter().any(|&g| g != 0) {
            return Err(ReplayError::Unsupported {
                feature: "policy maps (ADR-0079)",
            });
        }
        if self.header.has_curriculum {
            return Err(ReplayError::Unsupported {
                feature: "curricula (ADR-0081)",
            });
        }
        if self.header.chunk_size != 0 && self.header.chunk_size != chunk_size {
            return Err(PersistError::Incompatible {
                field: "chunk_size",
            }
            .into());
        }
        Ok(())
    }
}

/// Refuses a keyframe whose pool context names an archive (ADR-0075, not
/// implemented yet).
pub(super) fn fresh_only(source: &ResetSourceRecord) -> Result<(), ReplayError> {
    match source {
        ResetSourceRecord::Fresh => Ok(()),
        ResetSourceRecord::Archive { .. } => Err(ReplayError::Unsupported {
            feature: "archive reset sources (ADR-0075)",
        }),
    }
}

/// The refusal of an event whose feature is not implemented yet.
pub(super) const fn not_yet(tag: super::format::EventTag) -> ReplayError {
    use super::format::EventTag;
    ReplayError::Unsupported {
        feature: match tag {
            EventTag::Adaptive => "adaptive events (ADR-0081)",
            EventTag::ResetSource | EventTag::ResetFrom => "archive resets (ADR-0075)",
            _ => "this event",
        },
    }
}
