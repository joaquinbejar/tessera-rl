// SPDX-License-Identifier: MIT OR Apache-2.0

//! `HostRecord`: the continuation record of one host instance (ADR-0073
//! item 3).
//!
//! An identity digest, the engine's host header in canonical encoding, the
//! environment's `SNAPSHOT_VERSION` and its bytes. A record is never
//! trusted: every restore checks it (`check_host_record`), so a record
//! from bytes and one from `snapshot` take the same path.

use tessera_rl_core::encoding::{Canonical, StateHasher, decode_from_slice, encode_to_vec};
use tessera_rl_core::{DecodeError, RecordLayout, StateHash};

use super::kernels::HostHeader;

/// The domain tag hashed first in every host `StateHash`, so a host hash
/// never equals an `EnvPool` record's hash of the same bytes.
const HOST_HASH_DOMAIN: &[u8; 8] = b"TESHOST\x00";

/// The continuation record of one host instance (ADR-0073).
#[derive(Clone)]
pub struct HostRecord<const A: usize> {
    pub(crate) identity: u64,
    pub(crate) header: HostHeader<A>,
    pub(crate) version: u32,
    pub(crate) bytes: Vec<u8>,
}

impl<const A: usize> core::fmt::Debug for HostRecord<A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostRecord")
            .field("identity", &self.identity)
            .field("version", &self.version)
            .field("len", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

/// Fixed bytes of an encoded record besides the header: the identity
/// digest, the version and the length of the author bytes.
const FIXED: usize = 8 + 4 + 4;

impl<const A: usize> HostRecord<A> {
    /// An empty record, for `HostPool::snapshot_all` to fill.
    #[must_use]
    pub fn new() -> Self {
        Self {
            identity: 0,
            header: tessera_rl_core::StateInit::state_default(),
            version: 0,
            bytes: Vec::new(),
        }
    }

    /// The identity digest: XXH3-64 of the environment id, `ENV_VERSION`,
    /// `SNAPSHOT_VERSION` and the config hash.
    #[must_use]
    pub const fn identity(&self) -> u64 {
        self.identity
    }

    /// The environment's `SNAPSHOT_VERSION` that wrote the bytes.
    #[must_use]
    pub const fn snapshot_version(&self) -> u32 {
        self.version
    }

    /// The environment's bytes. Privileged: they hold full state
    /// (ADR-0035).
    #[must_use]
    pub fn state_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Whether the instance is pending a reset.
    #[must_use]
    pub fn pending_reset(&self) -> bool {
        self.header.engine().pending_reset.get()
    }

    /// The host `StateHash` (ADR-0073 item 4): a host domain tag, the
    /// header's canonical encoding, the version, the length and the bytes.
    /// It carries no tier and may differ across platforms.
    #[must_use]
    pub fn hash(&self) -> StateHash {
        let mut hasher = StateHasher::new();
        hasher.add_bytes(HOST_HASH_DOMAIN);
        hasher.add(&self.header);
        hasher.add(&self.version);
        hasher.add(&u64::try_from(self.bytes.len()).unwrap_or(u64::MAX));
        hasher.add_bytes(&self.bytes);
        hasher.finish()
    }

    /// The record's encoding: identity (`u64`), header (canonical), version
    /// (`u32`), length (`u32`) and bytes, little-endian. For files,
    /// replays and Python; never on the step path. `None` when the bytes
    /// are longer than `u32::MAX`, which `HostPool::snapshot` never makes.
    #[must_use]
    pub fn to_bytes(&self) -> Option<Vec<u8>> {
        let len = u32::try_from(self.bytes.len()).ok()?;
        let header = encode_to_vec(&self.header);
        let mut out = Vec::with_capacity(
            FIXED
                .checked_add(header.len())?
                .checked_add(self.bytes.len())?,
        );
        out.extend_from_slice(&self.identity.to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&self.bytes);
        Some(out)
    }

    /// The encoded length of the header.
    #[must_use]
    pub const fn header_len() -> usize {
        <HostHeader<A> as Canonical>::ENCODED_LEN
    }

    /// Decodes one record from exactly `input`, refusing author bytes over
    /// `max_bytes` before allocating them. The record is untrusted until a
    /// pool checks it on restore.
    ///
    /// # Errors
    ///
    /// [`RecordDecodeError`] for a truncated or overlong input, author bytes
    /// over `max_bytes`, or an invalid header field.
    pub fn from_bytes(input: &[u8], max_bytes: usize) -> Result<Self, RecordDecodeError> {
        let (record, rest) = Self::read(input, max_bytes)?;
        if rest.is_empty() {
            Ok(record)
        } else {
            Err(RecordDecodeError::TrailingBytes)
        }
    }

    /// Reads one record from the front of `input`; returns it and the rest.
    pub(crate) fn read(input: &[u8], max_bytes: usize) -> Result<(Self, &[u8]), RecordDecodeError> {
        let (identity, rest) = split::<8>(input)?;
        let (header, rest) = rest
            .split_at_checked(Self::header_len())
            .ok_or(RecordDecodeError::Truncated)?;
        let header =
            decode_from_slice::<HostHeader<A>>(header).map_err(RecordDecodeError::Header)?;
        let (version, rest) = split::<4>(rest)?;
        let (len, rest) = split::<4>(rest)?;
        let len =
            usize::try_from(u32::from_le_bytes(len)).map_err(|_| RecordDecodeError::Truncated)?;
        if len > max_bytes {
            return Err(RecordDecodeError::TooLong {
                len,
                max: max_bytes,
            });
        }
        let (bytes, rest) = rest
            .split_at_checked(len)
            .ok_or(RecordDecodeError::Truncated)?;
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(len)
            .map_err(|_| RecordDecodeError::Allocation)?;
        owned.extend_from_slice(bytes);
        Ok((
            Self {
                identity: u64::from_le_bytes(identity),
                header,
                version: u32::from_le_bytes(version),
                bytes: owned,
            },
            rest,
        ))
    }
}

impl<const A: usize> Default for HostRecord<A> {
    fn default() -> Self {
        Self::new()
    }
}

fn split<const N: usize>(input: &[u8]) -> Result<([u8; N], &[u8]), RecordDecodeError> {
    let (head, rest) = input
        .split_at_checked(N)
        .ok_or(RecordDecodeError::Truncated)?;
    let head = head.try_into().map_err(|_| RecordDecodeError::Truncated)?;
    Ok((head, rest))
}

/// Why bytes are not a host record.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RecordDecodeError {
    /// The input ends inside the record.
    #[error("the host record is truncated")]
    Truncated,
    /// Bytes follow the record.
    #[error("bytes follow the host record")]
    TrailingBytes,
    /// The author bytes are longer than allowed.
    #[error("the host record holds {len} state bytes, over the limit of {max}")]
    TooLong {
        /// Their length.
        len: usize,
        /// The limit.
        max: usize,
    },
    /// A header field is invalid.
    #[error("host record header: {0}")]
    Header(DecodeError),
    /// The allocator refused.
    #[error("could not allocate the host record")]
    Allocation,
}
