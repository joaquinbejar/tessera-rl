// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host snapshot files (ADR-0073 item 5).
//!
//! A host file has its own magic, [`HOST_MAGIC`], followed by the ADR-0011
//! header fields in ADR-0011's order (shared by every file kind, ADR-0072):
//! the magic is the container kind, and the format version keeps its
//! ADR-0011 meaning, a newer layout of the same container.
//!
//! - **Header fields.** `payload_encoding` raw canonical, `tier` 2 (host: no
//!   tier, the only value a host file allows), `mask_provider` and
//!   `illegal_*` 0, `auto_reset`, `max_episode_steps`, `env_version`,
//!   `config_hash`, the instance count, `record_len` = the encoded length
//!   of the host header, the env id, the library version (diagnostic only)
//!   and no assets.
//! - **Descriptor.** In place of ADR-0011's platform descriptor (a host
//!   file promises nothing across platforms), the host identity:
//!   descriptor version, `SNAPSHOT_VERSION`, `MAX_AGENTS`,
//!   `initial_agents`, and the `SPAWNS` and `MASKS` flags. Every field is
//!   checked equal on decode.
//! - **Payload.** `count` [`HostRecord`] encodings, each refused over
//!   `max_snapshot_len` and `DecodeLimits::max_host_record_bytes` before
//!   its bytes are allocated; the decoded total counts toward
//!   `max_decoded_bytes`.
//! - **Privileged**, like every persisted file (ADR-0035).
//!
//! Decoding returns records; they enter a pool only through
//! [`HostPool::import`] or [`HostPool::restore_all`], which check them.

use tessera_rl_core::HostSnapshot;
use tessera_rl_core::config::{CanonicalConfig, config_hash};
use tessera_rl_core::encoding::checksum;

use super::HostPool;
use super::record::{HostRecord, RecordDecodeError};
use crate::options::AutoReset;
use crate::persist::{
    CHECKSUM_LEN, DecodeLimits, FORMAT_VERSION, FileHeader, MAX_DESCRIPTOR, MAX_ENV_ID,
    MAX_LIBRARY_VERSION, PersistError, RAW_CANONICAL, Reader, push_capped, same, u64_of,
};

/// The first bytes of every host snapshot file (ADR-0073): 8 bytes, as
/// ADR-0011's `MAGIC`.
pub const HOST_MAGIC: [u8; 8] = *b"TESHOST\x1a";

/// The `tier` field of a host file: host, no tier.
const HOST_TIER: u8 = 2;
/// The version of the host descriptor below.
const HOST_DESCRIPTOR_VERSION: u8 = 1;

/// The host identity a file must match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HostIdentity {
    auto_reset: u8,
    max_episode_steps: u32,
    env_version: u32,
    config_hash: u64,
    record_len: u32,
    descriptor: [u8; 14],
}

impl<E, const A: usize> HostPool<E, A>
where
    E: HostSnapshot,
    E::Config: CanonicalConfig,
{
    fn host_identity(&self) -> Result<HostIdentity, PersistError> {
        let record_len = u32::try_from(HostRecord::<A>::header_len()).map_err(|_| {
            PersistError::Incompatible {
                field: "record_len",
            }
        })?;
        let agents =
            u16::try_from(A).map_err(|_| PersistError::Incompatible { field: "agents" })?;
        let initial = u16::try_from(self.initial).map_err(|_| PersistError::Incompatible {
            field: "initial_agents",
        })?;
        let mut descriptor = [0u8; 14];
        let fields = [
            &[HOST_DESCRIPTOR_VERSION][..],
            &E::SNAPSHOT_VERSION.to_le_bytes(),
            &agents.to_le_bytes(),
            &initial.to_le_bytes(),
            &[u8::from(E::SPAWNS), u8::from(E::MASKS)],
            &[0, 0, 0],
        ];
        let mut at = 0;
        for field in fields {
            if let Some(dst) = descriptor.get_mut(at..at + field.len()) {
                dst.copy_from_slice(field);
            }
            at += field.len();
        }
        Ok(HostIdentity {
            auto_reset: match self.options.auto_reset {
                AutoReset::NextStep => 0,
                AutoReset::SameStep => 1,
                AutoReset::Disabled => 2,
            },
            max_episode_steps: self
                .options
                .max_episode_steps
                .map_or(0, core::num::NonZeroU32::get),
            env_version: E::ENV_VERSION,
            config_hash: config_hash(E::ENV_ID, E::ENV_VERSION, &*self.cfg)
                .map_err(|e| PersistError::Engine(e.into()))?,
            record_len,
            descriptor,
        })
    }

    /// The host file holding the records of `ids`, in that order
    /// (ADR-0073 item 5).
    ///
    /// # Errors
    ///
    /// A [`PersistError`] on a poisoned pool or an unknown id, or the
    /// errors of [`HostPool::snapshot`].
    pub fn encode(&mut self, ids: &[u32]) -> Result<Vec<u8>, PersistError> {
        self.encode_with_version(ids, env!("CARGO_PKG_VERSION"))
    }

    /// [`HostPool::encode`] with the library version as a parameter, for
    /// fixtures: the version is diagnostic only and never checked.
    ///
    /// # Errors
    ///
    /// As [`HostPool::encode`].
    #[cfg(feature = "test-hooks")]
    pub fn encode_with(
        &mut self,
        ids: &[u32],
        library_version: &str,
    ) -> Result<Vec<u8>, PersistError> {
        self.encode_with_version(ids, library_version)
    }

    fn encode_with_version(
        &mut self,
        ids: &[u32],
        library_version: &str,
    ) -> Result<Vec<u8>, PersistError> {
        if self.is_poisoned() {
            return Err(crate::error::EngineError::Poisoned.into());
        }
        let id = self.host_identity()?;
        let count = u32::try_from(ids.len()).map_err(|_| PersistError::InvalidField {
            field: "instance_count",
        })?;
        let mut out = Vec::new();
        out.extend_from_slice(&HOST_MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&[RAW_CANONICAL, HOST_TIER, 0, id.auto_reset, 0, 0, 0, 0]);
        out.extend_from_slice(&0u32.to_le_bytes()); // penalty bits
        out.extend_from_slice(&id.max_episode_steps.to_le_bytes());
        out.extend_from_slice(&id.env_version.to_le_bytes());
        out.extend_from_slice(&id.config_hash.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&id.record_len.to_le_bytes());
        push_capped(&mut out, "env_id", E::ENV_ID.as_bytes(), MAX_ENV_ID)?;
        push_capped(
            &mut out,
            "library_version",
            library_version.as_bytes(),
            MAX_LIBRARY_VERSION,
        )?;
        out.extend_from_slice(&0u16.to_le_bytes()); // asset_count
        push_capped(&mut out, "descriptor", &id.descriptor, MAX_DESCRIPTOR)?;
        let mut record = HostRecord::new();
        for &i in ids {
            self.snapshot_to(i, &mut record)?;
            let bytes = record.to_bytes().ok_or(PersistError::InvalidField {
                field: "state bytes",
            })?;
            out.extend_from_slice(&bytes);
        }
        let sum = checksum(&out);
        out.extend_from_slice(&sum.to_le_bytes());
        Ok(out)
    }

    /// Checks a host file against this pool and returns its records, still
    /// to be checked by [`HostPool::import`]. It reads only the pool's
    /// configuration and runs no environment code.
    ///
    /// # Errors
    ///
    /// A [`PersistError`] for anything but a sound, compatible host file
    /// within `limits`.
    pub fn decode(
        &self,
        bytes: &[u8],
        limits: DecodeLimits,
    ) -> Result<Vec<HostRecord<A>>, PersistError> {
        if u64_of(bytes.len()) > limits.max_decoded_bytes {
            return Err(PersistError::TooLarge {
                bytes: u64_of(bytes.len()),
                limit: limits.max_decoded_bytes,
            });
        }
        let mut r = Reader(bytes);
        if r.array::<8>()? != HOST_MAGIC {
            return Err(PersistError::BadMagic);
        }
        let found = r.u32()?;
        if found != FORMAT_VERSION {
            return Err(PersistError::UnsupportedFormat { found });
        }
        let body_len = bytes
            .len()
            .checked_sub(CHECKSUM_LEN)
            .ok_or(PersistError::Truncated)?;
        let (body, stored) = bytes
            .split_at_checked(body_len)
            .ok_or(PersistError::Truncated)?;
        if checksum(body).to_le_bytes() != stored {
            return Err(PersistError::Checksum);
        }
        let mut r = Reader(body);
        let file = FileHeader::read(&mut r)?;
        check_host_fields(&file)?;
        let id = self.host_identity()?;
        same("env_id", &file.env_id, &E::ENV_ID.as_bytes())?;
        same("env_version", &file.env_version, &id.env_version)?;
        same("config_hash", &file.hash, &id.config_hash)?;
        same("auto_reset", &file.auto_reset, &id.auto_reset)?;
        same(
            "max_episode_steps",
            &file.max_episode_steps,
            &id.max_episode_steps,
        )?;
        same("record_len", &file.record_len, &id.record_len)?;
        same("descriptor", &file.descriptor, &&id.descriptor[..])?;
        let count = file.count;
        let limit = limits.max_instances.min(self.options.num_envs.get());
        if count > limit {
            return Err(PersistError::TooManyInstances { count, limit });
        }
        let n = usize::try_from(count).map_err(|_| PersistError::PayloadLength)?;
        let per_record = usize::try_from(limits.max_host_record_bytes)
            .unwrap_or(usize::MAX)
            .min(self.max_snapshot_len());
        let mut out = Vec::new();
        out.try_reserve_exact(n)
            .map_err(|_| PersistError::Allocation)?;
        let mut held = u64_of(bytes.len());
        let mut rest = r.0;
        for index in 0..n {
            let (record, tail) =
                HostRecord::<A>::read(rest, per_record).map_err(|e| record_error(index, &e))?;
            held = held
                .checked_add(u64_of(size_of::<HostRecord<A>>()))
                .and_then(|h| h.checked_add(u64_of(record.bytes.len())))
                .unwrap_or(u64::MAX);
            if held > limits.max_decoded_bytes {
                return Err(PersistError::TooLarge {
                    bytes: held,
                    limit: limits.max_decoded_bytes,
                });
            }
            out.push(record);
            rest = tail;
        }
        if !rest.is_empty() {
            return Err(PersistError::PayloadLength);
        }
        Ok(out)
    }
}

/// The header fields every host file must hold.
fn check_host_fields(file: &FileHeader<'_>) -> Result<(), PersistError> {
    let invalid = |field| Err(PersistError::InvalidField { field });
    if file.encoding != RAW_CANONICAL {
        return invalid("payload_encoding");
    }
    if file.reserved != [0; 3] {
        return invalid("reserved");
    }
    if file.tier != HOST_TIER {
        return invalid("tier");
    }
    if file.mask_provider != 0 {
        return invalid("mask_provider");
    }
    if file.auto_reset > 2 {
        return invalid("auto_reset");
    }
    if file.illegal_kind != 0 || file.penalty_bits != 0 {
        return invalid("illegal_policy");
    }
    if file.asset_count != 0 {
        return Err(PersistError::Incompatible { field: "assets" });
    }
    Ok(())
}

/// A host record's decode error as a file error.
fn record_error(index: usize, e: &RecordDecodeError) -> PersistError {
    match *e {
        RecordDecodeError::Header(source) => PersistError::Record { index, source },
        RecordDecodeError::TooLong { len, max } => PersistError::FieldTooLong {
            field: "host record state bytes",
            len,
            cap: max,
        },
        RecordDecodeError::Allocation => PersistError::Allocation,
        RecordDecodeError::TrailingBytes => PersistError::PayloadLength,
        _ => PersistError::Truncated,
    }
}
