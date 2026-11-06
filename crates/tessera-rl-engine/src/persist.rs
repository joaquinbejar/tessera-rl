// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persisted snapshots: `encode` and `decode` of SPEC 9.2, under ADR-0011.
//!
//! A file holds canonical records (ADR-0029) behind a header that identifies
//! what can continue them. It holds full state, so it is privileged (SPEC 9.3,
//! ADR-0035): a Leduc file holds hidden cards, and it must never reach a
//! player-perspective path. It is not a pool continuation snapshot (SPEC 9.4):
//! the pool's branch counter is not in it.
//!
//! `decode` returns untrusted records, which enter a pool only through
//! `import` or `import_all`. Nothing here casts bytes, and `decode` allocates
//! nothing before the bounds are checked. The checksum detects accidents; it
//! is not authentication, and a match skips no check.
//!
//! A `same-platform` file requires an explicit `TESSERA_BUILD_ID`, whose
//! producer guarantees that it distinguishes source, dependencies, flags and
//! settings for every crate in the build. The engine cannot see those, and it
//! does not prove the OS version or runtime libraries either. The ISA part is
//! compile-time until runtime dispatch exists.

use tessera_rl_core::config::{CanonicalConfig, config_hash};
use tessera_rl_core::encoding::{Canonical, checksum, decode_from_slice, encode_to_vec};
use tessera_rl_core::{AdaptiveValues, DecodeError, Persist, RecordLayout, Tier};

use crate::error::EngineError;
use crate::masks::{IllegalActionPolicy, PersistMask};
use crate::options::AutoReset;
use crate::pool::EnvPool;
use crate::record::DecodedRecord;

/// The first bytes of every file.
pub const MAGIC: [u8; 8] = *b"TESSERA\x1a";
/// The format this library reads and writes.
pub const FORMAT_VERSION: u32 = 1;

/// `TESSERA_BUILD_ID` as compiled into this engine, for the rebuild check.
#[doc(hidden)]
pub const BUILD_ID: Option<&str> = option_env!("TESSERA_BUILD_ID");

const RAW_CANONICAL: u8 = 0;
const MAX_ENV_ID: usize = 256;
const MAX_LIBRARY_VERSION: usize = 64;
const MAX_DESCRIPTOR: usize = 256;
const MAX_BUILD_ID: usize = 128;
const DESCRIPTOR_VERSION: u8 = 1;
const CHECKSUM_LEN: usize = 8;

/// What a reader accepts from a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeLimits {
    /// Most instances; a secondary ceiling beside `max_decoded_bytes`.
    pub max_instances: u32,
    /// Most bytes held at once: the input plus the decoded records.
    pub max_decoded_bytes: u64,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_instances: 1 << 24,
            max_decoded_bytes: 1 << 30,
        }
    }
}

/// Why a file could not be written or read.
#[derive(Debug, thiserror::Error)]
pub enum PersistError {
    /// The pool refused: poisoned or an unknown id, or its config.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// A record's field is invalid.
    #[error("record {index}: {source}")]
    Record {
        /// The record.
        index: usize,
        /// The field error.
        #[source]
        source: DecodeError,
    },
    /// The input ends inside a field.
    #[error("the file is truncated")]
    Truncated,
    /// The first bytes are not [`MAGIC`].
    #[error("not a Tessera snapshot file")]
    BadMagic,
    /// A format this library does not read.
    #[error("format version {found}, this library reads {FORMAT_VERSION}")]
    UnsupportedFormat {
        /// The file's version.
        found: u32,
    },
    /// The checksum does not match: the file was damaged.
    #[error("checksum mismatch: the file is damaged")]
    Checksum,
    /// A header field holds a value it may not.
    #[error("invalid header field {field}")]
    InvalidField {
        /// The field.
        field: &'static str,
    },
    /// A variable field is over its cap.
    #[error("{field} is {len} bytes, over the cap of {cap}")]
    FieldTooLong {
        /// The field.
        field: &'static str,
        /// Its length.
        len: usize,
        /// The cap.
        cap: usize,
    },
    /// The file cannot continue in this pool.
    #[error("the file does not match this pool: {field}")]
    Incompatible {
        /// The field that differs.
        field: &'static str,
    },
    /// A `same-platform` file needs `TESSERA_BUILD_ID`, and this build has none.
    #[error("same-platform files need an explicit TESSERA_BUILD_ID at build time")]
    BuildIdRequired,
    /// The file holds more instances than allowed.
    #[error("{count} instances, over the limit of {limit}")]
    TooManyInstances {
        /// The file's count.
        count: u32,
        /// The limit.
        limit: u32,
    },
    /// Holding the file and its records would pass `max_decoded_bytes`.
    #[error("decoding needs {bytes} bytes, over the limit of {limit}")]
    TooLarge {
        /// Bytes needed (`u64::MAX` on overflow).
        bytes: u64,
        /// The limit.
        limit: u64,
    },
    /// The payload's length is not the count times the record length.
    #[error("the payload does not hold exactly the declared records")]
    PayloadLength,
    /// The allocator refused.
    #[error("could not allocate the decoded records")]
    Allocation,
}

/// The compatibility fields, the same on the writer and the reader.
struct Identity {
    tier: u8,
    mask_provider: u8,
    auto_reset: u8,
    illegal_kind: u8,
    penalty_bits: u32,
    max_episode_steps: u32,
    env_version: u32,
    config_hash: u64,
    record_len: u32,
    env_id: &'static str,
}

/// Compile-time target features this descriptor names, sorted. A partial list,
/// and `target_env` (gnu, musl) is not in the descriptor either: the build id's
/// producer carries the guarantee for all flags and settings.
const FEATURES: [(&str, bool); 13] = [
    ("aes", cfg!(target_feature = "aes")),
    ("avx", cfg!(target_feature = "avx")),
    ("avx2", cfg!(target_feature = "avx2")),
    ("avx512f", cfg!(target_feature = "avx512f")),
    ("bmi1", cfg!(target_feature = "bmi1")),
    ("bmi2", cfg!(target_feature = "bmi2")),
    ("fma", cfg!(target_feature = "fma")),
    ("lzcnt", cfg!(target_feature = "lzcnt")),
    ("neon", cfg!(target_feature = "neon")),
    ("popcnt", cfg!(target_feature = "popcnt")),
    ("sse2", cfg!(target_feature = "sse2")),
    ("sse4.1", cfg!(target_feature = "sse4.1")),
    ("sse4.2", cfg!(target_feature = "sse4.2")),
];

fn features() -> impl Iterator<Item = &'static str> {
    FEATURES.iter().filter(|f| f.1).map(|f| f.0)
}

fn push_part(out: &mut Vec<u8>, part: &[u8]) {
    let len = u16::try_from(part.len()).unwrap_or(u16::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(part);
}

/// The build id a `same-platform` file needs, with no fallback.
fn required_id(build_id: Option<&str>) -> Result<&str, PersistError> {
    let id = build_id
        .filter(|id| !id.is_empty())
        .ok_or(PersistError::BuildIdRequired)?;
    if id.len() > MAX_BUILD_ID {
        return Err(PersistError::FieldTooLong {
            field: "build id",
            len: id.len(),
            cap: MAX_BUILD_ID,
        });
    }
    Ok(id)
}

/// The descriptor of a tier, for writing.
fn descriptor(tier: Tier, build_id: Option<&str>) -> Result<Vec<u8>, PersistError> {
    match tier {
        Tier::CrossPlatform => Ok(Vec::new()),
        Tier::SamePlatform => {
            let id = required_id(build_id)?;
            let mut out = vec![DESCRIPTOR_VERSION];
            push_part(&mut out, std::env::consts::OS.as_bytes());
            push_part(&mut out, std::env::consts::ARCH.as_bytes());
            let joined = features().collect::<Vec<_>>().join(",");
            push_part(&mut out, joined.as_bytes());
            push_part(&mut out, id.as_bytes());
            Ok(out)
        }
    }
}

/// Whether a file's descriptor is this build's. Allocates nothing: it runs
/// before the bounds are checked.
fn descriptor_matches(
    file: &[u8],
    tier: Tier,
    build_id: Option<&str>,
) -> Result<bool, PersistError> {
    match tier {
        Tier::CrossPlatform => Ok(file.is_empty()),
        Tier::SamePlatform => {
            let id = required_id(build_id)?;
            let mut r = Reader(file);
            let part = |r: &mut Reader<'_>, want: &[u8]| -> bool {
                r.capped("descriptor", MAX_DESCRIPTOR)
                    .is_ok_and(|got| got == want)
            };
            if r.u8().ok() != Some(DESCRIPTOR_VERSION)
                || !part(&mut r, std::env::consts::OS.as_bytes())
                || !part(&mut r, std::env::consts::ARCH.as_bytes())
            {
                return Ok(false);
            }
            let Ok(mut rest) = r.capped("descriptor", MAX_DESCRIPTOR) else {
                return Ok(false);
            };
            for (i, feature) in features().enumerate() {
                if i > 0 {
                    let Some(tail) = rest.strip_prefix(b",") else {
                        return Ok(false);
                    };
                    rest = tail;
                }
                let Some(tail) = rest.strip_prefix(feature.as_bytes()) else {
                    return Ok(false);
                };
                rest = tail;
            }
            Ok(rest.is_empty() && part(&mut r, id.as_bytes()) && r.0.is_empty())
        }
    }
}

fn identity<E, M>(pool: &EnvPool<E, M>) -> Result<Identity, PersistError>
where
    E: Persist,
    E::Config: CanonicalConfig,
    M: PersistMask<E>,
{
    let (illegal_kind, penalty_bits) = match (M::ENABLED, pool.options.illegal_action) {
        (true, IllegalActionPolicy::Penalize(r)) => (1, r.to_bits()),
        _ => (0, 0),
    };
    let record_len = u32::try_from(<DecodedRecord<E> as Canonical>::ENCODED_LEN).map_err(|_| {
        PersistError::Incompatible {
            field: "record_len",
        }
    })?;
    Ok(Identity {
        tier: match E::TIER {
            Tier::CrossPlatform => 0,
            Tier::SamePlatform => 1,
        },
        mask_provider: M::PROVIDER_ID,
        auto_reset: match pool.options.auto_reset {
            AutoReset::NextStep => 0,
            AutoReset::SameStep => 1,
            AutoReset::Disabled => 2,
        },
        illegal_kind,
        penalty_bits,
        max_episode_steps: pool
            .options
            .max_episode_steps
            .map_or(0, core::num::NonZeroU32::get),
        env_version: E::ENV_VERSION,
        config_hash: config_hash(E::ENV_ID, E::ENV_VERSION, &pool.cfg)
            .map_err(|e| PersistError::Engine(e.into()))?,
        record_len,
        env_id: E::ENV_ID,
    })
}

fn push_capped(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &[u8],
    cap: usize,
) -> Result<(), PersistError> {
    if value.len() > cap {
        return Err(PersistError::FieldTooLong {
            field,
            len: value.len(),
            cap,
        });
    }
    push_part(out, value);
    Ok(())
}

/// Reads fields by offset; never casts.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], PersistError> {
        let (head, tail) = self.0.split_at_checked(n).ok_or(PersistError::Truncated)?;
        self.0 = tail;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], PersistError> {
        self.take(N)?
            .try_into()
            .map_err(|_| PersistError::Truncated)
    }

    fn u8(&mut self) -> Result<u8, PersistError> {
        Ok(u8::from_le_bytes(self.array()?))
    }

    fn u16(&mut self) -> Result<u16, PersistError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, PersistError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, PersistError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn capped(&mut self, field: &'static str, cap: usize) -> Result<&'a [u8], PersistError> {
        let len = usize::from(self.u16()?);
        if len > cap {
            return Err(PersistError::FieldTooLong { field, len, cap });
        }
        self.take(len)
    }
}

fn same<T: PartialEq>(field: &'static str, a: &T, b: &T) -> Result<(), PersistError> {
    if a == b {
        Ok(())
    } else {
        Err(PersistError::Incompatible { field })
    }
}

/// The header as read, before any check.
struct FileHeader<'a> {
    encoding: u8,
    tier: u8,
    mask_provider: u8,
    auto_reset: u8,
    illegal_kind: u8,
    reserved: [u8; 3],
    penalty_bits: u32,
    max_episode_steps: u32,
    env_version: u32,
    hash: u64,
    count: u32,
    record_len: u32,
    env_id: &'a [u8],
    asset_count: u16,
    descriptor: &'a [u8],
}

impl<'a> FileHeader<'a> {
    fn read(r: &mut Reader<'a>) -> Result<Self, PersistError> {
        r.take(12)?;
        let encoding = r.u8()?;
        let tier = r.u8()?;
        let mask_provider = r.u8()?;
        let auto_reset = r.u8()?;
        let illegal_kind = r.u8()?;
        let reserved = r.array::<3>()?;
        let penalty_bits = r.u32()?;
        let max_episode_steps = r.u32()?;
        let env_version = r.u32()?;
        let hash = r.u64()?;
        let count = r.u32()?;
        let record_len = r.u32()?;
        let env_id = r.capped("env_id", MAX_ENV_ID)?;
        r.capped("library_version", MAX_LIBRARY_VERSION)?;
        let asset_count = r.u16()?;
        let descriptor = r.capped("descriptor", MAX_DESCRIPTOR)?;
        Ok(Self {
            encoding,
            tier,
            mask_provider,
            auto_reset,
            illegal_kind,
            reserved,
            penalty_bits,
            max_episode_steps,
            env_version,
            hash,
            count,
            record_len,
            env_id,
            asset_count,
            descriptor,
        })
    }

    fn check_fields(&self) -> Result<(), PersistError> {
        let invalid = |field| Err(PersistError::InvalidField { field });
        if self.encoding != RAW_CANONICAL {
            return invalid("payload_encoding");
        }
        if self.reserved != [0; 3] {
            return invalid("reserved");
        }
        if self.tier > 1 {
            return invalid("tier");
        }
        if self.mask_provider > 1 {
            return invalid("mask_provider");
        }
        if self.auto_reset > 2 {
            return invalid("auto_reset");
        }
        if self.illegal_kind > 1 {
            return invalid("illegal_kind");
        }
        if self.illegal_kind == 0 && self.penalty_bits != 0 {
            return invalid("illegal_penalty_bits");
        }
        // Format 1 knows no asset hash: zero means none, never unchecked.
        if self.asset_count != 0 {
            return Err(PersistError::Incompatible { field: "assets" });
        }
        Ok(())
    }

    fn check_against(
        &self,
        id: &Identity,
        tier: Tier,
        build_id: Option<&str>,
    ) -> Result<(), PersistError> {
        same("env_id", &self.env_id, &id.env_id.as_bytes())?;
        same("env_version", &self.env_version, &id.env_version)?;
        same("tier", &self.tier, &id.tier)?;
        same("config_hash", &self.hash, &id.config_hash)?;
        same("mask_provider", &self.mask_provider, &id.mask_provider)?;
        same("auto_reset", &self.auto_reset, &id.auto_reset)?;
        same("illegal_policy", &self.illegal_kind, &id.illegal_kind)?;
        same("illegal_policy", &self.penalty_bits, &id.penalty_bits)?;
        same(
            "max_episode_steps",
            &self.max_episode_steps,
            &id.max_episode_steps,
        )?;
        same("record_len", &self.record_len, &id.record_len)?;
        if descriptor_matches(self.descriptor, tier, build_id)? {
            Ok(())
        } else {
            Err(PersistError::Incompatible {
                field: "descriptor",
            })
        }
    }
}

fn u64_of(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

impl<E, M> EnvPool<E, M>
where
    E: Persist,
    E::Config: CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    M: PersistMask<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    /// The file holding the records of `ids`, in that order (SPEC 9.2).
    ///
    /// A borrow window excludes it, as `encode` cannot run while a step's
    /// outputs are held:
    ///
    /// ```compile_fail,E0502
    /// use tessera_rl_core::Persist;
    /// use tessera_rl_engine::{ActionInput, EnvPool, NoMasks};
    /// fn f<E: Persist>(p: &mut EnvPool<E, NoMasks>, a: ActionInput<'_>)
    /// where
    ///     E::Config: tessera_rl_core::config::CanonicalConfig + Sync,
    ///     E::Header: Send,
    ///     E::State: Send,
    ///     <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
    ///         tessera_rl_core::AdaptiveValues<E::Config>,
    /// {
    ///     let held = p.step(a).unwrap();
    ///     let _ = p.encode(&[0]);
    ///     let _ = held.rewards();
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`PersistError`] on a poisoned pool or an unknown id, or
    /// [`PersistError::BuildIdRequired`] for a `same-platform` environment
    /// built without `TESSERA_BUILD_ID`.
    pub fn encode(&self, ids: &[u32]) -> Result<Vec<u8>, PersistError> {
        self.encode_inner(ids, env!("CARGO_PKG_VERSION"), BUILD_ID)
    }

    /// Checks a file against this pool and returns its records, still
    /// untrusted: this is no restore. It reads only the config, so it works
    /// on a poisoned pool, whose recovery is `import_all`.
    ///
    /// # Errors
    ///
    /// Returns a [`PersistError`] for anything but a sound, compatible file
    /// within `limits`.
    pub fn decode(
        &self,
        bytes: &[u8],
        limits: DecodeLimits,
    ) -> Result<Vec<DecodedRecord<E>>, PersistError> {
        self.decode_inner(bytes, limits, BUILD_ID)
    }

    /// [`EnvPool::encode`] with its build inputs as parameters. It substitutes the
    /// build identity, so it exists only for fixtures, tests and fuzzing, behind the
    /// `test-hooks` feature, and must never be exposed to Python.
    ///
    /// # Errors
    ///
    /// As [`EnvPool::encode`].
    #[cfg(feature = "test-hooks")]
    pub fn encode_with(
        &self,
        ids: &[u32],
        library_version: &str,
        build_id: Option<&str>,
    ) -> Result<Vec<u8>, PersistError> {
        self.encode_inner(ids, library_version, build_id)
    }

    /// [`EnvPool::decode`] with the build id as a parameter; see
    /// [`EnvPool::encode_with`].
    ///
    /// # Errors
    ///
    /// As [`EnvPool::decode`].
    #[cfg(feature = "test-hooks")]
    pub fn decode_with(
        &self,
        bytes: &[u8],
        limits: DecodeLimits,
        build_id: Option<&str>,
    ) -> Result<Vec<DecodedRecord<E>>, PersistError> {
        self.decode_inner(bytes, limits, build_id)
    }

    fn encode_inner(
        &self,
        ids: &[u32],
        library_version: &str,
        build_id: Option<&str>,
    ) -> Result<Vec<u8>, PersistError> {
        if self.poisoned {
            return Err(EngineError::Poisoned.into());
        }
        let id = identity(self)?;
        let file_descriptor = descriptor(E::TIER, build_id)?;
        let count = u32::try_from(ids.len()).map_err(|_| PersistError::InvalidField {
            field: "instance_count",
        })?;
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&[
            RAW_CANONICAL,
            id.tier,
            id.mask_provider,
            id.auto_reset,
            id.illegal_kind,
            0,
            0,
            0,
        ]);
        out.extend_from_slice(&id.penalty_bits.to_le_bytes());
        out.extend_from_slice(&id.max_episode_steps.to_le_bytes());
        out.extend_from_slice(&id.env_version.to_le_bytes());
        out.extend_from_slice(&id.config_hash.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&id.record_len.to_le_bytes());
        push_capped(&mut out, "env_id", id.env_id.as_bytes(), MAX_ENV_ID)?;
        push_capped(
            &mut out,
            "library_version",
            library_version.as_bytes(),
            MAX_LIBRARY_VERSION,
        )?;
        out.extend_from_slice(&0u16.to_le_bytes()); // asset_count
        push_capped(&mut out, "descriptor", &file_descriptor, MAX_DESCRIPTOR)?;
        for &i in ids {
            let record = DecodedRecord::<E>::from(self.snapshot(i)?);
            out.extend_from_slice(&encode_to_vec(&record));
        }
        let sum = checksum(&out);
        out.extend_from_slice(&sum.to_le_bytes());
        Ok(out)
    }

    fn decode_inner(
        &self,
        bytes: &[u8],
        limits: DecodeLimits,
        build_id: Option<&str>,
    ) -> Result<Vec<DecodedRecord<E>>, PersistError> {
        // Before scanning a file no limit could admit.
        if u64_of(bytes.len()) > limits.max_decoded_bytes {
            return Err(PersistError::TooLarge {
                bytes: u64_of(bytes.len()),
                limit: limits.max_decoded_bytes,
            });
        }
        let mut r = Reader(bytes);
        if r.array::<8>()? != MAGIC {
            return Err(PersistError::BadMagic);
        }
        // Before the checksum, so a later format reports its version.
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
        file.check_fields()?;
        file.check_against(&identity(self)?, E::TIER, build_id)?;
        let (count, record_len) = (file.count, file.record_len);
        let limit = limits.max_instances.min(self.options.num_envs.get());
        if count > limit {
            return Err(PersistError::TooManyInstances { count, limit });
        }
        let n = usize::try_from(count).map_err(|_| PersistError::PayloadLength)?;
        let len = usize::try_from(record_len).map_err(|_| PersistError::PayloadLength)?;
        if n.checked_mul(len) != Some(r.0.len()) {
            return Err(PersistError::PayloadLength);
        }
        let held = n
            .checked_mul(size_of::<DecodedRecord<E>>())
            .and_then(|b| b.checked_add(bytes.len()))
            .map_or(u64::MAX, u64_of);
        if held > limits.max_decoded_bytes {
            return Err(PersistError::TooLarge {
                bytes: held,
                limit: limits.max_decoded_bytes,
            });
        }
        let mut out = Vec::new();
        out.try_reserve_exact(n)
            .map_err(|_| PersistError::Allocation)?;
        for (index, chunk) in r.0.chunks_exact(len.max(1)).enumerate() {
            let record = decode_from_slice(chunk)
                .map_err(|source| PersistError::Record { index, source })?;
            out.push(record);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{PersistError, Tier, descriptor};

    #[test]
    fn test_descriptor_cross_platform_is_empty() {
        assert_eq!(descriptor(Tier::CrossPlatform, None).ok(), Some(Vec::new()));
    }

    #[test]
    fn test_descriptor_same_platform_refuses_no_or_empty_id() {
        for id in [None, Some("")] {
            assert!(matches!(
                descriptor(Tier::SamePlatform, id),
                Err(PersistError::BuildIdRequired)
            ));
        }
    }

    #[test]
    fn test_descriptor_differs_by_id_alone() {
        assert_ne!(
            descriptor(Tier::SamePlatform, Some("a")).ok(),
            descriptor(Tier::SamePlatform, Some("b")).ok()
        );
    }
}
