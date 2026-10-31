// SPDX-License-Identifier: MIT OR Apache-2.0

//! Instance records: the continuation state of one instance (ADR-0010,
//! ADR-0036).
//!
//! A [`Record`] is trusted: only the engine creates one (by snapshotting),
//! so it can be restored directly. Bytes from outside decode into a
//! [`DecodedRecord`], which is untrusted and enters a pool only through
//! `EnvPool::import`, after the engine and the environment validated it
//! (ADR-0029).

use tessera_rl_core::encoding::{Canonical, Decoder, Encoder, Sink, StateHasher, encode_to_vec};
use tessera_rl_core::{DecodeError, Environment, StateHash};

/// The continuation state of one instance: its engine header (RNG
/// identity, auto-reset progress, accumulators, generations, adaptive
/// values) and its author state. Copying a record is two `memcpy`s. The
/// `same_step` terminal observation is a step output, not part of the
/// record (ADR-0039).
pub struct Record<E: Environment> {
    pub(crate) header: E::Header,
    pub(crate) state: E::State,
}

impl<E: Environment> Clone for Record<E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Environment> Copy for Record<E> {}

impl<E: Environment> core::fmt::Debug for Record<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Record")
            .field("hash", &self.hash())
            .finish()
    }
}

impl<E: Environment> Record<E> {
    /// The engine header.
    #[must_use]
    pub const fn header(&self) -> &E::Header {
        &self.header
    }

    /// The author state.
    #[must_use]
    pub const fn state(&self) -> &E::State {
        &self.state
    }

    /// The `StateHash` of the record: XXH3-64 of header then state.
    #[must_use]
    pub fn hash(&self) -> StateHash {
        let mut hasher = StateHasher::new();
        hasher.add(&self.header);
        hasher.add(&self.state);
        hasher.finish()
    }

    /// The canonical encoding: header then state. For persistence; never
    /// on the hot path.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        encode_to_vec(&DecodedRecord::<E>::from(*self))
    }

    /// Size of the continuation record in bytes (header plus state), as
    /// opposed to the author state alone.
    #[must_use]
    pub const fn record_size() -> usize {
        size_of::<E::Header>() + size_of::<E::State>()
    }
}

/// A record decoded from bytes. Its fields have valid bit patterns, but
/// it is not trusted: restore it with `EnvPool::import`, which checks the
/// engine header and runs `Validate` first.
pub struct DecodedRecord<E: Environment> {
    header: E::Header,
    state: E::State,
}

impl<E: Environment> Clone for DecodedRecord<E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Environment> Copy for DecodedRecord<E> {}

impl<E: Environment> core::fmt::Debug for DecodedRecord<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecodedRecord").finish_non_exhaustive()
    }
}

impl<E: Environment> DecodedRecord<E> {
    /// The decoded engine header.
    #[must_use]
    pub const fn header(&self) -> &E::Header {
        &self.header
    }

    /// The decoded author state.
    #[must_use]
    pub const fn state(&self) -> &E::State {
        &self.state
    }

    /// Builds an untrusted record from parts, for tests and tools. The parts
    /// go through exactly the canonical path that bytes do: they are encoded
    /// and decoded with the generated field policies (`Bool8`, enums,
    /// `Reserved`, NaN unless `#[state(allow_nan)]`), so no invalid field
    /// can get in this way. Allocates; never on the hot path.
    ///
    /// # Errors
    ///
    /// Returns the [`DecodeError`] of the first invalid field.
    pub fn from_parts(header: E::Header, state: E::State) -> Result<Self, DecodeError> {
        let parts = Self { header, state };
        tessera_rl_core::encoding::decode_from_slice(&encode_to_vec(&parts))
    }

    pub(crate) const fn trust(self) -> Record<E> {
        Record {
            header: self.header,
            state: self.state,
        }
    }
}

impl<E: Environment> From<Record<E>> for DecodedRecord<E> {
    fn from(record: Record<E>) -> Self {
        Self {
            header: record.header,
            state: record.state,
        }
    }
}

impl<E: Environment> Canonical for DecodedRecord<E> {
    const ENCODED_LEN: usize =
        <E::Header as Canonical>::ENCODED_LEN + <E::State as Canonical>::ENCODED_LEN;

    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        self.header.encode(out);
        self.state.encode(out);
    }

    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        Ok(Self {
            header: Canonical::decode(input)?,
            state: Canonical::decode(input)?,
        })
    }
}

/// Pool-level context captured by pool snapshots (ADR-0010).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolContext {
    /// Calls to `fork_branched` so far; part of every derived salt.
    pub branch_counter: u64,
}
