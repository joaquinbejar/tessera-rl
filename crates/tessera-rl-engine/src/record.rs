// SPDX-License-Identifier: MIT OR Apache-2.0

//! Instance records: the continuation state of one instance (ADR-0010,
//! ADR-0036).

use tessera_rl_core::encoding::{Canonical, Decoder, Encoder, Sink, StateHasher};
use tessera_rl_core::{DecodeError, Environment, StateHash};

/// The continuation state of one instance: its engine header (RNG
/// identity, auto-reset progress, accumulators, generations, adaptive
/// values) and its author state. Copying a record is two `memcpy`s, and
/// its canonical encoding is the header followed by the state.
pub struct Record<E: Environment> {
    /// Engine header.
    pub header: E::Header,
    /// Author state.
    pub state: E::State,
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
    /// The `StateHash` of the record: XXH3-64 of header then state.
    #[must_use]
    pub fn hash(&self) -> StateHash {
        let mut hasher = StateHasher::new();
        hasher.add(&self.header);
        hasher.add(&self.state);
        hasher.finish()
    }

    /// Size of the continuation record in bytes (header plus state), as
    /// opposed to the author state alone.
    #[must_use]
    pub const fn record_size() -> usize {
        size_of::<E::Header>() + size_of::<E::State>()
    }
}

impl<E: Environment> Canonical for Record<E> {
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
