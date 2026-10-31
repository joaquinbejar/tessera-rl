// SPDX-License-Identifier: MIT OR Apache-2.0

//! The engine-managed record header (SPEC 5.1, ADR-0036).
//!
//! Each instance record is a header plus the author's state, stored in two
//! parallel arrays. The header is plain data under the same rules as state
//! (ADR-0029): `#[repr(C)]`, padding-free and canonically encoded. Authors
//! name it in `Environment::Header` as `RecordHeader<MAX_AGENTS, Adaptive>`;
//! kernels never receive it. Its fields are public for the engine; changing
//! them from environment code is not supported.
//!
//! Layout: [`EngineHeader`] (40 bytes), then one [`AgentRow`] (16 bytes) per
//! agent slot, then the adaptive config values (a multiple of 8 bytes, or
//! [`NoAdaptive`]).

use tessera_rl_num::identity::RngIdentity;

use crate::bool8::Bool8;
use crate::encoding::{Canonical, Decoder, Encoder, Sink};
use crate::error::DecodeError;
use crate::reserved::Reserved;
use crate::state_init::StateInit;

/// Per-instance engine fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct EngineHeader {
    /// RNG identity, including the seed (ADR-0007).
    pub identity: RngIdentity,
    /// Transitions taken in the current episode.
    pub episode_steps: u32,
    /// The episode ended and the instance waits for its reset.
    pub pending_reset: Bool8,
    /// How the last episode ended: 0 none, 1 terminated, 2 truncated.
    pub end_kind: u8,
    /// No reset has used the current `episode_index` yet (construction or
    /// reseeding), so the next reset keeps it instead of incrementing it.
    pub fresh: Bool8,
    /// Always zero.
    pub reserved: Reserved<1>,
}

/// Per-agent engine fields.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct AgentRow {
    /// Incremented when a new occupant spawns in the slot.
    pub generation: u32,
    /// Sum of the agent's rewards in the current episode.
    pub episode_return: f32,
    /// Illegal-but-valid actions replaced in the current episode. A
    /// statistic: it saturates at `u32::MAX` (a documented bound).
    pub illegal_actions: u32,
    /// The slot holds a living agent.
    pub alive: Bool8,
    /// The agent acts this step.
    pub acting: Bool8,
    /// Spawns requested on this slot in the current episode while it was
    /// alive or dying, and therefore not applied (SPEC 9.1). A statistic
    /// like `illegal_actions`: it saturates at `u16::MAX` (a documented
    /// bound).
    pub invalid_spawns: u16,
}

/// Adaptive values built from a config `C`: `#[derive(EnvConfig)]`
/// generates `<Name>Adaptive` implementing it, and [`NoAdaptive`]
/// implements it for every config.
pub trait AdaptiveValues<C>: Canonical + StateInit + Copy + 'static {
    /// The values of the adaptive fields of `cfg`.
    fn from_config(cfg: &C) -> Self;
}

impl<C> AdaptiveValues<C> for NoAdaptive {
    #[inline]
    fn from_config(_: &C) -> Self {
        Self
    }
}

/// The adaptive values of a config without adaptive fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NoAdaptive;

impl Canonical for NoAdaptive {
    const ENCODED_LEN: usize = 0;

    #[inline]
    fn encode<S: Sink>(&self, _: &mut Encoder<S>) {}

    #[inline]
    fn decode(_: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        Ok(Self)
    }
}

impl StateInit for NoAdaptive {
    #[inline]
    fn state_default() -> Self {
        Self
    }
}

/// The record header of an environment with `A` agent slots and adaptive
/// config values `Ad`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct RecordHeader<const A: usize, Ad> {
    /// Engine fields.
    pub engine: EngineHeader,
    /// One row per agent slot.
    pub agents: [AgentRow; A],
    /// Effective adaptive config values (ADR-0030).
    pub adaptive: Ad,
}

/// What the engine needs from a record header type.
pub trait RecordLayout: Canonical + StateInit + Copy + 'static {
    /// Number of agent slots; must equal `Environment::MAX_AGENTS`.
    const AGENTS: usize;

    /// The adaptive config values stored in the header.
    type Adaptive: Canonical + StateInit + Copy + 'static;

    /// The adaptive values, mutably.
    fn adaptive_mut(&mut self) -> &mut Self::Adaptive;

    /// The engine fields.
    fn engine(&self) -> &EngineHeader;

    /// The engine fields, mutably.
    fn engine_mut(&mut self) -> &mut EngineHeader;

    /// The agent rows.
    fn agents(&self) -> &[AgentRow];

    /// The agent rows, mutably.
    fn agents_mut(&mut self) -> &mut [AgentRow];
}

impl<const A: usize, Ad> RecordHeader<A, Ad> {
    // Padding-free (ADR-0036): agent rows are 16 bytes and the engine part
    // 40, so `Ad` starts 8-aligned; `Ad` must be a multiple of its
    // alignment, which `#[derive(EnvConfig)]` guarantees.
    const LAYOUT_OK: () = assert!(
        size_of::<Self>()
            == size_of::<EngineHeader>() + A * size_of::<AgentRow>() + size_of::<Ad>(),
        "RecordHeader has padding: the adaptive struct must have a size that is a multiple of 8"
    );
}

impl<const A: usize, Ad: Canonical + StateInit + Copy + 'static> RecordLayout
    for RecordHeader<A, Ad>
{
    const AGENTS: usize = {
        let () = Self::LAYOUT_OK;
        A
    };

    type Adaptive = Ad;

    #[inline]
    fn adaptive_mut(&mut self) -> &mut Ad {
        &mut self.adaptive
    }

    #[inline]
    fn engine(&self) -> &EngineHeader {
        &self.engine
    }

    #[inline]
    fn engine_mut(&mut self) -> &mut EngineHeader {
        &mut self.engine
    }

    #[inline]
    fn agents(&self) -> &[AgentRow] {
        &self.agents
    }

    #[inline]
    fn agents_mut(&mut self) -> &mut [AgentRow] {
        &mut self.agents
    }
}

impl<const A: usize, Ad: StateInit> StateInit for RecordHeader<A, Ad> {
    #[inline]
    fn state_default() -> Self {
        let () = Self::LAYOUT_OK;
        Self {
            engine: EngineHeader::state_default(),
            agents: <[AgentRow; A]>::state_default(),
            adaptive: Ad::state_default(),
        }
    }
}

impl StateInit for EngineHeader {
    #[inline]
    fn state_default() -> Self {
        Self::default()
    }
}

impl StateInit for AgentRow {
    #[inline]
    fn state_default() -> Self {
        Self::default()
    }
}

impl Canonical for EngineHeader {
    const ENCODED_LEN: usize = 40;

    #[inline]
    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        self.identity.encode(out);
        self.episode_steps.encode(out);
        self.pending_reset.encode(out);
        self.end_kind.encode(out);
        self.fresh.encode(out);
        self.reserved.encode(out);
    }

    #[inline]
    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        let identity = RngIdentity::decode(input)?;
        let episode_steps = u32::decode(input)?;
        let pending_reset = Bool8::decode(input)?;
        let end_kind = u8::decode(input)?;
        if end_kind > 2 {
            return Err(DecodeError::InvalidEnum {
                name: "EpisodeEnd",
                value: i64::from(end_kind),
            });
        }
        Ok(Self {
            identity,
            episode_steps,
            pending_reset,
            end_kind,
            fresh: Bool8::decode(input)?,
            reserved: Reserved::decode(input)?,
        })
    }
}

impl Canonical for AgentRow {
    const ENCODED_LEN: usize = 16;

    #[inline]
    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        self.generation.encode(out);
        self.episode_return.encode(out);
        self.illegal_actions.encode(out);
        self.alive.encode(out);
        self.acting.encode(out);
        self.invalid_spawns.encode(out);
    }

    #[inline]
    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        Ok(Self {
            generation: u32::decode(input)?,
            episode_return: f32::decode(input)?,
            illegal_actions: u32::decode(input)?,
            alive: Bool8::decode(input)?,
            acting: Bool8::decode(input)?,
            invalid_spawns: u16::decode(input)?,
        })
    }
}

impl<const A: usize, Ad: Canonical + StateInit> Canonical for RecordHeader<A, Ad> {
    const ENCODED_LEN: usize = {
        let () = Self::LAYOUT_OK;
        let len = EngineHeader::ENCODED_LEN + A * AgentRow::ENCODED_LEN + Ad::ENCODED_LEN;
        assert!(
            len == size_of::<Self>(),
            "RecordHeader size differs from its encoded length: the adaptive struct has hidden padding"
        );
        len
    };

    #[inline]
    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        self.engine.encode(out);
        self.agents.encode(out);
        self.adaptive.encode(out);
    }

    #[inline]
    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        Ok(Self {
            engine: EngineHeader::decode(input)?,
            agents: Canonical::decode(input)?,
            adaptive: Ad::decode(input)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{AgentRow, EngineHeader, NoAdaptive, RecordHeader, RecordLayout};
    use crate::encoding::{Canonical, decode_from_slice, encode_to_vec};
    use crate::state_init::StateInit;

    #[test]
    fn test_header_parts_are_padding_free() {
        assert_eq!(size_of::<EngineHeader>(), 40);
        assert_eq!(size_of::<AgentRow>(), 16);
        assert_eq!(size_of::<RecordHeader<1, NoAdaptive>>(), 56);
        assert_eq!(size_of::<RecordHeader<3, [u64; 2]>>(), 40 + 48 + 16);
        assert_eq!(<RecordHeader<3, NoAdaptive> as RecordLayout>::AGENTS, 3);
    }

    #[test]
    fn test_header_encoded_len_matches_layout() {
        type H = RecordHeader<2, NoAdaptive>;
        assert_eq!(H::ENCODED_LEN, size_of::<H>());
        let mut h = H::state_default();
        h.engine.end_kind = 2;
        h.agents[1].generation = 7;
        let bytes = encode_to_vec(&h);
        assert_eq!(bytes.len(), H::ENCODED_LEN);
        assert_eq!(decode_from_slice::<H>(&bytes), Ok(h));
    }

    #[test]
    fn test_header_decode_rejects_invalid_end_kind() {
        let mut bytes = encode_to_vec(&RecordHeader::<1, NoAdaptive>::state_default());
        bytes[37] = 3;
        assert!(decode_from_slice::<RecordHeader<1, NoAdaptive>>(&bytes).is_err());
    }
}
