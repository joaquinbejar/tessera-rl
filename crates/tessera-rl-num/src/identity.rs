// SPDX-License-Identifier: MIT OR Apache-2.0

//! RNG identity types (SPEC 8.1, ADR-0007).
//!
//! The identity lives in the instance record header and is part of the
//! continuation state. It is independent of the storage slot.

/// An instance seed, any value in `[0, 2^64)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Seed(u64);

impl Seed {
    /// Wraps a raw seed.
    #[must_use]
    #[inline]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw seed.
    #[must_use]
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns the seed of instance `index` in a pool seeded with `self`:
    /// `self + index` modulo 2^64, as SPEC 8.1 documents.
    #[must_use]
    #[inline]
    pub const fn offset(self, index: u64) -> Self {
        // Documented wrapping: SPEC 8.1 defines pool seeds modulo 2^64.
        Self(self.0.wrapping_add(index))
    }
}

/// A branch salt: 0 for root lineages, derived by `fork_branched`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct BranchSalt(u64);

impl BranchSalt {
    /// The salt of root lineages.
    pub const ROOT: Self = Self(0);

    /// Wraps a raw salt.
    #[must_use]
    #[inline]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw salt.
    #[must_use]
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A lineage id. 0 by default; never derived from the storage slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct LogicalId(u32);

impl LogicalId {
    /// The default lineage.
    pub const DEFAULT: Self = Self(0);

    /// Wraps a raw lineage id.
    #[must_use]
    #[inline]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the raw lineage id.
    #[must_use]
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A named RNG stream. Engine streams are 0 to 15; environment streams
/// start at 16.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct StreamId(u32);

impl StreamId {
    /// Environment dynamics.
    pub const DYNAMICS: Self = Self(0);
    /// Procedural map generation.
    pub const MAP_GEN: Self = Self(1);
    /// Scripted opponents.
    pub const OPPONENT: Self = Self(2);
    /// Observation noise, used only through `ObsRng`.
    pub const OBS_NOISE: Self = Self(3);
    /// The first id available to environments.
    pub const FIRST_ENV: u32 = 16;

    /// Returns the environment-defined stream `FIRST_ENV + offset`, or
    /// `None` if that overflows `u32`.
    #[must_use]
    #[inline]
    pub const fn env(offset: u32) -> Option<Self> {
        match Self::FIRST_ENV.checked_add(offset) {
            Some(id) => Some(Self(id)),
            None => None,
        }
    }

    /// Returns the raw stream id.
    #[must_use]
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The 32-byte RNG identity of an instance (ADR-0007).
///
/// Layout, little-endian in the canonical encoding: `seed`, `branch_salt`,
/// `logical_id`, `episode_index`, `step_index`, then four zero bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct RngIdentity {
    seed: Seed,
    branch_salt: BranchSalt,
    logical_id: LogicalId,
    episode_index: u32,
    step_index: u32,
    reserved: u32,
}

impl RngIdentity {
    /// Creates a root identity: the given seed and lineage, salt 0, at
    /// episode 0 and step 0.
    #[must_use]
    #[inline]
    pub const fn new(seed: Seed, logical_id: LogicalId) -> Self {
        Self::from_parts(seed, BranchSalt::ROOT, logical_id, 0, 0)
    }

    /// Creates an identity from all of its fields.
    #[must_use]
    #[inline]
    pub const fn from_parts(
        seed: Seed,
        branch_salt: BranchSalt,
        logical_id: LogicalId,
        episode_index: u32,
        step_index: u32,
    ) -> Self {
        Self {
            seed,
            branch_salt,
            logical_id,
            episode_index,
            step_index,
            reserved: 0,
        }
    }

    /// The instance seed.
    #[must_use]
    #[inline]
    pub const fn seed(&self) -> Seed {
        self.seed
    }

    /// The branch salt.
    #[must_use]
    #[inline]
    pub const fn branch_salt(&self) -> BranchSalt {
        self.branch_salt
    }

    /// The lineage id.
    #[must_use]
    #[inline]
    pub const fn logical_id(&self) -> LogicalId {
        self.logical_id
    }

    /// The episode index, incremented on each reset.
    #[must_use]
    #[inline]
    pub const fn episode_index(&self) -> u32 {
        self.episode_index
    }

    /// The step index: 0 during reset, `t` for transition `t` (1-based).
    #[must_use]
    #[inline]
    pub const fn step_index(&self) -> u32 {
        self.step_index
    }
}

#[cfg(test)]
mod tests {
    use super::{LogicalId, RngIdentity, Seed, StreamId};

    #[test]
    fn test_rng_identity_layout_is_32_bytes() {
        assert_eq!(size_of::<RngIdentity>(), 32);
        assert_eq!(align_of::<RngIdentity>(), 8);
    }

    #[test]
    fn test_seed_offset_wraps_modulo_2_pow_64() {
        assert_eq!(Seed::new(u64::MAX).offset(2), Seed::new(1));
    }

    #[test]
    fn test_stream_id_env_overflow_returns_none() {
        assert_eq!(StreamId::env(0).map(StreamId::get), Some(16));
        assert_eq!(StreamId::env(u32::MAX), None);
    }

    #[test]
    fn test_rng_identity_new_defaults_root_lineage() {
        let id = RngIdentity::new(Seed::new(7), LogicalId::DEFAULT);
        assert_eq!(id.branch_salt().get(), 0);
        assert_eq!((id.episode_index(), id.step_index()), (0, 0));
    }
}
