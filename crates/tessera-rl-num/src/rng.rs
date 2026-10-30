// SPDX-License-Identifier: MIT OR Apache-2.0

//! `StepRng` and `ObsRng`: the only randomness environment code may use
//! (SPEC 8.1, ADR-0007).

use core::num::{NonZeroU32, NonZeroU64};

use crate::draw::Draws;
use crate::identity::{RngIdentity, StreamId};
use crate::key::{derive_key, key_words};

/// Proof that the caller is the engine.
///
/// Environment code never constructs an RNG: the engine builds `StepRng`
/// and `ObsRng` from the instance identity and passes them to kernels.
/// This token exists so that the constructors are not part of the
/// author-facing API.
#[doc(hidden)]
#[derive(Debug)]
pub struct RngAuthority(());

impl RngAuthority {
    /// Engine-only. Not part of the public API.
    #[doc(hidden)]
    #[must_use]
    #[inline]
    pub const fn engine_only() -> Self {
        Self(())
    }
}

fn draws(identity: &RngIdentity, stream: StreamId) -> Draws {
    let key = derive_key(identity.seed(), stream, identity.branch_salt());
    Draws::new(
        key_words(key),
        identity.logical_id().get(),
        identity.episode_index(),
        identity.step_index(),
    )
}

macro_rules! forward_draws {
    () => {
        /// Returns whether more than 2^32 Philox calls were made in this
        /// phase, so counters were reused. The engine checks this at the
        /// validation boundary.
        #[must_use]
        #[inline]
        pub const fn exhausted(&self) -> bool {
            self.draws.exhausted()
        }

        /// Returns the next 32-bit word.
        #[inline]
        pub fn next_u32(&mut self) -> u32 {
            self.draws.next_u32()
        }

        /// Returns the next `u64`, built from two words little-endian.
        #[inline]
        pub fn next_u64(&mut self) -> u64 {
            self.draws.next_u64()
        }

        /// Returns a uniform integer in `[0, bound)`.
        #[inline]
        pub fn below_u32(&mut self, bound: NonZeroU32) -> u32 {
            self.draws.below_u32(bound)
        }

        /// Returns a uniform integer in `[0, bound)`.
        #[inline]
        pub fn below_u64(&mut self, bound: NonZeroU64) -> u64 {
            self.draws.below_u64(bound)
        }

        /// Returns a uniform `f32` in `[0, 1)` (24 bits).
        #[inline]
        pub fn unit_f32(&mut self) -> f32 {
            self.draws.unit_f32()
        }

        /// Returns a uniform `f64` in `[0, 1)` (53 bits).
        #[inline]
        pub fn unit_f64(&mut self) -> f64 {
            self.draws.unit_f64()
        }
    };
}

/// Randomness for one reset or transition of one instance on one stream.
#[derive(Debug, Clone)]
pub struct StepRng {
    draws: Draws,
}

impl StepRng {
    /// Engine-only: the RNG for `stream` at the identity's current
    /// episode and step, with `draw_index` starting at 0.
    #[doc(hidden)]
    #[must_use]
    #[inline]
    pub fn new(_authority: &RngAuthority, identity: &RngIdentity, stream: StreamId) -> Self {
        Self {
            draws: draws(identity, stream),
        }
    }

    forward_draws!();
}

/// Observation randomness: the `obs_noise` stream at the current step,
/// restarting at `draw_index = 0` on every construction, so repeated
/// `observe` calls are identical.
#[derive(Debug, Clone)]
pub struct ObsRng {
    draws: Draws,
}

impl ObsRng {
    /// Engine-only: the observation RNG at the identity's current step.
    #[doc(hidden)]
    #[must_use]
    #[inline]
    pub fn new(_authority: &RngAuthority, identity: &RngIdentity) -> Self {
        Self {
            draws: draws(identity, StreamId::OBS_NOISE),
        }
    }

    forward_draws!();
}

#[cfg(test)]
mod tests {
    use super::{ObsRng, RngAuthority, StepRng};
    use crate::identity::{BranchSalt, LogicalId, RngIdentity, Seed, StreamId};

    fn id(seed: u64, lineage: u32, episode: u32, step: u32) -> RngIdentity {
        RngIdentity::from_parts(
            Seed::new(seed),
            BranchSalt::ROOT,
            LogicalId::new(lineage),
            episode,
            step,
        )
    }

    fn first(identity: &RngIdentity, stream: StreamId) -> u64 {
        StepRng::new(&RngAuthority::engine_only(), identity, stream).next_u64()
    }

    #[test]
    fn test_step_rng_same_identity_same_draws() {
        let a = id(42, 0, 1, 5);
        assert_eq!(first(&a, StreamId::DYNAMICS), first(&a, StreamId::DYNAMICS));
    }

    #[test]
    fn test_step_rng_reset_and_first_step_differ() {
        assert_ne!(
            first(&id(42, 0, 1, 0), StreamId::DYNAMICS),
            first(&id(42, 0, 1, 1), StreamId::DYNAMICS)
        );
    }

    #[test]
    fn test_step_rng_streams_lineages_and_seeds_differ() {
        let base = first(&id(42, 0, 0, 1), StreamId::DYNAMICS);
        assert_ne!(base, first(&id(42, 0, 0, 1), StreamId::MAP_GEN));
        assert_ne!(base, first(&id(42, 1, 0, 1), StreamId::DYNAMICS));
        assert_ne!(base, first(&id(43, 0, 0, 1), StreamId::DYNAMICS));
    }

    #[test]
    fn test_obs_rng_repeated_construction_is_identical() {
        let auth = RngAuthority::engine_only();
        let i = id(7, 0, 3, 9);
        let a: Vec<u32> = {
            let mut r = ObsRng::new(&auth, &i);
            (0..9).map(|_| r.next_u32()).collect()
        };
        let mut r = ObsRng::new(&auth, &i);
        let b: Vec<u32> = (0..9).map(|_| r.next_u32()).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn test_obs_rng_matches_obs_noise_stream() {
        let auth = RngAuthority::engine_only();
        let i = id(7, 0, 3, 9);
        assert_eq!(
            ObsRng::new(&auth, &i).next_u64(),
            first(&i, StreamId::OBS_NOISE)
        );
    }
}
