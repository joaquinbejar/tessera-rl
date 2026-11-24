// SPDX-License-Identifier: MIT OR Apache-2.0

//! `StepRng` and `ObsRng`: the only randomness environment code may use
//! (SPEC 8.1, ADR-0007).

use core::num::{NonZeroU32, NonZeroU64};

use crate::draw::Draws;
use crate::identity::{RngIdentity, StreamId};
#[cfg(test)]
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

/// The eagerly keyed sequence: the reference the deferred one must equal.
#[cfg(test)]
fn draws(identity: &RngIdentity, stream: StreamId) -> Draws {
    let key = derive_key(identity.seed(), stream, identity.branch_salt());
    Draws::new(
        key_words(key),
        identity.logical_id().get(),
        identity.episode_index(),
        identity.step_index(),
    )
}

/// The sequence for `identity` on `stream`, with its key derived on the
/// first draw (see `Draws::new_deferred`).
#[inline]
const fn deferred(identity: &RngIdentity, stream: StreamId) -> Draws {
    Draws::new_deferred(
        identity.seed(),
        stream,
        identity.branch_salt(),
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
            draws: deferred(identity, stream),
        }
    }

    /// Engine-only: the RNG of an instance's league assignment draw at the
    /// identity's episode (ADR-0088 item 3.1): the `LEAGUE` stream under
    /// the league salt, at step 0, so the draw is a function of the
    /// instance (seed, salt, logical id) and its episode counter only.
    #[doc(hidden)]
    #[must_use]
    #[inline]
    pub fn league(_authority: &RngAuthority, identity: &RngIdentity) -> Self {
        Self {
            draws: Draws::new_deferred(
                identity.seed(),
                StreamId::LEAGUE,
                crate::key::league_salt(identity.branch_salt()),
                identity.logical_id().get(),
                identity.episode_index(),
                0,
            ),
        }
    }

    /// Engine-only: the RNG of heuristic slot `slot` at the identity's
    /// current episode and step (ADR-0079 item 3.3): the `HEURISTIC`
    /// stream under a per-slot salt, so it shares no key with the
    /// environment's streams nor with another slot.
    #[doc(hidden)]
    #[must_use]
    #[inline]
    pub fn heuristic(_authority: &RngAuthority, identity: &RngIdentity, slot: u16) -> Self {
        Self {
            draws: Draws::new_deferred(
                identity.seed(),
                StreamId::HEURISTIC,
                crate::key::heuristic_salt(identity.branch_salt(), slot),
                identity.logical_id().get(),
                identity.episode_index(),
                identity.step_index(),
            ),
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
            draws: deferred(identity, StreamId::OBS_NOISE),
        }
    }

    forward_draws!();
}

#[cfg(test)]
mod tests {
    use super::{ObsRng, RngAuthority, StepRng};
    use crate::identity::{BranchSalt, LogicalId, RngIdentity, Seed, StreamId};
    use core::num::{NonZeroU32, NonZeroU64};

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
    fn test_step_rng_heuristic_slots_and_dynamics_differ() {
        let auth = RngAuthority::engine_only();
        let i = id(42, 0, 0, 1);
        let h = |slot| StepRng::heuristic(&auth, &i, slot).next_u64();
        assert_ne!(h(0), first(&i, StreamId::DYNAMICS));
        assert_ne!(h(0), first(&i, StreamId::HEURISTIC));
        assert_ne!(h(0), h(1));
        assert_eq!(h(3), h(3));
        // Exactly the `HEURISTIC` stream under the slot's heuristic salt.
        let salted = RngIdentity::from_parts(
            Seed::new(42),
            crate::key::heuristic_salt(BranchSalt::ROOT, 0),
            LogicalId::new(0),
            0,
            1,
        );
        assert_eq!(h(0), first(&salted, StreamId::HEURISTIC));
        assert_ne!(h(0), first(&salted, StreamId::DYNAMICS));
        // Keyed by the step index too.
        assert_ne!(
            h(0),
            StepRng::heuristic(&auth, &id(42, 0, 0, 2), 0).next_u64()
        );
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

    /// The deferred sequence equals one keyed eagerly at construction, under
    /// the root branch salt and a non-root one (a forked branch): same
    /// words, in order, for every stream, across a Philox block boundary.
    #[test]
    fn test_step_rng_lazy_sequence_equals_eager_derivation() {
        let auth = RngAuthority::engine_only();
        let b32 = NonZeroU32::new(7).unwrap_or(NonZeroU32::MIN);
        let b64 = NonZeroU64::new(1_000_003).unwrap_or(NonZeroU64::MIN);
        for stream in [
            StreamId::DYNAMICS,
            StreamId::MAP_GEN,
            StreamId::OPPONENT,
            StreamId::OBS_NOISE,
            StreamId::SEARCH,
        ] {
            for salt in [BranchSalt::ROOT, BranchSalt::new(0x9e37_79b9_7f4a_7c15)] {
                let i = RngIdentity::from_parts(Seed::new(11), salt, LogicalId::new(2), 4, 6);
                // Each draw method first, so the lazy derivation happens on it,
                // then a mix across a Philox block boundary.
                for first in 0..6 {
                    let mut eager = super::draws(&i, stream);
                    let mut lazy = StepRng::new(&auth, &i, stream);
                    for k in 0..14 {
                        match (first + k) % 6 {
                            0 => assert_eq!(lazy.next_u32(), eager.next_u32()),
                            1 => assert_eq!(lazy.next_u64(), eager.next_u64()),
                            2 => assert_eq!(lazy.below_u32(b32), eager.below_u32(b32)),
                            3 => assert_eq!(lazy.below_u64(b64), eager.below_u64(b64)),
                            4 => assert_eq!(lazy.unit_f32().to_bits(), eager.unit_f32().to_bits()),
                            _ => assert_eq!(lazy.unit_f64().to_bits(), eager.unit_f64().to_bits()),
                        }
                    }
                    assert_eq!(lazy.exhausted(), eager.exhausted());
                }
            }
        }
    }

    /// `exhausted()` reports the derived sequence's state once drawn from:
    /// one Philox call past `draw_index = u32::MAX` turns it true, for both
    /// RNG kinds.
    #[test]
    fn test_rng_exhausted_turns_true_after_the_last_counter() {
        let auth = RngAuthority::engine_only();
        let i = id(5, 1, 2, 3);
        let mut s = StepRng::new(&auth, &i, StreamId::DYNAMICS);
        s.draws.set_draw_index(u32::MAX);
        assert!(!s.exhausted());
        let _ = s.next_u32();
        assert!(s.exhausted());
        let mut o = ObsRng::new(&auth, &i);
        o.draws.set_draw_index(u32::MAX);
        assert!(!o.exhausted());
        let _ = o.next_u32();
        assert!(o.exhausted());
    }

    #[test]
    fn test_step_rng_unused_is_not_exhausted() {
        let r = StepRng::new(
            &RngAuthority::engine_only(),
            &id(1, 0, 0, 0),
            StreamId::DYNAMICS,
        );
        assert!(!r.exhausted());
        let o = ObsRng::new(&RngAuthority::engine_only(), &id(1, 0, 0, 0));
        assert!(!o.exhausted());
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
