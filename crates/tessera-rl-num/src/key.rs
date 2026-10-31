// SPDX-License-Identifier: MIT OR Apache-2.0

//! Key and branch salt derivation (ADR-0007).
//!
//! A Philox key is the `u64` from [`derive_key`] split little-endian into
//! two words: `[lo, hi]`.

use crate::identity::{BranchSalt, Seed, StreamId};
use crate::philox::Key;
use crate::splitmix::splitmix64;

/// Domain separation constant for branch salts: ASCII `TSRSALT1`.
pub const SALT_DOMAIN: u64 = 0x5453_5253_414C_5431;

/// Derives the 64-bit stream key:
/// `splitmix64(seed ^ splitmix64(branch_salt ^ splitmix64(stream_id)))`.
#[must_use]
#[inline]
pub const fn derive_key(seed: Seed, stream: StreamId, branch_salt: BranchSalt) -> u64 {
    splitmix64(seed.get() ^ splitmix64(branch_salt.get() ^ splitmix64(stream.get() as u64)))
}

/// Splits a 64-bit key into Philox key words `[lo, hi]`.
#[must_use]
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "splits a u64 into its two 32-bit halves"
)]
pub const fn key_words(key: u64) -> Key {
    [key as u32, (key >> 32) as u32]
}

/// Derives the salt of a `fork_branched` destination:
/// `splitmix64(splitmix64(splitmix64(SALT_DOMAIN ^ parent) ^ branch_counter) ^ label)`.
///
/// Inputs are absorbed one at a time at full width, never packed. Checking
/// the counter increment and label uniqueness is the engine's job.
#[must_use]
#[inline]
pub const fn derive_branch_salt(parent: BranchSalt, branch_counter: u64, label: u64) -> BranchSalt {
    BranchSalt::new(splitmix64(
        splitmix64(splitmix64(SALT_DOMAIN ^ parent.get()) ^ branch_counter) ^ label,
    ))
}

#[cfg(test)]
mod tests {
    use super::{derive_branch_salt, derive_key, key_words};
    use crate::identity::{BranchSalt, Seed, StreamId};

    fn key(seed: u64, stream: u32, salt: u64) -> u64 {
        let stream = match stream {
            0 => StreamId::DYNAMICS,
            3 => StreamId::OBS_NOISE,
            n => StreamId::env(n - 16).unwrap(),
        };
        derive_key(Seed::new(seed), stream, BranchSalt::new(salt))
    }

    // Reference vectors, computed independently from the ADR-0007 formula.
    #[test]
    fn test_derive_key_reference_vectors_match() {
        assert_eq!(key(0, 0, 0), 0x2382_75bc_38fc_be91);
        assert_eq!(key(1, 0, 0), 0x2f32_a784_96c6_7c60);
        assert_eq!(key(0, 3, 0), 0x546f_3606_1b6b_1313);
        assert_eq!(
            key(0xDEAD_BEEF, 16, 0x0123_4567_89AB_CDEF),
            0x5789_077f_8553_bd97
        );
    }

    #[test]
    fn test_derive_key_max_seed_and_salt_match() {
        let k = derive_key(
            Seed::new(u64::MAX),
            StreamId::env(u32::MAX - 16).unwrap(),
            BranchSalt::new(u64::MAX),
        );
        // The stream id widens to 0x0000_0000_FFFF_FFFF, not u64::MAX.
        assert_eq!(k, 0xf153_f841_731e_c67a);
    }

    #[test]
    fn test_derive_branch_salt_reference_vectors_match() {
        let s = |p, c, l| derive_branch_salt(BranchSalt::new(p), c, l).get();
        assert_eq!(s(0, 0, 0), 0x9d9d_2974_8059_342d);
        assert_eq!(s(0, 1, 0), 0x91fc_1c21_aa47_2b96);
        assert_eq!(s(0, 0, 1), 0x9f5e_9d7a_3b8a_77f9);
        assert_eq!(s(0x0123_4567_89AB_CDEF, 42, 7), 0x3aed_24af_eaae_a58c);
        assert_eq!(s(u64::MAX, u64::MAX, u64::MAX), 0x5fb1_18d4_a0b5_fcaf);
    }

    #[test]
    fn test_key_words_splits_little_endian() {
        assert_eq!(key_words(0x0123_4567_89AB_CDEF), [0x89AB_CDEF, 0x0123_4567]);
    }
}
