// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `splitmix64` mixer used by key and salt derivation (ADR-0007).
//!
//! This is one output of Vigna's `SplitMix64` generator for state `x`: add the
//! golden-ratio increment, then apply the finalizer. It is a bijection on
//! `u64`.

const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_0: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_1: u64 = 0x94D0_49BB_1331_11EB;

/// Mixes `x` with the `SplitMix64` increment and finalizer.
///
/// Arithmetic is modulo 2^64 by definition of the mixer.
#[must_use]
#[inline]
pub const fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(GAMMA);
    z = (z ^ (z >> 30)).wrapping_mul(MIX_0);
    z = (z ^ (z >> 27)).wrapping_mul(MIX_1);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::splitmix64;

    // First outputs of Vigna's reference splitmix64.c seeded with 0: each
    // output is splitmix64 of the previous state, and the state advances by
    // the increment.
    #[test]
    fn test_splitmix64_reference_sequence_matches_vigna() {
        let gamma = 0x9E37_79B9_7F4A_7C15_u64;
        assert_eq!(splitmix64(0), 0xe220_a839_7b1d_cdaf);
        assert_eq!(splitmix64(gamma), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(splitmix64(gamma.wrapping_mul(2)), 0x06c4_5d18_8009_454f);
    }
}
