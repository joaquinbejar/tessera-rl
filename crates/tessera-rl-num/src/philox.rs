// SPDX-License-Identifier: MIT OR Apache-2.0

//! Philox 4x32-10 block function (SPEC 8.1, ADR-0007).
//!
//! Salmon, Moraes, Dror and Shaw, "Parallel random numbers: as easy as
//! 1, 2, 3", SC 2011. Constants and round structure follow the Random123
//! reference implementation, and the known-answer tests below use its
//! published vectors.
//!
//! Tessera's counter words are `[logical_id, episode_index, step_index,
//! draw_index]` in that order, and the key words are the stream key split
//! little-endian into `[lo, hi]` (see `crate::key`).

const MUL_0: u32 = 0xD251_1F53;
const MUL_1: u32 = 0xCD9E_8D57;
const WEYL_0: u32 = 0x9E37_79B9;
const WEYL_1: u32 = 0xBB67_AE85;
const ROUNDS: usize = 10;

/// A Philox counter: four 32-bit words.
pub type Counter = [u32; 4];

/// A Philox key: two 32-bit words.
pub type Key = [u32; 2];

#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "splits a 64-bit product into its two 32-bit halves"
)]
const fn mul_hi_lo(a: u32, b: u32) -> (u32, u32) {
    let product = (a as u64) * (b as u64);
    ((product >> 32) as u32, product as u32)
}

#[inline]
const fn round(ctr: Counter, key: Key) -> Counter {
    let (hi0, lo0) = mul_hi_lo(MUL_0, ctr[0]);
    let (hi1, lo1) = mul_hi_lo(MUL_1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// Computes one Philox 4x32-10 block: ten rounds over `ctr` under `key`.
///
/// The key schedule adds the Weyl constants modulo 2^32 between rounds, as
/// the algorithm defines.
#[must_use]
#[inline]
pub const fn philox4x32_10(ctr: Counter, key: Key) -> Counter {
    let mut ctr = ctr;
    let mut key = key;
    let mut i = 0;
    while i < ROUNDS {
        if i > 0 {
            // Key schedule of the algorithm: addition modulo 2^32.
            key = [key[0].wrapping_add(WEYL_0), key[1].wrapping_add(WEYL_1)];
        }
        ctr = round(ctr, key);
        i += 1;
    }
    ctr
}

#[cfg(test)]
mod tests {
    use super::philox4x32_10;

    // Random123 kat_vectors, philox4x32_10.
    #[test]
    fn test_philox4x32_10_zero_inputs_match_random123() {
        assert_eq!(
            philox4x32_10([0; 4], [0; 2]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
    }

    #[test]
    fn test_philox4x32_10_all_ones_match_random123() {
        assert_eq!(
            philox4x32_10([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
    }

    #[test]
    fn test_philox4x32_10_pi_digits_match_random123() {
        assert_eq!(
            philox4x32_10(
                [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
                [0xa409_3822, 0x299f_31d0]
            ),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }
}
