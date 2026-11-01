// SPDX-License-Identifier: MIT OR Apache-2.0

//! `SplitMix64`: the small, fully specified generator of the reference tools
//! (openings, tie breaks, random players). Not the engine's RNG.

/// `SplitMix64` state.
#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// A generator from `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in `[0, n)` (`n > 0`), by multiply-shift.
    pub fn below(&mut self, n: usize) -> usize {
        let n64 = n as u64;
        usize::try_from((u128::from(self.next_u64()) * u128::from(n64)) >> 64).unwrap_or(0)
    }
}
