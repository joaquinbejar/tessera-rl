// SPDX-License-Identifier: MIT OR Apache-2.0

//! `SplitMix64`: the example's own sampler and shuffler, seeded, so a run is
//! repeatable on one machine without a `rand` dependency.

/// A `SplitMix64` generator.
pub struct Rng(u64);

impl Rng {
    /// A generator seeded with `seed`.
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A float in `[0, 1)`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "53 random bits into an f64 mantissa"
    )]
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Shuffles `v` (Fisher-Yates).
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = usize::try_from(self.next_u64() % (i as u64 + 1)).unwrap_or(0);
            v.swap(i, j);
        }
    }
}
