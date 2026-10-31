// SPDX-License-Identifier: MIT OR Apache-2.0

//! The draw API over one Philox stream (SPEC 8.1, ADR-0007).
//!
//! The counter is `[logical_id, episode_index, step_index, draw_index]`, in
//! that word order. `draw_index` counts Philox calls within one reset or
//! transition, starts at 0 and is not stored. The words of each block are
//! consumed in order 0 to 3.

use core::num::{NonZeroU32, NonZeroU64};

use crate::philox::{Counter, Key, philox4x32_10};

const WORDS: u8 = 4;
const F32_SCALE: f32 = 1.0 / 16_777_216.0; // 2^-24
const F64_SCALE: f64 = 1.0 / 9_007_199_254_740_992.0; // 2^-53

/// A sequence of draws from one stream at one counter prefix.
///
/// A stream may make at most 2^32 Philox calls per reset or transition.
/// Exceeding that reuses counters: the output stays deterministic, but its
/// blocks are no longer independent. Debug builds assert the limit.
#[derive(Debug, Clone)]
pub struct Draws {
    key: Key,
    prefix: [u32; 3],
    draw_index: u32,
    exhausted: bool,
    block: Counter,
    next_word: u8,
}

impl Draws {
    /// Starts a draw sequence at `draw_index = 0`.
    #[must_use]
    #[inline]
    pub(crate) const fn new(
        key: Key,
        logical_id: u32,
        episode_index: u32,
        step_index: u32,
    ) -> Self {
        Self {
            key,
            prefix: [logical_id, episode_index, step_index],
            draw_index: 0,
            exhausted: false,
            block: [0; 4],
            next_word: WORDS,
        }
    }

    /// Returns whether this sequence has made more than 2^32 Philox calls,
    /// so counters were reused. The engine checks this at the boundary and
    /// poisons the instance.
    #[must_use]
    #[inline]
    pub const fn exhausted(&self) -> bool {
        self.exhausted
    }

    #[inline]
    fn refill(&mut self) {
        debug_assert!(!self.exhausted, "more than 2^32 Philox calls in one phase");
        let [a, b, c] = self.prefix;
        self.block = philox4x32_10([a, b, c, self.draw_index], self.key);
        if let Some(next) = self.draw_index.checked_add(1) {
            self.draw_index = next;
        } else {
            // Documented limit; see the type docs.
            self.exhausted = true;
            self.draw_index = 0;
        }
        self.next_word = 0;
    }

    /// Returns the next 32-bit word.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.next_word >= WORDS {
            self.refill();
        }
        let word = match self.next_word {
            0 => self.block[0],
            1 => self.block[1],
            2 => self.block[2],
            _ => self.block[3],
        };
        self.next_word += 1;
        word
    }

    /// Returns a `u64` from the next two words, little-endian.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let lo = u64::from(self.next_u32());
        let hi = u64::from(self.next_u32());
        lo | (hi << 32)
    }

    /// Returns a uniform integer in `[0, bound)` by Lemire's method with
    /// rejection.
    #[inline]
    pub fn below_u32(&mut self, bound: NonZeroU32) -> u32 {
        let n = bound.get();
        let mut m = u64::from(self.next_u32()) * u64::from(n);
        if low_u32(m) < n {
            // Lemire's threshold, 2^32 mod n, computed modulo 2^32.
            let threshold = n.wrapping_neg() % n;
            while low_u32(m) < threshold {
                m = u64::from(self.next_u32()) * u64::from(n);
            }
        }
        high_u32(m)
    }

    /// Returns a uniform integer in `[0, bound)` by Lemire's method with
    /// rejection.
    #[inline]
    pub fn below_u64(&mut self, bound: NonZeroU64) -> u64 {
        let n = bound.get();
        let mut m = u128::from(self.next_u64()) * u128::from(n);
        if low_u64(m) < n {
            // Lemire's threshold, 2^64 mod n, computed modulo 2^64.
            let threshold = n.wrapping_neg() % n;
            while low_u64(m) < threshold {
                m = u128::from(self.next_u64()) * u128::from(n);
            }
        }
        high_u64(m)
    }

    /// Returns a uniform `f32` in `[0, 1)` from the top 24 bits of a word.
    #[inline]
    #[expect(
        clippy::cast_precision_loss,
        reason = "a 24-bit integer is exact in f32"
    )]
    pub fn unit_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * F32_SCALE
    }

    /// Returns a uniform `f64` in `[0, 1)` from the top 53 bits of a `u64`.
    #[inline]
    #[expect(
        clippy::cast_precision_loss,
        reason = "a 53-bit integer is exact in f64"
    )]
    pub fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * F64_SCALE
    }
}

#[inline]
#[expect(clippy::cast_possible_truncation, reason = "takes the low half")]
const fn low_u32(x: u64) -> u32 {
    x as u32
}

#[inline]
const fn high_u32(x: u64) -> u32 {
    (x >> 32) as u32
}

#[inline]
#[expect(clippy::cast_possible_truncation, reason = "takes the low half")]
const fn low_u64(x: u128) -> u64 {
    x as u64
}

#[inline]
const fn high_u64(x: u128) -> u64 {
    (x >> 64) as u64
}

#[cfg(test)]
mod tests {
    use core::num::{NonZeroU32, NonZeroU64};

    use super::Draws;
    use crate::philox::philox4x32_10;

    #[test]
    fn test_draws_exhausted_after_counter_wrap() {
        let mut d = Draws::new([1, 2], 0, 0, 0);
        assert!(!d.exhausted());
        d.draw_index = u32::MAX;
        d.next_u32();
        assert!(d.exhausted());
    }

    #[test]
    fn test_draws_word_order_follows_counter_layout() {
        let key = [11, 22];
        let mut d = Draws::new(key, 1, 2, 3);
        let b0 = philox4x32_10([1, 2, 3, 0], key);
        let b1 = philox4x32_10([1, 2, 3, 1], key);
        let got: Vec<u32> = (0..8).map(|_| d.next_u32()).collect();
        assert_eq!(got, [b0, b1].concat());
    }

    #[test]
    fn test_draws_next_u64_is_little_endian() {
        let key = [5, 6];
        let b0 = philox4x32_10([0, 0, 0, 0], key);
        let mut d = Draws::new(key, 0, 0, 0);
        assert_eq!(d.next_u64(), u64::from(b0[0]) | (u64::from(b0[1]) << 32));
    }

    #[test]
    fn test_draws_below_u32_stays_in_range() {
        let mut d = Draws::new([1, 2], 0, 0, 0);
        for bound in [1, 2, 3, 7, 1000, u32::MAX] {
            let b = NonZeroU32::new(bound).unwrap();
            for _ in 0..1000 {
                assert!(d.below_u32(b) < bound);
            }
        }
    }

    #[test]
    fn test_draws_below_u64_stays_in_range() {
        let mut d = Draws::new([3, 4], 0, 0, 0);
        for bound in [1, 3, 1 << 40, u64::MAX] {
            let b = NonZeroU64::new(bound).unwrap();
            for _ in 0..1000 {
                assert!(d.below_u64(b) < bound);
            }
        }
    }

    #[test]
    fn test_draws_below_u32_rejection_hits_threshold() {
        // bound 3: threshold = 2^32 mod 3 = 1; only low word 0 is rejected.
        // A uniform check over many draws keeps every residue populated.
        let mut d = Draws::new([9, 9], 0, 0, 0);
        let mut counts = [0u32; 3];
        for _ in 0..30_000 {
            counts[d.below_u32(NonZeroU32::new(3).unwrap()) as usize] += 1;
        }
        assert!(counts.iter().all(|&c| c > 9_000));
    }

    #[test]
    fn test_draws_unit_floats_in_half_open_unit_interval() {
        let mut d = Draws::new([7, 8], 0, 0, 0);
        for _ in 0..10_000 {
            let x = d.unit_f32();
            let y = d.unit_f64();
            assert!((0.0..1.0).contains(&x));
            assert!((0.0..1.0).contains(&y));
        }
    }

    #[test]
    #[expect(clippy::cast_precision_loss, reason = "the value has 53 bits")]
    fn test_draws_unit_f64_uses_top_53_bits() {
        let key = [1, 1];
        let b0 = philox4x32_10([0, 0, 0, 0], key);
        let raw = u64::from(b0[0]) | (u64::from(b0[1]) << 32);
        let mut d = Draws::new(key, 0, 0, 0);
        assert_eq!(
            d.unit_f64().to_bits(),
            ((raw >> 11) as f64 / 9_007_199_254_740_992.0).to_bits()
        );
    }
}
