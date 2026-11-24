// SPDX-License-Identifier: MIT OR Apache-2.0

//! The one float operation of the environment (`docs/reviews/381/DESIGN.md`,
//! section 2): an `i64` equity change converted to `f32` once, by IEEE
//! round to nearest, then multiplied by `2^-reward_shift`, which is exact.
//! Both steps are deterministic on every platform. This module is the only
//! one that does not deny `clippy::float_arithmetic`.

/// `diff x 2^-shift` as `f32`; `shift` is at most 32 (the config range).
#[must_use]
#[inline]
#[expect(
    clippy::cast_precision_loss,
    reason = "the single, documented i64 to f32 rounding (round to nearest)"
)]
pub fn scaled(diff: i64, shift: u32) -> f32 {
    // The biased exponent of 2^-shift: 127 - shift, in 95..=127, normal.
    let scale = f32::from_bits((127 - shift.min(32)) << 23);
    diff as f32 * scale
}

#[cfg(test)]
mod tests {
    use super::scaled;

    #[test]
    fn test_scaled_divides_by_power_of_two_exactly() {
        assert_eq!(scaled(12, 0).to_bits(), 12.0_f32.to_bits());
        assert_eq!(scaled(12, 2).to_bits(), 3.0_f32.to_bits());
        assert_eq!(scaled(-5, 1).to_bits(), (-2.5_f32).to_bits());
        assert_eq!(scaled(1 << 32, 32).to_bits(), 1.0_f32.to_bits());
        assert_eq!(scaled(0, 7).to_bits(), 0.0_f32.to_bits());
    }
}
