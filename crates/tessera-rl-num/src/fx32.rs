// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Fx32`: Q16.16 fixed point (SPEC 8.2, ADR-0008).

crate::fixed::fixed_point! {
    /// A Q16.16 fixed-point number: 16 integer bits (with sign) and 16
    /// fractional bits. See the crate docs of `fixed` for the arithmetic
    /// rules.
    Fx32, i32, i64, 16, scale = 65_536.0, bound = 2_147_483_648.0
}

impl Fx32 {
    /// Converts to `f64`. Always exact.
    #[must_use]
    #[inline]
    pub fn to_f64(self) -> f64 {
        f64::from(self.0) / Self::SCALE
    }
}

#[cfg(test)]
mod tests {
    use super::Fx32;

    fn fx(x: f64) -> Fx32 {
        Fx32::checked_from_f64(x).unwrap()
    }

    #[test]
    fn test_fx32_mul_reference_vectors_match() {
        assert_eq!(fx(1.5).checked_mul(fx(2.25)), Some(fx(3.375)));
        // -2^-16 * 0.5 = -2^-17 rounds toward negative infinity.
        assert_eq!(
            Fx32::from_raw(-1).checked_mul(fx(0.5)),
            Some(Fx32::from_raw(-1))
        );
        assert_eq!(Fx32::from_raw(1).checked_mul(fx(0.5)), Some(Fx32::ZERO));
    }

    #[test]
    fn test_fx32_mul_overflow_returns_none_or_saturates() {
        assert_eq!(fx(200.0).checked_mul(fx(200.0)), None);
        assert_eq!(fx(200.0).saturating_mul(fx(200.0)), Fx32::MAX);
        assert_eq!(fx(-200.0).saturating_mul(fx(200.0)), Fx32::MIN);
        // 40000 * 2^16 modulo 2^32, read as i32: 2_621_440_000 - 2^32.
        assert_eq!(fx(200.0).wrapping_mul(fx(200.0)).to_raw(), -1_673_527_296);
    }

    #[test]
    fn test_fx32_div_truncates_toward_zero() {
        assert_eq!(fx(1.0).checked_div(fx(3.0)), Some(Fx32::from_raw(21_845)));
        assert_eq!(fx(-1.0).checked_div(fx(3.0)), Some(Fx32::from_raw(-21_845)));
        assert_eq!(Fx32::from_raw(-1).checked_div(fx(3.0)), Some(Fx32::ZERO));
    }

    #[test]
    fn test_fx32_div_by_zero_or_overflow_returns_none() {
        assert_eq!(fx(1.0).checked_div(Fx32::ZERO), None);
        assert_eq!(Fx32::MIN.checked_div(fx(-1.0)), None);
    }

    #[test]
    fn test_fx32_add_sub_overflow_behaviour() {
        assert_eq!(Fx32::MAX.checked_add(Fx32::EPSILON), None);
        assert_eq!(Fx32::MAX.saturating_add(Fx32::EPSILON), Fx32::MAX);
        assert_eq!(Fx32::MAX.wrapping_add(Fx32::EPSILON), Fx32::MIN);
        assert_eq!(Fx32::MIN.checked_sub(Fx32::EPSILON), None);
        assert_eq!(Fx32::MIN.checked_neg(), None);
    }

    #[test]
    fn test_fx32_from_f64_rounds_half_to_even() {
        let ulp = 1.0 / 65_536.0;
        assert_eq!(fx(0.5 * ulp).to_raw(), 0);
        assert_eq!(fx(1.5 * ulp).to_raw(), 2);
        assert_eq!(fx(2.5 * ulp).to_raw(), 2);
        assert_eq!(fx(-0.5 * ulp).to_raw(), 0);
        assert_eq!(fx(-1.5 * ulp).to_raw(), -2);
    }

    #[test]
    fn test_fx32_from_f64_bounds_and_nan() {
        assert_eq!(Fx32::checked_from_f64(-32_768.0), Some(Fx32::MIN));
        assert_eq!(Fx32::checked_from_f64(32_768.0), None);
        assert_eq!(Fx32::saturating_from_f64(32_768.0), Fx32::MAX);
        assert_eq!(Fx32::saturating_from_f64(-1e10), Fx32::MIN);
        assert_eq!(Fx32::checked_from_f64(f64::NAN), None);
        assert_eq!(Fx32::saturating_from_f64(f64::NAN), Fx32::ZERO);
        assert_eq!(Fx32::saturating_from_f64(f64::INFINITY), Fx32::MAX);
        assert_eq!(Fx32::saturating_from_f32(f32::NEG_INFINITY), Fx32::MIN);
    }

    #[test]
    fn test_fx32_sub_wrapping_and_saturating() {
        assert_eq!(fx(1.0).wrapping_sub(fx(2.5)), fx(-1.5));
        assert_eq!(Fx32::MIN.wrapping_sub(Fx32::EPSILON), Fx32::MAX);
        assert_eq!(fx(1.0).saturating_sub(fx(2.5)), fx(-1.5));
        assert_eq!(Fx32::MIN.saturating_sub(Fx32::EPSILON), Fx32::MIN);
        assert_eq!(fx(1.0).checked_sub(fx(2.5)), Some(fx(-1.5)));
        assert_eq!(fx(1.5).checked_add(fx(-0.25)), Some(fx(1.25)));
        assert_eq!(fx(1.5).checked_neg(), Some(fx(-1.5)));
    }

    #[test]
    fn test_fx32_from_f32_rounds_without_saturating() {
        assert_eq!(Fx32::checked_from_f32(1.25), Some(fx(1.25)));
        assert_eq!(Fx32::checked_from_f32(-3.5), Some(fx(-3.5)));
        assert_eq!(Fx32::saturating_from_f32(0.1).to_raw(), 6_554);
        assert_eq!(Fx32::checked_from_f32(f32::NAN), None);
    }

    #[test]
    fn test_fx32_to_f64_is_exact() {
        assert_eq!(Fx32::MAX.to_f64(), 32_768.0 - 1.0 / 65_536.0);
        assert_eq!(Fx32::MIN.to_f64(), -32_768.0);
        assert_eq!(Fx32::EPSILON.to_f64(), 1.0 / 65_536.0);
    }

    #[test]
    fn test_fx32_from_int_range() {
        assert_eq!(Fx32::checked_from_int(3), Some(fx(3.0)));
        assert_eq!(Fx32::checked_from_int(32_767), Some(fx(32_767.0)));
        assert_eq!(Fx32::checked_from_int(32_768), None);
        assert_eq!(Fx32::checked_from_int(-32_768), Some(Fx32::MIN));
        assert_eq!(Fx32::checked_from_int(-32_769), None);
        assert_eq!(Fx32::checked_from_int(-7), Some(fx(-7.0)));
    }
}
