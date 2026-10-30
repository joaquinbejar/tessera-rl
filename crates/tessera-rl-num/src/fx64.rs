// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Fx64`: Q32.32 fixed point (SPEC 8.2, ADR-0008).

crate::fixed::fixed_point! {
    /// A Q32.32 fixed-point number: 32 integer bits (with sign) and 32
    /// fractional bits. See the crate docs of `fixed` for the arithmetic
    /// rules.
    Fx64, i64, i128, 32, scale = 4_294_967_296.0, bound = 9_223_372_036_854_775_808.0
}

impl Fx64 {
    /// Converts to `f64`. Exact whenever the raw value fits in 53 bits;
    /// otherwise rounded to nearest, ties to even.
    #[must_use]
    #[inline]
    #[expect(clippy::cast_precision_loss, reason = "documented rounding")]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE
    }
}

#[cfg(test)]
mod tests {
    use super::Fx64;

    fn fx(x: f64) -> Fx64 {
        Fx64::checked_from_f64(x).unwrap()
    }

    #[test]
    fn test_fx64_mul_reference_vectors_match() {
        assert_eq!(fx(1.5).checked_mul(fx(2.25)), Some(fx(3.375)));
        assert_eq!(
            Fx64::from_raw(-1).checked_mul(fx(0.5)),
            Some(Fx64::from_raw(-1))
        );
        assert_eq!(Fx64::from_raw(1).checked_mul(fx(0.5)), Some(Fx64::ZERO));
    }

    #[test]
    fn test_fx64_mul_overflow_returns_none_or_saturates() {
        assert_eq!(fx(65_536.0).checked_mul(fx(65_536.0)), None);
        assert_eq!(fx(65_536.0).saturating_mul(fx(65_536.0)), Fx64::MAX);
        assert_eq!(fx(-65_536.0).saturating_mul(fx(65_536.0)), Fx64::MIN);
    }

    #[test]
    fn test_fx64_wrapping_mul_and_add_sub_rules() {
        // 2^16 * 2^16 = 2^32, raw 2^64 wraps to 0.
        assert_eq!(fx(65_536.0).wrapping_mul(fx(65_536.0)), Fx64::ZERO);
        assert_eq!(fx(-0.5).wrapping_mul(fx(3.0)), fx(-1.5));
        assert_eq!(Fx64::MAX.checked_add(Fx64::EPSILON), None);
        assert_eq!(Fx64::MAX.wrapping_add(Fx64::EPSILON), Fx64::MIN);
        assert_eq!(Fx64::MAX.saturating_add(Fx64::EPSILON), Fx64::MAX);
        assert_eq!(Fx64::MIN.checked_sub(Fx64::EPSILON), None);
        assert_eq!(Fx64::MIN.wrapping_sub(Fx64::EPSILON), Fx64::MAX);
        assert_eq!(Fx64::MIN.saturating_sub(Fx64::EPSILON), Fx64::MIN);
        assert_eq!(Fx64::MIN.checked_neg(), None);
    }

    #[test]
    fn test_fx64_from_int_and_f32() {
        assert_eq!(Fx64::checked_from_int(-2_147_483_648), Some(Fx64::MIN));
        assert_eq!(
            Fx64::checked_from_int(2_147_483_647),
            Some(fx(2_147_483_647.0))
        );
        assert_eq!(Fx64::checked_from_int(2_147_483_648), None);
        assert_eq!(Fx64::checked_from_int(-2_147_483_649), None);
        assert_eq!(Fx64::checked_from_f32(-0.75), Some(fx(-0.75)));
        assert_eq!(Fx64::saturating_from_f32(f32::INFINITY), Fx64::MAX);
    }

    #[test]
    fn test_fx64_div_truncates_toward_zero() {
        assert_eq!(
            fx(1.0).checked_div(fx(3.0)),
            Some(Fx64::from_raw(1_431_655_765))
        );
        assert_eq!(
            fx(-1.0).checked_div(fx(3.0)),
            Some(Fx64::from_raw(-1_431_655_765))
        );
        assert_eq!(Fx64::from_raw(-1).checked_div(fx(3.0)), Some(Fx64::ZERO));
        assert_eq!(fx(1.0).checked_div(Fx64::ZERO), None);
        assert_eq!(Fx64::MIN.checked_div(fx(-1.0)), None);
    }

    #[test]
    fn test_fx64_from_f64_rounds_half_to_even_and_saturates() {
        let ulp = 1.0 / 4_294_967_296.0;
        assert_eq!(fx(0.5 * ulp).to_raw(), 0);
        assert_eq!(fx(1.5 * ulp).to_raw(), 2);
        assert_eq!(fx(2.5 * ulp).to_raw(), 2);
        assert_eq!(Fx64::checked_from_f64(-2_147_483_648.0), Some(Fx64::MIN));
        assert_eq!(Fx64::checked_from_f64(2_147_483_648.0), None);
        assert_eq!(Fx64::saturating_from_f64(2_147_483_648.0), Fx64::MAX);
        assert_eq!(Fx64::saturating_from_f64(f64::NAN), Fx64::ZERO);
    }

    #[test]
    fn test_fx64_to_f64_exact_when_representable() {
        assert_eq!(fx(3.375).to_f64(), 3.375);
        assert_eq!(Fx64::MIN.to_f64(), -2_147_483_648.0);
        // MAX has 63 significant bits: rounds to nearest, 2^31.
        assert_eq!(Fx64::MAX.to_f64(), 2_147_483_648.0);
    }
}
