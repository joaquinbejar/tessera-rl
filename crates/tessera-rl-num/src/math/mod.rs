// SPDX-License-Identifier: MIT OR Apache-2.0

//! Frozen transcendental functions for simulation code (SPEC 8.2, ADR-0008).
//!
//! These are the only transcendental functions environment code may call;
//! the `std` methods are banned by `clippy.toml`. Results are bit-identical
//! on every supported platform, provided the floating-point state is
//! canonical (see `crate::fp`).
//!
//! # Versioning
//!
//! [`MATH_VERSION`] identifies the exact results of every function here.
//! Environment versions record the math version they use.
//!
//! - Changing the result of any existing function for any input requires a
//!   new math version; the old implementation stays available for the
//!   environment versions that use it.
//! - Adding a new function does not change any existing result, so it does
//!   not change the math version.
//!
//! Version 1 is `libm` 0.2.16, vendored in `vendor` verbatim except for the
//! listed adaptations: `force_eval!` uses `black_box`, `i!` uses checked
//! indexing, `select_implementation!` expands to nothing (x87 only), and
//! `scalbn` is musl's algorithm, checked against `libm` 0.2.16.
//!
//! Not yet available (added later without a version change): `asin`,
//! `acos`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `exp2`, `exp_m1`,
//! `log2`, `log10`, `ln_1p`, `cbrt`, `pow`, `gamma` and `ln_gamma`.

pub mod lane;
mod vendor;

/// The current math version.
pub const MATH_VERSION: u32 = 1;

/// Sine of `x` (radians).
#[must_use]
#[inline]
pub fn sin(x: f64) -> f64 {
    vendor::sin(x)
}

/// Cosine of `x` (radians).
#[must_use]
#[inline]
pub fn cos(x: f64) -> f64 {
    vendor::cos(x)
}

/// Tangent of `x` (radians).
#[must_use]
#[inline]
pub fn tan(x: f64) -> f64 {
    vendor::tan(x)
}

/// Arctangent of `x`, in `[-pi/2, pi/2]`.
#[must_use]
#[inline]
pub fn atan(x: f64) -> f64 {
    vendor::atan(x)
}

/// Four-quadrant arctangent of `y / x`, in `[-pi, pi]`.
#[must_use]
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    vendor::atan2(y, x)
}

/// `e^x`.
#[must_use]
#[inline]
pub fn exp(x: f64) -> f64 {
    vendor::exp(x)
}

/// Natural logarithm of `x`.
#[must_use]
#[inline]
pub fn ln(x: f64) -> f64 {
    vendor::log(x)
}

/// Hyperbolic tangent of `x`.
#[must_use]
#[inline]
pub fn tanh(x: f64) -> f64 {
    vendor::tanh(x)
}

/// `sqrt(x^2 + y^2)` without undue overflow or underflow.
#[must_use]
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    vendor::hypot(x, y)
}

#[cfg(test)]
mod vectors;

#[cfg(test)]
mod tests {
    use super::vectors::{ATAN, ATAN2, COS, EXP, HYPOT, INPUTS, LN, SIN, TAN, TANH};
    use super::{atan, atan2, cos, exp, hypot, ln, sin, tan, tanh};

    // NaN payloads are not part of the contract (ADR-0008); every other
    // result must match the libm 0.2.16 oracle bit for bit.
    fn same(got: f64, want: u64) -> bool {
        let want = f64::from_bits(want);
        (got.is_nan() && want.is_nan()) || got.to_bits() == want.to_bits()
    }

    fn check_unary(name: &str, f: fn(f64) -> f64, expected: &[u64]) {
        assert_eq!(INPUTS.len(), expected.len());
        for (&x, &want) in INPUTS.iter().zip(expected) {
            let x = f64::from_bits(x);
            let got = f(x);
            assert!(
                same(got, want),
                "{name}({x:e}) = {got:e}, want {:e}",
                f64::from_bits(want)
            );
        }
    }

    fn check_binary(name: &str, f: fn(f64, f64) -> f64, expected: &[(u64, u64, u64)]) {
        for &(a, b, want) in expected {
            let (a, b) = (f64::from_bits(a), f64::from_bits(b));
            let got = f(a, b);
            assert!(
                same(got, want),
                "{name}({a:e}, {b:e}) = {got:e}, want {:e}",
                f64::from_bits(want)
            );
        }
    }

    #[test]
    fn test_math_sin_matches_libm_0_2_16() {
        check_unary("sin", sin, SIN);
    }

    #[test]
    fn test_math_cos_matches_libm_0_2_16() {
        check_unary("cos", cos, COS);
    }

    #[test]
    fn test_math_tan_matches_libm_0_2_16() {
        check_unary("tan", tan, TAN);
    }

    #[test]
    fn test_math_atan_matches_libm_0_2_16() {
        check_unary("atan", atan, ATAN);
    }

    #[test]
    fn test_math_exp_matches_libm_0_2_16() {
        check_unary("exp", exp, EXP);
    }

    #[test]
    fn test_math_ln_matches_libm_0_2_16() {
        check_unary("ln", ln, LN);
    }

    #[test]
    fn test_math_tanh_matches_libm_0_2_16() {
        check_unary("tanh", tanh, TANH);
    }

    #[test]
    fn test_math_atan2_matches_libm_0_2_16() {
        check_binary("atan2", atan2, ATAN2);
    }

    #[test]
    fn test_math_hypot_matches_libm_0_2_16() {
        check_binary("hypot", hypot, HYPOT);
    }

    #[test]
    fn test_math_signed_zero_edges_are_preserved() {
        assert_eq!(sin(-0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(tan(-0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(atan(-0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(tanh(-0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(atan2(-0.0, -1.0), -core::f64::consts::PI);
        assert_eq!(atan2(0.0, -1.0), core::f64::consts::PI);
        assert_eq!(ln(0.0), f64::NEG_INFINITY);
        assert!(ln(-1.0).is_nan());
        assert_eq!(exp(f64::NEG_INFINITY).to_bits(), 0);
        assert_eq!(hypot(f64::INFINITY, f64::NAN), f64::INFINITY);
    }
}
