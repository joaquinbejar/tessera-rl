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

mod fmod;
pub mod lane;
mod vendor;

pub use fmod::fmod;

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

/// The error function, `2/sqrt(pi) * integral from 0 to x of exp(-t^2) dt`.
/// Added after version 1 shipped, without changing any existing result
/// (ADR-0088: the standard normal cdf of `TrueSkill` ratings).
#[must_use]
#[inline]
pub fn erf(x: f64) -> f64 {
    vendor::erf(x)
}

/// The complementary error function, `1 - erf(x)`, accurate where `erf(x)`
/// is close to 1.
#[must_use]
#[inline]
pub fn erfc(x: f64) -> f64 {
    vendor::erfc(x)
}

/// `sqrt(x^2 + y^2)` without undue overflow or underflow.
#[must_use]
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    vendor::hypot(x, y)
}

/// The math version 1 reference table (`tools/math-oracle`, libm 0.2.16),
/// for this crate's tests and, with `test-hooks`, other crates' tests.
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub mod vectors;

#[cfg(test)]
mod tests {
    use super::vectors::{
        ATAN, ATAN2, COS, ERF, ERF_EDGE, ERF_INPUTS, ERFC, ERFC_EDGE, EXP, HYPOT, INPUTS, LN, SIN,
        TAN, TANH,
    };
    use super::{atan, atan2, cos, erf, erfc, exp, hypot, ln, sin, tan, tanh};

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

    fn check_on(name: &str, f: fn(f64) -> f64, inputs: &[u64], expected: &[u64]) {
        assert_eq!(inputs.len(), expected.len());
        for (&x, &want) in inputs.iter().zip(expected) {
            let x = f64::from_bits(x);
            let got = f(x);
            assert!(
                same(got, want),
                "{name}({x:e}) = {got:e}, want {:e}",
                f64::from_bits(want)
            );
        }
    }

    #[test]
    fn test_math_erf_matches_libm_0_2_16() {
        check_unary("erf", erf, ERF);
        check_on("erf", erf, ERF_INPUTS, ERF_EDGE);
    }

    #[test]
    fn test_math_erfc_matches_libm_0_2_16() {
        check_unary("erfc", erfc, ERFC);
        check_on("erfc", erfc, ERF_INPUTS, ERFC_EDGE);
    }

    /// `MATH_VERSION` 1 names the committed reference table in `vectors`
    /// (tools/math-oracle, libm 0.2.16) that every `test_math_*` test here
    /// checks bit for bit. The functions use only IEEE-754 `+ - * /` and bit
    /// operations, so the results do not depend on the ISA (ADR-0008); the
    /// cross-ISA check is CI running these tests on x86-64 and aarch64. A
    /// new version keeps this table and adds its own.
    #[test]
    fn test_math_version_1_is_the_libm_0_2_16_table() {
        assert_eq!(super::MATH_VERSION, 1);
        // The table's shape: one output per input for every unary function.
        for table in [SIN, COS, TAN, ATAN, EXP, LN, TANH, ERF, ERFC] {
            assert_eq!(table.len(), INPUTS.len());
        }
        assert_eq!(ERF_EDGE.len(), ERF_INPUTS.len());
        assert_eq!(ERFC_EDGE.len(), ERF_INPUTS.len());
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
