// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact floating-point remainder, truncating toward zero: the integer
//! shift-subtract method on the significands, with no division. Written from
//! the IEEE-754 definition; structured like `fmod.c` of musl libc (MIT, see
//! `vendor/LICENSE.txt` for the musl notice). Rust's `%` lowers to the
//! platform's `fmod` and is not used in kernels (ADR-0008).

const SIGN: u64 = 1 << 63;
const IMPLICIT: u64 = 1 << 52;
const EXPONENT_MASK: u64 = 0x7ff;

/// The significand, with its implicit bit, and the exponent, normalized.
#[inline]
fn normalize(bits: u64) -> (u64, i32) {
    let exponent = i32::try_from((bits >> 52) & EXPONENT_MASK).unwrap_or(0);
    if exponent != 0 {
        return ((bits & (IMPLICIT - 1)) | IMPLICIT, exponent);
    }
    // A subnormal: shift until the implicit position is set. `bits` is
    // nonzero here, the caller having handled zero.
    let magnitude = bits & !SIGN;
    let shift = i32::try_from(magnitude.leading_zeros()).unwrap_or(0) - 11;
    (magnitude << shift, 1 - shift)
}

/// `x - n * y` for the integer `n` nearest zero such that the result has the
/// sign of `x`, computed exactly.
///
/// `fmod(±0, y)` is `±0` for nonzero `y`; `fmod(x, ±inf)` is `x` for finite
/// `x`; `y == ±0`, infinite `x` and any NaN operand give the canonical NaN. A
/// zero result has the sign of `x`.
#[must_use]
pub fn fmod(x: f64, y: f64) -> f64 {
    let (ux, uy) = (x.to_bits(), y.to_bits());
    let sign = ux & SIGN;
    let (ax, ay) = (ux & !SIGN, uy & !SIGN);
    if ay == 0 || y.is_nan() || !x.is_finite() {
        return f64::NAN;
    }
    if ax < ay {
        return x;
    }
    if ax == ay {
        return f64::from_bits(sign);
    }
    let (mut mx, mut ex) = normalize(ax);
    let (my, ey) = normalize(ay);
    // `ax > ay`, so `ex >= ey`, and `mx` keeps its implicit bit or more.
    while ex > ey {
        if mx >= my {
            mx -= my;
            if mx == 0 {
                return f64::from_bits(sign);
            }
        }
        mx <<= 1;
        ex -= 1;
    }
    if mx >= my {
        mx -= my;
        if mx == 0 {
            return f64::from_bits(sign);
        }
    }
    let shift = i32::try_from(mx.leading_zeros()).unwrap_or(0) - 11;
    mx <<= shift;
    ex -= shift;
    let magnitude = if ex > 0 {
        (mx & (IMPLICIT - 1)) | (u64::try_from(ex).unwrap_or(0) << 52)
    } else {
        mx >> (1 - ex)
    };
    f64::from_bits(sign | magnitude)
}

#[cfg(test)]
mod tests {
    use super::fmod;

    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    #[test]
    fn test_fmod_special_values_follow_the_contract() {
        assert_eq!(fmod(0.0, 3.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(fmod(-0.0, 3.0).to_bits(), (-0.0f64).to_bits());
        assert!(fmod(1.0, 0.0).is_nan() && fmod(1.0, -0.0).is_nan());
        assert!(fmod(f64::INFINITY, 2.0).is_nan() && fmod(f64::NEG_INFINITY, 2.0).is_nan());
        assert!(fmod(f64::NAN, 2.0).is_nan() && fmod(2.0, f64::NAN).is_nan());
        assert_eq!(fmod(5.5, f64::INFINITY), 5.5);
        assert_eq!(fmod(-5.5, f64::NEG_INFINITY), -5.5);
        assert_eq!(fmod(3.0, 3.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(fmod(-3.0, 3.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(fmod(2.0, 3.0), 2.0);
        assert_eq!(fmod(7.5, 2.0), 1.5);
        assert_eq!(fmod(-7.5, 2.0), -1.5);
        assert_eq!(fmod(7.5, -2.0), 1.5);
    }

    #[test]
    fn test_fmod_matches_the_platform_on_edges() {
        let edges = [
            0.0,
            -0.0,
            f64::from_bits(1),
            f64::from_bits(2),
            f64::from_bits(0x000f_ffff_ffff_ffff),
            f64::MIN_POSITIVE,
            1.0,
            core::f64::consts::PI,
            2.0 * core::f64::consts::PI,
            1e300,
            f64::MAX,
            f64::INFINITY,
            f64::NAN,
        ];
        for a in edges {
            for b in edges {
                for (x, y) in [(a, b), (-a, b), (a, -b), (-a, -b)] {
                    assert!(same(fmod(x, y), x % y), "{x:e} {y:e}");
                }
            }
        }
    }

    #[test]
    fn test_fmod_matches_the_platform_on_random_bits() {
        let mut s = 0x1234_5678_9abc_def1_u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..300_000 {
            let (x, y) = (f64::from_bits(next()), f64::from_bits(next()));
            assert!(
                same(fmod(x, y), x % y),
                "{:x} {:x}",
                x.to_bits(),
                y.to_bits()
            );
            // Close exponents, the common case.
            let (x, y) = (
                f64::from_bits((next() & 0x800f_ffff_ffff_ffff) | ((0x3f0 + next() % 32) << 52)),
                f64::from_bits((next() & 0x800f_ffff_ffff_ffff) | ((0x3f0 + next() % 32) << 52)),
            );
            assert!(
                same(fmod(x, y), x % y),
                "{:x} {:x}",
                x.to_bits(),
                y.to_bits()
            );
        }
    }
}
