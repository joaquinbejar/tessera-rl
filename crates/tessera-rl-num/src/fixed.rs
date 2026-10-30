// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared implementation of the fixed-point types (SPEC 8.2, ADR-0008).
//!
//! Rules, identical for `Fx32` (Q16.16) and `Fx64` (Q32.32):
//!
//! - addition and subtraction are checked by default; `wrapping_*` and
//!   `saturating_*` are explicitly named alternatives for game rules;
//! - multiplication uses a full-width intermediate and an arithmetic right
//!   shift, so it rounds toward negative infinity;
//! - division uses a full-width intermediate and truncates toward zero;
//! - conversion from float rounds half to even and saturates at the bounds
//!   (NaN converts to zero), or is rejected by `checked_from_f64`;
//! - conversion to `f64` (`to_f64`, defined per type) is exact whenever
//!   the value is representable.
//!
//! No operator traits are implemented: every operation names its overflow
//! behaviour.

macro_rules! fixed_point {
    (
        $(#[$doc:meta])*
        $name:ident, $raw:ty, $wide:ty, $frac:expr, scale = $scale:expr, bound = $bound:expr
    ) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        #[repr(transparent)]
        pub struct $name($raw);

        impl $name {
            /// Number of fractional bits.
            pub const FRAC_BITS: u32 = $frac;
            /// Zero.
            pub const ZERO: Self = Self(0);
            /// One.
            pub const ONE: Self = Self(1 << $frac);
            /// The smallest positive value, `2^-FRAC_BITS`.
            pub const EPSILON: Self = Self(1);
            /// The smallest representable value.
            pub const MIN: Self = Self(<$raw>::MIN);
            /// The largest representable value.
            pub const MAX: Self = Self(<$raw>::MAX);

            // 2^FRAC_BITS, exact.
            const SCALE: f64 = $scale;
            // 2^(raw bits - 1), exact: the exclusive upper bound of the raw
            // range.
            const RAW_BOUND: f64 = $bound;

            /// Builds a value from its raw two's complement representation.
            #[must_use]
            #[inline]
            pub const fn from_raw(raw: $raw) -> Self {
                Self(raw)
            }

            /// Returns the raw representation.
            #[must_use]
            #[inline]
            pub const fn to_raw(self) -> $raw {
                self.0
            }

            /// Converts an integer, or returns `None` if it is out of range.
            #[must_use]
            #[inline]
            pub const fn checked_from_int(value: $raw) -> Option<Self> {
                match value.checked_mul(1 << $frac) {
                    Some(raw) => Some(Self(raw)),
                    None => None,
                }
            }

            /// Checked addition.
            #[must_use]
            #[inline]
            pub const fn checked_add(self, rhs: Self) -> Option<Self> {
                match self.0.checked_add(rhs.0) {
                    Some(raw) => Some(Self(raw)),
                    None => None,
                }
            }

            /// Checked subtraction.
            #[must_use]
            #[inline]
            pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
                match self.0.checked_sub(rhs.0) {
                    Some(raw) => Some(Self(raw)),
                    None => None,
                }
            }

            /// Checked negation.
            #[must_use]
            #[inline]
            pub const fn checked_neg(self) -> Option<Self> {
                match self.0.checked_neg() {
                    Some(raw) => Some(Self(raw)),
                    None => None,
                }
            }

            /// Addition modulo the raw width, for game rules that document it.
            #[must_use]
            #[inline]
            pub const fn wrapping_add(self, rhs: Self) -> Self {
                Self(self.0.wrapping_add(rhs.0))
            }

            /// Subtraction modulo the raw width, for game rules that document it.
            #[must_use]
            #[inline]
            pub const fn wrapping_sub(self, rhs: Self) -> Self {
                Self(self.0.wrapping_sub(rhs.0))
            }

            /// Addition clamped to `[MIN, MAX]`, for game rules that document it.
            #[must_use]
            #[inline]
            pub const fn saturating_add(self, rhs: Self) -> Self {
                Self(self.0.saturating_add(rhs.0))
            }

            /// Subtraction clamped to `[MIN, MAX]`, for game rules that document it.
            #[must_use]
            #[inline]
            pub const fn saturating_sub(self, rhs: Self) -> Self {
                Self(self.0.saturating_sub(rhs.0))
            }

            #[inline]
            const fn wide_mul(self, rhs: Self) -> $wide {
                // Arithmetic shift: rounds toward negative infinity.
                ((self.0 as $wide) * (rhs.0 as $wide)) >> $frac
            }

            #[inline]
            const fn narrow(wide: $wide) -> Option<Self> {
                if wide < <$raw>::MIN as $wide || wide > <$raw>::MAX as $wide {
                    None
                } else {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "range checked above"
                    )]
                    Some(Self(wide as $raw))
                }
            }

            /// Checked multiplication, rounding toward negative infinity.
            #[must_use]
            #[inline]
            pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
                Self::narrow(self.wide_mul(rhs))
            }

            /// Multiplication clamped to `[MIN, MAX]`, rounding toward
            /// negative infinity.
            #[must_use]
            #[inline]
            pub const fn saturating_mul(self, rhs: Self) -> Self {
                let wide = self.wide_mul(rhs);
                match Self::narrow(wide) {
                    Some(v) => v,
                    None if wide < 0 => Self::MIN,
                    None => Self::MAX,
                }
            }

            /// Multiplication modulo the raw width, rounding toward negative
            /// infinity, for game rules that document it.
            #[must_use]
            #[inline]
            #[expect(
                clippy::cast_possible_truncation,
                reason = "wrapping is the documented behaviour"
            )]
            pub const fn wrapping_mul(self, rhs: Self) -> Self {
                Self(self.wide_mul(rhs) as $raw)
            }

            /// Checked division, truncating toward zero. Returns `None` for a
            /// zero divisor or an out-of-range quotient.
            #[must_use]
            #[inline]
            pub const fn checked_div(self, rhs: Self) -> Option<Self> {
                if rhs.0 == 0 {
                    return None;
                }
                // Integer division of the widened values truncates toward zero.
                Self::narrow(((self.0 as $wide) << $frac) / (rhs.0 as $wide))
            }

            /// Converts from `f64`, rounding half to even. Returns `None` for
            /// NaN and for values outside `[MIN, MAX]` after rounding.
            #[must_use]
            #[inline]
            pub fn checked_from_f64(value: f64) -> Option<Self> {
                let scaled = (value * Self::SCALE).round_ties_even();
                if scaled.is_nan() || scaled < -Self::RAW_BOUND || scaled >= Self::RAW_BOUND {
                    None
                } else {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "integral and range checked above"
                    )]
                    Some(Self(scaled as $raw))
                }
            }

            /// Converts from `f64`, rounding half to even and saturating at
            /// `[MIN, MAX]`. NaN converts to zero.
            #[must_use]
            #[inline]
            pub fn saturating_from_f64(value: f64) -> Self {
                if value.is_nan() {
                    return Self::ZERO;
                }
                match Self::checked_from_f64(value) {
                    Some(v) => v,
                    None if value < 0.0 => Self::MIN,
                    None => Self::MAX,
                }
            }

            /// Converts from `f32` (exactly widened to `f64`), rounding half
            /// to even. Returns `None` for NaN and out-of-range values.
            #[must_use]
            #[inline]
            pub fn checked_from_f32(value: f32) -> Option<Self> {
                Self::checked_from_f64(f64::from(value))
            }

            /// Converts from `f32`, rounding half to even and saturating.
            /// NaN converts to zero.
            #[must_use]
            #[inline]
            pub fn saturating_from_f32(value: f32) -> Self {
                Self::saturating_from_f64(f64::from(value))
            }
        }
    };
}

pub(crate) use fixed_point;
