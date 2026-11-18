// SPDX-License-Identifier: MIT OR Apache-2.0

//! An exact decimal parser for the fixed-point types (ADR-0065 item 3):
//! `Fx32::from_decimal` and `Fx64::from_decimal`. Integer arithmetic only,
//! no floating point, so a config value given as a decimal string means
//! the same raw value on every platform.
//!
//! Grammar: an optional sign (`-` or `+`), one or more ASCII digits, and an
//! optional `.` followed by one or more ASCII digits. No exponent,
//! separators or whitespace. A value is accepted only when it is exactly
//! representable: `value * 2^frac` is an integer that fits the raw type.

use core::fmt;

/// Why a decimal string is not an exact fixed-point value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecimalError {
    /// Not a decimal of the accepted grammar.
    Syntax,
    /// Outside the type's range.
    OutOfRange,
    /// In range, but not a multiple of `2^-FRAC_BITS`.
    Inexact,
}

impl fmt::Display for DecimalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Syntax => "not a decimal number (digits, an optional sign and fraction)",
            Self::OutOfRange => "outside the fixed-point range",
            Self::Inexact => "not exactly representable in the fixed-point format",
        })
    }
}

impl core::error::Error for DecimalError {}

/// Most integer digits read before the value is out of range for any
/// fixed-point type (`Fx64` holds below 2^31, ten digits).
const MAX_INT_DIGITS: usize = 30;

/// The raw value `s * 2^frac`, exact, as `i128`; the caller narrows it to
/// the raw type. `frac` is at most 32.
pub(crate) fn parse(s: &str, frac: u32) -> Result<i128, DecimalError> {
    let (negative, digits) = match s.as_bytes() {
        [b'-', rest @ ..] => (true, rest),
        [b'+', rest @ ..] | rest => (false, rest),
    };
    let (int, fraction) = match digits.iter().position(|&b| b == b'.') {
        Some(dot) => {
            let (int, fraction) = digits.split_at(dot);
            match fraction {
                [_, rest @ ..] if !rest.is_empty() => (int, rest),
                _ => return Err(DecimalError::Syntax),
            }
        }
        None => (digits, &[][..]),
    };
    if int.is_empty() || !int.iter().chain(fraction).all(u8::is_ascii_digit) {
        return Err(DecimalError::Syntax);
    }
    // Leading zeros of the integer part and trailing zeros of the fraction
    // change nothing.
    let int = match int.iter().position(|&b| b != b'0') {
        Some(first) => int.get(first..).unwrap_or_default(),
        None => &[],
    };
    let fraction = match fraction.iter().rposition(|&b| b != b'0') {
        Some(last) => fraction.get(..=last).unwrap_or_default(),
        None => &[],
    };
    if int.len() > MAX_INT_DIGITS {
        return Err(DecimalError::OutOfRange);
    }
    // After trimming, the last fraction digit is not 0, so the fraction
    // `F / 10^k` reduces to an odd numerator over `2^k` at best: exact only
    // if `k <= frac`.
    let k = u32::try_from(fraction.len()).map_err(|_| DecimalError::Inexact)?;
    if k > frac {
        return Err(DecimalError::Inexact);
    }
    let int = int
        .iter()
        .try_fold(0i128, |acc, &d| {
            acc.checked_mul(10)?.checked_add(i128::from(d - b'0'))
        })
        .ok_or(DecimalError::OutOfRange)?;
    // `F < 10^k <= 10^32 < 2^107`: no overflow.
    let f = fraction
        .iter()
        .try_fold(0u128, |acc, &d| {
            acc.checked_mul(10)?.checked_add(u128::from(d - b'0'))
        })
        .ok_or(DecimalError::Inexact)?;
    // `F / 10^k * 2^frac = (F / 5^k) * 2^(frac - k)`: exact iff `5^k | F`.
    let five = 5u128.checked_pow(k).ok_or(DecimalError::Inexact)?;
    if f % five != 0 {
        return Err(DecimalError::Inexact);
    }
    let frac_raw = (f / five)
        .checked_shl(frac - k)
        .and_then(|v| i128::try_from(v).ok())
        .ok_or(DecimalError::Inexact)?;
    let magnitude = int
        .checked_shl(frac)
        .filter(|v| v >> frac == int)
        .and_then(|v| v.checked_add(frac_raw))
        .ok_or(DecimalError::OutOfRange)?;
    Ok(if negative { -magnitude } else { magnitude })
}

#[cfg(test)]
mod tests {
    use super::{DecimalError, parse};

    #[test]
    fn test_decimal_parse_exact_values() {
        assert_eq!(parse("0", 16), Ok(0));
        assert_eq!(parse("-0", 16), Ok(0));
        assert_eq!(parse("+1", 16), Ok(1 << 16));
        assert_eq!(parse("1.5", 16), Ok(3 << 15));
        assert_eq!(parse("-2.25", 16), Ok(-(9 << 14)));
        assert_eq!(parse("007.50000", 16), Ok(15 << 15));
        // 2^-16 needs 16 fraction digits.
        assert_eq!(parse("0.0000152587890625", 16), Ok(1));
        assert_eq!(parse("0.00000000023283064365386962890625", 32), Ok(1));
    }

    #[test]
    fn test_decimal_parse_inexact_values() {
        assert_eq!(parse("0.1", 16), Err(DecimalError::Inexact));
        assert_eq!(parse("0.00000762939453125", 16), Err(DecimalError::Inexact));
        assert_eq!(
            parse("0.000000000116415321826934814453125", 32),
            Err(DecimalError::Inexact)
        );
    }

    #[test]
    fn test_decimal_parse_syntax_errors() {
        for bad in [
            "", "-", "+", ".5", "1.", "1e3", "1_000", " 1", "1 ", "--1", "0x10", "1.2.3", "١",
        ] {
            assert_eq!(parse(bad, 16), Err(DecimalError::Syntax), "{bad:?}");
        }
    }

    #[test]
    fn test_decimal_parse_huge_integer_is_out_of_range() {
        assert_eq!(parse(&"9".repeat(31), 16), Err(DecimalError::OutOfRange));
        assert_eq!(
            parse(&format!("1{}", "0".repeat(29)), 32),
            Err(DecimalError::OutOfRange)
        );
    }
}
